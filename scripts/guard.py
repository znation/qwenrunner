#!/usr/bin/env python3
"""Guarded subprocess runner for Apple Silicon (Metal / unified memory).

The Intel build of this harness existed to protect a *display* GPU: it read
sysfs DRM counters (and NVML for the eGPU) every 50 ms and SIGKILLed the tree
when VRAM exceeded 5000 MiB. None of that survives the port:

  * macOS has no separate VRAM. Metal allocates from the same 128 GB unified
    pool the OS uses, so "VRAM" is just RSS plus GPU page tables.
  * There is no `/sys/class/drm`, no `/proc`, no O_DIRECT. Every counter here
    comes from a supported macOS API (`host_statistics`, `sysctl`, `ps`,
    `ioreg`).

What is still worth guarding against is *swap*: with 128 GB unified and a
16 GB model there is no cap to breach, but a run that spills to swap is a run
whose latency numbers are worthless. So the cap is reinterpreted -- kill only
if `--swap-cap-mib` of new swap appears (default 0: never).

The measurement fields are kept because the old traps are still real:

  * `bg_cpu_pct` -- host CPU busy excluding the guarded tree. With `-ngl 99`
    the Metal device runs the matmuls, so CPU contention matters far less than
    it did on the Intel host, but a contended arm is still detectable.
  * `gpu_busy_pct_peak` -- peak `Device Utilization %` from ioreg. This is the
    Mac analogue of background CPU load: another GPU tenant (a browser, a
    video call) steals Metal time even though our own process looks idle.
  * `gpu_mem_peak_mib` -- peak "In use system memory" across accelerators, so a
    phantom baseline left by a just-exited arm is visible.

Stdlib only.
"""

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

# ------------------------------------------------------------------ probes

POLL_S = 1.0

_TIME_RX = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def _parse_cputime(s):
    """Parse `ps -o time=` ([[dd-]hh:]mm:ss[.ss]) into seconds, or None."""
    m = _TIME_RX.match(s.strip())
    if not m:
        return None
    days, hours, mins, secs = m.groups()
    return (int(days or 0) * 86400 + int(hours or 0) * 3600
            + int(mins) * 60 + float(secs))


_PAGE = os.sysconf("SC_PAGE_SIZE")
_NCPU = os.sysconf("SC_NPROCESSORS_ONLN")


def ps_snapshot():
    """Return {pid: (ppid, rss_kib, cpu_s)} for every process.

    One `ps -A` per tick is the whole CPU story on macOS. `host_statistics
    (HOST_CPU_LOAD_INFO)` is the /proc/stat analogue and looks like the obvious
    primitive, but on this kernel the idle tick counter is frozen -- it never
    advances, so a busy fraction computed from it reads 100 % on an idle
    machine. `ps -o time=` carries each process's CPU time in seconds, which
    needs no tick rate and sums to host CPU time directly. Returns {} on
    failure.
    """
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,time="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    info = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        info[pid] = (ppid, rss, _parse_cputime(parts[3]))
    return info


def _tree_of(info, root_pid):
    children = {}
    for pid, (ppid, _, _) in info.items():
        children.setdefault(ppid, []).append(pid)
    tree, stack = set(), [root_pid]
    while stack:
        p = stack.pop()
        if p in tree:
            continue
        tree.add(p)
        stack.extend(children.get(p, ()))
    return tree


def proc_tree(root_pid, info=None):
    """Return ({pid: (ppid, rss_kib, cpu_s)}, set(pids)) for root_pid's tree."""
    if info is None:
        info = ps_snapshot()
    if not info:
        return {}, {root_pid}
    return info, _tree_of(info, root_pid)


def tree_rss_kib(root_pid):
    info, tree = proc_tree(root_pid)
    return sum(info[p][1] for p in tree if p in info)


def tree_cpu_s(root_pid, info=None):
    if info is None:
        info = ps_snapshot()
    _, tree = proc_tree(root_pid, info)
    return sum((info[p][2] or 0.0) for p in tree if p in info)


def host_cpu_s(info=None):
    """Host-wide CPU seconds consumed by all live processes."""
    if info is None:
        info = ps_snapshot()
    return sum((v[2] or 0.0) for v in info.values())


_RE_SWAP = re.compile(r"used\s*=\s*([0-9.]+)M")


def swap_used_mib():
    """Host swap in use (MiB) from `sysctl vm.swapusage`, or None."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _RE_SWAP.search(out)
    return None if m is None else float(m.group(1))


_RE_VMSTAT = {}


def _fill_vmstat_patterns():
    if _RE_VMSTAT:
        return
    for key in ("Pages free", "Pages active", "Pages inactive",
                "Pages wired down", "Pages occupied by compressor"):
        _RE_VMSTAT[key] = re.compile(re.escape(key) + r":\s+(\d+)\.")


def mem_used_mib():
    """Host memory in use (MiB): active + wired + compressed.

    `free` alone is misleading on macOS -- inactive and purgeable pages are
    reclaimable, so the honest "is the machine under pressure" number is the
    non-reclaimable set. Returns None if vm_stat is unavailable.
    """
    _fill_vmstat_patterns()
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pages = {}
    for key, rx in _RE_VMSTAT.items():
        m = rx.search(out)
        if not m:
            return None
        pages[key] = int(m.group(1))
    used = (pages["Pages active"] + pages["Pages wired down"]
            + pages["Pages occupied by compressor"])
    return used * _PAGE / 1048576.0


_RE_GPU_UTIL = re.compile(r'"Device Utilization %"=(\d+)')
_RE_GPU_MEM = re.compile(r'"In use system memory"=(\d+)')


def gpu_stats():
    """Return (busy_pct, used_mib) for the busiest IOAccelerator, or (None, None).

    `ioreg -c IOAccelerator` costs ~30 ms, so it is sampled once per tick and
    shared by every consumer rather than called per metric.
    """
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0",
                              "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    utils = [int(x) for x in _RE_GPU_UTIL.findall(out)]
    mems = [int(x) for x in _RE_GPU_MEM.findall(out)]
    util = max(utils) if utils else None
    # Sum across accelerators: one process's footprint may span them, and the
    # old harness's bug was reading a *stale client* as a fresh baseline.
    mem = sum(mems) / 1048576.0 if mems else None
    return util, mem


def settle_gpu(target_mib, timeout=60.0, slack_mib=250.0):
    """Block until accelerator memory stops falling and is near target.

    A just-exited arm releases its Metal buffers asynchronously; sampling a
    baseline in that window makes the next arm read a phantom footprint. If
    the reading stabilises above target the wait is extended and then returns
    anyway -- another tenant may legitimately hold that much.
    """
    deadline = time.time() + timeout
    _, v = gpu_stats()
    while v is not None and v > target_mib + slack_mib and time.time() < deadline:
        time.sleep(1.0)
        _, v = gpu_stats()
    return v


# ------------------------------------------------------------------ monitor


class Monitor(threading.Thread):
    """Polls host + tree counters once a second; SIGKILLs on a swap breach.

    Attributes after finish(): rss_peak_kib, swap_baseline, swap_peak,
    swap_delta, mem_peak, gpu_peak, gpu_util_peak, cpu_busy_pct, bg_cpu_pct,
    tree_cpu_pct, violation, samples.
    """

    def __init__(self, pid, swap_cap_mib=0.0, poll_s=POLL_S):
        super().__init__(daemon=True)
        self.pid = pid
        self.swap_cap_mib = swap_cap_mib or 0.0
        self.poll_s = poll_s
        self.rss_peak_kib = 0
        self.mem_peak = None
        self.swap_baseline = swap_used_mib()
        self.swap_peak = self.swap_baseline
        self.gpu_peak = None
        self.gpu_util_peak = 0
        self.cpu_busy_pct = 0.0
        self.tree_cpu_pct = 0.0
        self.bg_cpu_pct = 0.0
        self.violation = False
        self.samples = 0
        self._stop_evt = threading.Event()
        self._host_cpu0 = None
        self._tree_cpu0 = None
        self._t0 = None

    def run(self):
        self._t0 = time.time()
        info = ps_snapshot()
        self._host_cpu0 = host_cpu_s(info)
        self._tree_cpu0 = tree_cpu_s(self.pid, info)
        while not self._stop_evt.wait(self.poll_s):
            self._tick()

    def _tick(self):
        self.samples += 1
        info = ps_snapshot()
        _, tree = proc_tree(self.pid, info)
        rss = sum(info[p][1] for p in tree if p in info)
        self.rss_peak_kib = max(self.rss_peak_kib, rss)
        tree_cpu = sum((info[p][2] or 0.0) for p in tree if p in info)

        if not tree:
            # Process exited: nothing left to measure, and the remaining
            # counters would attribute other tenants' work to this run.
            self._stop_evt.set()
            return

        mem = mem_used_mib()
        if mem is not None:
            self.mem_peak = mem if self.mem_peak is None else max(self.mem_peak, mem)

        swap = swap_used_mib()
        if swap is not None:
            if self.swap_baseline is None:
                self.swap_baseline = swap
            self.swap_peak = swap if self.swap_peak is None else max(self.swap_peak, swap)
            delta = self.swap_peak - self.swap_baseline
            if self.swap_cap_mib and delta > self.swap_cap_mib:
                self.violation = True
                self._kill()

        util, gmem = gpu_stats()
        if util is not None:
            self.gpu_util_peak = max(self.gpu_util_peak, util)
        if gmem is not None:
            self.gpu_peak = gmem if self.gpu_peak is None else max(self.gpu_peak, gmem)

        wall = time.time() - self._t0
        if wall > 0 and self._host_cpu0 is not None:
            # CPU seconds consumed / (wall x cores) = fraction of the machine's
            # capacity. Same normalization as the Intel harness, so numbers are
            # comparable across the port.
            self.cpu_busy_pct = 100.0 * (host_cpu_s(info) - self._host_cpu0) / (wall * _NCPU)
            self.tree_cpu_pct = 100.0 * (tree_cpu - self._tree_cpu0) / (wall * _NCPU)
            self.bg_cpu_pct = max(0.0, self.cpu_busy_pct - self.tree_cpu_pct)

    def _kill(self):
        try:
            pgid = os.getpgid(self.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass

    def snapshot(self):
        return {
            "rss_peak_mib": round(self.rss_peak_kib / 1024.0, 1),
            "mem_peak_mib": None if self.mem_peak is None else round(self.mem_peak, 1),
            "swap_baseline_mib": None if self.swap_baseline is None else round(self.swap_baseline, 1),
            "swap_peak_mib": None if self.swap_peak is None else round(self.swap_peak, 1),
            "swap_delta_mib": None if (self.swap_peak is None or self.swap_baseline is None)
                               else round(self.swap_peak - self.swap_baseline, 1),
            "swap_violation": self.violation,
            "gpu_mem_peak_mib": None if self.gpu_peak is None else round(self.gpu_peak, 1),
            "gpu_busy_pct_peak": self.gpu_util_peak,
            "cpu_busy_pct": round(self.cpu_busy_pct, 1),
            "tree_cpu_pct": round(self.tree_cpu_pct, 1),
            "bg_cpu_pct": round(self.bg_cpu_pct, 1),
            "samples": self.samples,
        }

    def finish(self):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=3.0)
        return self.snapshot()


# ------------------------------------------------------------------ runner


def run_guarded(cmd, swap_cap_mib=0.0, timeout=None, log_path=None, cwd=None,
                env=None, quiet=False):
    """Run `cmd` in its own process group; monitor it; return a result dict.

    SIGINT/SIGTERM to this process are forwarded to the child's group, so a
    Ctrl-C on serve.sh stops llama-server rather than orphaning it.
    """
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w") if log_path else subprocess.DEVNULL
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=cwd, env=env,
                         start_new_session=True)
    mon = Monitor(p.pid, swap_cap_mib=swap_cap_mib)
    mon.start()

    timed_out = False
    forward = []

    def _forward(signum, _frame):
        forward.append(signum)
        try:
            os.killpg(os.getpgid(p.pid), signum)
        except OSError:
            pass

    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old[sig] = signal.signal(sig, _forward)
        except (ValueError, OSError):
            pass

    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except OSError:
            pass
        p.wait(timeout=30)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        snap = mon.finish()
        if not quiet:
            print("guard: " + json.dumps(snap, sort_keys=True), file=sys.stderr)
        log_f.close()

    stdout = ""
    if log_path:
        try:
            with open(log_path, errors="replace") as f:
                stdout = f.read()
        except OSError:
            pass

    res = {
        "exit_code": p.returncode,
        "wall_s": round(time.time() - t0, 3),
        "timed_out": timed_out,
        "signals": forward,
        "log": log_path,
        "stdout": stdout,
    }
    res.update(snap)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap-cap-mib", type=float, default=0.0,
                    help="SIGKILL the group if this much new swap appears "
                         "(0 = never; the default, since 128 GB unified holds "
                         "this model whole)")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (use: guard.py [opts] -- cmd args...)")

    log = args.log
    if log and args.label:
        root, ext = os.path.splitext(log)
        log = "%s.%s%s" % (root, args.label, ext)

    res = run_guarded(cmd, swap_cap_mib=args.swap_cap_mib, timeout=args.timeout,
                      log_path=log, quiet=True)
    print(json.dumps({k: v for k, v in res.items() if k != "stdout"}, sort_keys=True))
    return 0 if res["exit_code"] == 0 and not res["swap_violation"] else 1


if __name__ == "__main__":
    sys.exit(main())
