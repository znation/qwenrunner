#!/usr/bin/env python3
"""VRAM/RSS-guarded subprocess runner with per-process VRAM attribution.

Hard rule for this machine: the display GPU (Radeon Pro Vega 48, 8 GiB total)
is shared with the compositor, so total `mem_info_vram_used` MUST stay under
5 GB. This module runs a command, polls VRAM while it runs, and kills it if the
cap is exceeded, so no experiment can wedge the desktop.

Three numbers are recorded, because they answer different questions:

  * `vram_peak_mib`   -- system-wide `mem_info_vram_used`. This is the real
                         display constraint and is what the cap is enforced on.
  * `vram_delta_mib`  -- `vram_peak_mib` minus the system value sampled before
                         launch. Diagnostic only: the compositor reclaims under
                         pressure, so this can understate the app's own peak.
  * `client_vram_peak_mib` -- VRAM attributable to the guarded process tree,
                         summed from `/proc/<pid>/fdinfo/*` amdgpu client
                         counters, deduped by `drm-client-id`. This is the
                         model's actual footprint; the system counter also
                         contains the desktop, which drifts by >1 GiB.

The system and client numbers do not add up exactly: kernel-side allocations
(page tables, kernel buffers) have no DRM client, so
sum(clients) < mem_info_vram_used. Measured idle gap on this host is ~400 MiB
and stable.

Do NOT subtract a system-wide client baseline from `client_vram_peak_mib`: the
guarded tree has no DRM client before it execs, so its own baseline is zero by
construction and the peak already IS the footprint. (An earlier version did
subtract one and produced meaningless negative deltas.)

Stdlib only. Usable as a library (run_guarded) or as a CLI:

    scripts/guard.py [--cap-mib 5000] [--timeout 600] [--out results/guard.jsonl] \
        [--label NAME] -- <cmd> [args...]
"""

import argparse
import ctypes
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time

DRM = "/sys/class/drm/card0/device"
VRAM_USED = DRM + "/mem_info_vram_used"
VRAM_TOTAL = DRM + "/mem_info_vram_total"
GTT_USED = DRM + "/mem_info_gtt_used"
GTT_TOTAL = DRM + "/mem_info_gtt_total"

PCI_DEV = "0000:03:00.0"

DEFAULT_CAP_MIB = 5000.0
POLL_S = 0.05


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return None


def _mib(v):
    return None if v is None else v / 1048576.0


def vram_used_mib():
    return _mib(_read_int(VRAM_USED))


def vram_total_mib():
    return _mib(_read_int(VRAM_TOTAL))


def gtt_used_mib():
    return _mib(_read_int(GTT_USED))


class _NvmlMem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


_NVML = None
_NVML_TRIED = False


def nvidia_vram_mib():
    """Return (used_mib, total_mib) for NVIDIA GPU 0, or None if unavailable.

    The display GPU's VRAM is visible via sysfs (`card0`), but a second, discrete
    GPU — e.g. a Thunderbolt eGPU — is not. `nvidia-smi` costs ~150 ms per
    invocation, far too much to poll at 20 Hz, whereas NVML is a shared library
    so the query is a few microseconds.
    """
    global _NVML, _NVML_TRIED
    if not _NVML_TRIED:
        _NVML_TRIED = True
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() == 0:
                _NVML = lib
        except (OSError, AttributeError):
            _NVML = None
    if _NVML is None:
        return None
    try:
        h = ctypes.c_void_p()
        if _NVML.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return None
        m = _NvmlMem()
        if _NVML.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) != 0:
            return None
        return m.used / 1048576.0, m.total / 1048576.0
    except Exception:
        return None


def nvidia_util_pct():
    """GPU utilization % for NVIDIA device 0, or None. Cheap NVML query."""
    global _NVML, _NVML_TRIED
    if not _NVML_TRIED:
        nvidia_vram_mib()
    if _NVML is None:
        return None
    try:
        h = ctypes.c_void_p()
        if _NVML.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return None
        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        u = _Util()
        if _NVML.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) != 0:
            return None
        return u.gpu
    except Exception:
        return None


def drm_clients(pids=None):
    """Per-DRM-client VRAM/GTT usage in KiB, deduped by drm-client-id.

    amdgpu reports the same client totals for every fd of a process, so the
    dedupe is mandatory -- summing raw fdinfo entries over-counts by ~4x here.

    `pids` limits the scan to those processes. The unrestricted scan opens
    every fdinfo file of every process on the machine (thousands of files),
    which is enough CPU to perturb the run being measured; the guarded monitor
    therefore always passes its tree.

    Returns {client_id: {"pid", "vram_kib", "shared_vram_kib", "gtt_kib",
                         "shared_gtt_kib", "gfx_ns", "compute_ns"}}.
    """
    out = {}
    if pids is None:
        paths = glob.glob("/proc/[0-9]*/fdinfo/*")
    else:
        paths = []
        for p in pids:
            paths.extend(glob.glob("/proc/%d/fdinfo/*" % p))
    for path in paths:
        try:
            with open(path) as f:
                txt = f.read()
        except OSError:
            continue
        if PCI_DEV not in txt:
            continue
        vals = {}
        for line in txt.splitlines():
            if not line.startswith("drm-"):
                continue
            key, _, rest = line.partition(":")
            rest = rest.strip().split()
            if rest:
                vals[key.strip()] = rest[0]
        cid = vals.get("drm-client-id")
        if cid is None:
            continue
        pid = int(path.split("/")[2])

        def num(k):
            try:
                return int(vals.get(k, "0"))
            except ValueError:
                return 0

        rec = {
            "pid": pid,
            "vram_kib": num("drm-memory-vram"),
            "shared_vram_kib": num("drm-shared-vram"),
            "gtt_kib": num("drm-memory-gtt"),
            "shared_gtt_kib": num("drm-shared-gtt"),
        }
        prev = out.get(cid)
        # Same client seen via several fds: keep the largest reading, but
        # prefer a pid inside our own tree when both are present.
        if prev is None or rec["vram_kib"] + rec["gtt_kib"] > \
                prev["vram_kib"] + prev["gtt_kib"]:
            out[cid] = rec
    return out


_PAGE_KIB = os.sysconf("SC_PAGE_SIZE") // 1024


def _cpu_stat():
    """Return (busy_jiffies, total_jiffies) summed over all CPUs, or None."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
    except OSError:
        return None
    try:
        v = [int(x) for x in parts]
    except ValueError:
        return None
    idle = v[3] + (v[4] if len(v) > 4 else 0)   # idle + iowait
    return sum(v) - idle, sum(v)


def _proc_tree(root_pid):
    """Return ({pid: (ppid, rss_kib, cpu_jiffies)}, set_of_pids_in_tree)."""
    info = {}
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return info, set()
    for p in pids:
        try:
            with open("/proc/%d/stat" % p) as f:
                data = f.read()
        except OSError:
            continue
        # comm may contain spaces/parens; everything after the last ')' is safe.
        end = data.rfind(")")
        if end < 0:
            continue
        f_ = data[end + 2:].split()
        try:
            ppid = int(f_[1])
            cpu = int(f_[11]) + int(f_[12])   # utime + stime, in jiffies
            rss = int(f_[21]) * _PAGE_KIB
        except (IndexError, ValueError):
            continue
        info[p] = (ppid, rss, cpu)
    tree = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in tree:
            continue
        tree.add(pid)
        for p, (pp, _, _) in info.items():
            if pp == pid and p not in tree:
                stack.append(p)
    return info, tree


def _proc_tree_rss_kib(root_pid):
    info, tree = _proc_tree(root_pid)
    return sum(info.get(p, (0, 0))[1] for p in tree)


def client_vram_mib(pids=None):
    """VRAM (MiB) attributable to `pids` (all DRM clients if None)."""
    cl = drm_clients(pids)
    return sum(r["vram_kib"] + r["shared_vram_kib"] for r in cl.values()) / 1024.0


def client_gtt_mib(pids=None):
    """GTT (MiB) attributable to `pids` (all DRM clients if None)."""
    cl = drm_clients(pids)
    return sum(r["gtt_kib"] + r["shared_gtt_kib"] for r in cl.values()) / 1024.0


class Monitor(threading.Thread):
    """Polls VRAM (system + per-client) and child RSS; kills on cap breach."""

    def __init__(self, pid, cap_mib, poll_s=POLL_S):
        super().__init__(daemon=True)
        self.pid = pid
        self.cap_mib = cap_mib
        self.poll_s = poll_s
        self.stop_flag = threading.Event()
        self.vram_peak = 0.0
        self.client_peak = 0.0
        self.gtt_peak = 0.0
        self.rss_peak_kib = 0
        self.violation = False
        self.samples = 0
        self.killed_at = None
        self.top_clients = {}
        # Host CPU contention. Throughput on this box is CPU-bound whenever
        # layers are left on the CPU, so an arm measured while the desktop is
        # busy is not comparable to one measured while it is idle.
        self._cpu_prev = None
        self._cpu_busy = 0.0
        self._cpu_total = 0.0
        self._child_cpu0 = None
        self._child_cpu1 = None
        # Discrete GPU (e.g. Thunderbolt eGPU). Not covered by the sysfs counter
        # above, which tracks only the display GPU.
        self.nvidia_peak = 0.0
        self.nvidia_total = None

    # Ticks between the expensive /proc walks. The system-VRAM read that
    # enforces the cap is a single sysfs file and stays on every tick; walking
    # /proc/*/status and scanning fdinfo costs orders of magnitude more and
    # steals CPU from the very run being measured. Client VRAM is allocated at
    # model load and then plateaus, so 2 Hz sampling does not lose the peak.
    PROC_EVERY = 10

    def run(self):
        tick = 0
        while not self.stop_flag.is_set():
            cur = _cpu_stat()
            if cur is not None:
                if self._cpu_prev is not None:
                    db = cur[0] - self._cpu_prev[0]
                    dt = cur[1] - self._cpu_prev[1]
                    if dt > 0 and db >= 0:
                        self._cpu_busy += db
                        self._cpu_total += dt
                self._cpu_prev = cur
            v = vram_used_mib()
            nv = nvidia_vram_mib()
            if nv is not None:
                self.nvidia_peak = max(self.nvidia_peak, nv[0])
                self.nvidia_total = nv[1]
            if v is not None:
                self.vram_peak = max(self.vram_peak, v)
                self.samples += 1
                if v > self.cap_mib and not self.violation:
                    self.violation = True
                    self.killed_at = v
                    try:
                        os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                    except OSError:
                        pass
            if tick % self.PROC_EVERY == 0:
                info, tree = _proc_tree(self.pid)
                rss = sum(info.get(p, (0, 0))[1] for p in tree)
                if rss:
                    self.rss_peak_kib = max(self.rss_peak_kib, rss)
                child_cpu = sum(info.get(p, (0, 0, 0))[2] for p in tree)
                if self._child_cpu0 is None:
                    self._child_cpu0 = child_cpu
                self._child_cpu1 = child_cpu
                mine = drm_clients(tree)
                csum = sum(r["vram_kib"] + r["shared_vram_kib"]
                           for r in mine.values()) / 1024.0
                gsum = sum(r["gtt_kib"] + r["shared_gtt_kib"]
                           for r in mine.values()) / 1024.0
                if csum > self.client_peak:
                    self.client_peak = csum
                    self.top_clients = {
                        c: {k: r[k] / 1024.0 for k in
                            ("vram_kib", "shared_vram_kib", "gtt_kib",
                             "shared_gtt_kib")}
                        for c, r in mine.items()}
                self.gtt_peak = max(self.gtt_peak, gsum)
            tick += 1
            self.stop_flag.wait(self.poll_s)

    @property
    def cpu_busy_pct(self):
        """Mean CPU busy % across all cores during the run (benchmark included)."""
        if not self._cpu_total:
            return None
        return 100.0 * self._cpu_busy / self._cpu_total

    @property
    def bg_cpu_pct(self):
        """Mean CPU busy % across all cores, EXCLUDING the guarded process tree.

        This is the contention the measurement had to fight for; it is the
        number to quote when judging whether two arms are comparable.
        """
        if not self._cpu_total or self._child_cpu0 is None:
            return None
        child = max(0, (self._child_cpu1 or 0) - self._child_cpu0)
        return 100.0 * max(0.0, self._cpu_busy - child) / self._cpu_total

    def finish(self):
        self.stop_flag.set()
        self.join(timeout=2.0)


def run_guarded(cmd, cap_mib=DEFAULT_CAP_MIB, timeout=None, log_path=None,
                cwd=None, env=None, stdin_text=None, quiet=False):
    """Run cmd under the VRAM cap. Returns a result dict (JSON-serializable)."""
    base = vram_used_mib()
    t0 = time.time()
    out_f = open(log_path, "wb") if log_path else None
    try:
        p = subprocess.Popen(
            cmd,
            stdout=out_f or subprocess.PIPE,
            stderr=subprocess.STDOUT if out_f else subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except OSError as e:
        if out_f:
            out_f.close()
        return {"cmd": cmd, "error": str(e), "exit_code": None, "wall_s": 0.0}

    mon = Monitor(p.pid, cap_mib)
    mon.start()

    # The child runs in its own session, so a Ctrl-C / SIGTERM aimed at us never
    # reaches it. Forward the signal to the child's process group instead, or an
    # interrupted serve.sh would leave llama-server holding VRAM forever.
    def _forward(signum, _frame):
        try:
            os.killpg(os.getpgid(p.pid), signum)
        except OSError:
            pass

    prev_handlers = {}
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev_handlers[_sig] = signal.signal(_sig, _forward)
        except (ValueError, OSError):
            pass

    stdout = b""
    timed_out = False
    try:
        if stdin_text is not None:
            try:
                p.stdin.write(stdin_text.encode())
                p.stdin.close()
            except OSError:
                pass
        if timeout:
            try:
                stdout, _ = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout, _ = p.communicate()
        else:
            stdout, _ = p.communicate()
    finally:
        for _sig, _prev in prev_handlers.items():
            try:
                signal.signal(_sig, _prev)
            except (ValueError, OSError):
                pass
        mon.finish()
        if out_f:
            out_f.close()

    wall = time.time() - t0
    text = stdout.decode("utf-8", "replace") if stdout else ""
    if out_f and log_path:
        try:
            with open(log_path, "r", errors="replace") as f:
                text = f.read()
        except OSError:
            pass
    if not quiet and text and not out_f:
        sys.stdout.write(text)

    return {
        "cmd": cmd,
        "exit_code": p.returncode,
        "wall_s": round(wall, 3),
        "vram_baseline_mib": None if base is None else round(base, 1),
        "vram_peak_mib": round(mon.vram_peak, 1),
        "vram_delta_mib": None if base is None else round(mon.vram_peak - base, 1),
        "vram_cap_mib": cap_mib,
        "vram_violation": mon.violation,
        "vram_killed_at_mib": None if mon.killed_at is None else round(mon.killed_at, 1),
        "client_vram_peak_mib": round(mon.client_peak, 1),
        "client_gtt_peak_mib": round(mon.gtt_peak, 1),
        "client_top": mon.top_clients,
        "rss_peak_mib": round(mon.rss_peak_kib / 1024.0, 1),
        "cpu_busy_pct": None if mon.cpu_busy_pct is None else round(mon.cpu_busy_pct, 1),
        "bg_cpu_pct": None if mon.bg_cpu_pct is None else round(mon.bg_cpu_pct, 1),
        "nvidia_vram_peak_mib": None if not mon.nvidia_total else round(mon.nvidia_peak, 1),
        "nvidia_vram_total_mib": None if not mon.nvidia_total else round(mon.nvidia_total, 1),
        "timed_out": timed_out,
        "log": log_path,
        "stdout": text if out_f is None else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-mib", type=float, default=DEFAULT_CAP_MIB)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=None, help="append result JSON to this file")
    ap.add_argument("--label", default=None)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    res = run_guarded(cmd, cap_mib=args.cap_mib, timeout=args.timeout,
                      log_path=args.log, cwd=args.cwd)
    res["label"] = args.label
    res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    sys.stderr.write(json.dumps(res, indent=2) + "\n")
    return 0 if res.get("exit_code") == 0 and not res.get("vram_violation") else 1


if __name__ == "__main__":
    sys.exit(main())
