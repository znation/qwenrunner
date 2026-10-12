#!/usr/bin/env python3
"""wait_idle.py — block until a background workload goes quiet, then exit 0.

Why this exists on the Intel host and still matters here: a contended arm is
not a slow arm, it is a *wrong* one. The measured spread between repeats of the
identical configuration there was up to 4.6x. On the M5 Max the CPU matters
less (every layer runs on the Metal device), so the load that corrupts a number
has moved: the shared GPU. `--gpu` therefore gates on `Device Utilization %`
from ioreg, which is what a browser, a video call or a second model load looks
like from inside our process. `--system` and `--pattern` remain for CPU-side
load.

    python3 scripts/wait_idle.py --gpu --hold 30 && ./scripts/bench.py ...

Exits 0 once the watched signal has stayed under the threshold for --hold
seconds. Exits 1 on --timeout without ever seeing an idle window.

Stdlib only (imports guard.py for the counters).
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard  # noqa: E402



def cpu_of(pattern):
    """Return (cpu_seconds_total, n_procs) for processes named `pattern`.

    macOS has no /proc; `ps -o comm=,time=` gives the same two facts the old
    reader got from /proc/<pid>/stat (comm and cumulative cpu time), and the
    delta between two samples is the busy fraction, in cores.
    """
    try:
        out = subprocess.run(["ps", "-A", "-o", "comm=,time="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0, 0
    total = 0.0
    n = 0
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        if os.path.basename(parts[0]) != pattern:
            continue
        t = guard._parse_cputime(parts[1])
        if t is None:
            continue
        total += t
        n += 1
    return total, n


def system_cpu():
    """Host-wide CPU seconds consumed by all live processes."""
    return guard.host_cpu_s()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=None,
                    help="process name to watch (exact basename match)")
    ap.add_argument("--system", action="store_true",
                    help="watch total host CPU instead of one process name; "
                         "use when the load comes from many unrelated processes")
    ap.add_argument("--gpu", action="store_true",
                    help="watch GPU `Device Utilization %%` (the shared "
                         "resource that actually gates throughput here)")
    ap.add_argument("--idle-cores", type=float, default=0.5)
    ap.add_argument("--idle-gpu-pct", type=float, default=None,
                    help="absolute ceiling for GPU %% (default: baseline + "
                         "--gpu-margin, because the desktop itself holds the "
                         "GPU 20-30%% busy on this machine)")
    ap.add_argument("--gpu-margin", type=float, default=10.0,
                    help="allowed rise above the first GPU reading")
    ap.add_argument("--hold", type=float, default=60.0,
                    help="seconds the workload must stay idle")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()

    if not (args.gpu or args.system or args.pattern):
        ap.error("pick one of --gpu, --system or --pattern")

    t0 = time.time()
    prev_wall = time.time()
    prev_cpu = cpu_of(args.pattern)[0] if args.pattern else system_cpu()
    gpu_base = None
    if args.gpu:
        util, _ = guard.gpu_stats()
        gpu_base = 0.0 if util is None else float(util)
        print("gpu baseline %.0f%%; ceiling %s" % (
            gpu_base, "%.0f%%" % args.idle_gpu_pct if args.idle_gpu_pct is not None
            else "%.0f%%" % (gpu_base + args.gpu_margin)))
    idle_since = None

    while True:
        time.sleep(args.interval)
        now = time.time()
        dt = now - prev_wall

        if args.gpu:
            util, _ = guard.gpu_stats()
            value = 0.0 if util is None else float(util)
            limit = (args.idle_gpu_pct if args.idle_gpu_pct is not None
                     else gpu_base + args.gpu_margin)
            label, unit = "gpu", "%"
        elif args.pattern:
            j, n_procs = cpu_of(args.pattern)
            value = (j - prev_cpu) / dt
            prev_cpu = j
            limit = args.idle_cores
            label, unit = "%s (%d procs)" % (args.pattern, n_procs), " cores"
        else:
            j = system_cpu()
            value = (j - prev_cpu) / dt
            prev_cpu = j
            limit = args.idle_cores
            label, unit = "host", " cores"
        prev_wall = now

        if value <= limit:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= args.hold:
                print("%s idle (%.2f%s) for %.0fs; proceeding"
                      % (label, value, unit, args.hold))
                return 0
        else:
            if idle_since is not None:
                print("%s resumed (%.2f%s); resetting hold" % (label, value, unit))
            idle_since = None
        if now - t0 > args.timeout:
            print("%s never stayed under %.2f%s for %.0fs within %.0fs"
                  % (label, limit, unit, args.hold, args.timeout))
            return 1


if __name__ == "__main__":
    sys.exit(main())
