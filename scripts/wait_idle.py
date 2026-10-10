#!/usr/bin/env python3
"""wait_idle.py — block until a background workload goes quiet, then exit 0.

Benchmarks on this host are CPU-bound (most layers run on the CPU), so a
background job that bursts to ~10 of 16 cores depresses throughput by 2-4x and
makes arms incomparable. `funes index` is exactly such a job: it never exits, it
fans out to short-lived workers, and it alternates between idle and saturated.
Watching for the process to *disappear* therefore never fires — watch its CPU
instead.

    python3 scripts/wait_idle.py --pattern funes --hold 60 && ./scripts/bench.py ...

Exits 0 once the matched processes have stayed below --idle-cores for --hold
seconds. Exits 1 on --timeout (default 2 h) without ever seeing an idle window.
"""

import argparse
import os
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")


def cpu_of(pattern):
    """Return (cores_busy, n_procs) for processes whose comm matches pattern."""
    jiffies = 0
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % pid) as f:
                data = f.read()
        except OSError:
            continue
        end = data.rfind(")")
        if end < 0:
            continue
        comm = data[data.find("(") + 1:end]
        if comm != pattern:
            continue
        f_ = data[end + 2:].split()
        try:
            jiffies += int(f_[11]) + int(f_[12])
        except (IndexError, ValueError):
            continue
        n += 1
    return jiffies, n


def system_cpu():
    """Return (busy_jiffies, total_jiffies) summed over all CPUs."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
    except OSError:
        return 0, 0
    v = [int(x) for x in parts]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v) - idle, sum(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="funes",
                    help="process comm to watch (exact match)")
    ap.add_argument("--system", action="store_true",
                    help="watch total host CPU instead of one process name; "
                         "use when the load comes from many unrelated processes")
    ap.add_argument("--idle-cores", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=60.0,
                    help="seconds the workload must stay idle")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()

    t0 = time.time()
    if args.system:
        prev_j, prev_t = system_cpu()
    else:
        prev_j, prev_t = cpu_of(args.pattern)
    idle_since = None
    while True:
        time.sleep(args.interval)
        now = time.time()
        if args.system:
            j, _ = system_cpu()
            cores = (j - prev_j) / CLK / (now - prev_t)
            label = "host"
        else:
            j, n = cpu_of(args.pattern)
            cores = (j - prev_j) / CLK / (now - prev_t)
            label = "%s (%d procs)" % (args.pattern, n)
        prev_j, prev_t = j, now
        if cores <= args.idle_cores:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= args.hold:
                print("%s idle (%.2f cores) for %.0fs; proceeding"
                      % (label, cores, args.hold))
                return 0
        else:
            if idle_since is not None:
                print("%s resumed (%.2f cores); resetting hold" % (label, cores))
            idle_since = None
        if now - t0 > args.timeout:
            print("%s never stayed under %.2f cores for %.0fs within %.0fs"
                  % (label, args.idle_cores, args.hold, args.timeout))
            return 1


if __name__ == "__main__":
    sys.exit(main())
