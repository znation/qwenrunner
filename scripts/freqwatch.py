#!/usr/bin/env python3
"""Sample CPU frequencies + GPU clocks every 0.5 s to a JSONL file.

Run alongside a benchmark to answer "was the CPU actually boosting?".

    python3 scripts/freqwatch.py results/freq.jsonl &
    ... run the benchmark ...
    kill %1

Stdlib only.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import guard  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/freq.jsonl"


def read_cpu_mhz():
    mhz = []
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("cpu MHz"):
                try:
                    mhz.append(float(line.split(":")[1]))
                except ValueError:
                    pass
    return mhz


def read_thermal():
    out = {"pkg_temp_c": None, "pkg_w": None}
    try:
        with open("/sys/class/thermal/thermal_zone1/temp") as f:
            out["pkg_temp_c"] = int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
            out["pkg_w"] = int(f.read().strip())
    except (OSError, ValueError):
        pass
    return out


def read_gpu():
    out = {"nvidia": None, "vega": None, "vega_sclk_mhz": None}
    try:
        with open("/sys/class/drm/card0/device/gpu_busy_percent") as f:
            out["vega"] = int(f.read().strip())
    except OSError:
        pass
    try:
        with open("/sys/class/drm/card1/device/gpu_busy_percent") as f:
            out["vega"] = int(f.read().strip())
    except OSError:
        pass
    # current SCLK level (the line marked with *)
    for card in ("card0", "card1"):
        try:
            with open("/sys/class/drm/%s/device/pp_dpm_sclk" % card) as f:
                for line in f:
                    if "*" in line:
                        out["vega_sclk_mhz"] = int(line.split(":")[1].split("Mhz")[0].strip())
                        break
                break
        except (OSError, ValueError):
            pass
    try:
        nv = guard.nvidia_vram_mib()
        if nv is not None:
            out["nvidia"] = {"used_mib": nv[0], "total_mib": nv[1]}
    except Exception:
        pass
    return out


def main():
    with open(OUT, "w") as f:
        t0 = time.time()
        while True:
            mhz = read_cpu_mhz()
            rec = {"t": round(time.time() - t0, 2),
                   "avg_mhz": round(sum(mhz) / len(mhz), 1) if mhz else None,
                   "max_mhz": max(mhz) if mhz else None,
                   "n_over_4ghz": sum(1 for m in mhz if m > 4000)}
            rec.update(read_thermal())
            rec.update(read_gpu())
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(0.5)


if __name__ == "__main__":
    main()
