#!/usr/bin/env python3
"""Instrument one live completion: per-thread CPU%, GPU util, CPU freq.

Usage: python3 scripts/instrument_run.py <port> <out.jsonl>
Reads the prompt from data/long_prompt.txt, POSTs /completion with n_predict=64,
and samples llama-server's thread CPU usage + GPU clocks every 0.5 s.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import guard  # noqa: E402

PORT = sys.argv[1]
OUT = sys.argv[2]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID_ARG = sys.argv[3] if len(sys.argv) > 3 else None

prompt = open(os.path.join(ROOT, "data", "long_prompt.txt")).read()


def server_pid():
    if PID_ARG:
        return PID_ARG
    out = subprocess.run(["pgrep", "-f", "port %s" % PORT],
                         capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def sample_threads(pid):
    """per-thread CPU time for the process (jiffies), total."""
    if not pid:
        return None
    try:
        total = 0
        for t in os.listdir("/proc/%s/task" % pid):
            try:
                with open("/proc/%s/task/%s/stat" % (pid, t)) as f:
                    fields = f.read().split()
                    total += int(fields[13]) + int(fields[14])
            except (OSError, IndexError, ValueError):
                pass
        return total
    except OSError:
        return None


def cpu_mhz():
    mhz = []
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("cpu MHz"):
                try:
                    mhz.append(float(line.split(":")[1]))
                except ValueError:
                    pass
    return sum(mhz) / len(mhz) if mhz else 0


def gpu_util():
    r = {"nvidia": {"util": guard.nvidia_util_pct()}, "vega": None}
    for card in ("card0", "card1"):
        try:
            with open("/sys/class/drm/%s/device/gpu_busy_percent" % card) as f:
                v = int(f.read().strip())
                r["vega"] = v
                break
        except OSError:
            pass
    return r


def main():
    pid = server_pid()
    if not pid:
        sys.exit("no server on port %s" % PORT)
    # launch the completion in the background
    payload = {"prompt": prompt, "n_predict": 64, "temperature": 0.0,
               "seed": 42, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(
        "http://127.0.0.1:%s/completion" % PORT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys,urllib.request,json;"
         "req=urllib.request.Request(sys.argv[1],data=json.dumps(json.loads(sys.argv[2])).encode(),headers={'Content-Type':'application/json'});"
         "print(urllib.request.urlopen(req,timeout=2000).read().decode())",
         "http://127.0.0.1:%s/completion" % PORT, json.dumps(payload)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    prev = None
    with open(OUT, "w") as f:
        while proc.poll() is None:
            cur = sample_threads(pid)
            g = gpu_util()
            rec = {"t": round(time.time() - t0, 2), "avg_mhz": round(cpu_mhz(), 0)}
            rec.update(g)
            if prev is not None and cur is not None:
                # jiffies per 0.5s; HZ=100 → 100 jiffies = 1 core
                rec["cpu_cores"] = round((cur - prev) / 50.0, 2)
            prev = cur
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(0.5)
        out = proc.stdout.read().decode()
        try:
            resp = json.loads(out)
            t = resp.get("timings") or {}
            print("RESULT pp=%.2f tg=%.3f draft=%s/%s" % (
                t.get("prompt_per_second", 0), t.get("predicted_per_second", 0),
                t.get("draft_n"), t.get("draft_n_accepted")))
        except (ValueError, AttributeError):
            print("RAW:", out[:200])


if __name__ == "__main__":
    main()
