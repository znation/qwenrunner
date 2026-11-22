#!/usr/bin/env python3
"""Parallelism sweep for the Inkling pure-CPU serving path.

Goal: a team runs several agents against one local server; maximize TOTAL
token throughput. Each arm starts llama-server with the arm's -np/-t/-c,
then fires N = np concurrent /completion requests (distinct ~900-token
prompts so expert routing differs across streams, 256 generated tokens,
greedy), and records per-request timings plus wall-clock aggregates.

Raw rows append to results/inkling_np.jsonl.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8084
OUT = os.path.join(ROOT, "results", "inkling_np.jsonl")
MODEL = os.path.join(ROOT, "models", "Inkling-2-Mini-IQ4_NL.gguf")
CORPUS = os.path.join(ROOT, "data", "ppl_corpus.txt")

os.environ["PATH"] = os.path.join(ROOT, "tools/local/usr/bin") + ":" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = (os.path.join(ROOT, "build/bin") + ":"
                                 + os.path.join(ROOT, "tools/local/usr/lib/x86_64-linux-gnu")
                                 + ":" + os.environ.get("LD_LIBRARY_PATH", ""))

text = open(CORPUS, encoding="utf-8", errors="replace").read()
words = text.split()
# 4 distinct ~900-token prompts (different corpus regions -> different routing)
PROMPTS = [" ".join(words[i * 700:(i + 1) * 700]) for i in range(4)]
NGEN = 256


def avail_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def start_server(np, threads, ctx, pin=True):
    # a previous arm's server may linger if something above this line raised;
    # it holds 17 GiB of mmap'd model and would poison the next arm's RAM
    subprocess.run(["pkill", "-f", f"--port {PORT}"], capture_output=True)
    time.sleep(2)
    cmd = (["taskset", "-c", "8-15"] if pin else []) + [
           os.path.join(ROOT, "build/bin/llama-server"),
           "-m", MODEL, "-a", "bench", "-c", str(ctx), "-t", str(threads),
           "-np", str(np), "-dev", "none", "-ngl", "0", "-cmoe", "--fit", "off",
           "-lm", "mmap", "-b", "2048", "-ub", "128",
           "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--jinja",
           "--reasoning-effort", "low",
           "--host", "127.0.0.1", "--port", str(PORT)]
    log = open(os.path.join(ROOT, "logs", f"bench_np_{np}t{threads}c{ctx}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        r = subprocess.run(["curl", "-sf", f"http://127.0.0.1:{PORT}/health"],
                           capture_output=True, timeout=2)
        if r.returncode == 0:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: see logs/bench_np_{np}t{threads}c{ctx}.log")
        time.sleep(2)
    proc.terminate()
    proc.wait()
    raise RuntimeError("server did not become healthy in 180 s")


def one_request(prompt, ngen, out, idx, barrier):
    body = json.dumps({"prompt": prompt, "n_predict": ngen,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                 data=body, headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.load(r)
    out[idx] = {"wall_s": round(time.time() - t0, 2), "timings": resp["timings"]}


def loadavg():
    return round(float(open("/proc/loadavg").read().split()[0]), 1)


def run_arm(label, np, threads, ctx, rep=0, pin=True):
    la0 = loadavg()
    proc = start_server(np, threads, ctx, pin)
    try:
        # warm-up: page in the mmap'd experts once
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                     data=json.dumps({"prompt": "2 + 2 =", "n_predict": 16,
                                                      "temperature": 0.0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()
        avail_before = avail_mib()

        barrier = threading.Barrier(np)
        out = {}
        threads_list = [threading.Thread(target=one_request,
                                         args=(PROMPTS[i % len(PROMPTS)], NGEN, out, i, barrier))
                        for i in range(np)]
        wall0 = time.time()
        for t in threads_list:
            t.start()
        for t in threads_list:
            t.join()
        wall = time.time() - wall0

        tot_gen = sum(o["timings"]["predicted_n"] for o in out.values())
        tot_prompt = sum(o["timings"]["prompt_n"] for o in out.values())
        per_stream_gen = [round(o["timings"]["predicted_n"] / (o["timings"]["predicted_ms"] / 1000), 2)
                          for o in out.values()]
        per_stream_prompt = [round(o["timings"]["prompt_n"] / (o["timings"]["prompt_ms"] / 1000), 2)
                             for o in out.values()]
        row = {
            "label": label, "np": np, "threads": threads, "ctx": ctx, "rep": rep, "pin": pin,
            "loadavg1_start": la0, "loadavg1_end": loadavg(),
            "wall_s": round(wall, 2),
            "agg_gen_ts": round(tot_gen / wall, 2),
            "agg_prompt_ts": round(tot_prompt / wall, 2),
            "per_stream_gen_ts": per_stream_gen,
            "per_stream_prompt_ts": per_stream_prompt,
            "tot_gen_tokens": tot_gen,
            "avail_drop_mib": (avail_before - avail_mib()) if avail_before else None,
        }
        print(json.dumps(row))
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
    finally:
        proc.terminate()
        proc.wait()
        time.sleep(4)


ARMS = [
    # interleaved reps cancel the drifting desktop load (rustc builds etc.):
    # np1/np2/np4 twice, then the thread-count variants once.
    ("np1_t8_c32k", 1, 8, 32768, 0),
    ("np2_t8_c32k", 2, 8, 32768, 0),
    ("np4_t8_c32k", 4, 8, 32768, 0),
    ("np1_t8_c32k", 1, 8, 32768, 1),
    ("np2_t8_c32k", 2, 8, 32768, 1),
    ("np4_t8_c32k", 4, 8, 32768, 1),
    ("np1_t16_c32k", 1, 16, 32768, 0),
    ("np2_t16_c64k", 2, 16, 65536, 0),
    # pin A/B: does confining the server to one SMT lane per core buy
    # throughput under the real desktop load, or cost it? Interleaved
    # pin/nopin, two reps, to cancel load drift (the first single-pass
    # attempt was voided by a load spike: nopin arms ran at loadavg 39-45
    # vs 11-20 for every pinned arm).
    ("np1_pin", 1, 8, 32768, 0, True),
    ("np1_nopin", 1, 8, 32768, 0, False),
    ("np2_pin", 2, 8, 32768, 0, True),
    ("np2_nopin", 2, 8, 32768, 0, False),
    ("np1_pin", 1, 8, 32768, 1, True),
    ("np1_nopin", 1, 8, 32768, 1, False),
    ("np2_pin", 2, 8, 32768, 1, True),
    ("np2_nopin", 2, 8, 32768, 1, False),
]

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    arms = ARMS
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        arms = [a for a in ARMS if a[0] in want]
    for arm in arms:
        label, np, threads, ctx, rep = arm[:5]
        pin = arm[5] if len(arm) > 5 else True
        print(f"=== {label} rep{rep} ===", flush=True)
        run_arm(label, np, threads, ctx, rep, pin)
    print("=== done ===")
    print(open(OUT).read())
