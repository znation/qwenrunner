#!/usr/bin/env python3
"""Fit probe for the two-GPU layer split: how many layers actually fit.

`llama-bench` cannot answer this -- its `-dev` lists devices to benchmark
*separately*, so it loads one model per device instead of splitting one across
both (see RESULTS.md "Traps"). So this drives `llama-server`: start it under
`scripts/guard.py`'s Monitor, wait for /health, issue **one real completion**,
then kill the group. Loading is not enough -- measured here, `-ngl 30` at 128k
loads clean and then aborts in `cudaMalloc` on the first long prefill.

The numbers that matter are per-device peaks, because the two GPUs have
different ceilings:

  * `nvidia_vram_peak_mib` against 8192 -- the GTX 1070, which sets the layer
    ceiling.
  * `client_vram_peak_mib` against `CAP - desktop_baseline` -- the Vega 10,
    which is also the display GPU and is what the 5000 MiB cap protects.

A short prompt is enough: the compute buffers are reserved at load from
`-b`/`-ub`/`-np`, and the completion only has to prove the config survives a
request. Use `--prompt-chars` to keep each rung cheap while sweeping.

    scripts/split_fit.py --ngl-ladder 26,28,30 --ts 6,1 --ctx 131072 \
        --prompt-chars 400 --out results/split_fit.jsonl

Stdlib only.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wait_health(port, deadline):
    url = "http://127.0.0.1:%d/health" % port
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def http_json(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_rung(args, prompt, ngl):
    log = os.path.join(ROOT, "logs", "split_fit-%d.log" % ngl)
    cmd = [os.path.join(ROOT, "build", "bin", "llama-server"),
           "-m", args.model, "-ngl", str(ngl), "-c", str(args.ctx),
           "-dev", args.dev, "-sm", "layer", "-ts", args.ts,
           "-t", str(args.threads), "-np", str(args.np),
           "--host", "127.0.0.1", "--port", str(args.port),
           "--seed", "42", "-lv", "2", "--log-file", log]
    cmd += args.server_extra.split()
    if args.mtp:
        cmd += ["--spec-type", "draft-mtp", "-md", args.draft,
                "-devd", args.draft_dev, "-ngld", str(args.ngld),
                "-ctkd", args.ctkd, "-ctvd", args.ctvd]

    rec = {"ngl": ngl, "ts": args.ts, "ctx": args.ctx, "mtp": args.mtp,
           "cmd": cmd, "log": os.path.relpath(log, ROOT)}
    t0 = time.time()
    out_f = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=ROOT,
                         start_new_session=True)
    mon = guard.Monitor(p.pid, args.cap_mib)
    mon.start()
    try:
        if not wait_health(args.port, time.time() + args.load_timeout):
            rec["error"] = "health timeout (exit %s)" % p.poll()
        else:
            rec["load_s"] = round(time.time() - t0, 2)
            try:
                resp = http_json(
                    "http://127.0.0.1:%d/completion" % args.port,
                    {"prompt": prompt, "n_predict": args.n_predict,
                     "temperature": 0.0, "seed": 42,
                     "cache_prompt": False, "stream": False},
                    args.gen_timeout)
                t = resp.get("timings") or {}
                rec["prompt_per_second"] = round(t.get("prompt_per_second", 0), 2)
                rec["predicted_per_second"] = round(t.get("predicted_per_second", 0), 3)
                rec["draft_n"] = t.get("draft_n")
                rec["draft_n_accepted"] = t.get("draft_n_accepted")
            except Exception as e:  # noqa: BLE001
                # A load that survives /health can still die here -- that is the
                # whole reason this probe issues a request.
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:80])
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except OSError:
            pass
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        mon.finish()
        out_f.close()

    rec["exit_code"] = p.returncode
    rec["vram_peak_mib"] = round(mon.vram_peak, 1)
    rec["client_vram_peak_mib"] = round(mon.client_peak, 1)
    rec["nvidia_vram_peak_mib"] = round(mon.nvidia_peak, 1)
    rec["vram_violation"] = mon.violation
    rec["bg_cpu_pct"] = None if mon.bg_cpu_pct is None else round(mon.bg_cpu_pct, 1)
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def summarize(rec):
    print("ngl=%-3s ts=%-4s client=%-8s nvidia=%-8s pp=%-7s tg=%-7s %s" % (
        rec["ngl"], rec["ts"], rec["client_vram_peak_mib"],
        rec["nvidia_vram_peak_mib"], rec.get("prompt_per_second", "-"),
        rec.get("predicted_per_second", "-"),
        ("ERR " + rec["error"]) if rec.get("error") else "ok"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--draft", default="models/mtp-Qwen3.8-27B-Q4_0.gguf")
    ap.add_argument("--ngl-ladder", default="28,30",
                    help="comma-separated -ngl values to try, in order")
    ap.add_argument("--ts", default="6,1")
    ap.add_argument("--dev", default="CUDA0,Vulkan0")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--draft-dev", default="Vulkan0")
    ap.add_argument("--ngld", type=int, default=99)
    ap.add_argument("--ctkd", default="q4_0")
    ap.add_argument("--ctvd", default="q4_0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--prompt-file", default="data/long_prompt.txt")
    ap.add_argument("--prompt-chars", type=int, default=600,
                    help="truncate the prompt file; 0 = use all of it. A short "
                         "prompt keeps rungs cheap but can miss an abort that "
                         "only shows up on a long prefill (e.g. -ngl 30 at 128k)")
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--cap-mib", type=float, default=5000.0)
    ap.add_argument("--load-timeout", type=float, default=300.0)
    ap.add_argument("--gen-timeout", type=float, default=1800.0)
    ap.add_argument("--server-extra",
                    default="-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on")
    ap.add_argument("--out", default="results/split_fit.jsonl")
    args = ap.parse_args(argv)

    text = open(os.path.join(ROOT, args.prompt_file)).read()
    prompt = text[:args.prompt_chars] if args.prompt_chars > 0 else text
    print("baseline VRAM: %.1f MiB (cap %.1f); prompt %d chars"
          % (guard.vram_used_mib(), args.cap_mib, len(prompt)))

    for ngl in [int(x) for x in args.ngl_ladder.split(",") if x.strip()]:
        rec = run_rung(args, prompt, ngl)
        summarize(rec)
        if args.out:
            with open(os.path.join(ROOT, args.out), "a") as f:
                f.write(json.dumps(rec) + "\n")
        if rec["vram_violation"]:
            print("stopping: -ngl %d breached the display cap" % ngl)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
