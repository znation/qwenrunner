#!/usr/bin/env python3
"""MTP (speculative decoding) A/B harness for Qwen-3.8-27B on Apple Silicon.

Why llama-server and not llama-bench / llama-completion:

  * `llama-bench` exposes no speculative options at all.
  * `--spec-type` is declared `.set_examples({LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI})` (common/arg.cpp) --
    `llama-completion` is NOT in the set and rejects the flag outright.
  * `llama-server` returns structured `timings` including `draft_n` /
    `draft_n_accepted`, which is the only way to see whether the MTP head
    actually fired and how often its draft token was accepted.

What changed for this host: the Intel arms ran the draft head on the CPU
(`-ngld 0`) because the 1070 had no room for it, and paid 6 layers of offload
for the privilege. Here the draft goes on the Metal device too (`-ngld 99`), so
the expectation is a *larger* win than the +29 % measured on the two-GPU split
-- the CPU-resident draft was the bottleneck in that configuration.

`-md` (the sidecar) is required, not optional: common/speculative.cpp leaves
`ctx_dft` null without it and the MTP driver is never enabled. The `mtp-nomd`
arm exists precisely to show that -- it loads the main file's own
blk.64.nextn.* tensors and still measures base speed.

Each arm runs the server under `scripts/guard.py`'s Monitor, waits for /health,
issues one POST /completion at a fixed context depth, records the timings, then
kills the whole process group. Arms are serialized -- never two model
workloads at once on this host.

Arms:
  base          no speculative flags.
  mtp-sidecar   `--spec-type draft-mtp -md <sidecar.gguf>`: the real MTP path.
  mtp-nomd      `--spec-type draft-mtp` with no draft model.

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


def arms_for(name, args):
    if name == "base":
        return []
    if name == "mtp-sidecar":
        extra = ["--spec-type", "draft-mtp", "-md", args.draft]
        if args.ngld is not None:
            extra += ["-ngld", str(args.ngld)]
        return extra
    if name == "mtp-nomd":
        return ["--spec-type", "draft-mtp"]
    raise SystemExit("unknown arm %r" % name)


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(port, deadline):
    url = "http://127.0.0.1:%d/health" % port
    while time.time() < deadline:
        try:
            st = http_json(url, timeout=5)
            if st.get("status") in ("ok", "no slot available"):
                return True, None
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False, "health timeout"


def run_arm(name, args, prompt, idle0):
    extra = arms_for(name, args)
    log = os.path.join(ROOT, "logs", "mtp_ab-%s.log" % name)
    cmd = [os.path.join(ROOT, "build", "bin", "llama-server"),
           "-m", args.model, "-ngl", str(args.ngl), "-c", str(args.ctx),
           "-t", str(args.threads), "--host", "127.0.0.1",
           "--port", str(args.port), "--seed", "42",
           "-np", "1", "-lv", "2"]
    if args.server_extra:
        cmd += args.server_extra.split()
    cmd += extra

    base = guard.settle_gpu(idle0)
    t0 = time.time()
    out_f = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=ROOT,
                         start_new_session=True)
    mon = guard.Monitor(p.pid, swap_cap_mib=args.swap_cap_mib)
    mon.start()

    rec = {"arm": name, "cmd": cmd, "model": args.model,
           "flags": {"ngl": args.ngl, "ngld": args.ngld, "ctx": args.ctx,
                     "threads": args.threads, "n_predict": args.n_predict,
                     "server_extra": args.server_extra},
           "gpu_mem_baseline_mib": None if base is None else round(base, 1),
           "log": os.path.relpath(log, ROOT)}

    loaded = None
    timings = None
    err = None
    try:
        ok, why = wait_health(args.port, time.time() + args.load_timeout)
        if not ok:
            err = why
        else:
            loaded = round(time.time() - t0, 2)
            payload = {"prompt": prompt, "n_predict": args.n_predict,
                       "temperature": 0.0, "seed": 42,
                       "cache_prompt": False, "stream": False}
            g0 = time.time()
            resp = http_json("http://127.0.0.1:%d/completion" % args.port,
                             payload, timeout=args.gen_timeout)
            gen_wall = round(time.time() - g0, 3)
            timings = resp.get("timings")
            rec["gen_wall_s"] = gen_wall
            rec["content_head"] = (resp.get("content") or "")[:64]
    except Exception as e:  # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, e)
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

    snap = mon.snapshot()
    rec["exit_code"] = p.returncode
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["load_s"] = loaded
    rec["error"] = err
    rec["timings"] = timings
    rec.update(snap)
    rec["gpu_mem_delta_mib"] = (
        None if (snap["gpu_mem_peak_mib"] is None or base is None)
        else round(snap["gpu_mem_peak_mib"] - base, 1))
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def summarize(rec):
    t = rec.get("timings") or {}
    print("%-12s gpu_peak=%-7s exit=%-3s load=%-6s bg_cpu=%-5s gpu%%=%-3s "
          "pp=%-8s tg=%-8s draft=%s/%s %s" % (
              rec["arm"], rec["gpu_mem_peak_mib"],
              rec["exit_code"], rec["load_s"], rec.get("bg_cpu_pct"),
              rec.get("gpu_busy_pct_peak"),
              round(t.get("prompt_per_second", 0), 2),
              round(t.get("predicted_per_second", 0), 3),
              t.get("draft_n_accepted", "-"), t.get("draft_n", "-"),
              ("ERR " + rec["error"]) if rec.get("error") else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_S.gguf")
    ap.add_argument("--draft", default="models/mtp-Qwen3.8-27B-Q4_0.gguf")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ngld", type=int, default=99,
                    help="GPU layers for the draft model (-ngld); default 99 "
                         "puts the whole draft head on the Metal device")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--prompt-file", default="data/long_prompt.txt")
    ap.add_argument("--arms", default="base,mtp-sidecar,mtp-nomd")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--swap-cap-mib", type=float, default=0.0)
    ap.add_argument("--load-timeout", type=float, default=900.0)
    ap.add_argument("--gen-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="results/mtp_ab.jsonl")
    ap.add_argument("--server-extra", default="",
                    help="extra llama-server flags for every arm, "
                         "e.g. '-b 4096 -ub 1024'")
    args = ap.parse_args(argv)

    prompt = open(os.path.join(ROOT, args.prompt_file)).read()
    _, idle0 = guard.gpu_stats()
    print("baseline GPU memory: %s MiB; prompt %d chars"
          % ("n/a" if idle0 is None else "%.1f" % idle0, len(prompt)))

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    recs = []
    for rep in range(args.reps):
        for name in names:
            rec = run_arm(name, args, prompt, idle0)
            rec["rep"] = rep
            recs.append(rec)
            if args.out:
                # Flush per arm: a later hang or SIGKILL must not lose this one.
                with open(os.path.join(ROOT, args.out), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            summarize(rec)
            if rec["swap_violation"]:
                print("stopping: %s breached the swap cap" % name)
                break
            if rec.get("error"):
                print("stopping: %s failed (%s)" % (name, rec["error"]))
                break

    if args.out:
        print("wrote %d records to %s" % (len(recs), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
