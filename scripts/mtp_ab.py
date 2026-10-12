#!/usr/bin/env python3
"""MTP (speculative decoding) A/B harness for qwen35 on this host.

Why llama-server and not llama-bench / llama-completion:

  * `llama-bench` exposes no speculative options at all.
  * `--spec-type` is declared `.set_examples({LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI})` (common/arg.cpp:4291) --
    `llama-completion` is NOT in the set and rejects the flag outright
    ("error: invalid argument: --spec-type", verified live).
  * `llama-server` returns structured `timings` including `draft_n` /
    `draft_n_accepted` (tools/server/server-common.cpp:100), which is the
    only way to see whether the MTP head actually fired and how often its
    draft token was accepted.

Each arm runs the server under `scripts/guard.py`'s Monitor (hard 5000 MiB
system-VRAM cap, SIGKILL on breach), waits for /health, issues one
POST /completion at a fixed context depth, records the timings, then kills the
whole process group. Arms are serialized -- never two model workloads at once
on this host.

Arms:
  base          no speculative flags; the main GGUF's blk.64.nextn.* tensors
                are unused (loader prints "unused tensor" warnings).
  mtp-sidecar   `--spec-type draft-mtp -md <sidecar.gguf>`: the real MTP path.
  mtp-nomd      `--spec-type draft-mtp` with no draft model. `common.cpp`
                still sets load_mtp=true, so the main file's own MTP tensors
                load, but the draft context stays null and
                `add_config_if_enabled(..., params.draft.ctx_dft != nullptr)`
                never enables the driver -- this arm isolates the memory cost
                of MTP tensors from the speed benefit.

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
    if name == "mtp-sidecar-qkv":
        # mtp-sidecar with a quantized draft KV cache. At 128k the draft's f16
        # KV is ~500 MiB per draft layer, and the draft is a full attention
        # block, so this is the cheapest VRAM/RAM lever on the MTP path.
        extra = ["--spec-type", "draft-mtp", "-md", args.draft,
                 "-ctkd", "q4_0", "-ctvd", "q4_0"]
        if args.ngld is not None:
            extra += ["-ngld", str(args.ngld)]
        return extra
    if name == "mtp-draftgpu":
        # Same sidecar as mtp-sidecar, but the draft head is offloaded to a
        # specific device (-devd) instead of following -dev. On this host the
        # point is the Vega 10, which has VRAM to spare under the display cap
        # while the 1070 is full of trunk layers. Uses its own -ngld so the
        # trunk's --ngld stays comparable to the CPU-draft arm.
        extra = ["--spec-type", "draft-mtp", "-md", args.draft,
                 "-devd", args.draft_dev, "-ngld", str(args.ngld_gpu)]
        return extra
    raise SystemExit("unknown arm %r" % name)


def settle_vram(target_mib, timeout=60.0, slack_mib=250.0):
    """Wait for the previous arm's VRAM to be released before sampling a baseline.

    The guard reads the system counter, so a stale DRM client from the arm that
    just exited silently inflates the next arm's baseline (and can trip the cap).
    """
    deadline = time.time() + timeout
    v = guard.vram_used_mib()
    while v > target_mib + slack_mib and time.time() < deadline:
        time.sleep(2.0)
        v = guard.vram_used_mib()
    return v


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

    base = settle_vram(idle0)
    t0 = time.time()
    out_f = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=ROOT,
                         start_new_session=True)
    mon = guard.Monitor(p.pid, args.cap_mib)
    mon.start()

    rec = {"arm": name, "cmd": cmd, "model": args.model,
           "flags": {"ngl": args.ngl, "ngld": args.ngld, "ctx": args.ctx,
                     "threads": args.threads, "n_predict": args.n_predict,
                     "server_extra": args.server_extra},
           "vram_baseline_mib": None if base is None else round(base, 1),
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

    rec["exit_code"] = p.returncode
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["load_s"] = loaded
    rec["error"] = err
    rec["timings"] = timings
    rec["vram_peak_mib"] = round(mon.vram_peak, 1)
    rec["vram_delta_mib"] = None if base is None else round(mon.vram_peak - base, 1)
    rec["vram_violation"] = mon.violation
    rec["client_vram_peak_mib"] = round(mon.client_peak, 1)
    rec["client_gtt_peak_mib"] = round(mon.gtt_peak, 1)
    rec["client_top"] = mon.top_clients
    rec["rss_peak_mib"] = round(mon.rss_peak_kib / 1024.0, 1)
    # Throughput on this host is CPU-bound, so an arm measured against a busy
    # desktop is not comparable to an idle one. Quote bg_cpu_pct with the rate.
    rec["cpu_busy_pct"] = None if mon.cpu_busy_pct is None else round(mon.cpu_busy_pct, 1)
    rec["bg_cpu_pct"] = None if mon.bg_cpu_pct is None else round(mon.bg_cpu_pct, 1)
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def summarize(rec):
    t = rec.get("timings") or {}
    print("%-12s vram_peak=%-7s client=%-7s exit=%-3s load=%-6s "
          "pp=%-8s tg=%-8s draft=%s/%s %s" % (
              rec["arm"], rec["vram_peak_mib"], rec["client_vram_peak_mib"],
              rec["exit_code"], rec["load_s"],
              round(t.get("prompt_per_second", 0), 2),
              round(t.get("predicted_per_second", 0), 3),
              t.get("draft_n_accepted", "-"), t.get("draft_n", "-"),
              ("ERR " + rec["error"]) if rec.get("error") else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--draft", default="models/mtp-Qwen3.8-27B-Q4_0.gguf")
    ap.add_argument("--ngl", type=int, default=8)
    ap.add_argument("--ngld", type=int, default=None,
                    help="GPU layers for the draft model (-ngld); unset = default")
    ap.add_argument("--draft-dev", default="Vulkan0",
                    help="device for the mtp-draftgpu arm's -devd")
    ap.add_argument("--ngld-gpu", type=int, default=99,
                    help="draft GPU layers for the mtp-draftgpu arm")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--prompt-file", default="data/long_prompt.txt")
    ap.add_argument("--arms", default="base,mtp-sidecar,mtp-nomd")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cap-mib", type=float, default=5000.0)
    ap.add_argument("--load-timeout", type=float, default=600.0)
    ap.add_argument("--gen-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="results/mtp_ab.jsonl")
    ap.add_argument("--server-extra", default="",
                    help="extra llama-server flags for every arm, "
                         "e.g. '-lm dio -b 2048 -ub 512'")
    args = ap.parse_args(argv)

    prompt = open(os.path.join(ROOT, args.prompt_file)).read()
    idle0 = guard.vram_used_mib()
    print("baseline VRAM: %.1f MiB (cap %.1f); prompt %d chars"
          % (idle0, args.cap_mib, len(prompt)))

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
            if rec["vram_violation"]:
                print("stopping: %s breached the cap" % name)
                break
            if rec.get("error"):
                print("stopping: %s failed (%s)" % (name, rec["error"]))
                break

    if args.out:
        print("wrote %d records to %s" % (len(recs), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
