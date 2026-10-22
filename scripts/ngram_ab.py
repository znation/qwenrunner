#!/usr/bin/env python3
"""Draft-free speculative decoding A/B harness (ngram / lookup) for Flint.

Flint ships no MTP head, so the Qwen trick in scripts/mtp_ab.py has no
counterpart here -- but llama.cpp at a37cc4610 has model-free drafters that
generate proposal tokens from the *text itself*:

  ngram-simple    fixed n-gram lookup over the prompt + generated history
  ngram-map-k     hash-map variant, k-size lookup, m-token draft
  ngram-map-k4v   same with 4-value keys
  ngram-mod       modular hashing, tunable match length (n-match)

They cost one batched verify pass per step instead of one matvec per token, so
on this workload (2.5B active parameters, ~1.4 GB of weights read per token)
they attack precisely the thing that makes decode slow. The gain is a property
of the text, not the machine: high-acceptance on code/markdown/JSON, near zero
on prose -- the same split RESULTS.md measured for MTP (+180 % patterned,
-11 % prose).

Arms:
  base        no speculative flags.
  <spec-type> --spec-type <spec-type> with the tunables for that type.
  synth-L     --spec-synth-len L with a registered drafter (--synth-base, default
              ngram-simple): the server fakes an acceptance length of L while the
              drafter still proposes real tokens. Measured here, this is NOT a
              ceiling: at L=2/4/8 on copy-heavy text it came out *slower* than
              base (54/77/112 t/s against 119) because the verify pass still runs
              with the drafter's proposals, so the arm charges the drafter's cost
              without its benefit. Read a synth arm as "this drafter, acceptance
              forced to L", not as "machinery at acceptance L". The honest
              ceiling has to come from a real drafter with real acceptance.
  novr        ngram-simple with --min-hits 99: a registered spec context whose
              drafter can never fire. Separates "speculative context allocated"
              overhead from "drafted and rejected" overhead.

Serialization matters: arms run one at a time, interleaved by --reps, each under
guard.Monitor, each with `-np 1`. Greedy (--temperature 0) is the default because
it makes arms comparable; --temperature -1 omits the field so the server applies
the GGUF's own general.sampling.* (Flint: 1.0/0.95/20), which is what real
agentic traffic uses -- note that at temp 0 this model's chat output collapses
into terse non-code reasoning, so prompt-shape conclusions must be re-checked at
the card's sampling. Stdlib only.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DEFAULT = "third_party/llama-flint-build/bin/llama-server"

# Per-type tunables, typed exactly as common/arg.cpp declares them.
#   lookup length   : ngram-simple/map-*  --spec-ngram-*-size-n   ngram-mod --spec-ngram-mod-n-match
#   draft length    : ngram-simple/map-*  --spec-ngram-*-size-m   ngram-mod --spec-ngram-mod-n-max
#   min draft length: ngram-mod --spec-ngram-mod-n-min (a draft shorter than this is dropped)
#   min occurrences : ngram-simple/map-*  --spec-ngram-*-min-hits
NG_TUNABLES = {
    "ngram-simple":  ("--spec-ngram-simple-size-n", "--spec-ngram-simple-size-m",
                      "--spec-ngram-simple-min-hits"),
    "ngram-map-k":   ("--spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-m",
                      "--spec-ngram-map-k-min-hits"),
    "ngram-map-k4v": ("--spec-ngram-map-k4v-size-n", "--spec-ngram-map-k4v-size-m",
                      "--spec-ngram-map-k4v-min-hits"),
    "ngram-mod":     ("--spec-ngram-mod-n-match", "--spec-ngram-mod-n-max",
                      "--spec-ngram-mod-n-min"),
}


def arms_for(name, args):
    if name == "base":
        return []
    m = re.fullmatch(r"synth-([0-9.]+)", name)
    if m:
        # Synthetic acceptance is not a drafter: it needs a *registered* spec
        # context to size its rates from (--spec-type none fails model load with
        # "synthetic acceptance requires an initialized speculative decoding
        # context"), so pair it with a cheap real impl and let the injected
        # acceptance replace its tokens.
        return ["--spec-type", args.synth_base, "--spec-synth-len", m.group(1)]
    if name == "novr":
        # Registered spec context whose drafter can never fire: no key has 99 hits.
        return ["--spec-type", "ngram-simple", "--spec-ngram-simple-min-hits", "99"]
    if name == "ngram-cache":
        return ["--spec-type", "ngram-cache"]
    if name not in NG_TUNABLES:
        raise SystemExit("unknown arm %r" % name)
    f_n, f_m, f_hits = NG_TUNABLES[name]
    extra = ["--spec-type", name]
    if args.lookup_n is not None:
        extra += [f_n, str(args.lookup_n)]
    if args.draft_m is not None and name != "ngram-mod":
        extra += [f_m, str(args.draft_m)]
    if args.min_hits is not None and name != "ngram-mod":
        extra += [f_hits, str(args.min_hits)]
    if name == "ngram-mod":
        extra += ["--spec-ngram-mod-n-max", str(args.draft_max_mod)]
        if args.draft_min is not None:
            extra += ["--spec-ngram-mod-n-min", str(args.draft_min)]
    return extra


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
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    log = os.path.join(ROOT, "logs", "ngram_ab-%s.log" % tag)
    cmd = [args.server, "-m", args.model, "-ngl", str(args.ngl),
           "-c", str(args.ctx), "-t", str(args.threads), "--alias", args.alias,
           "--host", "127.0.0.1", "--port", str(args.port), "--seed", "42",
           "-np", "1", "-lv", "2"]
    if args.ctx_type:
        cmd += ["-ctk", args.ctx_type, "-ctv", args.ctx_type, "-fa", "on"]
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
           "flags": {"ngl": args.ngl, "ctx": args.ctx, "threads": args.threads,
                     "n_predict": args.n_predict, "lookup_n": args.lookup_n,
                     "draft_m": args.draft_m, "draft_min": args.draft_min,
                     "min_hits": args.min_hits, "ctx_type": args.ctx_type,
                     "chat": args.chat, "temperature": args.temperature,
                     "server_extra": args.server_extra},
           "gpu_mem_baseline_mib": None if base is None else round(base, 1),
           "log": os.path.relpath(log, ROOT)}

    loaded, timings, err = None, None, None
    try:
        ok, why = wait_health(args.port, time.time() + args.load_timeout)
        if not ok:
            err = why
        else:
            loaded = round(time.time() - t0, 2)
            payload = {"prompt": prompt, "n_predict": args.n_predict,
                       "seed": 42, "cache_prompt": False, "stream": False}
            if args.temperature >= 0:
                payload["temperature"] = args.temperature
            # temperature < 0 omits the field, so the server applies the GGUF's own
            # general.sampling.* values (Flint: temp 1.0, top-p 0.95, top-k 20) --
            # the card's evaluation setting, and what agentic traffic runs at.
            if args.chat:
                # Real agent traffic goes through the chat template (reasoning,
                # tool framing). Same timings block, one message.
                path = "/v1/chat/completions"
                payload = {"model": args.alias, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": args.n_predict, "seed": 42, "cache_prompt": False,
                           "stream": False}
                if args.temperature >= 0:
                    payload["temperature"] = args.temperature
            else:
                path = "/completion"
            g0 = time.time()
            resp = http_json("http://127.0.0.1:%d%s" % (args.port, path),
                             payload, timeout=args.gen_timeout)
            rec["gen_wall_s"] = round(time.time() - g0, 3)
            timings = resp.get("timings")
            if args.chat:
                msg = (resp.get("choices") or [{}])[0].get("message") or {}
                rec["content_head"] = (msg.get("content") or "")[:64]
                rec["reasoning_head"] = (msg.get("reasoning_content") or "")[:64]
            else:
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


def acceptance(t):
    """accepted / drafted, or None when the arm used no drafter."""
    n, a = t.get("draft_n"), t.get("draft_n_accepted")
    if not n:
        return None
    return round(a / n, 4)


def summarize(rec):
    t = rec.get("timings") or {}
    acc = acceptance(t)
    print("%-14s gpu_peak=%-7s exit=%-3s load=%-6s bg_cpu=%-5s gpu%%=%-3s "
          "pp=%-9s tg=%-9s draft=%s/%s%s %s" % (
              rec["arm"], rec.get("gpu_mem_peak_mib"), rec["exit_code"],
              rec.get("load_s"), rec.get("bg_cpu_pct"),
              rec.get("gpu_busy_pct_peak"),
              round(t.get("prompt_per_second", 0), 2),
              round(t.get("predicted_per_second", 0), 3),
              t.get("draft_n_accepted", "-"), t.get("draft_n", "-"),
              "" if acc is None else "(%.1f%%)" % (100 * acc),
              ("ERR " + rec["error"]) if rec.get("error") else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=SERVER_DEFAULT)
    ap.add_argument("--model", default="models/flint-int4-q4_1.gguf")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (deterministic A/B); negative = omit the field so "
                         "the server uses the GGUF's own sampling defaults")
    ap.add_argument("--prompt-file", default="data/eval_corpus.txt")
    ap.add_argument("--arms", default="base,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--lookup-n", type=int, default=None,
                    help="n-gram length looked up (size-n / n-match)")
    ap.add_argument("--draft-m", type=int, default=None,
                    help="draft length for ngram-simple/map-* (size-m)")
    ap.add_argument("--draft-min", type=int, default=None,
                    help="ngram-mod n-min: drafts shorter than this are dropped")
    ap.add_argument("--draft-max-mod", type=int, default=64,
                    help="ngram-mod n-max (default 64)")
    ap.add_argument("--min-hits", type=int, default=None,
                    help="minimum occurrences for a lookup key (ngram-simple/map-*)")
    ap.add_argument("--synth-base", default="ngram-simple",
                    help="drafter to pair with synth-N arms (synthetic acceptance "
                         "needs a registered spec context)")
    ap.add_argument("--chat", action="store_true",
                    help="drive /v1/chat/completions (flint chat template, reasoning) "
                         "instead of raw /completion")
    ap.add_argument("--alias", default="flint", help="server model id for chat requests")
    ap.add_argument("--ctx-type", default="f16",
                    help="KV dtype (f16 keeps -fa optional; quantized needs -fa)")
    ap.add_argument("--swap-cap-mib", type=float, default=0.0)
    ap.add_argument("--load-timeout", type=float, default=180.0)
    ap.add_argument("--gen-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="results/ngram_ab.jsonl")
    ap.add_argument("--server-extra", default="")
    args = ap.parse_args(argv)

    server = args.server if os.path.isabs(args.server) else os.path.join(ROOT, args.server)
    args.server = server
    if not os.path.exists(server):
        raise SystemExit("no llama-server at %s -- run ./scripts/build_flint.sh" % server)

    prompt = open(os.path.join(ROOT, args.prompt_file)).read()
    _, idle0 = guard.gpu_stats()
    print("baseline GPU memory: %s MiB; prompt %d chars; server %s"
          % ("n/a" if idle0 is None else "%.1f" % idle0, len(prompt),
             os.path.relpath(server, ROOT)))

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    recs = []
    dead = set()
    for rep in range(args.reps):
        for name in names:
            if name in dead:
                continue
            rec = run_arm(name, args, prompt, idle0)
            rec["rep"] = rep
            rec["prompt_file"] = args.prompt_file
            recs.append(rec)
            if args.out:
                with open(os.path.join(ROOT, args.out), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            summarize(rec)
            if rec["swap_violation"]:
                print("skipping %s for the remaining reps: swap cap breached" % name)
                dead.add(name)
            if rec.get("error"):
                print("skipping %s for the remaining reps: %s" % (name, rec["error"]))
                dead.add(name)

    if args.out:
        print("wrote %d records to %s" % (len(recs), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
