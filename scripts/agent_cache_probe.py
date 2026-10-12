#!/usr/bin/env python3
"""agent_cache_probe.py — does a growing agent transcript get re-prefilled?

An agent loop re-sends a transcript that grows every turn. If the server can
reuse the KV prefix it already has, a turn costs the delta; if it cannot, every
turn pays a full prefill of the whole context (at 128k that is minutes, not
seconds). llama-server reports this directly: `timings.prompt_n` is how many
tokens it actually had to process, so a cached prefix shows up as a tiny
prompt_n on an identical re-send.

The trap this measures is slot *eviction*. With N conversations and fewer slots
than conversations, each conversation's prefix gets pushed out by the next one,
and every turn re-prefills. That is the real argument for parallelism in
agentic use -- not throughput, which saturates.

Usage:
    python3 scripts/agent_cache_probe.py --port 8090 --rounds 2
    python3 scripts/agent_cache_probe.py --port 8090 --interleave   # A B A
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(port, prompt, n_predict, timeout):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port,
        data=json.dumps({"prompt": prompt, "n_predict": n_predict,
                         "temperature": 0.0, "seed": 42,
                         "cache_prompt": True, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    t = d.get("timings") or {}
    return {"wall_s": round(time.time() - t0, 2),
            "prompt_n": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_per_second": t.get("predicted_per_second")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--prompt-a", default="data/prompt_30k.txt")
    ap.add_argument("--prompt-b", default="data/prompt_30k_b.txt")
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--interleave", action="store_true",
                    help="A B A: what a second conversation does to the first "
                         "one's cached prefix (default is A A A)")
    ap.add_argument("--grow", action="store_true",
                    help="A then A+delta: the agent loop, where each turn "
                         "re-sends a transcript that grew")
    args = ap.parse_args()

    a = open(os.path.join(ROOT, args.prompt_a)).read()
    b = open(os.path.join(ROOT, args.prompt_b)).read()
    grow = "\n\nUSER: continue where you left off.\nASSISTANT:\n"
    if args.grow:
        seq = [("A", a), ("A+grow", a + grow)]
    elif args.interleave:
        seq = [("A", a), ("B", b), ("A", a)]
    else:
        seq = [("A", a), ("A", a), ("A", a)]

    _, idle_gpu = guard.gpu_stats()
    print("port %d, ~%d-token prompts, GPU idle %.0f MiB"
          % (args.port, len(a) and 30000, idle_gpu or 0))
    print("%-4s %-9s %-11s %-12s %s" % ("req", "wall_s", "prompt_n", "prefill t/s", "gen t/s"))
    for i, (tag, prompt) in enumerate(seq, 1):
        try:
            r = post(args.port, prompt, args.n_predict, args.timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as e:
            print("%-4s FAILED %s: %s" % (i, type(e).__name__, e))
            return 1
        print("%-4s %-9s %-11s %-12s %s"
              % ("%d%s" % (i, tag), r["wall_s"], r["prompt_n"],
                 "%.1f" % (r["prompt_per_second"] or 0),
                 "%.1f" % (r["predicted_per_second"] or 0)))
        sys.stdout.flush()
    print("\nprompt_n is how many tokens the server had to process. A cached "
          "prefix shows up as a small prompt_n on a re-send.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
