#!/usr/bin/env python3
"""Streaming benchmark against any OpenAI-compatible endpoint.

Why this exists next to scripts/ngram_ab.py: llama-server reports a `timings`
block, but a non-llama.cpp runtime (e.g. Recursive's own `rec`, which is MLX
under the hood) does not. Comparing two runtimes on llama.cpp's own timing
fields is not a fair comparison anyway, so this measures both stacks the same
way -- from the wire:

  TTFT        time from request to the first streamed token (the prefill cost)
  decode t/s  (n_out - 1) / (last_token_time - first_token_time), i.e. pure
              generation rate with prefill divided out
  wall        total request time, for the latency a user actually feels

Token counts come from the number of streamed deltas, cross-checked against the
`usage` block when the server sends one (llama.cpp does not in streaming mode
unless asked, MLX servers usually do).

Stdlib only. Serialized by design: run one arm at a time.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stream_once(url, payload, api_key):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=body, headers=headers)

    t0 = time.time()
    t_first = None
    t_last = None
    n_delta = 0
    text = []
    usage = None
    finish = None
    first_abs = None
    last_abs = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                # llama.cpp streams `reasoning_content`, Recursive's `rec` streams
                # `reasoning`; both stream `content`. Count all three.
                piece = ((d.get("content") or "") + (d.get("reasoning_content") or "")
                         + (d.get("reasoning") or ""))
                if piece:
                    n_delta += 1
                    if t_first is None:
                        t_first = time.time()
                        first_abs = t_first
                    t_last = time.time()
                    last_abs = t_last
                    text.append(piece)
    t_end = time.time()

    n_out = (usage or {}).get("completion_tokens") or n_delta
    decode_s = (t_last - t_first) if (t_first and t_last) else None
    return {
        "ttft_s": None if t_first is None else round(t_first - t0, 3),
        "wall_s": round(t_end - t0, 3),
        "n_out": n_out,
        "n_delta": n_delta,
        "finish": finish,
        "usage": usage,
        "decode_s": None if decode_s is None else round(decode_s, 3),
        "decode_tps": None if not decode_s or n_out < 2 else round((n_out - 1) / decode_s, 2),
        "first_abs": first_abs,
        "last_abs": last_abs,
        "char_head": ("".join(text))[:64].replace("\n", " "),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://127.0.0.1:8091/v1 or http://127.0.0.1:11500/v1")
    ap.add_argument("--model", default="flint")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--label", default="", help="name for this arm in the output")
    ap.add_argument("--prompt-file", default="data/flint_agentic_30k.txt")
    ap.add_argument("--chat", action="store_true", default=True)
    ap.add_argument("--raw", dest="chat", action="store_false",
                    help="use /completions (raw prompt) instead of /chat/completions")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="fire N requests at once; each gets a distinct prompt marker. "
                         "With batching (-np N in llama-server) the streams overlap and "
                         "aggregate_tps is the combined rate; with a single slot, or an "
                         "engine that queues, it collapses to the single-stream rate")
    ap.add_argument("--prompt-slices", type=int, default=0,
                    help="split the prompt file into N slices and give session i slice i, "
                         "so concurrent requests carry *different* content (different MoE "
                         "expert unions) instead of near-identical prompts")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="negative omits the field (server default)")
    ap.add_argument("--warmup", action="store_true",
                    help="send max_tokens=1 first, so the measured rep has weights "
                         "and kernels warm (page cache / metallib / cudnn-style init)")
    ap.add_argument("--nonce", action="store_true",
                    help="append a per-rep marker so prompt caching cannot hide prefill")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prompt = open(os.path.join(ROOT, args.prompt_file)).read()
    url = args.base_url.rstrip("/") + ("/chat/completions" if args.chat else "/completions")
    print("%s: %s model=%s prompt=%d chars max_tokens=%d"
          % (args.label or "arm", url, args.model, len(prompt), args.max_tokens))

    def make_payload(text):
        if args.chat:
            p = {"model": args.model, "stream": True,
                 "messages": [{"role": "user", "content": text}],
                 "max_tokens": args.max_tokens, "seed": 42}
        else:
            p = {"model": args.model, "stream": True, "prompt": text,
                 "n_predict": args.max_tokens, "seed": 42}
        if args.temperature >= 0:
            p["temperature"] = args.temperature
        return p

    if args.warmup:
        w = dict(make_payload(prompt))
        if args.chat:
            w["max_tokens"] = 1
        else:
            w["n_predict"] = 1
        try:
            r = stream_once(url, w, args.api_key)
            print("  warmup: ttft=%s wall=%s" % (r["ttft_s"], r["wall_s"]))
        except Exception as e:  # noqa: BLE001
            print("  warmup FAILED: %s: %s" % (type(e).__name__, e))

    recs = []
    for rep in range(args.reps):
        # The nonce goes at the FRONT: llama.cpp's prompt cache matches on the
        # prefix, so appending it would leave the whole prompt a cache hit and
        # measure nothing.
        base_text = prompt if not args.nonce else "<!-- probe %d -->\n" % rep + prompt

        results = [None] * args.concurrency
        errors = [None] * args.concurrency

        slices = None
        if args.prompt_slices:
            n = args.prompt_slices
            size = (len(prompt) + n - 1) // n
            slices = [prompt[i * size:(i + 1) * size] for i in range(n)]

        def worker(i):
            if slices:
                text = slices[i % len(slices)]
            else:
                text = base_text if args.concurrency == 1 else "<!-- session %d -->\n" % i + base_text
            try:
                results[i] = stream_once(url, make_payload(text), args.api_key)
            except Exception as e:  # noqa: BLE001
                errors[i] = "%s: %s" % (type(e).__name__, e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
        wall0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - wall0

        if any(results):
            wins = [r for r in results if r]
            # Aggregate over the window in which *any* stream was producing: the
            # union of the per-request decode windows. With batching the streams
            # overlap and this is the honest combined rate; serialized (a single
            # slot, or an engine that queues) the windows barely overlap and the
            # aggregate collapses to the single-stream rate.
            first = min(r["first_abs"] for r in wins)
            last = max(r["last_abs"] for r in wins)
            n_tok = sum(r["n_out"] for r in wins)
            agg = round(n_tok / (last - first), 2) if last > first else None
            agg_wall = round(n_tok / wall, 2) if wall > 0 else None
            r = dict(wins[0])
            r.update({
                "concurrency": args.concurrency,
                "requests_ok": len(wins),
                "per_request_tps": sorted(round(x["decode_tps"] or 0, 1) for x in wins),
                "aggregate_tps": agg,
                "aggregate_wall_tps": agg_wall,
                "aggregate_tokens": n_tok,
                "window_s": round(last - first, 3),
                "wall_all_s": round(wall, 3),
                "errors": [e for e in errors if e],
                "rep": rep,
                "label": args.label,
                "base_url": args.base_url,
                "prompt_file": args.prompt_file,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            for k in ("ttft_s", "wall_s", "decode_s", "n_delta"):
                r.pop(k, None)
            recs.append(r)
            print("  rep%d conc=%-2d ok=%-2d per-req=%-22s aggregate=%-8s wall-rate=%-8s (%d tok / %ss, wall %ss)%s"
                  % (rep, args.concurrency, len(wins), r["per_request_tps"], agg, agg_wall, n_tok,
                     r["window_s"], r["wall_all_s"],
                     "" if not r["errors"] else "  ERR " + r["errors"][0][:60]))
            if args.out:
                with open(os.path.join(ROOT, args.out), "a") as f:
                    f.write(json.dumps(r) + "\n")
        else:
            print("  rep%d conc=%-2d FAILED: %s" % (rep, args.concurrency, errors[0]))
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
