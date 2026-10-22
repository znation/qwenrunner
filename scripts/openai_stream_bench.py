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
                    t_last = time.time()
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
        text = prompt if not args.nonce else "<!-- probe %d -->\n" % rep + prompt
        payload = make_payload(text)
        try:
            r = stream_once(url, payload, args.api_key)
        except Exception as e:  # noqa: BLE001
            print("  rep%d FAILED: %s: %s" % (rep, type(e).__name__, e))
            break
        r.update({"rep": rep, "label": args.label, "base_url": args.base_url,
                  "prompt_file": args.prompt_file, "max_tokens": args.max_tokens,
                  "temperature": args.temperature, "warmup": args.warmup,
                  "nonce": args.nonce,
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        recs.append(r)
        print("  rep%d ttft=%-7s wall=%-7s out=%-5s decode=%-8s finish=%-6s %r"
              % (rep, r["ttft_s"], r["wall_s"], r["n_out"], r["decode_tps"],
                 r["finish"], r["char_head"][:40]))
        if args.out:
            with open(os.path.join(ROOT, args.out), "a") as f:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
