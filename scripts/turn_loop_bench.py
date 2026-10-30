#!/usr/bin/env python3
"""Agentic turn-loop benchmark: N concurrent sessions, each with a growing transcript.

What this models, and why a simpler harness doesn't: `openai_stream_bench.py
--concurrency N` sends N copies of one prompt. A real coding harness does not.
It holds a transcript, sends it every turn, gets back reasoning + a patch/tool
call, appends both, and does it
again -- so every turn is a *different* prompt, sharing a prefix with the previous
turn, with a short output. That is the workload the slot and prefill behaviour
has to be judged on.

Per session: an initial "here is the code I am working on" context, then
`--turns` rounds of (generate -> append assistant text -> append a tool
observation drawn from the same corpus slice). Sessions run in their own threads
and fire as soon as they are ready, so the server sees up to N overlapping
requests, exactly like N harnesses pointed at one endpoint.

Reported: per-turn TTFT and wall for every turn (from a streaming client, so TTFT
is what a user waits for), p50/p90/max across turns, per-session means, and the
aggregate output/prefill rates. A turn whose TTFT is small relative to its prompt
is a slot/prefix-cache hit; one that is large is a re-prefill. That contrast is
the signal to watch.

Stdlib only. Serialized by design: one server at a time.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREAMBLE = ("You are a coding agent working in a checkout. The user has given you a "
            "task; work on it in small steps. Reply with your reasoning followed by "
            "the concrete change you would make.")


def stream_chat(url, api_key, model, messages, max_tokens, temperature, timeout=3600):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "stream": True, "seed": 42, "stream_options": {"include_usage": True}}
    if temperature >= 0:
        payload["temperature"] = temperature
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)

    t0 = time.time()
    t_first = t_last = None
    content, reasoning = [], []
    n_delta = 0
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
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
                # llama.cpp streams `reasoning_content`, rec streams `reasoning`.
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                cc = d.get("content") or ""
                if rc or cc:
                    n_delta += 1
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time()
                    if rc:
                        reasoning.append(rc)
                    if cc:
                        content.append(cc)
    t_end = time.time()
    n_out = (usage or {}).get("completion_tokens") or n_delta
    return {
        "ttft_s": None if t_first is None else round(t_first - t0, 3),
        "wall_s": round(t_end - t0, 3),
        "decode_s": None if not (t_first and t_last) else round(t_last - t_first, 3),
        "n_out": n_out,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "finish": finish,
        "text": "".join(content),
        "reasoning": "".join(reasoning),
    }


def chunker(text, want_chars):
    """Yield successive observation chunks of roughly want_chars from the slice."""
    i = 0
    while True:
        piece = text[i:i + want_chars]
        if len(piece) < want_chars // 4:
            i = 0
            piece = text[i:i + want_chars]
        yield piece
        i += len(piece)
        if i >= len(text):
            i = 0


def session_loop(sid, slice_text, args, records, errors):
    try:
        obs_gen = chunker(slice_text, args.obs_chars)
        start = slice_text[:args.initial_chars]
        messages = [{"role": "system", "content": PREAMBLE},
                    {"role": "user", "content":
                     "Session %d. Here is the code I am working on:\n\n%s\n\n"
                     "Summarise what it does, then suggest one concrete improvement."
                     % (sid, start)}]
        for turn in range(args.turns):
            r = stream_chat(args.base_url, args.api_key, args.model, messages,
                            args.max_tokens, args.temperature)
            rec = {
                "session": sid, "turn": turn, "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
                "decode_s": r["decode_s"], "n_out": r["n_out"],
                "prompt_tokens": r["prompt_tokens"],
                "decode_tps": None if not r["decode_s"] or r["n_out"] < 2
                              else round((r["n_out"] - 1) / r["decode_s"], 2),
                "finish": r["finish"],
                "context_chars": sum(len(m["content"]) for m in messages),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            records.append(rec)
            if args.out:
                with open(os.path.join(ROOT, args.out), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if not (r["text"] or r["reasoning"]):
                errors.append("session %d turn %d: empty reply (%s)" % (sid, turn, r["finish"]))
                break

            # The harness's own bookkeeping: assistant text (with the reasoning the
            # model card says to preserve), then the next tool observation.
            amsg = {"role": "assistant", "content": r["text"]}
            if r["reasoning"]:
                amsg["reasoning_content"] = r["reasoning"]
            messages.append(amsg)
            messages.append({"role": "user", "content":
                             "Output of the last command:\n\n%s\n\nContinue." % next(obs_gen)})
    except Exception as e:  # noqa: BLE001
        errors.append("session %d: %s: %s" % (sid, type(e).__name__, e))


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    i = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
    return round(vals[i], 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8090/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", default="")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="negative omits the field (server's own sampling defaults)")
    ap.add_argument("--corpus", default="data/prompt_100k.txt",
                    help="each session gets one slice of this as its 'repo'")
    ap.add_argument("--initial-chars", type=int, default=12000,
                    help="opening context per session (~3.4k tokens at 3.5 chars/token)")
    ap.add_argument("--obs-chars", type=int, default=2400,
                    help="size of each appended tool observation")
    ap.add_argument("--out", default="results/turn_loop.jsonl")
    args = ap.parse_args(argv)

    corpus = open(os.path.join(ROOT, args.corpus)).read()
    args.base_url = args.base_url.rstrip("/") + "/chat/completions"
    size = len(corpus) // args.sessions
    slices = [corpus[i * size:(i + 1) * size + args.initial_chars] for i in range(args.sessions)]
    print("%s: %s sessions=%d turns=%d max_tokens=%d corpus=%s (%.1fk chars/session)"
          % (args.label or "turn-loop", args.base_url, args.sessions, args.turns,
             args.max_tokens, os.path.basename(args.corpus), size / 1000))

    records, errors = [], []
    threads = [threading.Thread(target=session_loop, args=(i, slices[i], args, records, errors))
               for i in range(args.sessions)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    turns = sorted(records, key=lambda r: (r["turn"], r["session"]))
    print("  turns: %d in %.1fs wall" % (len(turns), wall))
    for r in turns:
        print("    turn%d s%d ttft=%-8s wall=%-8s out=%-4s ctx=%.1fk chars"
              % (r["turn"], r["session"], r["ttft_s"], r["wall_s"], r["n_out"],
                 r["context_chars"] / 1000))
    ttfts = [r["ttft_s"] for r in records]
    walls = [r["wall_s"] for r in records]
    n_out = sum(r["n_out"] for r in records)
    n_in = sum(r["prompt_tokens"] for r in records if r["prompt_tokens"])
    print("  TTFT   p50=%-7s p90=%-7s max=%-7s" % (pct(ttfts, .5), pct(ttfts, .9), pct(ttfts, 1.0)))
    print("  wall   p50=%-7s p90=%-7s max=%-7s" % (pct(walls, .5), pct(walls, .9), pct(walls, 1.0)))
    for s in range(args.sessions):
        sr = [r for r in records if r["session"] == s]
        if sr:
            print("    session %d: turns=%d mean_wall=%.2fs mean_ttft=%.2fs out=%d"
                  % (s, len(sr), statistics.mean(r["wall_s"] for r in sr),
                     statistics.mean(r["ttft_s"] for r in sr if r["ttft_s"] is not None),
                     sum(r["n_out"] for r in sr)))
    print("  aggregate: out=%.2f t/s  prefill=%s t/s  (out=%d tok, prompt=%s tok)"
          % (n_out / wall, "n/a" if not n_in else round(n_in / wall, 1), n_out, n_in or "?"))
    if errors:
        print("  errors: %s" % "; ".join(errors[:4]))

    if args.out:
        summary = {"kind": "turn_loop_summary", "label": args.label, "sessions": args.sessions,
                   "turns_per_session": args.turns, "max_tokens": args.max_tokens,
                   "wall_s": round(wall, 2), "ttft_p50": pct(ttfts, .5), "ttft_p90": pct(ttfts, .9),
                   "ttft_max": pct(ttfts, 1.0), "wall_p50": pct(walls, .5),
                   "wall_p90": pct(walls, .9), "wall_max": pct(walls, 1.0),
                   "out_tps": round(n_out / wall, 2),
                   "prefill_tps": None if not n_in else round(n_in / wall, 1),
                   "n_out": n_out, "n_in": n_in, "errors": errors,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(os.path.join(ROOT, args.out), "a") as f:
            f.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
