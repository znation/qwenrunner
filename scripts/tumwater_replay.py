#!/usr/bin/env python3
"""Replay real tumwater pi sessions against an OpenAI-compatible server, concurrently.

Why replay rather than synthesise: tumwater's load has a specific shape that a
synthetic loop gets wrong in ways that matter for engine choice. Measured over
19 358 recorded turns (508 sessions, ~/tumwater/.tumwater/sessions):

    first prompt     ~7.4k tokens          turns/session  p50 29, p90 78
    request prompt   p50 40k, p90 81k      new tokens/turn p50 747, p90 3.2k
    output           p50 333, p90 2.2k     prefix reused   ~92 %

and roughly one turn in 23 carries a context_edit (tumwater's context-shake
elides an old tool output in place), which invalidates the cached prefix from
that point on. So a turn is "decode a few hundred tokens at 20-80k depth, after
prefilling a ~750-token delta", four sessions at a time — prefix-cache retention
across four long sessions and batched decode at depth are what decide it.

Two subcommands:

  extract  pull S sessions out of the pi session store into a self-contained
           fixture (data/tumwater_replay.json): OpenAI chat messages per
           request point, the recorded output length, and context edits applied
           where they happened. Thinking is dropped from assistant history, as
           tumwater's strip-thinking extension does, so each request diverges
           from the server's cached state right where the real one does.
  run      replay the fixture: one thread per session, each sending its
           requests in order as soon as the previous one returns. The first
           --warm turns of each session are sent (to build the prefix) but not
           scored, so the scored turns sit at realistic depth.

Reported per arm: wall time of the scored region, aggregate output t/s, per-turn
TTFT and per-stream decode t/s percentiles, prompt tokens the server says it
cached (when it reports them), and peak wired memory.

Stdlib only. One server at a time.
"""

import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORE = os.path.expanduser("~/tumwater/.tumwater/sessions")


# --------------------------------------------------------------------- extract

def _text(content):
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        if c.get("type") == "text":
            out.append(c.get("text", ""))
        elif c.get("type") == "image":
            out.append("[image]")
    return "".join(out)


def _system(msg):
    if msg.get("content"):
        return _text(msg["content"])
    secs = msg.get("sections") or {}
    return "\n\n".join(v for v in secs.values() if isinstance(v, str) and v)


def session_requests(path):
    """Walk one pi session and return its request points.

    Each request point is (messages, recorded_output_tokens, recorded_prompt).
    Messages are kept by pi message id so context_edits can rewrite them.
    """
    order, by_id = [], {}
    reqs = []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "context_edit":
            t = by_id.get(r.get("targetId"))
            rep = (r.get("replacement") or {}).get("content")
            if t is not None and rep is not None:
                t["content"] = _text(rep)
            continue
        if r.get("type") != "message":
            continue
        m = r.get("message") or {}
        role = m.get("role")
        if role == "assistant":
            u = m.get("usage") or {}
            prompt = u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0)
            if prompt:
                snap = [dict(x) for x in (by_id[i] for i in order)]
                for x in snap:
                    if "tool_calls" in x:
                        x["tool_calls"] = [dict(tc) for tc in x["tool_calls"]]
                reqs.append({"messages": snap, "out": u.get("output", 0), "prompt": prompt})
            text = "".join(c.get("text", "") for c in m.get("content") or []
                           if c.get("type") == "text")
            calls = [{"id": c.get("id") or "call_%d" % k, "type": "function",
                      "function": {"name": c.get("name", "tool"),
                                   "arguments": json.dumps(c.get("arguments") or {})}}
                     for k, c in enumerate(m.get("content") or []) if c.get("type") == "toolCall"]
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
        elif role == "toolResult":
            msg = {"role": "tool", "tool_call_id": m.get("toolCallId", ""),
                   "content": _text(m.get("content"))}
        elif role == "system":
            msg = {"role": "system", "content": _system(m)}
        elif role == "user":
            msg = {"role": "user", "content": _text(m.get("content"))}
        else:
            continue
        mid = r.get("id") or str(len(order))
        by_id[mid] = msg
        order.append(mid)
    return reqs


def cmd_extract(args):
    files = sorted(glob.glob(os.path.join(args.store, "*", "*.jsonl")))
    cands = []
    for f in files:
        role = os.path.basename(os.path.dirname(f))
        if role.startswith("_"):
            continue
        reqs = session_requests(f)
        need = args.warm + args.turns
        if len(reqs) < need:
            continue
        # Score window depth: the request depth tumwater actually runs at.
        depth = reqs[need - 1]["prompt"]
        if not (args.min_depth <= depth <= args.max_depth):
            continue
        cands.append((f, role, reqs[:need]))
    if len(cands) < args.sessions:
        sys.exit("extract: only %d sessions qualify (need %d)" % (len(cands), args.sessions))
    rng = random.Random(args.seed)
    # Prefer distinct roles, the way a fleet mixes them.
    rng.shuffle(cands)
    picked, roles = [], set()
    for c in cands:
        if c[1] not in roles:
            picked.append(c)
            roles.add(c[1])
        if len(picked) == args.sessions:
            break
    for c in cands:
        if len(picked) == args.sessions:
            break
        if c not in picked:
            picked.append(c)

    # Store each session as a base transcript plus deltas, not a full snapshot per
    # request: request k is base[:prefix_k] (after edits applied so far) + tail.
    out = {"source": args.store, "warm": args.warm, "turns": args.turns,
           "extracted": time.strftime("%Y-%m-%d"), "sessions": []}
    for f, role, reqs in picked:
        out["sessions"].append({
            "file": os.path.basename(f), "role": role,
            "requests": [{"messages": r["messages"], "out": r["out"], "prompt": r["prompt"]}
                         for r in reqs]})
        print("  %-12s %s  depth %6d -> %6d  outs p50 %d"
              % (role, os.path.basename(f)[:30], reqs[0]["prompt"], reqs[-1]["prompt"],
                 statistics.median(r["out"] for r in reqs)))
    path = os.path.join(ROOT, args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("extract: %d sessions x %d requests -> %s (%.1f MB)"
          % (len(picked), args.warm + args.turns, args.out, os.path.getsize(path) / 1e6))
    return 0


# ------------------------------------------------------------------------- run

def flatten(messages):
    """Tool calls as plain text, for servers whose tool-message handling differs."""
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "user", "content": "<tool_response>\n%s\n</tool_response>"
                        % m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join("<tool_call>\n%s\n</tool_call>" % json.dumps(
                {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])})
                for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m["content"] + "\n" + calls).strip()})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    # Merge consecutive same-role messages (several tool results in a row).
    merged = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n" + m["content"]}
        else:
            merged.append(m)
    return merged


def stream_chat(url, args, messages, max_tokens):
    payload = {"model": args.model, "messages": messages, "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True}}
    if args.temperature >= 0:
        payload.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.extra_body:
        payload.update(json.loads(args.extra_body))
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    t0 = time.time()
    t_first = t_last = None
    n_delta, usage, finish = 0, None, None
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
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
                got = (d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                       or d.get("tool_calls"))
                if got:
                    n_delta += 1
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
    t_end = time.time()
    usage = usage or {}
    n_out = usage.get("completion_tokens") or n_delta
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft_s": None if t_first is None else t_first - t0, "wall_s": t_end - t0,
            "decode_s": None if t_first is None else t_last - t_first,
            "n_out": n_out, "n_delta": n_delta, "finish": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens")}


def session_worker(sid, sess, url, args, warm, records, errors, barrier, t_mark):
    try:
        for k, req in enumerate(sess["requests"]):
            if k == warm:
                barrier.wait()  # everyone enters the scored region together
                t_mark.setdefault("t0", time.time())
            msgs = flatten(req["messages"]) if args.flatten else req["messages"]
            mt = max(args.min_out, min(args.max_out, req["out"])) if args.max_out else req["out"]
            t_start = time.time()
            r = stream_chat(url, args, msgs, mt)
            r.update(session=sid, role=sess["role"], turn=k, scored=k >= warm,
                     max_tokens=mt, rec_prompt=req["prompt"], t_start=t_start, t_end=time.time())
            records.append(r)
            if args.verbose:
                print("    s%d t%-2d %s ttft=%6.2f out=%4d/%-4d dec=%6.1f t/s prompt=%s cached=%s"
                      % (sid, k, "*" if k >= warm else " ", r["ttft_s"] or -1, r["n_out"], mt,
                         (r["n_out"] - 1) / r["decode_s"] if r["decode_s"] else 0,
                         r["prompt_tokens"], r["cached_tokens"]), flush=True)
    except Exception as e:  # noqa: BLE001
        errors.append("session %d: %s: %s" % (sid, type(e).__name__, e))
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass


class MemSampler(threading.Thread):
    """Peak wired + compressor memory, sampled from vm_stat."""

    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop = every, 0, threading.Event()

    def sample(self):
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = 16384
        vals = {}
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                k, v = line.split(":", 1)
                try:
                    vals[k.strip()] = int(v.strip().rstrip("."))
                except ValueError:
                    pass
        return (vals.get("Pages wired down", 0) + vals.get("Pages occupied by compressor", 0)) \
            * page / 2**30

    def run(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.sample())
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(self.every)


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def cmd_run(args):
    fx = json.load(open(os.path.join(ROOT, args.fixture)))
    sessions = fx["sessions"][:args.sessions]
    warm = fx["warm"] if args.warm is None else args.warm
    turns = args.turns or (len(sessions[0]["requests"]) - warm)
    for s in sessions:
        s["requests"] = s["requests"][:warm + turns]
    url = args.base_url.rstrip("/") + "/chat/completions"
    print("%s: %s sessions=%d warm=%d scored=%d max_out=%s temp=%s flatten=%s"
          % (args.label or "replay", url, len(sessions), warm, turns, args.max_out,
             args.temperature, args.flatten), flush=True)

    load = os.getloadavg()[0]
    mem = MemSampler()
    mem.start()
    records, errors, t_mark = [], [], {}
    barrier = threading.Barrier(len(sessions))
    threads = [threading.Thread(target=session_worker,
                                args=(i, s, url, args, warm, records, errors, barrier, t_mark))
               for i, s in enumerate(sessions)]
    t_all = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.time()
    mem.stop.set()

    sc = [r for r in records if r["scored"]]
    warm_recs = [r for r in records if not r["scored"]]
    wall = t_end - t_mark.get("t0", t_all)
    n_out = sum(r["n_out"] for r in sc)
    dec = [(r["n_out"] - 1) / r["decode_s"] for r in sc if r["decode_s"] and r["n_out"] > 8]
    ttft = [r["ttft_s"] for r in sc]
    prompt = [r["prompt_tokens"] for r in sc if r["prompt_tokens"]]
    cached = [r["cached_tokens"] for r in sc if r["cached_tokens"] is not None]
    turn_s = [r["wall_s"] for r in sc]
    summary = {
        "kind": "replay_summary", "label": args.label, "sessions": len(sessions),
        "warm": warm, "scored_turns": len(sc), "errors": errors,
        "wall_s": round(wall, 1), "warm_wall_s": round(t_mark.get("t0", t_end) - t_all, 1),
        "out_tokens": n_out, "agg_out_tps": round(n_out / wall, 2) if wall else None,
        "turns_per_min": round(len(sc) / wall * 60, 2) if wall else None,
        "stream_tps_p50": round(pct(dec, .5), 2) if dec else None,
        "stream_tps_p10": round(pct(dec, .1), 2) if dec else None,
        "ttft_p50": round(pct(ttft, .5), 2) if ttft else None,
        "ttft_p90": round(pct(ttft, .9), 2) if ttft else None,
        "ttft_max": round(pct(ttft, 1.0), 2) if ttft else None,
        "turn_s_p50": round(pct(turn_s, .5), 2) if turn_s else None,
        "prompt_p50": pct(prompt, .5), "prompt_max": pct(prompt, 1.0),
        "cached_frac": round(sum(cached) / sum(prompt), 3) if cached and prompt else None,
        "hit_max_tokens": sum(1 for r in sc if r["finish"] == "length"),
        "warm_ttft_max": round(max((r["ttft_s"] or 0) for r in warm_recs), 1) if warm_recs else None,
        "peak_wired_compr_gib": round(mem.peak, 1), "loadavg_start": round(load, 2),
        "loadavg_end": round(os.getloadavg()[0], 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for k in ("wall_s", "agg_out_tps", "turns_per_min", "stream_tps_p50", "ttft_p50",
              "ttft_p90", "turn_s_p50", "prompt_p50", "cached_frac", "hit_max_tokens",
              "peak_wired_compr_gib", "loadavg_start", "loadavg_end"):
        print("  %-22s %s" % (k, summary[k]))
    if errors:
        print("  errors: %s" % "; ".join(errors[:4]))
    if args.out:
        path = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            for r in records:
                f.write(json.dumps(dict(r, kind="replay_turn", label=args.label)) + "\n")
            f.write(json.dumps(summary) + "\n")
    return 1 if errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--store", default=DEFAULT_STORE)
    ex.add_argument("--sessions", type=int, default=8)
    ex.add_argument("--warm", type=int, default=12)
    ex.add_argument("--turns", type=int, default=16)
    ex.add_argument("--min-depth", type=int, default=25000,
                    help="recorded prompt at the last request must be at least this")
    ex.add_argument("--max-depth", type=int, default=90000)
    ex.add_argument("--seed", type=int, default=7)
    ex.add_argument("--out", default="data/tumwater_replay.json")
    ex.set_defaults(fn=cmd_extract)

    rn = sub.add_parser("run")
    rn.add_argument("--base-url", required=True)
    rn.add_argument("--api-key", default="")
    rn.add_argument("--model", default="qwen3.8-27b")
    rn.add_argument("--label", default="")
    rn.add_argument("--fixture", default="data/tumwater_replay.json")
    rn.add_argument("--sessions", type=int, default=4)
    rn.add_argument("--warm", type=int, default=None, help="default: the fixture's")
    rn.add_argument("--turns", type=int, default=None, help="scored turns; default: all")
    rn.add_argument("--max-out", type=int, default=512,
                    help="cap each turn's max_tokens at this (0 = recorded length)")
    rn.add_argument("--min-out", type=int, default=128)
    rn.add_argument("--temperature", type=float, default=0.6,
                    help="Qwen's recommended thinking sampling; negative = server default")
    rn.add_argument("--top-p", type=float, default=0.95)
    rn.add_argument("--top-k", type=int, default=20)
    rn.add_argument("--seed", type=int, default=1234)
    rn.add_argument("--extra-body", default="", help="JSON merged into every request")
    rn.add_argument("--flatten", action="store_true",
                    help="send tool calls/results as plain text messages")
    rn.add_argument("--timeout", type=float, default=3600)
    rn.add_argument("--verbose", "-v", action="store_true")
    rn.add_argument("--out", default="results/replay.jsonl")
    rn.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
