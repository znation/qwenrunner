#!/usr/bin/env python3
"""Context-checkpoint probe: does a mid-transcript rewrite resume from a checkpoint?

This is the probe behind RESULTS.md's "Context checkpoints need
`message_delimiters`". Two things it establishes that a throughput harness
cannot:

  * `--mode raw` sends a bare /completion prompt. Mid-prompt checkpoints are
    gated on `spans.is_user_start(...)` (server-context.cpp:3828, :3885) and
    `spans` is built from the request's `message_delimiters`, so a raw prompt
    gets none -- only the pair the server always makes near the end of the
    prompt (server-context.cpp:3838, offsets 4+n_ubatch and 4). A rewrite then
    re-prefills everything, at every -cms, which is what the older
    "compaction pays full price" measurement actually showed.
  * `--mode chat` sends the same prompt as a Qwen-format transcript *with* the
    delimiters, which is what the differential autoparser does for every
    /v1/chat/completions request (common/chat.cpp:1363-1370) -- i.e. what pi and
    omp actually send. The checkpoint mechanism then fires and the readout
    collapses from the full prompt to a few hundred tokens.

Each arm issues two requests against one slot with cache_prompt=true: the
prompt, then the same prompt with one block replaced. timings.prompt_n is a
deterministic token count, so single reps are decisive -- but the *rate* is
still CPU-contention sensitive, so bg_cpu_pct is recorded with every arm.

Stdlib only, like the rest of the harness.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import guard  # noqa: E402

DELIMS = [
    {"role": "user",      "delimiter": "<|im_start|>user"},
    {"role": "assistant", "delimiter": "<|im_start|>assistant"},
    {"role": "system",    "delimiter": "<|im_start|>system"},
]


def build_prompts(args):
    """Return (base, edited). The two differ only inside one block."""
    body = open(os.path.join(ROOT, args.corpus)).read(args.turns * args.chunk * 2 + args.chunk)
    blocks = [body[i * args.chunk:(i + 1) * args.chunk] for i in range(args.turns)]
    spare = body[args.turns * args.chunk:args.turns * args.chunk + args.chunk]

    if args.mode == "raw":
        base = "".join(blocks)
        assert args.splice + len(spare) <= len(base), \
            "raw splice %d + block %d overruns the %d-char prompt" % (
                args.splice, len(spare), len(base))
        edited = base[:args.splice] + spare + base[args.splice + len(spare):]
        return base, edited

    def render(bs):
        p = "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
        for b in bs:
            p += "<|im_start|>user\n" + b + "<|im_end|>\n"
            p += "<|im_start|>assistant\nUnderstood, continuing.<|im_end|>\n"
        return p + "<|im_start|>assistant\n"

    edited_blocks = list(blocks)
    edited_blocks[args.rewrite_turn - 1] = spare
    return render(blocks), render(edited_blocks)


def http_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(port, deadline):
    while time.time() < deadline:
        try:
            st = http_json("http://127.0.0.1:%d/health" % port, timeout=5)
            if st.get("status") in ("ok", "no slot available"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def ask(args, prompt):
    payload = {"prompt": prompt, "n_predict": 4, "temperature": 0.0,
               "seed": 42, "cache_prompt": True, "stream": False}
    if args.mode == "chat":
        payload["message_delimiters"] = DELIMS
    g0 = time.time()
    r = http_json("http://127.0.0.1:%d/completion" % args.port, payload)
    t = r.get("timings") or {}
    return {"wall_s": round(time.time() - g0, 2), "prompt_n": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms")}


def run_arm(args, label, ckpt_flags, base, edited):
    log = os.path.join(ROOT, "logs", "ckpt_probe-%s.log" % label)
    cmd = [os.path.join(ROOT, "build/bin/llama-server"),
           "-m", args.model,
           "-ngl", str(args.ngl), "-c", str(args.ctx), "-t", str(args.threads),
           "-np", "1", "-lv", "4",
           "-dev", args.dev, "-sm", "layer", "-ts", args.ts,
           "-lm", args.lm, "-b", "2048", "-ub", "512",
           "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on",
           "--host", "127.0.0.1", "--port", str(args.port), "--seed", "42",
           "--spec-type", "draft-mtp", "-md", args.draft,
           "-devd", "Vulkan0", "-ngld", "99", "-ctkd", "q4_0", "-ctvd", "q4_0"] + ckpt_flags
    rec = {"arm": label, "mode": args.mode, "ckpt_flags": ckpt_flags, "cmd": cmd}
    print("### arm %s  %s" % (label, time.strftime("%H:%M:%S")))
    out = open(log, "wb")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=ROOT, start_new_session=True)
    mon = guard.Monitor(p.pid, args.cap_mib)
    mon.start()
    try:
        if not wait_health(args.port, time.time() + args.load_timeout):
            rec["error"] = "health timeout"
        else:
            rec["req1"] = ask(args, base)
            rec["req2"] = ask(args, edited)
    except Exception as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, e)
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
        out.close()
    rec["exit_code"] = p.returncode
    rec["vram_peak_mib"] = round(mon.vram_peak, 1)
    rec["client_vram_peak_mib"] = round(mon.client_peak, 1)
    rec["vram_violation"] = mon.violation
    rec["bg_cpu_pct"] = None if mon.bg_cpu_pct is None else round(mon.bg_cpu_pct, 1)
    rec["log"] = os.path.relpath(log, ROOT)
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    keep = []
    for line in open(log, errors="replace"):
        if ("created context checkpoint" in line or "restored context checkpoint" in line
                or "forcing full prompt" in line or "checking checkpoint" in line):
            keep.append(line.strip())
    rec["checkpoint_traces"] = keep[-60:]
    r1, r2 = rec.get("req1") or {}, rec.get("req2") or {}
    print("    req1 prompt_n=%-7s %-10s | req2 prompt_n=%-7s %-10s | %s" % (
        r1.get("prompt_n"), r1.get("prompt_ms"),
        r2.get("prompt_n"), r2.get("prompt_ms"), rec.get("error", "")))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("raw", "chat"), default="chat",
                    help="raw = bare /completion prompt (no message_delimiters); "
                         "chat = Qwen transcript + delimiters, i.e. what the clients send")
    ap.add_argument("--corpus", default="data/prompt_128k.txt")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=5200,
                    help="characters per turn (~1.2k tokens at 4.2 chars/token)")
    ap.add_argument("--rewrite-turn", type=int, default=9, help="1-based, --mode chat")
    ap.add_argument("--splice", type=int, default=45000,
                    help="character offset of the rewrite, --mode raw")
    ap.add_argument("--arms", default="",
                    help="';'-separated label:flags specs, flags space-separated, e.g. "
                         "\"cms8192:-ctxcp 32 -cms 8192;cms2048:-ctxcp 32 -cms 2048;off:-ctxcp 0\"")
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_M.gguf")
    ap.add_argument("--draft", default="models/mtp-Qwen3.8-27B-Q4_0.gguf")
    ap.add_argument("--ngl", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ts", default="6,1")
    ap.add_argument("--dev", default="CUDA0,Vulkan0")
    ap.add_argument("--lm", default="dio")
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--cap-mib", type=float, default=5000.0)
    ap.add_argument("--load-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="results/ckpt_probe.jsonl")
    args = ap.parse_args(argv)

    if not args.arms:
        args.arms = ("cms8192:-ctxcp 32 -cms 8192;"
                     "cms2048:-ctxcp 32 -cms 2048;"
                     "off:-ctxcp 0")

    arms = []
    for spec in args.arms.split(";"):
        if not spec.strip():
            continue
        label, _, flags = spec.partition(":")
        arms.append((label.strip(), flags.split()))

    base, edited = build_prompts(args)
    print("mode=%s prompts: %d / %d chars (~%d tokens), rewrite in turn %d"
          % (args.mode, len(base), len(edited), len(base) // 4, args.rewrite_turn))
    for label, flags in arms:
        rec = run_arm(args, label, flags, base, edited)
        if args.out:
            with open(os.path.join(ROOT, args.out), "a") as f:
                f.write(json.dumps(rec) + "\n")
        if rec.get("error"):
            print("stopping: %s failed" % label)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
