#!/usr/bin/env python3
"""Parameterized llama-bench sweep driver with a hard VRAM cap.

Every configuration runs as its own llama-bench invocation under
scripts/guard.py, so a VRAM-cap breach kills only that configuration and is
recorded as a violation instead of taking down the sweep.

Results are appended to results/bench.jsonl (one row per configuration) and
rendered to results/bench.csv + results/bench.md.

Examples
--------
  ./scripts/bench.py --stage ngl --reps 2 --ngl 0 --out results/bench.jsonl
  ./scripts/bench.py --stage kv --ngl 99 --fa on --reps 3
  ./scripts/bench.py --models a.gguf,b.gguf --stage quant
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard import run_guarded, vram_used_mib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_BIN = os.path.join(ROOT, "build", "bin", "llama-bench")

DEFAULT_MODEL = "models/Qwen3.8-27B-UD-Q4_K_M.gguf"

# A just-exited arm's DRM client lingers for seconds. Sampling the baseline in
# that window makes the next stage look already over the cap, and guard.py then
# kills every one of its arms at 0 s — observed losing an entire `threads` stage
# to a 5117.6 MiB phantom baseline left behind by an `-ngl 12` cap breach.
SETTLE_QUIET_S = 6.0
SETTLE_POLL_S = 2.0
SETTLE_SLACK_MIB = 32.0


def settle_vram(cap_mib=None, timeout=60.0):
    """Block until system VRAM stops falling *and* is back inside the cap.

    Waits for a stale DRM client from a just-exited arm to be reaped. If the
    reading stabilises at or above the cap the wait is extended to `timeout` and
    then returns anyway with a warning — the desktop may legitimately be using
    that much, and guard.py remains the backstop.
    """
    t0 = time.time()
    prev = None
    stable_since = None
    while time.time() - t0 < timeout:
        v = vram_used_mib()
        if v is None:
            return v
        inside = cap_mib is None or v <= cap_mib
        if inside and prev is not None and abs(v - prev) <= SETTLE_SLACK_MIB:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since >= SETTLE_QUIET_S:
                return v
        else:
            stable_since = None
        prev = v
        time.sleep(SETTLE_POLL_S)
    v = vram_used_mib()
    if cap_mib is not None and v is not None and v > cap_mib:
        print("warning: VRAM still %.1f MiB after %.0fs settle (cap %.1f); "
              "the desktop may be holding it" % (v, timeout, cap_mib))
    return v

# llama-bench JSON scalar fields we care about (see tools/llama-bench/llama-bench.cpp).
ROW_FIELDS = [
    "build_commit", "build_number", "cpu_info", "gpu_info", "backends",
    "model_filename", "model_type", "model_size", "model_n_params",
    "n_batch", "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll",
    "type_k", "type_v", "n_gpu_layers", "n_cpu_moe", "split_mode", "main_gpu",
    "no_kv_offload", "flash_attn", "devices", "tensor_split",
    "tensor_buft_overrides", "load_mode", "lazy_mode", "embeddings",
    "no_op_offload", "no_host", "repack", "fit_target", "fit_min_ctx",
    "n_prompt", "n_gen", "n_depth", "test_time",
    "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
]

CSV_FIELDS = [
    "timestamp", "stage", "label", "ok", "model_filename", "build_commit",
    "n_gpu_layers", "flash_attn", "type_k", "type_v", "n_threads",
    "n_batch", "n_ubatch", "load_mode", "no_kv_offload", "no_op_offload",
    "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts",
    "vram_peak_mib", "vram_delta_mib", "client_vram_peak_mib",
    "client_gtt_peak_mib", "vram_violation", "rss_peak_mib", "wall_s",
]

# Per-run guard fields copied verbatim into every record (and the CSV).
# `client_vram_peak_mib` is already the model's own footprint: the guarded tree
# has no DRM client before it execs, so it has no baseline to subtract. The
# system counter carries the drifting desktop, hence the separate delta.
GUARD_FIELDS = [
    "vram_baseline_mib", "vram_peak_mib", "vram_delta_mib", "vram_violation",
    "client_vram_peak_mib", "client_gtt_peak_mib",
    "rss_peak_mib",
    "cpu_busy_pct", "bg_cpu_pct",
    "nvidia_vram_peak_mib", "nvidia_vram_total_mib",
]


def parse_bench_json(text):
    """Extract the JSON array llama-bench prints, tolerating leading log noise."""
    if not text:
        return None
    lo = text.find("[")
    hi = text.rfind("]")
    if lo < 0 or hi < lo:
        return None
    try:
        val = json.loads(text[lo:hi + 1])
    except ValueError:
        return None
    return val if isinstance(val, list) else None


# ---------------------------------------------------------------- stage defs

# Measured on this host (UD-Q4_K_M, -p 64 -n 16 -r 1, guarded):
#   -ngl 0 -> 1063.0 MiB client VRAM, system peak 2421.9 (baseline 1358.9)
#   -ngl 4 -> 1643.8 MiB client VRAM, system peak 3002.7
# i.e. a ~1063 MiB Vulkan compute/scratch floor even with zero layers
# offloaded, then ~145 MiB per offloaded layer at low -ngl (cheaper than the
# 218.6 MiB/layer of on-disk weights, so placement shifts as -ngl rises --
# re-measure, never extrapolate). Remaining budget at -ngl 4 was ~2000 MiB,
# so the breach is expected in the -ngl 16..24 range. The ladder stops on the
# first cap breach so we never probe a value that would hard-freeze the shared
# display GPU.
NGL_LADDER = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]

# Context depths for the `depth` stage. llama-bench's n_ctx is
# n_prompt + n_gen + n_depth, so every earlier stage (which never passed -d)
# measured a ~192-token context, i.e. a KV cache of ~12 MiB -- the KV cost was
# invisible. This ladder is what makes the KV term real.
DEPTH_LADDER = [0, 2048, 4096, 8192, 16384, 32768]


def stage_configs(stage, base, ngl_ladder=None):
    """Return [(label, flags_dict)] for a stage, with base applied by caller."""
    if stage == "ngl":
        return [("ngl%d" % n, {"-ngl": n}) for n in (ngl_ladder or NGL_LADDER)]
    if stage == "fa":
        return [("fa=%s" % v, {"-fa": v}) for v in ("off", "on")]
    if stage == "kv":
        return [
            ("kv=f16/f16", {"-ctk": "f16", "-ctv": "f16"}),
            ("kv=q8_0/f16", {"-ctk": "q8_0", "-ctv": "f16"}),
            ("kv=q8_0/q8_0", {"-ctk": "q8_0", "-ctv": "q8_0"}),
            ("kv=q4_0/q4_0", {"-ctk": "q4_0", "-ctv": "q4_0"}),
            ("kv=bf16/bf16", {"-ctk": "bf16", "-ctv": "bf16"}),
        ]
    if stage == "threads":
        return [("t%d" % n, {"-t": n}) for n in (4, 6, 8, 12, 16)]
    if stage == "batch":
        return [
            ("b2048/ub512", {"-b": 2048, "-ub": 512}),
            ("b1024/ub256", {"-b": 1024, "-ub": 256}),
            ("b512/ub128", {"-b": 512, "-ub": 128}),
            ("b256/ub64", {"-b": 256, "-ub": 64}),
        ]
    if stage == "load":
        return [("lm=%s" % v, {"-lm": v}) for v in ("auto", "mmap", "none", "mlock", "dio")]
    if stage == "nkvo":
        return [("nkvo=%d" % v, {"-nkvo": v}) for v in (0, 1)]
    if stage == "nopo":
        return [("nopo=%d" % v, {"-nopo": v}) for v in (0, 1)]
    if stage == "ctx":
        return [
            ("pp512/tg128", {"-p": 512, "-n": 128}),
            ("pp2048/tg128", {"-p": 2048, "-n": 128}),
            ("pp128/tg512", {"-p": 128, "-n": 512}),
            ("tg128", {"-p": 0, "-n": 128}),
            ("pp512", {"-p": 512, "-n": 0}),
        ]
    if stage == "depth":
        # -d/--n-depth is the ONLY context/KV-size knob in llama-bench:
        # llama-bench.cpp:1308 sets n_ctx = n_prompt + n_gen + n_depth.
        # A fixed (p=128, n=128) probe at each depth isolates the KV cost.
        return [("p128/n128/d%d" % d, {"-p": 128, "-n": 128, "-d": d})
                for d in DEPTH_LADDER]
    if stage == "quant":
        return [("base", {})]
    raise SystemExit("unknown stage: %s" % stage)


def flags_to_argv(flags):
    out = []
    for k, v in flags.items():
        out.append(str(k))
        out.append(str(v))
    return out


def run_one(model, label, flags, args, stage):
    cmd = [BENCH_BIN, "-m", model, "-o", "json", "-r", str(args.reps)]
    if stage != "ctx":
        cmd += ["-p", str(args.pp), "-n", str(args.tg)]
    cmd += flags_to_argv(flags)
    log = os.path.join(ROOT, "logs", "bench", "%s-%s.log" % (stage, label.replace("/", "_")))
    os.makedirs(os.path.dirname(log), exist_ok=True)

    res = run_guarded(cmd, cap_mib=args.cap_mib, timeout=args.timeout,
                      log_path=log, cwd=ROOT, quiet=True)
    rows = parse_bench_json(res.get("stdout") or "")
    if rows is None:
        try:
            with open(log, errors="replace") as f:
                rows = parse_bench_json(f.read())
        except OSError:
            rows = None

    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "label": label,
        "model": model,
        "flags": flags,
        "cmd": cmd,
        "exit_code": res.get("exit_code"),
        "wall_s": res.get("wall_s"),
        "timed_out": res.get("timed_out"),
        "log": os.path.relpath(log, ROOT),
        "client_top": res.get("client_top"),
        "rows": [],
    }
    for k in GUARD_FIELDS:
        rec[k] = res.get(k)
    for r in (rows or []):
        rec["rows"].append({k: r.get(k) for k in ROW_FIELDS if k in r})
    if rows:
        first = rows[0]
        for k in ("build_commit", "build_number", "model_filename", "model_size",
                  "model_n_params"):
            rec[k] = first.get(k)
    return rec


def rec_to_csv_rows(rec):
    out = []
    if not rec["rows"]:
        r = {k: None for k in CSV_FIELDS}
        r.update({k: rec.get(k) for k in ("timestamp", "stage", "label", "model_filename",
                                          "build_commit", "wall_s")})
        r.update({k: rec.get(k) for k in GUARD_FIELDS})
        r["ok"] = False
        out.append(r)
        return out
    for row in rec["rows"]:
        r = {k: None for k in CSV_FIELDS}
        r.update({k: rec.get(k) for k in ("timestamp", "stage", "label",
                                          "model_filename", "build_commit",
                                          "wall_s")})
        r.update({k: rec.get(k) for k in GUARD_FIELDS})
        r.update({k: row.get(k) for k in CSV_FIELDS if k in row})
        r["ok"] = rec.get("exit_code") == 0 and not rec.get("vram_violation")
        out.append(r)
    return out


def load_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    return recs


def write_outputs(recs, args):
    csv_path = args.out.replace(".jsonl", ".csv")
    md_path = args.out.replace(".jsonl", ".md")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in recs:
            for row in rec_to_csv_rows(rec):
                w.writerow(row)

    lines = ["# llama-bench sweep", ""]
    lines.append("| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rec in recs:
        ok = rec.get("exit_code") == 0 and not rec.get("vram_violation")
        if not rec["rows"]:
            lines.append("| %s | %s | | | | | | | | | | | | %s | %s | %s |" % (
                rec.get("stage"), rec.get("label"), rec.get("vram_peak_mib"),
                rec.get("client_vram_peak_mib"), "NO" if not ok else "yes"))
            continue
        # One markdown row per llama-bench row so a multi-configuration record
        # (e.g. the `depth` stage) shows each point rather than only the first.
        for row in rec["rows"]:
            ts = row.get("avg_ts")
            ts = "%.2f" % ts if isinstance(ts, (int, float)) else ""
            npr, ngn = row.get("n_prompt"), row.get("n_gen")
            pp, tg = "", ""
            if npr and ngn:
                pp, tg = ts, ts
            elif ngn:
                tg = ts
            elif npr:
                pp = ts
            lines.append("| %s | %s | %s | %s | %s/%s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                rec.get("stage"), rec.get("label"), row.get("n_gpu_layers"),
                row.get("flash_attn"), row.get("type_k"), row.get("type_v"),
                row.get("n_threads"), row.get("n_batch"), row.get("n_ubatch"),
                row.get("load_mode"), npr, ngn, row.get("n_depth"),
                pp, tg, rec.get("vram_peak_mib"),
                rec.get("client_vram_peak_mib"), "yes" if ok else "NO"))
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--models", default=None,
                    help="comma-separated model list for --stage quant")
    ap.add_argument("--stage", action="append", default=None,
                    help="ngl|fa|kv|depth|threads|batch|load|nkvo|nopo|ctx|quant (repeatable)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--ngl", default=None, help="base -ngl applied to every stage except 'ngl'")
    ap.add_argument("--ngl-ladder", default=None,
                    help="comma-separated rungs for --stage ngl (default: NGL_LADDER)")
    ap.add_argument("--fa", default=None, help="base -fa (off|on|auto)")
    ap.add_argument("--ctk", default=None)
    ap.add_argument("--ctv", default=None)
    ap.add_argument("--threads", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--ubatch", default=None)
    ap.add_argument("--lm", default=None, help="base -lm load mode")
    ap.add_argument("--depth", default=None, type=int,
                    help="base -d/--n-depth; llama-bench ctx = pp + tg + depth")
    ap.add_argument("--cap-mib", type=float, default=5000.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    stages = args.stage or ["ngl"]
    if "all" in stages:
        stages = ["ngl", "fa", "kv", "depth", "ctx", "threads", "batch", "load",
                  "nkvo", "nopo"]

    base = {}
    for key, flag in (("ngl", "-ngl"), ("fa", "-fa"), ("ctk", "-ctk"),
                      ("ctv", "-ctv"), ("threads", "-t"), ("batch", "-b"),
                      ("ubatch", "-ub"), ("lm", "-lm"), ("depth", "-d")):
        val = getattr(args, key)
        if val is not None:
            base[flag] = val

    if not os.path.exists(BENCH_BIN):
        raise SystemExit("missing %s (run scripts/build.sh)" % BENCH_BIN)

    models = [args.model]
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    ngl_ladder = None
    if args.ngl_ladder:
        ngl_ladder = [int(x) for x in args.ngl_ladder.split(",") if x.strip()]

    if not args.dry_run:
        settle_vram(args.cap_mib)
    print("baseline VRAM: %.1f MiB (cap %.1f)" % (vram_used_mib() or 0, args.cap_mib))
    recs = []
    for stage in stages:
        if not args.dry_run:
            settle_vram(args.cap_mib)
        for model in (models if stage == "quant" else models[:1]):
            for label, flags in stage_configs(stage, base, ngl_ladder):
                merged = dict(base)
                merged.update(flags)
                if stage == "quant":
                    label = os.path.basename(model)
                cmd = [BENCH_BIN, "-m", model, "-o", "json", "-r", str(args.reps),
                       "-p", str(args.pp), "-n", str(args.tg)] + flags_to_argv(merged)
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                # Also settle between arms, not just between stages: a killed
                # arm's DRM client lingers, and without this the *next* arm in
                # the same stage reads a phantom baseline and dies at 0 s.
                settle_vram(args.cap_mib)
                t0 = time.time()
                rec = run_one(model, label, merged, args, stage)
                recs.append(rec)
                ok = rec["exit_code"] == 0 and not rec["vram_violation"]
                rates = " ".join(
                    "n_prompt=%s n_gen=%s %.2f t/s" % (r.get("n_prompt"), r.get("n_gen"),
                                                       r.get("avg_ts") or 0)
                    for r in rec["rows"]) or "NO RESULTS"
                print("[%s/%s] %s vram_peak=%s client=%s (+%s) exit=%s (%.0fs) %s" % (
                    stage, label, "OK" if ok else "FAIL",
                    rec["vram_peak_mib"], rec.get("client_vram_peak_mib"),
                    rec.get("vram_delta_mib"), rec["exit_code"],
                    time.time() - t0, rates))
                sys.stdout.flush()
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                if stage in ("ngl", "depth") and not ok:
                    print("[%s] stopping ladder: %s breached the cap" % (stage, label))
                    break

    if args.dry_run:
        return 0

    csv_path, md_path = write_outputs(load_jsonl(args.out), args)
    print("wrote %s, %s" % (csv_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
