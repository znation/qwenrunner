#!/usr/bin/env python3
"""Parameterized llama-bench sweep driver for Apple Silicon.

The Intel version of this file existed to find the largest `-ngl` that fit
under a 5000 MiB display-GPU cap, and its ladders were VRAM ladders: rungs at
0,2,4,... layers, a `depth` stage to make the KV term visible, and a hard stop
on the first breach. On the M5 Max there is no cap to breach and no offload
ladder worth climbing -- `-ngl 99` fits, so the interesting question changed
from "how many layers fit" to "what does the last layer buy, and what is the
KV/prefill cost at 128k". The ladders below reflect that:

  * `ngl`     -- the plateau: 0 vs 8..99. Shows where the CPU->GPU crossover is.
  * `depth`   -- up to 128k, the context this branch targets.
  * `threads` -- includes the 12 performance cores to show they cost, not pay.
  * `batch`   -- larger ubatch is the Metal prefill lever; the Intel host was
                 fastest at the 2048/512 default.

Per-run guard fields are now memory/serialization fields, not VRAM fields: see
scripts/guard.py. `bg_cpu_pct` and `gpu_busy_pct_peak` are what make a
contended arm detectable.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard import run_guarded, gpu_stats, settle_gpu  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_BIN = os.path.join(ROOT, "build", "bin", "llama-bench")

DEFAULT_MODEL = "models/Qwen3.8-27B-UD-Q4_K_S.gguf"

# A just-exited arm keeps its Metal buffers alive for a moment. Sampling a
# baseline inside that window makes the next arm look like it started with a
# full GPU. The old harness lost a whole `threads` stage to exactly this, so
# the settle exists here too -- against accelerator memory instead of DRM.
SETTLE_QUIET_S = 3.0
SETTLE_POLL_S = 1.0
SETTLE_SLACK_MIB = 128.0


def settle(gpu_baseline=None, timeout=20.0):
    """Block until accelerator memory stops falling, then return it.

    `gpu_baseline` is the idle footprint sampled before the first arm (the
    desktop's own GPU tenant). If the reading stabilises above it, the wait is
    extended and then returns anyway -- another app may legitimately hold it.
    """
    t0 = time.time()
    prev = None
    stable_since = None
    while time.time() - t0 < timeout:
        _, v = gpu_stats()
        if v is None:
            return None
        if prev is not None and abs(v - prev) <= SETTLE_SLACK_MIB:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since >= SETTLE_QUIET_S:
                return v
        else:
            stable_since = None
        prev = v
        time.sleep(SETTLE_POLL_S)
    return prev


# llama-bench JSON scalar fields we care about (tools/llama-bench/llama-bench.cpp).
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
    "gpu_mem_peak_mib", "gpu_mem_delta_mib", "swap_delta_mib", "swap_violation",
    "rss_peak_mib", "mem_peak_mib", "gpu_busy_pct_peak", "cpu_busy_pct",
    "bg_cpu_pct", "wall_s",
]

# Per-run guard fields copied verbatim into every record (and the CSV).
GUARD_FIELDS = [
    "gpu_mem_baseline_mib", "gpu_mem_peak_mib", "gpu_mem_delta_mib",
    "gpu_busy_pct_peak",
    "swap_baseline_mib", "swap_peak_mib", "swap_delta_mib", "swap_violation",
    "mem_peak_mib", "rss_peak_mib",
    "cpu_busy_pct", "tree_cpu_pct", "bg_cpu_pct",
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

# Where does Metal stop paying? With every layer offloaded the model is purely
# bandwidth-bound on the GPU; before that it is CPU-bound and slow. The rungs
# bracket the crossover rather than searching for a cap that no longer exists.
NGL_LADDER = [0, 8, 16, 32, 48, 64, 99]

# Context depths for the `depth` stage. llama-bench sets n_ctx = n_prompt +
# n_gen + n_depth, so only this ladder makes the KV term real -- every other
# stage runs with a ~256-token context and a KV cache of a few hundred KiB.
# 131072 is the context this branch serves at.
DEPTH_LADDER = [0, 4096, 16384, 32768, 65536, 131072]


def stage_configs(stage, base, ngl_ladder=None):
    """Return [(label, flags_dict)] for a stage, with base applied by caller."""
    if stage == "ngl":
        return [("ngl%d" % n, {"-ngl": n}) for n in (ngl_ladder or NGL_LADDER)]
    if stage == "fa":
        return [("fa=%s" % v, {"-fa": v}) for v in ("off", "on", "auto")]
    if stage == "kv":
        return [
            ("kv=f16/f16", {"-ctk": "f16", "-ctv": "f16"}),
            ("kv=bf16/bf16", {"-ctk": "bf16", "-ctv": "bf16"}),
            ("kv=q8_0/f16", {"-ctk": "q8_0", "-ctv": "f16"}),
            ("kv=q8_0/q8_0", {"-ctk": "q8_0", "-ctv": "q8_0"}),
            ("kv=q4_0/q4_0", {"-ctk": "q4_0", "-ctv": "q4_0"}),
        ]
    if stage == "threads":
        # 6 = super cores; 18 = all. The Intel result (-t 16 was ~8 % worse)
        # should reappear here with wider separation.
        return [("t%d" % n, {"-t": n}) for n in (4, 6, 8, 12, 18)]
    if stage == "batch":
        return [
            ("b2048/ub512", {"-b": 2048, "-ub": 512}),
            ("b4096/ub1024", {"-b": 4096, "-ub": 1024}),
            ("b4096/ub2048", {"-b": 4096, "-ub": 2048}),
            ("b8192/ub2048", {"-b": 8192, "-ub": 2048}),
            ("b1024/ub256", {"-b": 1024, "-ub": 256}),
        ]
    if stage == "load":
        # `dio` is gone: O_DIRECT is a Linux-only load mode and there is no
        # reason to want it here, since mmap is zero-copy into a Metal buffer.
        return [("lm=%s" % v, {"-lm": v}) for v in ("auto", "mmap", "none", "mlock")]
    if stage == "nkvo":
        return [("nkvo=%d" % v, {"-nkvo": v}) for v in (0, 1)]
    if stage == "nopo":
        return [("nopo=%d" % v, {"-nopo": v}) for v in (0, 1)]
    if stage == "ctx":
        return [
            ("pp512/tg128", {"-p": 512, "-n": 128}),
            ("pp2048/tg128", {"-p": 2048, "-n": 128}),
            ("pp8192/tg128", {"-p": 8192, "-n": 128}),
            ("pp128/tg512", {"-p": 128, "-n": 512}),
            ("tg128", {"-p": 0, "-n": 128}),
            ("pp512", {"-p": 512, "-n": 0}),
        ]
    if stage == "depth":
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


def run_one(model, label, flags, args, stage, gpu_baseline):
    cmd = [BENCH_BIN, "-m", model, "-o", "json", "-r", str(args.reps)]
    if stage != "ctx":
        cmd += ["-p", str(args.pp), "-n", str(args.tg)]
    cmd += flags_to_argv(flags)
    log = os.path.join(ROOT, "logs", "bench", "%s-%s.log" % (stage, label.replace("/", "_")))
    os.makedirs(os.path.dirname(log), exist_ok=True)

    res = run_guarded(cmd, swap_cap_mib=args.swap_cap_mib, timeout=args.timeout,
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
        "rows": [],
    }
    for k in GUARD_FIELDS:
        rec[k] = res.get(k)
    rec["gpu_mem_baseline_mib"] = None if gpu_baseline is None else round(gpu_baseline, 1)
    rec["gpu_mem_delta_mib"] = (
        None if (res.get("gpu_mem_peak_mib") is None or gpu_baseline is None)
        else round(res["gpu_mem_peak_mib"] - gpu_baseline, 1))
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
        r["ok"] = rec.get("exit_code") == 0 and not rec.get("swap_violation")
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

    lines = ["# llama-bench sweep (Apple Silicon / Metal)", ""]
    lines.append("| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | "
                 "pp t/s | tg t/s | gpu MiB | bg cpu %% | gpu %% | ok |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rec in recs:
        ok = rec.get("exit_code") == 0 and not rec.get("swap_violation")
        if not rec["rows"]:
            lines.append("| %s | %s | | | | | | | | | | | | %s | %s | %s | %s |" % (
                rec.get("stage"), rec.get("label"), rec.get("gpu_mem_peak_mib"),
                rec.get("bg_cpu_pct"), rec.get("gpu_busy_pct_peak"),
                "NO" if not ok else "yes"))
            continue
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
            lines.append("| %s | %s | %s | %s | %s/%s | %s | %s/%s | %s | %s | %s | %s | "
                         "%s | %s | %s | %s | %s | %s |" % (
                rec.get("stage"), rec.get("label"), row.get("n_gpu_layers"),
                row.get("flash_attn"), row.get("type_k"), row.get("type_v"),
                row.get("n_threads"), row.get("n_batch"), row.get("n_ubatch"),
                row.get("load_mode"), npr, ngn, row.get("n_depth"),
                pp, tg, rec.get("gpu_mem_peak_mib"),
                rec.get("bg_cpu_pct"), rec.get("gpu_busy_pct_peak"),
                "yes" if ok else "NO"))
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
    ap.add_argument("--swap-cap-mib", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
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

    gpu_baseline = None
    if not args.dry_run:
        gpu_baseline = settle()
    print("baseline GPU memory: %s MiB" % (
        "n/a" if gpu_baseline is None else "%.1f" % gpu_baseline))
    recs = []
    for stage in stages:
        if not args.dry_run:
            settle()
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
                # Settle between arms too: a just-exited arm's Metal buffers
                # linger, and without this the next arm reads a phantom
                # baseline (the old harness lost arms to exactly this).
                settle()
                t0 = time.time()
                rec = run_one(model, label, merged, args, stage, gpu_baseline)
                recs.append(rec)
                ok = rec["exit_code"] == 0 and not rec["swap_violation"]
                rates = " ".join(
                    "n_prompt=%s n_gen=%s %.2f t/s" % (r.get("n_prompt"), r.get("n_gen"),
                                                       r.get("avg_ts") or 0)
                    for r in rec["rows"]) or "NO RESULTS"
                print("[%s/%s] %s gpu_peak=%s (+%s) bg_cpu=%s gpu_util=%s exit=%s (%.0fs) %s" % (
                    stage, label, "OK" if ok else "FAIL",
                    rec["gpu_mem_peak_mib"], rec.get("gpu_mem_delta_mib"),
                    rec.get("bg_cpu_pct"), rec.get("gpu_busy_pct_peak"),
                    rec["exit_code"], time.time() - t0, rates))
                sys.stdout.flush()
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                if stage in ("ngl", "depth") and not ok:
                    print("[%s] stopping ladder: %s failed" % (stage, label))
                    break

    if args.dry_run:
        return 0

    csv_path, md_path = write_outputs(load_jsonl(args.out), args)
    print("wrote %s, %s" % (csv_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
