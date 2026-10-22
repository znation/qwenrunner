#!/usr/bin/env bash
# serve_flint_rec.sh — run Flint on Recursive's own Apple engine (`rec`).
#
# Why this exists: measured on this machine, the llama.cpp path
# (scripts/serve_flint.sh) sustains 193-220 GB/s of the 510-530 GB/s this M5 Max
# actually delivers to a streaming Metal kernel (measured, /tmp probe; the spec
# sheet says 614 GB/s). That is 38-42% of the achievable roofline, and it is not
# a bytes problem -- see FLINT.md, "Where the time goes":
#
#   * decode is not DRAM-bound: 1.56 GB of weights per token at 123.5 t/s is
#     193 GB/s, and at d100k (4.43 GB/token) it is 213 GB/s -- flat, and both
#     far below the roofline;
#   * the MoE expert matvec has no MMA/tensor-op path at this llama.cpp
#     revision (mul_mat_id only reaches the simdgroup kernels at >= 32 rows,
#     i.e. under long speculative drafts, never in plain decode).
#
# `rec` is Recursive's own engine for this model on Apple Silicon: MLX plus
# Metal kernels (the shipped runtime carries a Metal "megakernel" and
# per-architecture kernels), running the 4-bit MLX checkpoint (group-64 affine,
# 8-bit embeddings, bf16 KV) instead of the Q4_1 GGUF.
#
# Measured here -- same prompts, same sampling, same wire-level harness
# (scripts/openai_stream_bench.py, which divides prefill out of decode, so both
# stacks are measured identically rather than through llama.cpp's own timings):
#
#   workload                     llama.cpp   rec       ratio
#   copy-heavy, 4.1k, temp 0     121.6       226.8     1.9x
#   copy-heavy, 4.1k, temp 1     -           213.7     1.8x
#   agentic chat, 27k, temp 0     93.4       140.2     1.5x
#   agentic chat, 27k, temp 1     95.9       134.9     1.4x
#   prefill 27k (warm, TTFT)      24.3-24.7s  9.4-9.5s  2.6x
#
# The vendor's own README claims 1.4-2.6x over llama.cpp on an M4 Pro for this
# model, which brackets what was measured independently here.
#
# What this path does NOT have: speculative decoding. llama.cpp's draft-free
# n-gram drafter is a large lossless win on copy-heavy text (+68% shallow,
# +195-270% at 30k depth) and a small loss on reasoning-heavy agentic traffic
# (-6% at temp 1), so it stays a switch on the llama.cpp path (scripts/
# ngram_ab.py, FLINT.md) -- rec exposes no equivalent flag.
#
# Requires: rec installed (~/.local/bin/rec, from
# https://raw.githubusercontent.com/recursive-oss/rec/main/install.sh) and the
# model pulled (`rec pull`, 22.6 GB). Requirements are M4 Pro / M5 Pro / M5 Max
# with >= 48 GB unified memory.
#
# Overridable via env: PORT CTX REC_BIN
#   ./scripts/serve_flint_rec.sh                 # 262 144-token context on 11500
#   CTX=131072 ./scripts/serve_flint_rec.sh      # smaller context
#   ./scripts/serve_flint_rec.sh --port 11501    # extra args go to rec serve
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REC_BIN="${REC_BIN:-$(command -v rec || true)}"
PORT="${PORT:-11500}"
CTX="${CTX:-262144}"

if [ -z "$REC_BIN" ] || [ ! -x "$REC_BIN" ]; then
    echo "serve: rec not found. Install it (recursive-oss/rec):" >&2
    echo "serve:   curl -fsSL https://raw.githubusercontent.com/recursive-oss/rec/main/install.sh | bash" >&2
    exit 1
fi

mkdir -p "$ROOT/logs"

# rec manages its own model lifecycle (`rec stop`, `rec status`), so unlike
# serve.sh/serve_flint.sh this script deliberately does NOT wrap the server in
# scripts/guard.py: guard's job is to SIGKILL a process group on a swap breach,
# and that would leave rec's own instance tracking out of sync.
#
# doctor is run once and matched with a glob, not piped into grep -q: this
# script runs under `set -o pipefail`, and `grep -q` closing the pipe early
# makes doctor die of SIGPIPE, which would fail the pipeline and invert the
# check. doctor also hashes the 22.6 GB checkpoint, so running it twice is slow.
doctor_out="$("$REC_BIN" doctor 2>&1 || true)"
case "$doctor_out" in
    *"meets the requirements"*) ;;
    *)
        echo "serve: this machine does not meet rec's requirements:" >&2
        printf '%s\n' "$doctor_out" >&2
        exit 1
        ;;
esac
case "$doctor_out" in
    *"is not downloaded"*)
        echo "serve: the Flint checkpoint is not downloaded yet (~22.6 GB):" >&2
        echo "serve:   rec pull" >&2
        exit 1
        ;;
esac

echo "serve: engine=rec (mlx/metal) port=$PORT context=$CTX endpoint=http://127.0.0.1:$PORT/v1"
echo "serve: client needs model=flint and the key from: rec auth token"
echo "serve: this is the fast path; scripts/serve_flint.sh is the llama.cpp path"
exec "$REC_BIN" serve --port "$PORT" --context "$CTX" "$@"
