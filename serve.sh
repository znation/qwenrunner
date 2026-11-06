#!/usr/bin/env bash
# serve.sh — launch mlx-serve for Qwen-3.8-27B on Apple Silicon (MLX).
#
# The only entry point. OpenAI-compatible on $PORT, 128k context, one process,
# N concurrent requests, on MLX weights.
#
# Why this runtime: measured on this machine with scripts/openai_stream_bench.py
# and scripts/turn_loop_bench.py, same prompts, one stream and four agent
# sessions. Decode 73.4 t/s at 4k and 76-82 at 30k against llama.cpp's 28.7 and
# 25.2-25.7; 128k prefill 461 t/s and decode 46.9 t/s against 176-231 and 14.3;
# four agent sessions 49.9 t/s aggregate at TTFT p50 1.26 s against 37.4 at
# 2.28 s. The whole gain is speculative decoding (MTP + DFlash2 + PLD), not the
# MLX kernels: with MTP=0 this server decodes at parity with llama.cpp.
# RESULTS.md, "MLX engines".
#
# Overridable via env (anything unset is derived):
#   MODEL CTX NP PORT HOST MTP KVQ CACHE_MB LOG_DIR
#
#   ./serve.sh                 # the default: 2 concurrent, speculation on
#   NP=4 ./serve.sh            # the measured agent-fleet optimum
#   MTP=0 ./serve.sh           # speculation off (parity with llama.cpp)
#   KVQ=8 ./serve.sh           # 8-bit KV cache: ~2x less KV, ~10% slower
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The release binary, not a build. `brew install mlx-serve` from the ddalcu tap
# builds from source (Zig + MLX + llama.cpp, >30 min here); the release tarball
# is 98 MB and is what these numbers were measured on.
BIN="${MLX_SERVE_BIN:-$HOME/opt/mlx-serve/mlx-serve-macos-arm64/mlx-serve}"
if [ ! -x "$BIN" ]; then
  BIN="$(command -v mlx-serve || true)"
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "serve: mlx-serve not found. Install the release binary:" >&2
  echo "  curl -sL https://github.com/ddalcu/mlx-serve/releases/latest/download/mlx-serve-bin-macos-arm64.tar.gz | tar xz -C ~/opt/mlx-serve" >&2
  exit 1
fi

# The pack carries a DFlash2 drafter in its `drafter/` subdirectory, which the
# server loads with the model. The mlx-community 4-bit build has no drafter and
# no MTP head, and measures at the no-speculation rate.
MODEL="${MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit}"

# 128k is the working context. The model's own limit is 256k;
# the KV for 128k is ~9.4 GiB at f16 (16 full-attention layers of 64), which is
# what the server's hot-cache budget sizes itself to.
CTX="${CTX:-131072}"

# Concurrent requests. Continuous batching, so this is a cap, not an
# allocation: KV is taken as requests arrive. 2 is the RAM-first default; 4
# measured 49.9 t/s aggregate against 54.7 at 2 on the four-session shape, with
# a better wall p50 (6.3 s against 10.4 s) because the turns overlap instead of
# queueing.
NP="${NP:-2}"

PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"

# Speculation: MTP (the checkpoint's own head, depth 6), DFlash2 (the drafter in
# the pack) and PLD (prompt lookup). All three are on by default; MTP=0 turns
# all of them off, which is the apples-to-apples arm against llama.cpp.
MTP="${MTP:-1}"

# KV cache precision. off = f16. 8 halves it at ~10% decode; 4 quarters it.
KVQ="${KVQ:-off}"

# Prefix-cache budget. Default is one session at the working context (~9.4 GiB
# at 128k); raise it to hold more live conversations, which is what keeps an
# appending transcript from re-prefilling every turn.
CACHE_MB="${CACHE_MB:-}"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

ARGS=(--model "$MODEL" --serve --host "$HOST" --port "$PORT"
      --ctx-size "$CTX" --max-concurrent "$NP"
      --kv-quant "$KVQ"
      --log-file "$LOG_DIR/mlx-serve.$PORT.log")

if [ "$MTP" = "0" ]; then
  ARGS+=(--no-mtp --no-pld --no-drafter)
fi
if [ -n "$CACHE_MB" ]; then
  ARGS+=(--prefix-cache-mem "${CACHE_MB}MB")
fi

echo "serve: model=$(basename "$MODEL") ctx=$CTX concurrent=$NP mtp=$MTP kv=$KVQ"
echo "serve: http://$HOST:$PORT/v1  logs: $LOG_DIR/mlx-serve.$PORT.log"
exec "$BIN" "${ARGS[@]}"
