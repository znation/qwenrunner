#!/usr/bin/env bash
# serve.sh — serve Qwen-3.8-27B on this M5 Max for the tumwater fleet.
#
# The only entry point. OpenAI-compatible on $PORT (8090), 128k context, four
# concurrent sessions.
#
# Why this engine: measured on this machine by replaying real tumwater pi
# sessions (scripts/tumwater_replay.py: 4 sessions at once, ~42k-token median
# request, ~750 new tokens and a few hundred out per turn — the shape of 19k
# recorded tumwater turns). RESULTS.md, "Engine shoot-out".
#
#   engine (format)                       turns/min  agg out t/s  TTFT p90
#   Splash 1.6.0 (MLX 4-bit + DFlash2)       15.0       63.8        10.0 s
#   Kapitsa 0.2.31 (27b 4-bit + DFlash2)     10.7       45.4         4.7 s
#   mlx-serve 26.11.2, 32 GB prefix cache    10.7       17.3        31.0 s
#   oMLX 0.7.3 (MLX 4-bit + DFlash2)          6.6       26.1        12.8 s
#   mlx-serve 26.11.2, as shipped (NP=2)       3.9        6.3       128   s
#
# The difference is batched speculation: Splash runs its DFlash2 draft/verify
# cycle across all four lanes at once, so each stream keeps ~29 t/s at depth
# while three others decode beside it. mlx-serve's speculation is a
# single-stream win (73 t/s alone) that collapses to ~16 t/s per stream once
# four sessions overlap, and its default prefix-cache budget holds only one
# 128k session, so the other three re-prefill 40-55k tokens every turn.
#
# Env (anything unset is derived):
#   ENGINE   splash (default) | mlx-serve (the previous engine, kept runnable)
#   MODEL    the weights; per-engine default below
#   CTX      context per session, tokens (default 131072)
#   NP       concurrent sessions (default 4 — tumwater's fleet). Splash's lane
#            count is fixed at 4 by the engine; NP>4 queues there. mlx-serve
#            takes it as --max-concurrent and sizes its prefix cache from it.
#   PORT HOST LOG_DIR
#   CACHE_MB mlx-serve only: prefix-cache budget (default: NP sessions x 8 GB)
#   MTP      mlx-serve only: 0 turns speculation off
#
#   ./serve.sh                       # Splash, 4 sessions, 128k, :8090
#   ENGINE=mlx-serve ./serve.sh      # the previous engine, cache fixed for NP
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-splash}"
CTX="${CTX:-131072}"
# Four concurrent sessions: tumwater's fleet runs four role instances against
# this server, and every engine measured here is at its best with one lane per
# session (fewer lanes queue whole turns behind each other's decode).
NP="${NP:-4}"
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$ENGINE.$PORT.log"

# The ids clients send. pi and omp were configured against mlx-serve's id, so
# Splash announces Flash-Next as the primary and still accepts the old 27B ids
# the clients currently send, so a restart needs no client change.
ALIAS="${ALIAS:-Qwen3.8-Flash-Next}"
OLD_ID="Qwen3.8-27B"
LEGACY_ID="Qwen3.8-27B-MLX-Serve-4bit"

case "$ENGINE" in
splash)
  BIN="$(command -v splash || true)"
  if [ -z "$BIN" ]; then
    echo "serve: splash not found. Install it: brew install incoai/tap/splash" >&2
    exit 1
  fi
  # Qwen3.8-Flash-Next, GSQ-RCO GGUF at IQ3_S (83.6 GB). Measured on this
  # machine: 17.7/16.6 turns/min, 92-101 agg out t/s, 119-127 stream t/s at
  # 93 GiB peak (RESULTS.md). IQ3_S is task-parity with the unquantised model
  # (93.3 vs 93.1); the lighter Q2_0 (66.4 GB) costs ~4 points, unnecessary at
  # 128 GB. Splash pairs it with the incoai Flash-Next DFlash2 drafter and
  # downloads both on first start. Vision (mmproj) is on by default, which
  # tumwater needs for image/PDF turns. The previous 27B default was
  # mlx-community/Qwen3.8-27B-4bit; set MODEL=... to restore it.
  MODEL="${MODEL:-ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S}"
  if [ "$NP" -gt 4 ]; then
    echo "serve: note: Splash decodes 4 requests at once; NP=$NP queues the rest" >&2
  fi
  # --idle-release off: the default unwires the weights after 10 idle minutes,
  #   which makes the next tumwater turn after a quiet spell pay a reload.
  # --max-context: per conversation; Splash's own default is the model's 256k
  #   window when memory holds it, which tumwater does not need.
  ARGS=(serve --model "$MODEL" --host "$HOST" --port "$PORT"
        --max-context "$(( CTX / 1024 ))K" --no-webui --idle-release off
        --served-model-name "$ALIAS" --served-model-name "$OLD_ID"
        --served-model-name "$LEGACY_ID" --announce-served-name)
  echo "serve: splash model=$MODEL ctx=$CTX lanes=4 alias=$ALIAS (+$OLD_ID +$LEGACY_ID)"
  echo "serve: http://$HOST:$PORT/v1  log: $LOG"
  exec "$BIN" "${ARGS[@]}" "$@" > >(tee -a "$LOG") 2>&1
  ;;

mlx-serve)
  BIN="${MLX_SERVE_BIN:-$HOME/opt/mlx-serve/mlx-serve-macos-arm64/mlx-serve}"
  [ -x "$BIN" ] || BIN="$(command -v mlx-serve || true)"
  if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "serve: mlx-serve not found (see README, 'Install')" >&2
    exit 1
  fi
  MODEL="${MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit}"
  MTP="${MTP:-1}"
  # The prefix cache must hold every live session, or sessions evict each
  # other and re-prefill their whole transcript. A session at ~55k tokens is
  # ~4.8 GB here (KV + GDN snapshots), so 8 GB each leaves room to grow; the
  # shipped one-session budget (9.4 GB) measured 3.9 turns/min against 10.7.
  CACHE_MB="${CACHE_MB:-$(( NP * 8192 ))}"
  ARGS=(--model "$MODEL" --serve --host "$HOST" --port "$PORT"
        --ctx-size "$CTX" --max-concurrent "$NP"
        --prefix-cache-mem "${CACHE_MB}MB" --log-file "$LOG")
  if [ "$MTP" = "0" ]; then
    ARGS+=(--no-mtp --no-pld --no-drafter)
  fi
  echo "serve: mlx-serve model=$(basename "$MODEL") ctx=$CTX concurrent=$NP cache=${CACHE_MB}MB mtp=$MTP"
  echo "serve: http://$HOST:$PORT/v1  log: $LOG"
  exec "$BIN" "${ARGS[@]}" "$@"
  ;;

*)
  echo "serve: unknown ENGINE=$ENGINE (splash | mlx-serve)" >&2
  exit 1
  ;;
esac
