#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B on the Vega 48 iMac under the
# 5000 MiB VRAM watchdog (scripts/guard.py). guard.py SIGKILLs the whole process
# group if system-wide VRAM ever exceeds --cap-mib, so the display GPU is safe.
#
# Overridable via env:
#   MODEL, NGL, CTX, THREADS, LM, BATCH, UBATCH, CTK, CTV, FA, HOST, PORT, CAP
# Extra args are appended verbatim to the llama-server command line.
#
#   ./scripts/serve.sh                      # defaults
#   NGL=10 CTX=16384 ./scripts/serve.sh     # bigger context
#   ./scripts/serve.sh --spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf}"
NGL="${NGL:-8}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-8}"
# `-lm dio` reads the weights into anonymous memory with O_DIRECT; measured at
# matched host load it beats the default mmap path by ~60 % on prompt
# throughput (66.29 vs 41.27 t/s). `-lm none` is a close second. Needs enough
# RAM to hold the model (39 GiB here) and a filesystem with O_DIRECT.
LM="${LM:-dio}"
# Measured at depth 4096 with `-lm dio`: the llama-server default 2048/512 is
# fastest on prompt (65.60 vs 52.28 t/s for 1024/256). Smaller micro-batches
# only buy VRAM (3072 -> 2809 -> 2679 MiB client), not speed — lower them only
# if the cap is tight.
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
# Measured: KV dtype is free at depth 0 (all five within 3 %) and worth ~9.6 %
# generation at depth 8192. A quantized V cache REQUIRES flash attention.
CTK="${CTK:-q4_0}"
CTV="${CTV:-q4_0}"
FA="${FA:-on}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CAP="${CAP:-5000}"

mkdir -p "$ROOT/logs"

echo "serve: model=$MODEL ngl=$NGL ctx=$CTX threads=$THREADS cap=${CAP}MiB"
echo "serve: http://$HOST:$PORT  (log: $ROOT/logs/serve.log)"

exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
    --log "$ROOT/logs/serve.log" -- \
    "$ROOT/build/bin/llama-server" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
    -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
    -ctk "$CTK" -ctv "$CTV" -fa "$FA" \
    --host "$HOST" --port "$PORT" "$@"
