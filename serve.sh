#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B under the 5000 MiB VRAM watchdog.
#
# One model instance, split across the GTX 1070 (CUDA) and the Vega 10
# (Vulkan), with the layers that do not fit spilling to CPU: -dev CUDA0,Vulkan0
# -sm layer -ts 4,1. Measured 40 layers at 8k context, 32 at 128k; the KV cache
# competes with the weights for the same VRAM, so context length and layer count
# trade against each other. The layer count is derived from the context below.
#
# The eGPU is REQUIRED. The Vega 10 alone cannot hold this model at the default
# 128k context under the display cap, so serve.sh errors out without it rather
# than silently degrading to a configuration that the watchdog will kill.
#
# The 5000 MiB cap protects the *display* GPU (the Vega 10), which is why the
# watchdog can kill the whole thing. The 1070 has its own 8 GB and is reported
# separately as nvidia_vram_peak_mib.
#
# Overridable via env (anything unset is derived):
#   MODEL CTX THREADS TS NGL LM BATCH UBATCH CTK CTV FA HOST PORT CAP
#
#   ./serve.sh                      # 128k context across both GPUs
#   CTX=32768 ./serve.sh            # shorter context, more layers (faster)
#   NGL=36 TS=4,1 ./serve.sh        # override the split
# Extra args are appended verbatim to the llama-server command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf}"
# 128k is the working context: an agent client needs its system prompt, tool
# schemas and a long transcript resident at once. q4_0 KV plus flash attention
# keep that cache at ~2.4 GB, which the layer count below pays for by offloading
# ~8 layers to CPU. Lower CTX to buy layers back if generation speed matters
# more than depth.
CTX="${CTX:-131072}"
THREADS="${THREADS:-8}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CAP="${CAP:-5000}"

# The eGPU is required, not optional: /dev/nvidia0 exists only once the
# proprietary driver has bound the card, so this is an instant check (no
# nvidia-smi spawn).
if [ ! -c /dev/nvidia0 ] || [ ! -x "$ROOT/build/bin/llama-bench" ] \
   || ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  CUDA0:'; then
    echo "serve: no CUDA eGPU: /dev/nvidia0 is missing or no CUDA0 backend is built." >&2
    echo "serve: this configuration splits one 27B model across the GTX 1070 and the" >&2
    echo "serve: Vega 10; the Vega 10 alone cannot hold ${CTX} tokens of context under" >&2
    echo "serve: the ${CAP} MiB display cap. Build with GGML_CUDA=ON (see README)." >&2
    exit 1
fi

DEV="CUDA0,Vulkan0"
# -ts 4,1 keeps the Vega 10's share small enough to stay under the display
# cap. It is NOT the documented ratio: measured layer counts land nearer 2:1,
# because non-layer tensors go to the primary device. Tuned empirically.
TS="${TS:-4,1}"
# Measured: 40 layers fit at 8k context, 32 at 128k. Interpolate linearly in
# context length between those two points; override NGL to depart from it.
NGL="${NGL:-$(( 40 - (CTX - 8192) / 15360 ))}"
[ "$NGL" -lt 8 ] && NGL=8

# `-lm dio` reads the weights with O_DIRECT; measured ~12 % faster on prompt
# than the mmap default and ~40x lighter in RSS. `-lm none` is a close second.
LM="${LM:-dio}"
# Measured at depth 4096: the llama-server default 2048/512 is fastest on prompt.
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
# Measured: KV dtype is free at depth 0 and worth ~9.6 % generation at depth
# 8192; at 128k a quantized cache is mandatory (~8.6 GB f16 vs ~2.4 GB q4_0).
# A quantized V cache REQUIRES flash attention.
CTK="${CTK:-q4_0}"
CTV="${CTV:-q4_0}"
FA="${FA:-on}"

mkdir -p "$ROOT/logs"

echo "serve: model=$(basename "$MODEL") ctx=$CTX threads=$THREADS"
echo "serve: dev=$DEV -ts $TS -ngl=$NGL -lm=$LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
echo "serve: cap=${CAP}MiB (display GPU)  http://$HOST:$PORT  log: $ROOT/logs/serve.log"

exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
    --log "$ROOT/logs/serve.log" -- \
    "$ROOT/build/bin/llama-server" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
    -dev "$DEV" -sm layer -ts "$TS" \
    -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
    -ctk "$CTK" -ctv "$CTV" -fa "$FA" \
    --host "$HOST" --port "$PORT" "$@"
