#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B under the 5000 MiB VRAM watchdog.
#
# One model instance, spread across whatever GPUs are present, with the layers
# that do not fit spilling to CPU. Two configurations, chosen automatically:
#
#   split      eGPU present — the GTX 1070 (CUDA) and the Vega 10 (Vulkan) share
#              the layers. Measured 40 layers at 8k context, 32 at 128k; the KV
#              cache competes with the weights for the same VRAM, so context
#              length and layer count trade against each other.
#   vega-only  no eGPU — the original single-GPU configuration.
#
# The 5000 MiB cap protects the *display* GPU (the Vega 10), which is why the
# watchdog can kill the whole thing. The 1070 has its own 8 GB and is reported
# separately as nvidia_vram_peak_mib.
#
# Overridable via env (anything unset is derived):
#   MODEL CTX THREADS DEV TS NGL LM BATCH UBATCH CTK CTV FA HOST PORT CAP
#
#   ./scripts/serve.sh                      # picks the best available config
#   CTX=131072 ./scripts/serve.sh           # 128k context, fewer layers
#   NGL=40 TS=4,1 ./scripts/serve.sh        # override the split
#   DEV=Vulkan0 NGL=10 ./scripts/serve.sh   # force the Vega 10 alone
# Extra args are appended verbatim to the llama-server command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-8}"

# eGPU present? /dev/nvidia0 exists only once the proprietary driver has bound
# the card, so this is an instant check (no nvidia-smi spawn).
if [ -c /dev/nvidia0 ] && [ -x "$ROOT/build/bin/llama-bench" ] \
   && "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  CUDA0:'; then
    DEV="${DEV:-CUDA0,Vulkan0}"
else
    DEV="${DEV:-Vulkan0}"
fi

if [ "$DEV" = "CUDA0,Vulkan0" ]; then
    # -ts 4,1 keeps the Vega 10's share small enough to stay under the display
    # cap. It is NOT the documented ratio: measured layer counts land nearer 2:1,
    # because non-layer tensors go to the primary device. Tuned empirically.
    TS="${TS:-4,1}"
    # Measured: 40 layers fit at 8k context, 32 at 128k. Interpolate linearly in
    # context length between those two points; override NGL to depart from it.
    NGL="${NGL:-$(( 40 - (CTX - 8192) / 15360 ))}"
    [ "$NGL" -lt 8 ] && NGL=8
else
    TS="${TS:-}"
    # Single-GPU: ~252 MiB of client VRAM per layer against the display cap.
    NGL="${NGL:-8}"
fi

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

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CAP="${CAP:-5000}"

mkdir -p "$ROOT/logs"

echo "serve: model=$(basename "$MODEL") ctx=$CTX threads=$THREADS"
echo "serve: dev=$DEV ${TS:+-ts $TS }-ngl=$NGL -lm=$LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
echo "serve: cap=${CAP}MiB (display GPU)  http://$HOST:$PORT  log: $ROOT/logs/serve.log"

DEV_ARGS=(-dev "$DEV")
[ -n "$TS" ] && DEV_ARGS+=(-sm layer -ts "$TS")

exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
    --log "$ROOT/logs/serve.log" -- \
    "$ROOT/build/bin/llama-server" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
    "${DEV_ARGS[@]}" \
    -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
    -ctk "$CTK" -ctv "$CTV" -fa "$FA" \
    --host "$HOST" --port "$PORT" "$@"
