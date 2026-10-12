#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B under the 5000 MiB VRAM watchdog.
#
# One model instance, split across the GTX 1070 (CUDA) and the Vega 10
# (Vulkan), with the layers that do not fit spilling to CPU: -dev CUDA0,Vulkan0
# -sm layer. Measured 40 layers at 8k context, 32 at 128k without speculation;
# the KV cache competes with the weights for the same VRAM, so context length
# and layer count trade against each other. The layer count is derived from the
# context below.
#
# With MTP on (the default) the draft head runs on the *Vega 10*, not the CPU:
# the MTP block's forward pass is serial with the target's, so once 26+ trunk
# layers are offloaded it is the bottleneck, and the Vega has headroom the 1070
# does not. Measured +38 % generation over the previous CPU-draft default at
# 128k (6.224 vs 4.509 t/s) and +6 % prompt, for ~1.5 GB of the display GPU's
# budget -- which is why the tensor split becomes 6,1 rather than 4,1 (the
# draft's weights would otherwise sit on top of six trunk layers and breach).
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
#   MODEL CTX THREADS TS NGL LM BATCH UBATCH CTK CTV FA HOST PORT CAP NP
#   MTP MD DEVD NGLD DCTK DCTV
#
#   ./serve.sh                      # 128k context across both GPUs, MTP on
#   MTP=0 ./serve.sh                # the same split without speculative decoding
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
# llama-server's own default is -np 4, which reserves compute buffers for four
# sequences: measured 8117 MiB on the 1070 at -ngl 28 against 7541 at -np 1, and
# a real chat request aborted it there in `ggml_cuda_pool_vmm::alloc` (SIGABRT,
# exit -6, 8189 of 8192 MiB). One slot costs no concurrency for a single agent
# client and buys ~2 trunk layers; every number in RESULTS.md is -np 1.
NP="${NP:-1}"

# MTP speculation is ON by default and its draft head runs on the Vega 10, not
# the CPU: measured +38 % generation over the shipped CPU-draft default at 128k
# (6.224 vs 4.509 t/s at depth 5246) and +6 % prompt, or +34 % against the CPU
# draft at the same -ngl 28 (4.640). The MTP block's forward pass is serial with
# the target's, so at 26+ offloaded trunk layers the CPU draft is the
# bottleneck; the 1070 has no room for the draft, the Vega does.
# -ngld 99 (not 1) is deliberate: the draft declares nextn_predict_layers = 1,
# and offloading only that layer still leaves the draft's output projection on
# the CPU -- a 151k x 5120 matmul per draft step. A count above the layer count
# moves the non-repeating tensors too: measured +10 % generation (5.66 -> 6.22
# t/s) for +234 MiB of the display GPU (client 3151 -> 3384), 2/2 interleaved.
# -ctkd/-ctvd q4_0 is not optional at 128k: the draft's own f16 KV is ~537 MiB
# at 131072 tokens and q4_0 cuts that to ~134 MiB. `-devd` must name a device
# from -dev; the default (follow -dev) would put the draft's share on the 1070.
MTP="${MTP:-1}"
MD="${MD:-$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf}"
DEVD="${DEVD:-Vulkan0}"
NGLD="${NGLD:-99}"
DCTK="${DCTK:-q4_0}"
DCTV="${DCTV:-q4_0}"

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
# The safe -ngl is a function of the live desktop baseline, not a constant --
# measured 746-2188 MiB within one hour. At 128k this config's client footprint
# is ~3384 MiB, so the display cap is reached at a baseline near 1216 MiB. The
# watchdog is the backstop, not the plan: say so before it has to fire.
BASE_MIB=$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_used_mib()))" 2>/dev/null || true)
if [ -n "${BASE_MIB:-}" ] && [ "$MTP" = 1 ] && [ "$BASE_MIB" -gt 1200 ]; then
    echo "serve: WARNING display baseline is ${BASE_MIB} MiB; this config fits to about" >&2
    echo "serve: WARNING 1216 MiB. Lower NGL= by 2-4 (~158 MiB/layer at -ts 6,1) or" >&2
    echo "serve: WARNING close desktop GPU users before relying on it." >&2
fi
# -ts is tied to whether the draft is resident on the Vega 10: with MTP the
# draft costs ~1.5 GB of the display GPU, so the trunk's share there must stay
# small. 6,1 is the measured optimum at 128k (client 3384 MiB for 28 trunk
# layers plus the draft); 4,1 puts the draft's weights on top of ~6 trunk layers
# and breaches the cap (client 3644, measured). Without a draft, 4,1 is the
# measured split. It is NOT the documented ratio either way: a requested 6,1
# assigned 28 layers to CUDA0 and 4 to Vulkan0 at -ngl 32, i.e. 7:1, because
# non-layer tensors go to the primary device. Tuned empirically against the two
# VRAM counters.
if [ "$MTP" = 1 ]; then
    TS="${TS:-6,1}"
else
    TS="${TS:-4,1}"
fi
# Layer budget, from measured fit points (load + a real completion,
# scripts/split_fit.py, `-ts 6,1` with the draft on the Vega): 38 layers fit at
# 8k context, 36 at 32k, 30 at 128k -- the KV cache competes with the weights
# for the same VRAM. Interpolate between the 32k and 128k anchors, two layers
# under the last startable rung at both ends; below 32k that is deliberately
# conservative (38 is startable at 8k, but the extra layers buy little). The
# margin is for depth: those probes peak at 5246 tokens, and the flash-attention
# workspace grows with context -- the 1070 aborts near 8100-8189 MiB of its 8192
# (measured at -ngl 30 with the CPU draft, and at -np 4 even for -ngl 28).
# Without a draft the anchors are the original 40 at 8k / 32 at 128k. An
# explicit NGL is honoured.
if [ "$MTP" = 1 ]; then
    NGL="${NGL:-$(( 34 - (CTX - 32768) / 16384 ))}"
else
    NGL="${NGL:-$(( 40 - (CTX - 8192) / 15360 ))}"
fi
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
echo "serve: dev=$DEV -ts $TS -ngl=$NGL -np=$NP -lm=$LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
if [ "$MTP" = 1 ]; then
    echo "serve: mtp=on draft=$(basename "$MD") devd=$DEVD ngld=$NGLD ctkd=$DCTK ctvd=$DCTV"
else
    echo "serve: mtp=off"
fi
echo "serve: cap=${CAP}MiB (display GPU)  http://$HOST:$PORT  log: $ROOT/logs/serve.log"

MTP_ARGS=()
if [ "$MTP" = 1 ]; then
    MTP_ARGS=(--spec-type draft-mtp -md "$MD" -devd "$DEVD" -ngld "$NGLD"
              -ctkd "$DCTK" -ctvd "$DCTV")
fi

exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
    --log "$ROOT/logs/serve.log" -- \
    "$ROOT/build/bin/llama-server" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS" -np "$NP" \
    -dev "$DEV" -sm layer -ts "$TS" \
    -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
    -ctk "$CTK" -ctv "$CTV" -fa "$FA" \
    "${MTP_ARGS[@]}" \
    --host "$HOST" --port "$PORT" "$@"
