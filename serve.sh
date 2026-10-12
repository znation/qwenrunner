#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B under the 5000 MiB VRAM watchdog.
#
# One model instance, split across the GTX 1070 (CUDA) and the Vega 10
# (Vulkan), with the layers that do not fit spilling to CPU: -dev CUDA0,Vulkan0
# -sm layer. Measured 40 layers at 8k context, 32 at 128k without speculation;
# the KV cache competes with the weights for the same VRAM, so context length
# and layer count trade against each other. The layer count is derived from the
# context by the measured formula below; FIT=1 instead lets llama.cpp's own
# --fit derive it from live free VRAM (see the layer-budget block).
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
# Two slots by default (-np 2), so a second agent makes progress instead of
# queueing. -c is the *shared* pool: 131072 with -np 2 is 65536 per slot, and the
# client must be configured for that (pi's models.json says contextWindow
# 65536). Slots buy fairness, not throughput -- see the -np block below.
#
# The 5000 MiB cap protects the *display* GPU (the Vega 10), which is why the
# watchdog can kill the whole thing. The 1070 has its own 8 GB and is reported
# separately as nvidia_vram_peak_mib.
#
# Overridable via env (anything unset is derived):
#   MODEL ALIAS CTX THREADS TS NGL LM BATCH UBATCH CTK CTV FA HOST PORT CAP NP
#   MTP MD DEVD NGLD DCTK DCTV DNMAX DPMIN FIT FITT_CUDA
#
#   ./serve.sh                      # 2 slots x 64k, both GPUs, MTP on
#   NP=1 ./serve.sh                 # one slot, the whole 128k pool for one agent
#   MTP=0 ./serve.sh                # the same split without speculative decoding
#   CTX=65536 ./serve.sh            # shorter pool: 32k/slot, more layers (faster)
#   NGL=36 TS=4,1 ./serve.sh        # override the split (turns --fit off)
#   FIT=1 ./serve.sh                # let --fit derive -ngl from live free VRAM
# Extra args are appended verbatim to the llama-server command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf}"
# The name the API advertises (/v1/models). Without it llama-server reports the
# .gguf path, so a client's model id becomes
# "/home/zach/qwenrunner/models/Qwen3.8-27B-UD-Q4_K_M.gguf" -- which is what
# omp's llama.cpp discovery then shows as the selector. The alias gives both
# clients one stable id to refer to.
ALIAS="${ALIAS:-Qwen3.8-27B}"
# CTX is the *pool*, and llama-server divides it by NP: the default 131072 with
# NP=2 is 65536 per slot, which is what a client actually gets and what its
# contextWindow must say. q4_0 KV plus flash attention keep the 128k pool at
# ~2.4 GB, which the layer count below pays for by offloading ~8 layers to CPU.
# Raise CTX to buy depth back (CTX = NP * the slot size you want) -- at the cost
# of layers -- or lower it to buy layers back if generation speed matters more.
CTX="${CTX:-131072}"
THREADS="${THREADS:-8}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CAP="${CAP:-5000}"
# Two slots by default: a second agent (or a subagent) then makes progress
# instead of waiting behind the first. Measured at 128k pool / -np 2: n_ctx_slot
# 65536, two concurrent 1135-token requests both progressing (100.0 s wall
# together) against one waiting 44 s behind the other at -np 1 (91.4 s wall), so
# this buys fairness, not throughput -- the box is CPU/GPU-bound on one model.
# Cost: +160 MiB of the 1070's compute buffers (NVIDIA peak 7701 against 7541)
# and half the per-slot context. -np 4 fits at load (NVIDIA 8093) but leaves
# 32768 per slot and has aborted a real request in `ggml_cuda_pool_vmm::alloc`,
# which is why llama-server's own default of 4 is never used. The 128k-context
# measurements in RESULTS.md were all taken at -np 1.
NP="${NP:-2}"
CTX_SLOT=$(( CTX / NP ))
if [ $(( CTX % NP )) -ne 0 ]; then
    echo "serve: WARNING CTX=$CTX is not a multiple of NP=$NP; slots get ${CTX_SLOT}" >&2
fi

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
# The chain length and early-stop (DNMAX/DPMIN below) are the one part of this
# that upstream's defaults get wrong for this host -- see the block at DNMAX.
MTP="${MTP:-1}"
MD="${MD:-$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf}"
DEVD="${DEVD:-Vulkan0}"
NGLD="${NGLD:-99}"
DCTK="${DCTK:-q4_0}"
DCTV="${DCTV:-q4_0}"
# The MTP draft *chain* is the one knob left on the speculation path, and
# llama.cpp's defaults are a compromise that costs this host a lot: -n-max 3 and
# -p-min 0.00 mean the chain stops on length, never on confidence. Measured
# (results/draft_chain*.jsonl, 2 interleaved reps each, bg_cpu_pct <= 10):
#
#   arm                     feature-list (46/48 accepted)  mid-doc (32/90)
#   -n-max 3 (upstream)     5.98 t/s                        3.41-3.67 t/s
#   -n-max 6                7.05                           2.64
#   -n-max 6 -p-min 0.5     7.08 / 7.21                    3.63
#   -n-max 14 -p-min 0.5    7.71 / 7.76                    not measured
#
# So 6 + p-min 0.5 is a strict win -- +18 % where drafts land, no loss where
# they do not -- and 14 is worth +29-45 % on high-acceptance content at -13 % on
# low. p-min is NOT a good acceptance proxy (on the 26 %-acceptance prompt it
# still drafted 110 tokens for 38 accepted), so the chain length, not p-min, is
# what has to be chosen for the content. 6 is the safe default; raise DNMAX for
# a workload known to be high-acceptance, and MTP=0 disables speculation
# outright. Neither flag changes the output: greedy drafting plus target
# verification is exactly target-greedy decoding.
#
# Those figures are at -c 32768 -ngl 32, not at the shipped 128k/28, because the
# desktop was holding ~1.6 GiB of the Vega and left the shipped config ~16 MiB
# of cap margin. The transfer is expected to hold or improve -- the shipped
# config leaves *more* CPU work per target step, so a draft step is relatively
# cheaper there -- but it is an inference, not a measurement.
DNMAX="${DNMAX:-6}"
DPMIN="${DPMIN:-0.5}"

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
# Layer budget. The default (FIT=0) is the measured formula; FIT=1 hands the
# decision to llama.cpp's own --fit.
#
#   FIT=0  the measured formula, which is what RESULTS.md quotes: 34 -
#          (CTX-32768)/16384 with MTP, 40 - (CTX-8192)/15360 without, floored at
#          8 -- two layers under the last startable rung at the 32k and 128k
#          anchors. Measured fit points (load + a real completion,
#          scripts/split_fit.py, `-ts 6,1` with the draft on the Vega): 38
#          layers at 8k, 36 at 32k, 30 at 128k; the margin is for depth, because
#          those probes peak at 5246 tokens and the flash-attention workspace
#          grows with context. Without a draft the anchors are 40 at 8k / 32 at
#          128k. Deterministic, and the reserve it leaves (client 3384 MiB of a
#          5000 MiB cap on a ~1100 MiB baseline) is what absorbs desktop drift.
#   FIT=1  llama-server projects per-device weight + KV + compute + draft memory
#          against live *free* VRAM and picks -ngl and the tensor split itself.
#          The display cap is encoded as a margin: -fitt asks the fit to leave
#          (vram_total - CAP) free on the Vega, which is baseline-independent
#          because the fit already sees the desktop's usage in `free`. Measured
#          at 128k + MTP it picks 28-30 layers (client 3383-3648 MiB, NVIDIA
#          7479-7525 MiB) against the formula's 28 (3384/7541), and its
#          projection tracked measurement to ~25 MiB. It re-derives the count
#          from live free memory instead of assuming a baseline -- but the
#          choice is discrete and moves with a few MiB of desktop noise (28 at a
#          7059 MiB free Vega, 30 at 7059+10), and at 30 it spends the reserve
#          the formula keeps, leaving ~350 MiB of drift tolerance instead of
#          ~620. Hence opt-in, not default.
#
# NGL= or TS= takes over: either one turns --fit off, because llama.cpp's fit
# aborts rather than honour an explicit -ngl/-ts.
FIT="${FIT:-0}"
# Margin to leave free on the 1070; the fit's own default is 1024 MiB, which
# costs a layer or two. The 1070 is not the display GPU, so 256 is enough.
FITT_CUDA="${FITT_CUDA:-256}"
if [ -n "${NGL:-}" ] || [ -n "${TS:-}" ]; then
    FIT=0
fi

if [ "$MTP" = 1 ]; then
    NGL="${NGL:-$(( 34 - (CTX - 32768) / 16384 ))}"
    TS="${TS:-6,1}"
else
    NGL="${NGL:-$(( 40 - (CTX - 8192) / 15360 ))}"
    TS="${TS:-4,1}"
fi
[ "$NGL" -lt 8 ] && NGL=8

# -ts, when it is used at all, is tied to whether the draft is resident on the
# Vega 10: with MTP the draft costs ~1.5 GB of the display GPU, so the trunk's
# share there must stay small. 6,1 is the measured optimum at 128k (client 3384
# MiB for 28 trunk layers plus the draft); 4,1 puts the draft's weights on top
# of ~6 trunk layers and breaches the cap (client 3644, measured). Without a
# draft, 4,1 is the measured split. It is NOT the documented ratio either way: a
# requested 6,1 assigned 28 layers to CUDA0 and 4 to Vulkan0 at -ngl 32, i.e.
# 7:1, because non-layer tensors go to the primary device. Tuned empirically
# against the two VRAM counters.
VEGA_MARGIN=""
if [ "$FIT" = 1 ]; then
    VEGA_MARGIN=$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_total_mib() - $CAP))" 2>/dev/null || true)
    if [ -z "${VEGA_MARGIN:-}" ] || [ "$VEGA_MARGIN" -lt 256 ]; then
        echo "serve: could not derive the --fit margin (vram_total $CAP); using the formula" >&2
        FIT=0
    fi
fi

FIT_ARGS=()
if [ "$FIT" = 1 ]; then
    # -ngl/-ts must stay unset or the fit aborts ("already set by user").
    FIT_ARGS=(--fit on -fitt "$FITT_CUDA,$VEGA_MARGIN")
else
    FIT_ARGS=(-ngl "$NGL" -ts "$TS")
fi

# The safe layer count is a function of the live desktop baseline, not a
# constant -- measured 746-2188 MiB within one hour. At 128k this config's
# client footprint is ~3384 MiB at 28 layers, so a fixed -ngl reaches the cap at
# a baseline near 1216 MiB. The watchdog is the backstop, not the plan: say so
# before it has to fire.
BASE_MIB=$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_used_mib()))" 2>/dev/null || true)
if [ -n "${BASE_MIB:-}" ] && [ "$MTP" = 1 ] && [ "$BASE_MIB" -gt 1200 ]; then
    if [ "$FIT" = 1 ]; then
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB (> the ~1216 MiB the" >&2
        echo "serve: WARNING 28-layer config was measured to survive); --fit will pick" >&2
        echo "serve: WARNING fewer layers to hold the ${CAP} MiB cap." >&2
    else
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB; this config fits to about" >&2
        echo "serve: WARNING 1216 MiB. Lower NGL= by 2-4 (~158 MiB/layer at -ts 6,1), close" >&2
        echo "serve: WARNING desktop GPU users, or re-run with FIT=1 so llama.cpp's own" >&2
        echo "serve: WARNING --fit derives the layer count from live free VRAM." >&2
    fi
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

mkdir -p "$ROOT/logs"

echo "serve: model=$(basename "$MODEL") alias=$ALIAS ctx=$CTX (${CTX_SLOT}/slot x $NP) threads=$THREADS"
if [ "$FIT" = 1 ]; then
    echo "serve: dev=$DEV -ngl/-ts=auto(--fit, margins ${FITT_CUDA},${VEGA_MARGIN}) -np=$NP -lm=$LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
    echo "serve: fit would be -ngl $NGL -ts $TS from the measured formula"
else
    echo "serve: dev=$DEV -ts $TS -ngl=$NGL -np=$NP -lm=$LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
fi
if [ "$MTP" = 1 ]; then
    echo "serve: mtp=on draft=$(basename "$MD") devd=$DEVD ngld=$NGLD ctkd=$DCTK ctvd=$DCTV n-max=$DNMAX p-min=$DPMIN"
else
    echo "serve: mtp=off"
fi
echo "serve: cap=${CAP}MiB (display GPU)  http://$HOST:$PORT  log: $ROOT/logs/serve.log"

MTP_ARGS=()
if [ "$MTP" = 1 ]; then
    MTP_ARGS=(--spec-type draft-mtp -md "$MD" -devd "$DEVD" -ngld "$NGLD"
              -ctkd "$DCTK" -ctvd "$DCTV"
              --spec-draft-n-max "$DNMAX" --spec-draft-p-min "$DPMIN")
fi

exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
    --log "$ROOT/logs/serve.log" -- \
    "$ROOT/build/bin/llama-server" \
    -m "$MODEL" -a "$ALIAS" -c "$CTX" -t "$THREADS" -np "$NP" \
    -dev "$DEV" -sm layer "${FIT_ARGS[@]}" \
    -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
    -ctk "$CTK" -ctv "$CTV" -fa "$FA" \
    "${MTP_ARGS[@]}" \
    --host "$HOST" --port "$PORT" "$@"
