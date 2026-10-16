#!/usr/bin/env bash
# serve.sh — launch llama-server for the local agent stack.
#
# DEFAULT (and only mode): one Qwen3.8-27B instance split across the GPUs and
# CPU, fitted by llama.cpp's own --fit. The device list depends on whether the
# Thunderbolt eGPU is attached:
#   * eGPU present: -dev CUDA0,Vulkan0 -sm layer — spread across the GTX 1070
#     (eGPU), the Vega 10 (built-in display GPU), and the CPU for the rest.
#   * eGPU absent:  -dev Vulkan0 -sm layer — the Vega 10 plus the CPU.
# The built-in display GPU always respects two guards, deliberately different:
# --fit packs the model into FIT_CAP (default 4600 MiB) via a -fitt margin of
# (vram_total - FIT_CAP), so the fit holds the budget regardless of desktop
# drift (it sees the desktop in `free`); guard.py only SIGKILLs at the higher
# CAP (default 5500 MiB). The 900 MiB gap is desktop-drift tolerance, not model
# headroom: the model footprint is fixed at load, so the gap is how far the
# desktop may grow after load before the guard fires. The eGPU drives no
# display, so it is not capped — only a small fit margin (FITT_CUDA, default
# 256 MiB) keeps it from filling the card. MTP speculation is on by default,
# its draft head on the Vega 10.
#
# Overridable via env:
#   MODEL ALIAS CTX NP PORT CAP FIT_CAP THREADS HOST
#   MTP MD DEVD NGLD DCTK DCTV DNMAX DPMIN
#   LM BATCH UBATCH CTK CTV FA
#   FITT_CUDA (eGPU fit margin, MiB)
#   NGL=/TS= — explicit layer count / tensor split; turns --fit off (llama.cpp
#              aborts rather than honour either).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf}"
# The name the API advertises (/v1/models). Without it llama-server reports the
# .gguf path, so a client's model id becomes the path. The alias gives both
# clients one stable id to refer to.
ALIAS="${ALIAS:-Qwen3.8-27B}"
# CTX is the *pool*, and llama-server divides it by NP: the default 131072 with
# NP=2 is 65536 per slot, which is what a client actually gets and what its
# contextWindow must say. q4_0 KV plus flash attention keep the 128k pool at
# ~2.4 GB, which the layer count pays for by offloading layers to CPU. Raise
# CTX to buy depth back (CTX = NP * the slot size you want) — at the cost of
# layers — or lower it to buy layers back if generation speed matters more.
CTX="${CTX:-131072}"
THREADS="${THREADS:-8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# Two guards, deliberately different (see the header): --fit packs the model
# into FIT_CAP, guard.py only kills at CAP. The gap is desktop-drift tolerance
# after load, not model headroom (the footprint is fixed at load).
FIT_CAP="${FIT_CAP:-4600}"
CAP="${CAP:-5500}"
# Margin to leave free on the eGPU; the fit's own default is 1024 MiB, which
# costs a layer or two. The eGPU is not the display GPU, so 256 is enough.
FITT_CUDA="${FITT_CUDA:-256}"

# MTP speculation is ON by default and its draft head runs on the Vega 10, not
# the CPU: measured +38 % generation over the shipped CPU-draft default at 128k
# (6.224 vs 4.509 t/s at depth 5246) and +6 % prompt, or +34 % against the CPU
# draft at the same -ngl 28 (4.640). The MTP block's forward pass is serial with
# the target's, so with 26+ offloaded trunk layers the CPU is the bottleneck;
# the 1070 has no room for the draft, the Vega does.
# -ngld 99 (not 1) is deliberate: the draft declares nextn_predict_layers = 1,
# and offloading only that layer still leaves the draft's output projection on
# the CPU — a 151k x 5120 matmul per draft step. A count above the layer count
# moves the non-repeating tensors too: measured +10 % generation (5.66 -> 6.22
# t/s) for +234 MiB of the display GPU (client 3151 -> 3384), 2/2 interleaved.
# -ctkd/-ctvd q4_0 is not optional at 128k: the draft's own f16 KV is ~537 MiB
# at 131072 tokens and q4_0 cuts that to ~134 MiB. `-devd` must name a device
# from -dev; the default (follow -dev) would put the draft's share on the 1070.
# The chain length and early-stop (DNMAX/DPMIN below) are the one part of this
# that upstream's defaults get wrong for this host — see the block at DNMAX.
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
# So 6 + p-min 0.5 is a strict win — +18 % where drafts land, no loss where
# they do not — and 14 is worth +29-45 % on high-acceptance content at -13 % on
# low. p-min is NOT a good acceptance proxy (on the 26 %-acceptance prompt it
# still drafted 110 tokens for 38 accepted), so the chain length, not p-min, is
# what has to be chosen for the content. 6 is the safe default; raise DNMAX for
# a workload known to be high-acceptance, and MTP=0 disables speculation
# outright. Neither flag changes the output: greedy drafting plus target
# verification is exactly target-greedy decoding.
DNMAX="${DNMAX:-6}"
DPMIN="${DPMIN:-0.5}"

NP="${NP:-2}"
CTX_SLOT=$(( CTX / NP ))
if [ $(( CTX % NP )) -ne 0 ]; then
    echo "serve: WARNING CTX=$CTX is not a multiple of NP=$NP; slots get ${CTX_SLOT}" >&2
fi

# Detect the eGPU: /dev/nvidia0 exists only once the proprietary driver has
# bound the card, so this is an instant check (no nvidia-smi spawn). The CUDA0
# backend must also be built.
EGPU=0
if [ -c /dev/nvidia0 ] && [ -x "$ROOT/build/bin/llama-bench" ] \
   && "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  CUDA0:'; then
    EGPU=1
fi
if ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  Vulkan0:'; then
    echo "serve: no Vulkan0 backend (the Vega 10 display GPU) is built." >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "serve: missing $MODEL" >&2
    echo "serve: fetch the Qwen3.8-27B GGUF into models/ (see README)" >&2
    exit 1
fi
if [ "$MTP" = 1 ] && [ ! -f "$MD" ]; then
    echo "serve: missing MTP draft $MD" >&2
    echo "serve: set MTP=0 to disable speculation, or fetch the draft model" >&2
    exit 1
fi

if [ "$EGPU" = 1 ]; then
    DEV="CUDA0,Vulkan0"
else
    DEV="Vulkan0"
fi

# --fit always: llama-server derives -ngl and the tensor split from live free
# VRAM per device. The display GPU's margin is (vram_total - FIT_CAP), which is
# baseline-independent because the fit sees the desktop's usage in `free`; the
# eGPU (if present) only needs the small FITT_CUDA margin. No -ngl/-ts anywhere
# by default: the fit aborts if either is set.
VEGA_MARGIN=$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_total_mib() - $FIT_CAP))" 2>/dev/null || true)
if [ -z "${VEGA_MARGIN:-}" ] || [ "$VEGA_MARGIN" -lt 256 ]; then
    echo "serve: WARNING could not derive the display --fit margin (vram_total - fit_cap); using 256 MiB" >&2
    VEGA_MARGIN=256
fi

FIT_ON=1
FIT_ARGS=(--fit on)
if [ "$EGPU" = 1 ]; then
    FIT_ARGS+=( -fitt "$FITT_CUDA,$VEGA_MARGIN")
else
    FIT_ARGS+=( -fitt "$VEGA_MARGIN")
fi

# NGL=/TS= override: llama.cpp's fit aborts rather than honour an explicit
# -ngl/-ts, so setting either turns --fit off and pins the layer count.
if [ -n "${NGL:-}" ] || [ -n "${TS:-}" ]; then
    FIT_ON=0
    FIT_ARGS=()
    [ -n "${NGL:-}" ] && FIT_ARGS+=( -ngl "$NGL")
    [ -n "${TS:-}" ] && FIT_ARGS+=( -ts "$TS")
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

echo "serve: model=$(basename "$MODEL") alias=$ALIAS dev=$DEV ctx=$CTX (${CTX_SLOT}/slot x $NP) threads=$THREADS"
if [ "$EGPU" = 1 ]; then
    echo "serve: eGPU present — split across CUDA0 (1070) + Vulkan0 (Vega) + CPU"
else
    echo "serve: no eGPU — split across Vulkan0 (Vega) + CPU"
fi
if [ "$FIT_ON" = 1 ]; then
    if [ "$EGPU" = 1 ]; then
        echo "serve: --fit on, margins ${FITT_CUDA},${VEGA_MARGIN} MiB (fit-cap=${FIT_CAP}MiB) cap=${CAP}MiB"
    else
        echo "serve: --fit on, margin ${VEGA_MARGIN} MiB (fit-cap=${FIT_CAP}MiB) cap=${CAP}MiB"
    fi
else
    echo "serve: --fit off (NGL=/TS= set): -ngl ${NGL:-auto} -ts ${TS:-auto}"
fi
if [ "$MTP" = 1 ]; then
    echo "serve: mtp=on draft=$(basename "$MD") devd=$DEVD ngld=$NGLD ctkd=$DCTK ctvd=$DCTV n-max=$DNMAX p-min=$DPMIN"
else
    echo "serve: mtp=off"
fi
echo "serve: http://$HOST:$PORT  log: $ROOT/logs/serve.log"

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
