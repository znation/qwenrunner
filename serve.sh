#!/usr/bin/env bash
# serve.sh — launch llama-server for the local agent stack.
#
# DEFAULT (./serve.sh): one fully-resident instance on the Vega 10 display GPU
# only — no eGPU involved:
#   * Qwen3.5-4B Q4_K_M on Vulkan0 (:8080), under the 5000 MiB watchdog
# The 9B cannot fit under the display cap (its Q4_K_M weights alone exceed
# 5000 MiB), so the 4B is the default model. --fit always, target
# (vram_total - CAP) free, so the fit holds the cap regardless of desktop
# drift. Generation is ~19 t/s at the shipped 64k allocation (short prompt;
# ~16 at full depth) and climbs to ~35 t/s at a 32k pool — the *allocated*
# context is the first-order throughput lever, not the model (vs the 2-6 t/s
# the old 27B split managed).
#
# Overridable via env:
#   MODEL ALIAS CTX NP PORT CAP THREADS HOST FITT
#   EGPU=1   — two fully-resident instances: Qwen3.5-9B on the GTX 1070 eGPU
#              (:8080) + Qwen3.5-4B on the Vega (:8081); its own env:
#              MAIN_MODEL MAIN_ALIAS MAIN_CTX MAIN_NP MAIN_PORT MAIN_FITT
#              VEGA_MODEL VEGA_ALIAS VEGA_CTX VEGA_NP VEGA_PORT VEGA_CAP VEGA_FITT
#   LEGACY=1 — the old single Qwen3.8-27B split across both GPUs (requires the
#              eGPU; its own env:
#               MODEL ALIAS CTX TS NGL LM BATCH UBATCH CTK CTV FA HOST PORT
#               CAP NP MTP MD DEVD NGLD DCTK DCTV DNMAX DPMIN FIT FITT_CUDA),
#               e.g. LEGACY=1 ./serve.sh  or  LEGACY=1 MTP=0 ./serve.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

EGPU="${EGPU:-0}"
LEGACY="${LEGACY:-0}"
if [ "$EGPU" = 1 ]; then
    # ---- eGPU mode (EGPU=1): two fully-resident instances, one per GPU ----

    # eGPU (GTX 1070) main agent. 131072 with -np 2 is 65536 per slot — the
    # same per-slot context the clients already expect (pi contextWindow
    # 59392, omp 65536), so no client change is needed. At this pool the 1070
    # holds weights ~5.5 GiB + q4_0 KV ~1.1 GiB + compute ~0.7 GiB ≈ 7.3 GiB
    # of its 8 GiB.
    MAIN_MODEL="${MAIN_MODEL:-$ROOT/models/Qwen3.5-9B-Q4_K_M.gguf}"
    MAIN_ALIAS="${MAIN_ALIAS:-Qwen3.5-9B}"
    MAIN_CTX="${MAIN_CTX:-131072}"
    MAIN_NP="${MAIN_NP:-2}"
    MAIN_PORT="${MAIN_PORT:-8080}"

    # Display GPU (Vega 10) secondary. Q4_K_S is the measured best 4B quant on
    # this Vega (see the default block below); ~3.3 GiB at the 32768 pool
    # leaves ~1.7 GiB of the 5000 MiB cap for the desktop at a typical
    # baseline; guard.py still kills before any breach.
    VEGA_MODEL="${VEGA_MODEL:-$ROOT/models/Qwen3.5-4B-Q4_K_S.gguf}"
    VEGA_ALIAS="${VEGA_ALIAS:-Qwen3.5-4B}"
    VEGA_CTX="${VEGA_CTX:-32768}"
    VEGA_NP="${VEGA_NP:-2}"
    VEGA_PORT="${VEGA_PORT:-8081}"
    VEGA_CAP="${VEGA_CAP:-5000}"

    THREADS="${THREADS:-8}"
    HOST="${HOST:-127.0.0.1}"

    if [ ! -c /dev/nvidia0 ] || [ ! -x "$ROOT/build/bin/llama-bench" ] \
       || ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  CUDA0:'; then
        echo "serve: no CUDA eGPU: /dev/nvidia0 is missing or no CUDA0 backend is built." >&2
        echo "serve: the default needs the eGPU for the main agent. Build with GGML_CUDA=ON." >&2
        exit 1
    fi
    if ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  Vulkan0:'; then
        echo "serve: no Vulkan0 backend (the Vega 10 display GPU) is built." >&2
        exit 1
    fi
    for m in "$MAIN_MODEL" "$VEGA_MODEL"; do
        if [ ! -f "$m" ]; then
            echo "serve: missing $m" >&2
            echo "serve: run ./scripts/dl_dual.sh to fetch the Qwen3.5-9B + Qwen3.5-4B pair" >&2
            exit 1
        fi
    done

    mkdir -p "$ROOT/logs"

    # --fit on both instances, always, with a per-GPU --fit-target. The 1070 is
    # not the display GPU, so it only needs a small margin (256 MiB, the legacy
    # FITT_CUDA value). The Vega instance must hold the display cap no matter
    # what the desktop is doing: its target leaves (vram_total - CAP) free, and
    # since the fit sees the desktop's usage in `free`, that is
    # baseline-independent — a loaded desktop automatically costs it layers,
    # never the display. No -ngl/-ts anywhere: the fit aborts if either is set.
    MAIN_FITT="${MAIN_FITT:-256}"
    VEGA_FITT="${VEGA_FITT:-$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_total_mib() - $VEGA_CAP))" 2>/dev/null || true)}"
    if [ -z "${VEGA_FITT:-}" ] || [ "$VEGA_FITT" -lt 256 ]; then
        echo "serve: WARNING could not derive the Vega --fit target (vram_total - cap); using 256 MiB" >&2
        VEGA_FITT=256
    fi

    echo "serve: main  $(basename "$MAIN_MODEL") alias=$MAIN_ALIAS dev=CUDA0 ctx=$MAIN_CTX ($(( MAIN_CTX / MAIN_NP ))/slot x $MAIN_NP) --fit-target ${MAIN_FITT}MiB  http://$HOST:$MAIN_PORT"
    echo "serve: vega  $(basename "$VEGA_MODEL") alias=$VEGA_ALIAS dev=Vulkan0 ctx=$VEGA_CTX ($(( VEGA_CTX / VEGA_NP ))/slot x $VEGA_NP) --fit-target ${VEGA_FITT}MiB cap=${VEGA_CAP}MiB  http://$HOST:$VEGA_PORT"
    echo "serve: --fit on both (no split, no MTP); logs serve-main.log / serve-vega.log"

    "$ROOT/build/bin/llama-server" \
        -m "$MAIN_MODEL" -a "$MAIN_ALIAS" -c "$MAIN_CTX" -t "$THREADS" -np "$MAIN_NP" \
        -dev CUDA0 --fit on -fitt "$MAIN_FITT" -lm dio -b 2048 -ub 512 \
        -ctk q4_0 -ctv q4_0 -fa on \
        --host "$HOST" --port "$MAIN_PORT" "$@" >>"$ROOT/logs/serve-main.log" 2>&1 &
    MAIN_PID=$!

    python3 "$ROOT/scripts/guard.py" --cap-mib "$VEGA_CAP" --label serve-vega \
        --log "$ROOT/logs/serve-vega.log" -- \
        "$ROOT/build/bin/llama-server" \
        -m "$VEGA_MODEL" -a "$VEGA_ALIAS" -c "$VEGA_CTX" -t "$THREADS" -np "$VEGA_NP" \
        -dev Vulkan0 --fit on -fitt "$VEGA_FITT" -lm dio -b 2048 -ub 512 \
        -ctk q4_0 -ctv q4_0 -fa on \
        --host "$HOST" --port "$VEGA_PORT" &
    VEGA_PID=$!

    cleanup() {
        kill "$MAIN_PID" "$VEGA_PID" 2>/dev/null || true
        wait "$MAIN_PID" "$VEGA_PID" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    # Tear both down when either dies (e.g. the watchdog kills the Vega
    # instance on a display breach); one clean command restarts the pair.
    rc=0
    wait -n "$MAIN_PID" "$VEGA_PID" || rc=$?
    exit "$rc"
fi

if [ "$LEGACY" = 0 ]; then
    # ---- default: one fully-resident instance on the Vega 10 (display GPU),
    # no eGPU involved. The 9B cannot fit under the display cap (its Q4_K_M
    # weights alone exceed 5000 MiB), so the default model is the 4B; EGPU=1
    # restores the two-GPU mode. --fit always, leaving (vram_total - CAP) free
    # so the fit holds the cap regardless of desktop drift (it sees the
    # desktop in `free`); no -ngl/-ts anywhere (the fit aborts if either is
    # set). O_DIRECT is unsupported on the Vega (measured: dio falls back to
    # buffered IO with a warning), so -lm none.
    # Q4_K_S, not Q4_K_M: measured on this Vega (results/amd4b_*.md) it is
    # +2.6 % generation / +4-5 % prompt at statistically identical perplexity
    # (8.54 vs 8.47 on ppl_corpus.txt), and ~140 MiB lighter under the cap.
    # Q4_0 is faster still (+6.5 %) but costs a real quality hit (8.72).
    MODEL="${MODEL:-$ROOT/models/Qwen3.5-4B-Q4_K_S.gguf}"
    ALIAS="${ALIAS:-Qwen3.5-4B}"
    CTX="${CTX:-65536}"
    NP="${NP:-1}"
    PORT="${PORT:-8080}"
    CAP="${CAP:-5000}"
    HOST="${HOST:-127.0.0.1}"
    THREADS="${THREADS:-8}"
    FITT="${FITT:-$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_total_mib() - $CAP))" 2>/dev/null || true)}"
    if [ -z "${FITT:-}" ] || [ "$FITT" -lt 256 ]; then
        echo "serve: WARNING could not derive the --fit target (vram_total - cap); using 256 MiB" >&2
        FITT=256
    fi
    if ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -q '^  Vulkan0:'; then
        echo "serve: no Vulkan0 backend (the Vega 10 display GPU) is built." >&2
        exit 1
    fi
    if [ ! -f "$MODEL" ]; then
        echo "serve: missing $MODEL" >&2
        echo "serve: run ./scripts/dl_dual.sh to fetch the Qwen3.5-9B + Qwen3.5-4B pair" >&2
        exit 1
    fi
    mkdir -p "$ROOT/logs"
    echo "serve: model=$(basename "$MODEL") alias=$ALIAS dev=Vulkan0 ctx=$CTX ($(( CTX / NP ))/slot x $NP) --fit-target ${FITT}MiB cap=${CAP}MiB  http://$HOST:$PORT"
    exec python3 "$ROOT/scripts/guard.py" --cap-mib "$CAP" --label serve \
        --log "$ROOT/logs/serve.log" -- \
        "$ROOT/build/bin/llama-server" \
        -m "$MODEL" -a "$ALIAS" -c "$CTX" -t "$THREADS" -np "$NP" \
        -dev Vulkan0 --fit on -fitt "$FITT" -lm none -b 2048 -ub 512 \
        -ctk q4_0 -ctv q4_0 -fa on \
        --host "$HOST" --port "$PORT" "$@"
fi
# ---- legacy (LEGACY=1): one Qwen3.8-27B split across both GPUs ----

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
#
# THE CHAIN LENGTH COSTS 1070 VRAM, because n_outputs_max scales with it. At the
# shipped config (128k, -np 2), one run each:
#
#   DNMAX  client MiB  NVIDIA MiB  system peak  outcome
#   3       3282.7      7677.3       4764.2      loads
#   6       3314.2      8013.3       4943.7      loads, 179 MiB of 8192 left
#   14        --         --          --          FAILS TO LOAD: cudaMalloc of
#                                                 the 837 MiB compute buffer
#
# So 6 is affordable but it spends the 1070's reserve -- with n-max 3 the last
# startable rung at 128k is 30 and 28 leaves two layers of slack; with n-max 6
# the 1070 peaks at 8013 of 8192 and 28 is at the ceiling. DNMAX=14 is a 32k
# option only (it was measured at -c 32768 -ngl 32, where the compute buffer is
# smaller); it does not load at 128k. If the display or the 1070 is tight, the
# rung to drop to is -ngl 26, not 24 -- measured, 24 and 26 both hold 3 trunk
# layers on the Vega (client 2972.5 / 2971.5 MiB, i.e. the same display cost) but
# 26 keeps two more layers on the 1070 (7169 vs 7689 MiB).
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
#   FIT=0  the measured formula, which is what RESULTS.md quotes: 32 -
#          (CTX-32768)/16384 with MTP, 40 - (CTX-8192)/15360 without, floored at
#          8 -- two layers under the last startable rung at the 32k and 128k
#          anchors. Measured fit points (load + a real completion,
#          scripts/split_fit.py, `-ts 6,1` with the draft on the Vega): 38
#          layers at 8k, 36 at 32k, 30 at 128k; the margin is for depth, because
#          those probes peak at 5246 tokens and the flash-attention workspace
#          grows with context. Without a draft the anchors are 40 at 8k / 32 at
#          128k. The MTP form is 32 rather than 34 because those anchors were
#          measured with upstream's 3-token draft chain: the shipped 6 costs the
#          1070 ~1.3 layers, which moves the 128k ceiling from 30 to 28. See the
#          DNMAX block for the measurement. Deterministic, and the reserve it
#          leaves (client 2972 MiB of a 5000 MiB cap on a ~1480 MiB baseline,
#          7689 MiB of the 1070's 8192) is what absorbs desktop drift -- it is
#          safe to ~1880 MiB -- most, not all, of the 746-2188 MiB range the
#          baseline is known to occupy; 28 layers is safe only to ~1540, and its
#          two runs of the same command peaked 200 MiB apart, one of them over
#          the cap.
#   FIT=1  llama-server projects per-device weight + KV + compute + draft memory
#          against live *free* VRAM and picks -ngl and the tensor split itself.
#          The display cap is encoded as a margin: -fitt asks the fit to leave
#          (vram_total - CAP) free on the Vega, which is baseline-independent
#          because the fit already sees the desktop's usage in `free`. Measured
#          at 128k + MTP it picks 28-30 layers (client 3383-3648 MiB, NVIDIA
#          7479-7525 MiB) against the formula's 28 (3384/7541) -- those runs also
#          predate the 6-token chain, which is why the formula now ships 26 -- and
#          its projection tracked measurement to ~25 MiB. It re-derives the count
#          from live free memory instead of assuming a baseline -- but the
#          choice is discrete and moves with a few MiB of desktop noise (28 at a
#          7059 MiB free Vega, 30 at 7059+10), and at 30 it spends the reserve
#          the formula keeps, leaving ~350 MiB of drift tolerance instead of
#          ~620. Hence opt-in, not default -- but only while the desktop is
#          quiet: once the live display baseline passes ~1200 MiB a fixed
#          formula -ngl would breach the cap, so serve.sh warns and auto-falls
#          back to --fit (the only path that gives up layers to hold it). An
#          explicit FIT=0 pins the formula and suppresses the fallback;
#          NGL=/TS= also keep the formula, because the fit aborts on either.
#
# NGL= or TS= takes over: either one turns --fit off, because llama.cpp's fit
# aborts rather than honour an explicit -ngl/-ts.
# FIT=0 is the default; FIT=1 forces --fit. An *explicit* FIT=0 pins the
# formula (the auto-fallback below is suppressed); unset FIT lets serve.sh
# fall back to --fit when the live desktop baseline is too high for a fixed
# -ngl. NGL=/TS= also pin the formula (the fit aborts on either). Capture
# what the user actually set before the formula fills NGL/TS below.
FIT_SET=0
[ -n "${FIT+x}" ] && FIT_SET=1
NGL_TS_SET=0
FIT="${FIT:-0}"
# Margin to leave free on the 1070; the fit's own default is 1024 MiB, which
# costs a layer or two. The 1070 is not the display GPU, so 256 is enough.
FITT_CUDA="${FITT_CUDA:-256}"
if [ -n "${NGL:-}" ] || [ -n "${TS:-}" ]; then
    FIT=0
    NGL_TS_SET=1
fi

if [ "$MTP" = 1 ]; then
    # 32, not 34: the two-layer reserve is unchanged, but the *ceiling* it is
    # measured against moves with the draft chain. At 128k the last startable
    # rung is 30 with upstream's n-max 3 and 28 with the shipped 6, because
    # n_outputs_max scales with the chain and costs the 1070 336 MiB (~1.3
    # layers) -- see the DNMAX block. Shipping 28 with the 6-token chain puts the
    # 1070 at 8013 of 8192 MiB and leaves the config safe only while the desktop
    # baseline is <= ~1540 MiB; 26 leaves it at 7689 and safe to ~1880. Measured
    # at 28, two runs of the same command 3.5 MiB apart in baseline peaked 5053.3
    # (killed) and 4850.9 -- a coin flip; at 26, 4529.1 and 4721.1, clear both
    # times.
    # Measured: -ngl 24 buys nothing over 26 on the display (both hold 3 trunk
    # layers on the Vega, client 2972.5 vs 2971.5 MiB) and costs two more layers.
    # Raising DNMAX further must lower NGL further; DNMAX=14 does not load at all
    # at 128k.
    NGL="${NGL:-$(( 32 - (CTX - 32768) / 16384 ))}"
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

# The safe layer count is a function of the live desktop baseline, not a
# constant -- measured 746-2188 MiB within one hour. At 128k this config's
# client footprint is ~3384 MiB at 28 layers, so a fixed -ngl reaches the cap at
# a baseline near 1216 MiB. The watchdog is the backstop, not the plan: say so
# before it has to fire. Above that baseline the only safe default is to let
# --fit give up layers, so an unset FIT auto-falls back to it (with this
# warning); an explicit FIT=0 or an NGL=/TS= override pins the formula instead.
BASE_MIB=$(python3 -c "import sys; sys.path.insert(0, '$ROOT/scripts'); import guard; print(int(guard.vram_used_mib()))" 2>/dev/null || true)
if [ -n "${BASE_MIB:-}" ] && [ "$MTP" = 1 ] && [ "$BASE_MIB" -gt 1200 ]; then
    if [ "$FIT" = 1 ]; then
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB (> the ~1216 MiB the" >&2
        echo "serve: WARNING 28-layer config was measured to survive); --fit will pick" >&2
        echo "serve: WARNING fewer layers to hold the ${CAP} MiB cap." >&2
    elif [ "$NGL_TS_SET" = 1 ]; then
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB; this config fits to about" >&2
        echo "serve: WARNING 1216 MiB. NGL=/TS= is set, so --fit cannot take over (llama.cpp" >&2
        echo "serve: WARNING aborts rather than honour either); lower NGL= by 2-4 (~158" >&2
        echo "serve: WARNING MiB/layer at -ts 6,1) or close desktop GPU users." >&2
    elif [ "$FIT_SET" = 1 ]; then
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB (> the ~1216 MiB the" >&2
        echo "serve: WARNING 28-layer config was measured to survive); FIT=0 is set, so the" >&2
        echo "serve: WARNING formula is kept and may breach the ${CAP} MiB cap (the watchdog" >&2
        echo "serve: WARNING would kill the load). Unset FIT, or use FIT=1, to let --fit" >&2
        echo "serve: WARNING give up layers." >&2
    else
        echo "serve: WARNING display baseline is ${BASE_MIB} MiB (> the ~1216 MiB the" >&2
        echo "serve: WARNING 28-layer config was measured to survive); falling back to" >&2
        echo "serve: WARNING --fit, which picks fewer layers to hold the ${CAP} MiB cap." >&2
        echo "serve: WARNING Set FIT=0 or NGL= to keep the measured formula instead." >&2
        FIT=1
    fi
fi

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
