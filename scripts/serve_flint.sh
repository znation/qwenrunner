#!/usr/bin/env bash
# serve_flint.sh — launch llama-server for Recursive Flint on Apple Silicon (Metal).
#
# Flint is the second model this repo can host, and on this machine it is the
# *fast* one: 39.8B total / ~2.5B active parameters (49 layers: 2 dense + 47
# MoE, 256 routed experts with 6 selected per token plus 2 always-active shared
# experts). Measured here with llama-bench (-ngl 99 -fa 1 -ctk f16 -ctv f16
# -t 8, r=2), against the Qwen-3.8-27B operating point in RESULTS.md:
#
#   depth      flint pp512   flint tg128   qwen3.8-27B (README)
#   0            3706 t/s      123.5 t/s   705 / 24.4
#   30k          1589 t/s       90.9 t/s   -
#   100k          604 t/s       48.0 t/s   176-231 / 14.3
#
# That is ~5x both directions shallow, ~3x/3.4x at 100k, and the reason is the
# arithmetic: 2.5B active parameters per token rather than 27B, and a KV cache
# that does not grow on 35 of the 49 layers.
#
# Memory: weights are 24.21 GiB (Q4_1 codes with F32 embeddings — see the GGUF
# card). A token costs 14 (full-attention layers) x 4 kv heads x 128 dims x 2
# (K and V) x 2 B = 28 672 B, because the other 35 layers are sliding-window at
# 513 tokens and carry a fixed ~35 MiB per slot instead of a growing cache. The
# *full* 262 144-token context therefore costs 7.0 GiB, against ~8 GiB for
# Qwen's 128k. Model + one full-context slot ≈ 31 GiB; the default 8 slots at
# CTX=131072 measure 59.3 GiB of RSS all in, leaving ~68 GiB of the 128 GB for
# whatever else the machine is doing (see the CTX block below for the table).
#
# There is no draft head in this GGUF (the repo is one 26.0 GB file), so none of
# serve.sh's MTP machinery applies. What does apply is llama.cpp's model-free
# n-gram drafter, which SPEC below turns on by default -- it is lossless, worth
# +66-72 % on copy-heavy traffic, and neutral on reasoning-heavy traffic. The
# t/s table above was measured with speculation off, on llama-bench, which has
# no speculative support at all; FLINT.md has the spec A/B by workload.
#
# Requires llama.cpp with Flint support — the pinned checkout predates it. The
# binary is a second build; see scripts/build_flint.sh for the revision and why
# the pin did not move.
#
# Overridable via env:
#   MODEL CTX NP INSTANCES PORT HOST THREADS NGL LM BATCH UBATCH CTK CTV FA TENSOR CRAM ALIAS SWAPCAP
#
#   ./scripts/serve_flint.sh                  # one full-context slot on 8091
#   NP=4 ./scripts/serve_flint.sh             # four harnesses, -14 GiB KV
#   NP=1 ./scripts/serve_flint.sh             # one stream, fastest per token
#   CTX=65536 ./scripts/serve_flint.sh        # -14 GiB KV at 8 slots, shallower
#   CRAM=4096 ./scripts/serve_flint.sh        # more conversations than slots
#   CRAM=8192 ./scripts/serve_flint.sh        # host prompt cache, see below
# Extra args are appended verbatim to every llama-server command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/third_party/llama-flint-build/bin/llama-server"
BENCH="$ROOT/third_party/llama-flint-build/bin/llama-bench"

MODEL="${MODEL:-$ROOT/models/flint-int4-q4_1.gguf}"

# Model id llama-server advertises on /v1/models, and therefore the id every
# client config must reference -- the pi-models.json/opencode.json shipped with
# the GGUF both name "flint".
ALIAS="${ALIAS:-flint}"

# Per-slot context. The card documents 262 144 tokens, but with eight slots that
# is 56 GiB of f16 KV (7.0 GiB each), and real agent harnesses rarely carry more
# than ~100k tokens of transcript. 131 072 halves it to 3.5 GiB/slot, 28 GiB for
# eight slots, on top of 24.21 GiB of weights (measured RSS at the defaults:
# 59.3 GiB for 8 slots, 42.1 GiB for 4, 33.0 GiB for 4 slots at CTX=65536).
# Raise it if your agents really run to 200k+ tokens -- each +131 072 costs
# another 3.5 GiB *per slot* -- and note a session that outgrows its slot is
# truncated, not given a neighbour's.
CTX="${CTX:-131072}"

# Slots *within* the server (-np), each getting CTX of KV. Default 8: one per
# concurrent harness in the fleet this repo serves, because the slot count is
# the one knob that decides whether a fleet's throughput survives.
#
# llama-server puts up to N requests into one decode step, and a step reads the
# weights once regardless of how many rows it carries, so extra rows amortise the
# 1.56 GB/token weight read. Measured here, four concurrent requests, one
# instance at a time, chat, 256-token responses, temp 0, SPEC=0 (full tables in
# FLINT.md):
#
#   shallow, 4 concurrent    -np 1  105.7 agg (121-123 each, serialized)
#                            -np 2  161.3 agg (82.6-94.5 each)
#                            -np 4  301-310 agg (75.0-77.5 each)   <- 2.9x
#                            -np 8  304-310 agg  <- same as -np 4: slots beyond
#                            the real concurrency are wasted memory
#   shallow, 8 concurrent    -np 8  376-488 agg (47.4-61.6 each)   <- 1.3-1.5x
#                            the -np 4 / 4-concurrent figure
#
# The rule that matters is *slots >= concurrent sessions*, and the failure when
# it is violated is not graceful. Measured with the turn loop (8 appending
# transcripts, 256-token turns, one server at a time, interleaved reps):
#
#   -np 4, 4 sessions            123.3 / 117.2 agg out, prefill 2882 / 2739,
#                                TTFT p50 1.88 / 1.42 s, wall p50 7.3 / 8.3 s
#   -np 8, 8 sessions            143.4 / 140.0 agg out, prefill 3349 / 3271,
#                                TTFT p50 2.00 / 1.90 s, wall p50 12.8 / 13.0 s
#   -np 4, 8 sessions, CRAM=0     62.9 agg out, prefill 1470, TTFT p50 21.2 s,
#                                wall p50 32.2 s   <- every turn re-prefills
#   -np 4, 8 sessions, CRAM=4096 107.8 / 113.7 agg out, prefill 2518 / 2656,
#                                TTFT p50 9.4 / 9.2 s, wall p50 16.2 / 16.1 s
#
# So: NP=8 for an eight-harness fleet (+11-15 % aggregate over NP=4, same
# per-turn TTFT, at +17 GiB of KV); NP=4 for four harnesses or a single stream
# (NP=1 is the single-session latency configuration -- 123.5 t/s for the one
# stream against ~77 each for four batched). Slots beyond the real concurrency
# are wasted memory. -c is the total, so TOTAL_CTX scales with NP.
NP="${NP:-8}"
TOTAL_CTX=$(( CTX * NP ))

# Independent servers. Each is +24.21 GiB of weights (shared through the page
# cache via mmap) + 7 GiB of KV at full context, so prefer NP over INSTANCES.
INSTANCES="${INSTANCES:-1}"

# 8080 is held by Rancher Desktop's Lima port forwarder on this machine (see
# serve.sh) and 8090 is serve.sh's default. 8091 keeps both servers launchable
# at once.
PORT="${PORT:-8091}"
HOST="${HOST:-127.0.0.1}"

# All layers, including the output layer, on the Metal device. `-ngl all` is
# equivalent; 99 is what serve.sh uses and what the t/s above were measured at.
NGL="${NGL:-99}"

# The GGUF card's value. Swept at the fleet shape (8 slots, 8 appending
# sessions, interleaved, card sampling) once the batch below was fixed: -t 12
# measured 130.3 t/s aggregate out against 148.9 for -t 8 -- a loss, not a win,
# so the card's 8 stands. Decode here is Metal-bound (the CPU embeds tokens,
# dispatches and samples), and the Qwen-side finding that 4 beats 8 for prefill
# did not transfer either: single-stream pp512 at d20000 is the same at 4 and 8.
THREADS="${THREADS:-8}"

# mmap: zero-copy into Metal buffers and shared between instances.
LM="${LM:-mmap}"

# Prefill batch. The GGUF card suggests -b 256 -ub 128, and that is measurably
# worse at depth here: llama-server splits a prompt into -b-sized chunks, so the
# card's values cost 85 % of prefill at 27k -- TTFT 24.3-24.7 s against 13.2 s
# for 2048/512 (warm, cache-busted, same prompt, same server). Paired
# llama-bench at d30000 in single invocations:
#
#   -ub 128   pp512  802-939 t/s   tg128  76.9-94.4 t/s
#   -ub 512   pp512 1255-1618      tg128  94.6-95.0
#
# Shallow it is a wash (pp512 ~3700 either way, tg128 125 against 123). Only
# paired runs are comparable on this box -- it carries load 16-18 from other
# workloads, and an unpaired -ub 1024/2048 run measured decode 15-30 % low
# across the board, i.e. contamination rather than a -ub effect. The t/s table
# at the top of this file came from llama-bench, whose own batch defaults are
# 2048/512, so these defaults now match what that table describes.
#
# The *logical* batch is the one knob llama-bench cannot see, because it needs
# several slots in flight: -b caps how many prompt tokens one decode call may
# carry, and a fleet's turn deltas arrive in waves of NP x ~900 tokens. At -b
# 2048 the server splits that wave across several decodes and the slots' prefills
# run mostly in sequence; at -b 4096 two to four slots' chunks land in one decode.
# Turn loop, 8 slots, 8 sessions, card sampling, interleaved (medians of 3 reps,
# plus a second window):
#
#   -b 2048   agg out 131.3 / 121.0 / 120.6 / 145.3 / 111.3
#   -b 4096   agg out 148.9 / 141.6 / 157.2 / 137.6 / 128.7
#   -b 8192   agg out 143.1                   <- no further gain past 4096
#
# pooled over seven interleaved runs each (three windows): median 141.6 against
# 124.2 t/s, prefill 3591 against 3405, turn wall p50 10.3 against 11.3 s, TTFT
# +0.5 s -- 4096 wins six of the seven paired comparisons and one window is
# level, so read it as +12-14 % aggregate, at no RSS cost (56.2 GiB either way)
# and with single-stream pp/tg unchanged (pp512 at d20000 1651.6/1661.9 at 2048
# against 1682.1/1658.5 at 4096; tg identical). -ub 1024 with -b 4096 measured
# 139.6 against 148.9, so the physical batch stays 512.
BATCH="${BATCH:-4096}"
UBATCH="${UBATCH:-512}"

# f16 KV: the card's value, and it is what the t/s above were measured with.
# 28 672 B/token is cheap enough that there is no memory reason to quantize it,
# and a quantized V cache additionally requires -fa on (below) and was not
# measured for this model.
CTK="${CTK:-f16}"
CTV="${CTV:-f16}"
FA="${FA:-on}"

# Metal 4 tensor API on by default. It is worth ~2x on prefill at depth -- pp512
# @d20000 measures 1431-1591 t/s on it against 701-836 with
# GGML_METAL_TENSOR_DISABLE=1 -- and in this revision the one consumer that is
# wrong is the tensor-API flash-attention kernel: `test-backend-ops test -b MTL0
# -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128'` fails 41/558 with ERR = inf at nb >= 64
# (and a model that reaches it decodes word salad from the prompt), while MUL_MAT
# passes with the same path on. scripts/patches/0001-metal-flash-attn-disable-
# tensor-path.patch sends FA to the half8x8 kernels and leaves mul_mm on the
# tensor path, which is why this script no longer exports the variable: the old
# disable was all-or-nothing, so it spent the whole ~2x to avoid a path worth
# ~3.4% (d20000 against d20032, FLINT.md). Rebuild with scripts/build_flint.sh
# after touching that patch. TENSOR=0 restores the old global disable.
TENSOR="${TENSOR:-1}"
if [ "$TENSOR" = 0 ]; then export GGML_METAL_TENSOR_DISABLE=1; fi

# Server prompt cache in host RAM (--cache-ram). Default 0. Note what 0 does
# *not* disable -- a slot still reuses its own KV prefix, so an appending
# transcript prefills only the delta; CRAM only covers a conversation *displaced*
# by another one (and with CRAM=0 the server logs "--cache-idle-slots requires
# --cache-ram, disabling", which is that same limit, not a misconfiguration).
#
# Measured for this model (turn loop, appending transcripts, card sampling,
# interleaved reps, two reps per row):
#
#   sessions/slots  CRAM     agg out t/s     prefill t/s    TTFT p50     RSS
#   8 / 4           0        79.7 / 79.9     1836 / 1883    13.2 / 12.1 s  40.2 GiB
#   8 / 4           4096     130.4 / 142.9   3586 / 3425     5.3 /  5.3 s  47.0 GiB
#   12 / 8          0        73.7 / 76.7     1814 / 1854    14.4 / 12.7 s  54.0 GiB
#   12 / 8          4096     142.1 / 153.4   3489 / 3974     6.2 /  4.0 s  62.3 GiB
#   8 / 8           0        131.3 / 121.0 / 120.6 (three windows)          56.2 GiB
#   8 / 8           4096     109.9 / 109.8   2843 / 3042     1.0 /  0.8 s  59.0 GiB
#
# So CRAM is worth 1.6-2x whenever concurrent conversations exceed NP, and it
# *costs* ~9 % of throughput when they do not: with no conversation to displace,
# the server still saves each idle slot's KV to the cache on every new task,
# which is pure copying. Leave it 0 at NP >= conversations; set it when the
# fleet can outgrow the slots, sized as ~0.9 GiB per 32k-token conversation (a
# full 262 144-token one is 7 GiB). The corner worth knowing: 12 harnesses on 8
# slots with CRAM measured *more* aggregate than 8 on 8 (142-153 against
# 121-131 t/s) because twelve clients keep every slot busy -- what that shape
# costs is first-token latency (TTFT 4-6 s against 0.8-1.3 s).
CRAM="${CRAM:-0}"

# New-swap guard (see serve.sh / scripts/guard.py). Default off: 31 GiB of
# model + context leaves this machine's 128 GB plenty of room.
SWAPCAP="${SWAPCAP:-0}"

# Draft-free speculative decoding. Flint ships no MTP head and no EAGLE/Medusa
# head exists for it, so the only speculation available is the model-free
# n-gram family llama.cpp carries. Measured here (FLINT.md, "Speculative
# decoding"): a large *lossless* win when the output re-emits text that is
# already in context -- +66-72 % on a verbatim-reproduction prompt at both temp 0
# and the card's temp 1, +195-270 % on a markdown continuation at 30k depth --
# and neutral-to-slightly-positive on reasoning-heavy agentic traffic (+1-4 % at
# 27k, card sampling). At 100k it found nothing to copy and measured neutral.
#
# Default ngram-mod, because it refuses to draft shorter than NMIN tokens: that
# is what keeps the reasoning-heavy case neutral. ngram-simple drafts short
# sequences and measured -15 % on temp-0 agentic chat (drafting, never
# accepting) while being equal on copy-heavy text. The one configuration where
# the default is a real cost is *greedy* agentic traffic: at temp 0 the model's
# chat output is terse non-code reasoning, acceptance falls to zero, and
# ngram-mod measured -6 % to -19 % there -- so a greedy client (temperature 0
# per request, as scripts/mtp_ab.py drives its arms) wants SPEC=0.
#
#   SPEC=0 ./scripts/serve_flint.sh                 # off (greedy clients)
#   SPEC=ngram-simple ./scripts/serve_flint.sh      # alternative drafter
#   NMIN=32 ./scripts/serve_flint.sh                # draft more aggressively
# Default: ngram-mod for a single slot, off for several. Measured at -np 4 with
# four appending sessions, interleaved A-B-A-B: 133.95 and 97.70 t/s on against
# 125.50 and 129.18 t/s off -- no benefit, plausibly a small cost. At -np 1 the
# same switch was worth +68 % on verbatim reproduction and neutral-to-positive on
# agentic chat at the card's sampling.
#
# Re-measured at the fleet shape (8 slots, 8 appending sessions, card sampling,
# two interleaved reps each) including the *per-sequence* drafters, which exist
# because the simple/mod tables are shared across sequences in this revision
# ("shared across all sequences", common/speculative.cpp):
#
#   SPEC=0               113.8 / 123.3 t/s aggregate out
#   SPEC=ngram-map-k     107.7 / 119.0
#   SPEC=ngram-map-k4v   115.1 / 112.8
#   SPEC=ngram-mod       109.7 /  99.5
#   SPEC=ngram-cache      81.9 /  84.5   <- and 15-21 s turns
#
# Per-sequence tables do not rescue it: nothing beats off once eight sessions
# share one decode step, so the default stays 0 for NP > 1.
SPEC="${SPEC:-$([ "${NP:-8}" -gt 1 ] && echo 0 || echo ngram-mod)}"
NMATCH="${NMATCH:-24}"   # n-gram lookup length (--spec-ngram-mod-n-match)
NMIN="${NMIN:-48}"       # minimum draft length; shorter drafts are dropped
NMAX="${NMAX:-64}"       # maximum draft length

# Sampling. The Flint GGUF embeds general.sampling.{temp,top_p,top_k} = 1.0 /
# 0.95 / 20 and llama-server applies them to requests that omit them, so these
# are here to pin the card's evaluation setting explicitly and to set min-p,
# which llama.cpp otherwise defaults to 0.05.
SAMPLE_ARGS=(--temp 1 --top-p 0.95 --top-k 20 --min-p 0)

SPEC_ARGS=()
case "$SPEC" in
    0|off|none) ;;
    ngram-mod)
        SPEC_ARGS=(--spec-type ngram-mod
                   --spec-ngram-mod-n-match "$NMATCH"
                   --spec-ngram-mod-n-min "$NMIN"
                   --spec-ngram-mod-n-max "$NMAX") ;;
    *)  SPEC_ARGS=(--spec-type "$SPEC") ;;
esac

mkdir -p "$ROOT/logs"

if [ ! -x "$SERVER" ]; then
    echo "serve: $SERVER is missing. Build it: ./scripts/build_flint.sh" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "serve: $MODEL is missing. Fetch it (~26.0 GB):" >&2
    echo "serve:   hf download RecursiveSI/flint-int4-gguf flint-int4-q4_1.gguf --local-dir models" >&2
    exit 1
fi
if ! "$BENCH" --list-devices 2>/dev/null | grep -qi 'MTL'; then
    echo "serve: no Metal device reported by llama-bench --list-devices." >&2
    echo "serve: rebuild with ./scripts/build_flint.sh (GGML_METAL=ON)." >&2
    exit 1
fi

# KV size per slot, for the memory line. 14 full-attention layers x 4 kv heads
# x 128 dims x 2 tensors, at the dtype's bytes/element; the 35 sliding-window
# layers are a fixed 513-token cache instead (35 x 513 x 4 x 128 x 2).
kv_bytes_per_tok() {
    case "$1" in
        f16|bf16) echo 2 ;;
        q8_0)     echo 1 ;;
        q4_0)     echo 0.5 ;;
        *)        echo 2 ;;
    esac
}
BYTES=$(python3 -c "print(2*14*4*128*($(kv_bytes_per_tok "$CTK")+$(kv_bytes_per_tok "$CTV"))/2)")
KV_MIB=$(python3 -c "print(int($CTX*$BYTES/1048576))")
SWA_MIB=$(python3 -c "print(int(35*513*4*128*2*($(kv_bytes_per_tok "$CTK")+$(kv_bytes_per_tok "$CTV"))/2/1048576))")

echo "serve: model=$(basename "$MODEL") alias=$ALIAS ctx=$CTX/slot slots=$NP instances=$INSTANCES threads=$THREADS"
echo "serve: -ngl $NGL -lm $LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA metal-tensor=$TENSOR"
echo "serve: speculative=${SPEC}${NMATCH:+ n_match=$NMATCH n_min=$NMIN n_max=$NMAX}"
echo "serve: kv≈${KV_MIB} MiB/slot + swa ${SWA_MIB} MiB/slot -> weights 24.21 GiB shared + ${INSTANCES} server(s) x ${NP} slot(s) x (${KV_MIB} MiB kv)"
echo "serve: kv=$CTK/$CTV  prompt-cache=${CRAM}MiB  swap-cap=${SWAPCAP}MiB  http://$HOST:$PORT..$((PORT + INSTANCES - 1))  logs: $ROOT/logs/serve.flint.$PORT.log"

for i in $(seq 0 $((INSTANCES - 1))); do
    port=$((PORT + i))
    python3 "$ROOT/scripts/guard.py" --swap-cap-mib "$SWAPCAP" \
        --log "$ROOT/logs/serve.flint.$port.log" -- \
        "$SERVER" \
        -m "$MODEL" --alias "$ALIAS" -ngl "$NGL" -c "$TOTAL_CTX" -np "$NP" -t "$THREADS" \
        -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
        -ctk "$CTK" -ctv "$CTV" -fa "$FA" --cache-ram "$CRAM" \
        ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
        --jinja "${SAMPLE_ARGS[@]}" \
        --host "$HOST" --port "$port" "$@" &
done

# Ctrl-C must reach every server: kill the whole group, which lets each guard
# forward the signal to its own child. Disarm the trap first: a *second* signal
# (a supervisor's cleanup, a second Ctrl-C) used to run `kill 0` again, and a
# double interrupt is what trips llama.cpp's shutdown-path assertion in
# ggml_metal_device_free -- the server then aborts while still holding the port,
# which the next launch sees as "address already in use".
trap 'trap - INT TERM; kill 0' INT TERM
wait
