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
# Qwen's 128k. Model + full context ≈ 31 GiB.
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
#   MODEL CTX NP INSTANCES PORT HOST THREADS NGL LM BATCH UBATCH CTK CTV FA CRAM ALIAS SWAPCAP
#
#   ./scripts/serve_flint.sh                  # one full-context slot on 8091
#   NP=2 ./scripts/serve_flint.sh             # +7 GiB KV, two sessions
#   CTX=131072 ./scripts/serve_flint.sh       # -3.5 GiB KV, faster at depth
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

# Per-slot context. The card documents 262 144 tokens, but with four slots that
# is 28 GiB of f16 KV (7.0 GiB each), and real agent harnesses rarely carry more
# than ~100k tokens of transcript. 131 072 halves it to 3.5 GiB/slot, 14 GiB for
# four slots, on top of 24.21 GiB of weights. Raise it if your agents really run
# to 200k+ tokens -- each +131 072 costs another 3.5 GiB *per slot* -- and note a
# session that outgrows its slot is truncated, not given a neighbour's.
CTX="${CTX:-131072}"

# Slots *within* the server (-np), each getting CTX of KV. Default 4, one per
# concurrent harness, because that is the shape this repo serves: N harnesses or
# subagents pointed at one endpoint.
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
#   4 sessions, appending transcripts, quiet box:
#                            -np 4  171.6 agg, TTFT p50 1.19 s, turn wall 5.5 s
#                            rec     57.2 agg, TTFT p50 14.8 s  (rec serves one
#                            generation at a time, so it queues)
#
# NP=1 is the single-session latency configuration (123.5 t/s for the one stream
# against ~77 each for four batched); set NP=1 when there is only one agent and
# it matters. -c is the total, so TOTAL_CTX scales with NP.
NP="${NP:-4}"
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

# The GGUF card's value. Not swept on this machine: decode is Metal-bound here
# (the CPU only embeds tokens, dispatches and samples), so the Qwen-side
# finding that 4 beats 8 for prefill may or may not transfer -- if you sweep
# threads, sweep prefill, not generation.
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
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"

# f16 KV: the card's value, and it is what the t/s above were measured with.
# 28 672 B/token is cheap enough that there is no memory reason to quantize it,
# and a quantized V cache additionally requires -fa on (below) and was not
# measured for this model.
CTK="${CTK:-f16}"
CTV="${CTV:-f16}"
FA="${FA:-on}"

# Server prompt cache in host RAM (--cache-ram). Default 0, the card's value.
# The card's reason is host memory: this is a ceiling that costs RAM whether or
# not it is used. Note what 0 does *not* disable -- a slot still reuses its own
# KV prefix, so an appending transcript prefills only the delta; CRAM only
# covers a conversation *displaced* by another one (and with CRAM=0 the server
# logs "--cache-idle-slots requires --cache-ram, disabling", which is that same
# limit, not a misconfiguration). At f16 flint KV a full 262 144-token
# conversation is 7 GiB, so size CRAM as 7 GiB x (conversations kept warm), e.g.
# CRAM=7168 for one. Untested for this model (it is measured for Qwen in
# RESULTS.md, "Prefill cache").
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
# Default: ngram-mod for a single slot, off for several. The drafter's n-gram
# table is shared across sequences in this revision ("shared across all
# sequences", common/speculative.cpp), so its benefit is a single-stream,
# single-session one and it does not survive batching. Measured at -np 4 with
# four appending sessions, interleaved A-B-A-B: 133.95 and 97.70 t/s on against
# 125.50 and 129.18 t/s off -- no benefit, plausibly a small cost. At -np 1 the
# same switch was worth +68 % on verbatim reproduction and neutral-to-positive on
# agentic chat at the card's sampling.
SPEC="${SPEC:-$([ "${NP:-4}" -gt 1 ] && echo 0 || echo ngram-mod)}"
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
echo "serve: -ngl $NGL -lm $LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
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
# forward the signal to its own child.
trap 'kill 0' INT TERM
wait
