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
# There is no draft head in this GGUF (the repo is one 26.0 GB file): no MTP, no
# speculative decode, so the t/s above are plain decode, and none of serve.sh's
# MTP machinery applies here.
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

# The documented configuration is 262 144 tokens and this machine can afford it
# at the default f16 KV (7.0 GiB), so the context is not traded down the way it
# is on the Qwen side. Reduce it only to buy memory back or to go faster at
# depth, not to fit: 24.21 + 7.0 GiB is small here.
CTX="${CTX:-262144}"

# Slots *within* the server (-np), each getting CTX of KV. 1, because that is
# the GGUF card's default and because a single slot already saturates the Metal
# device (measured aggregate on the Qwen side is flat once one stream is GPU
# bound). Each extra slot costs 7 GiB of KV at full context plus Metal compute
# buffers; -c is the total, so TOTAL_CTX scales with NP.
NP="${NP:-1}"
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

# The card's operating point. Measured here only at this pair.
BATCH="${BATCH:-256}"
UBATCH="${UBATCH:-128}"

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

# Sampling. The Flint GGUF embeds general.sampling.{temp,top_p,top_k} = 1.0 /
# 0.95 / 20 and llama-server applies them to requests that omit them, so these
# are here to pin the card's evaluation setting explicitly and to set min-p,
# which llama.cpp otherwise defaults to 0.05.
SAMPLE_ARGS=(--temp 1 --top-p 0.95 --top-k 20 --min-p 0)

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
echo "serve: speculative=none (no draft head in this GGUF)"
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
        --jinja "${SAMPLE_ARGS[@]}" \
        --host "$HOST" --port "$port" "$@" &
done

# Ctrl-C must reach every server: kill the whole group, which lets each guard
# forward the signal to its own child.
trap 'kill 0' INT TERM
wait
