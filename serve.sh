#!/usr/bin/env bash
# serve.sh — launch llama-server for Qwen-3.8-27B on Apple Silicon (Metal).
#
# This is the M5 Max / 128 GB unified-memory entry point. The Intel build of
# this script split one model across a GTX 1070 and a Vega 10 under a 5000 MiB
# display cap, derived `-ngl` from the context, and refused to start without
# the eGPU. All of that is gone:
#
#   * There is no second device to split to. Metal sees one GPU with the whole
#     machine's memory behind it, so `-ngl 99` puts *every* layer (including
#     the draft head) on the GPU and no layer budget needs deriving.
#   * No 5000 MiB cap and no watchdog kill — but the *swap* guard is kept: a
#     run that spills to swap has worthless latency numbers, so `SWAPCAP` can
#     still SIGKILL on new swap (default off, since 128 GB holds this model
#     many times over).
#   * `-lm dio` is gone. dio is Linux O_DIRECT; the default mmap is both the
#     fastest path here and the reason N instances are cheap: every process
#     maps the same weight pages, so only the KV cache and compute buffers are
#     per-instance.
#
# INSTANCES launches that many independent llama-server processes on
# consecutive ports, each with its own full CTX of KV. Each is a complete
# 128k-context server; weights are shared through the page cache.
#
# Overridable via env (anything unset is derived):
#   MODEL CTX NP INSTANCES PORT THREADS HOST NGL LM BATCH UBATCH CTK CTV FA
#   MTP MD NGLD NMAX CRAM SLOT_SAVE SWAPCAP
#
#   ./serve.sh                      # 4 agent slots, 64 GiB prompt cache, NMAX=6
#   NP=1 CRAM=8192 ./serve.sh       # single session, minimal footprint
#   INSTANCES=4 ./serve.sh          # four servers, ports 8080-8083
#   NMAX=6 ./serve.sh               # deeper speculation for code/agent traffic
#   CRAM=65536 NP=8 ./serve.sh      # bigger prefill cache, more agent sessions
#   MTP=0 ./serve.sh                # no speculative decoding
#   CTK=q8_0 CTV=q8_0 ./serve.sh    # halve the KV cache per slot
# Extra args are appended verbatim to every llama-server command line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$ROOT/build/bin/llama-server"

MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-UD-Q4_K_S.gguf}"

# 128k is the working context the user asked for, and on this machine it is
# nearly free: only 16 of the 64 layers are full attention (the other 48 are
# linear-attention / SSM and carry a constant ~10 MiB state, not a growing
# cache), so f16 KV at 128k is ~8 GiB per instance, not the ~40 GiB a
# same-size all-attention model would need. Per token: 16 layers x 4 kv heads
# x 256 key/value dims x 2 (K and V) x 2 B = 64 KiB.
CTX="${CTX:-131072}"

# Independent servers. Memory per additional instance is its KV cache plus
# ~2-3 GiB of Metal compute buffers: the 15.4 GB of weights are mapped once and
# shared by every process. Budget = weights + INSTANCES x (KV + compute).
INSTANCES="${INSTANCES:-1}"

# Parallel slots *within* one server (-np). This is the cheaper way to serve N
# concurrent prompts: one process, one weight set, one set of compute buffers,
# one port. Measured at 128k per slot, -np 6 fits in 70 GiB where 6 *processes*
# die of a Metal out-of-memory (they sum to >112 GiB of per-process
# allocations) -- see RESULTS.md, "Parallel prompts". -c is the total, so it
# scales with NP to keep every slot at full CTX.
#
# Default 4: one slot per concurrent agent session, which is what this repo is
# for. Each slot reserves its full KV up front -- 8 GiB at 128k f16, so 32 GiB
# for four whether or not anyone is connected. Use NP=1 if you only ever run one
# session and want the RAM back.
NP="${NP:-4}"
TOTAL_CTX=$(( CTX * NP ))
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

# All layers, plus the output layer and the MTP head, on the Metal device.
NGL="${NGL:-99}"

# The CPU only touches the token embedding, the SSM/conv step of the
# linear-attention layers and sampling. Measured on this machine: 4 is best for
# prefill (758 t/s vs 716 at 6, 529 at 18), and generation is flat from 4 to 18
# because the Metal device does that work. Anything in 4..12 is equivalent; the
# 12 performance cores cost more in contention than they return.
THREADS="${THREADS:-4}"

# mmap is the default and the right answer: zero-copy into Metal buffers and
# shared between instances. `-lm none` costs a full 15.4 GB read per process.
LM="${LM:-mmap}"
# Measured at 98.6k depth with MTP on (the operating point this serves):
# -b 1024 -ub 256 generates 37.45 t/s against 28.52 for -b 2048 -ub 512, a
# +31 % win, with prefill level (349.0 vs 340.0). The same pair wins both axes
# at 64k depth. Only *shallow* prefill prefers 2048/512 (709.6 vs 633.3), and a
# server holding 128k contexts spends its time generating, not prefilling.
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"

# q8_0 KV, chosen for *capacity* rather than raw speed. It halves both the slot
# KV (4 GiB per 128k slot instead of 8) and every prompt-cache entry, which is
# what lets this configuration cover every simultaneous conversation at once --
# measured server footprint is 35.7 GiB against 50.3 for f16, and a cached 128k
# conversation costs 4 GiB against 8. The price is measured and small: -10 %
# generation at 128k depth (12.89 vs 14.34 t/s, no MTP) and -4 % prefill.
# A quantized V cache REQUIRES flash attention; -fa on below satisfies that.
# CTK=f16 CTV=f16 restores the faster, hungrier configuration.
CTK="${CTK:-q8_0}"
CTV="${CTV:-q8_0}"
FA="${FA:-on}"

# MTP speculation: the biggest single lever on the Intel host (+97.8 %) and it
# is cheaper here, because the draft head is offloaded to the GPU too (-ngld
# 99) instead of running on the CPU. Default draft file is the sidecar when it
# is present; otherwise the main GGUF's own blk.64.nextn.* tensors are used by
# `--spec-type draft-mtp` with no -md.
MTP="${MTP:-1}"
MD="${MD:-$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf}"
NGLD="${NGLD:-99}"
# Depth of speculation. 6, because this repo serves agent/code traffic and that
# is where deeper drafting pays (+40 % over the upstream default of 3 on
# structured text). MTP's gain is a property of the *text*, not the machine:
# measured at 128k context it is +180 % on patterned text (markdown/code/JSON,
# 99 % draft acceptance) and -11 % on free-form prose (14 % acceptance), and the
# depth that maximises one hurts the other. Prose traffic wants NMAX=3 or
# MTP=0. See RESULTS.md, "MTP's gain is a property of the text".
NMAX="${NMAX:-6}"

SWAPCAP="${SWAPCAP:-0}"

# Disk-backed tier for conversations beyond CRAM. With this set, a slot's KV
# state can be written to and read from a file, so a conversation evicted from
# both the slots and the RAM cache comes back for the cost of reading its state
# (seconds) instead of re-prefilling its transcript (a 128k prefill is ~4.5 min
# at ~490 t/s). The server is driven by its own API, not by a flag:
#
#   curl -X POST "http://127.0.0.1:8080/slots/0?action=save" \
#        -H 'Content-Type: application/json' -d '{"filename":"conv-a.bin"}'
#   curl -X POST "http://127.0.0.1:8080/slots/0?action=restore" \
#        -H 'Content-Type: application/json' -d '{"filename":"conv-a.bin"}' 
#
# Save the slot that currently holds the conversation, restore it before the
# request that would otherwise re-prefill. Files are written where SLOT_SAVE
# points, and a 128k q8_0 state is ~4 GiB each -- size the disk accordingly.
#
# NOT ENABLED BY DEFAULT, and the round trip was measured to NOT pay off yet
# (RESULTS.md, "Disk-backed slot states"): save and restore both work and are
# fast (1.11 GiB written in 346 ms, restored in 112 ms for a 30k-token
# conversation), but after a restore the slot reports n_prompt_tokens = None,
# so the next matching request still prefilled the whole 29 772 tokens (63.4 s)
# instead of reusing the state. The KV buffers come back; the slot's prompt
# bookkeeping does not. Until that is resolved this is plumbing, not a tier --
# treat SLOT_SAVE as untested.
SLOT_SAVE="${SLOT_SAVE:-}"

# Server prompt cache in host RAM (--cache-ram). This is the *prefill* cache,
# and for agentic use it is the most important knob here.
#
# Three layers exist: (1) a slot reuses its own KV prefix automatically, so a
# turn that appends to a transcript prefills only the delta -- measured: a
# 19-token delta cost 19 tokens and 0.41 s instead of a 60 s re-prefill of
# 29 772; (2) this RAM cache holds *evicted* slot states so a conversation can
# come back after another one displaced it -- measured A B A at 0.58 s vs 57 s
# when it cannot; (3) --slot-save-path persists states to disk (not used here).
#
# The upstream default is 8192 MiB, which at f16 (64 KiB/token) is ONE 128k
# conversation: with two, the second evicts the first and every turn re-prefills
# the whole transcript -- silently, because a cache miss looks exactly like a
# first turn. Size it as (conversations to keep warm) x (context KV), and note
# this is a *ceiling* that the rest of the budget has to absorb:
#
#   server footprint (measured, no traffic) = idle-others 39.3 + this server
#     NP=4 q8_0: 35.7 GiB   + CRAM 32 GiB  = ~107 GiB total   <- the default
#     NP=4 f16 : 50.3 GiB   + CRAM 64 GiB  = ~153 GiB         <- does not fit
#
# Default 24576 = SIX 128k conversations: the four live slots plus two
# displaced, which is the coverage this configuration is sized for. A q8_0
# 128k conversation costs ~4.3 GiB (64 KiB/token halved, plus the 1-in-16
# scale block), so six entries need ~26 GiB; entries are cut to the
# conversation's actual length, so shorter sessions fit more than six.
#
# Worst case with the shipped defaults: 39.3 GiB idle-others + 39.1 GiB server
# + 24 GiB cache = ~102 GiB, leaving ~25 GiB of headroom on a 128 GB machine.
# Raise it only against that ceiling; CRAM=0 disables the cache.
CRAM="${CRAM:-24576}"

mkdir -p "$ROOT/logs"

if [ ! -x "$SERVER" ]; then
    echo "serve: $SERVER is missing. Build it: ./scripts/build.sh" >&2
    exit 1
fi
if ! "$ROOT/build/bin/llama-bench" --list-devices 2>/dev/null | grep -qi 'MTL'; then
    echo "serve: no Metal device reported by llama-bench --list-devices." >&2
    echo "serve: rebuild with ./scripts/build.sh (GGML_METAL=ON)." >&2
    exit 1
fi

DISK_ARGS=()
if [ -n "$SLOT_SAVE" ]; then
    mkdir -p "$SLOT_SAVE"
    DISK_ARGS=(--slot-save-path "$SLOT_SAVE")
fi

MTP_ARGS=()
if [ "$MTP" = 1 ]; then
    if [ -f "$MD" ]; then
        MTP_ARGS=(--spec-type draft-mtp -md "$MD" -ngld "$NGLD" --spec-draft-n-max "$NMAX")
    else
        echo "serve: $MD not found; using the main GGUF's embedded MTP tensors" >&2
        MTP_ARGS=(--spec-type draft-mtp -ngld "$NGLD" --spec-draft-n-max "$NMAX")
    fi
fi

# KV cache size per instance, for the memory line. 16 full-attention layers x
# 4 kv heads x 256 dims x 2 tensors, at the dtype's bytes/element per 128k.
kv_bytes_per_tok() {
    case "$1" in
        f16|bf16) echo 2 ;;
        q8_0)     echo 1 ;;
        q4_0)     echo 0.5 ;;
        *)        echo 2 ;;
    esac
}
BYTES=$(python3 -c "print(2*16*4*256*($(kv_bytes_per_tok "$CTK")+$(kv_bytes_per_tok "$CTV"))/2)")
KV_MIB=$(python3 -c "print(int($CTX*$BYTES/1048576))")

echo "serve: model=$(basename "$MODEL") ctx=$CTX/slot slots=$NP instances=$INSTANCES threads=$THREADS"
echo "serve: -ngl $NGL -lm $LM -b $BATCH -ub $UBATCH -ctk $CTK -ctv $CTV -fa $FA"
if [ "$MTP" = 1 ]; then
    echo "serve: mtp=on (draft on GPU, -ngld $NGLD) $([ -f "$MD" ] && basename "$MD" || echo 'embedded nextn tensors')"
else
    echo "serve: mtp=off"
fi
echo "serve: kv≈${KV_MIB} MiB/slot -> weights 15.4 GB shared + ${INSTANCES} server(s) x ${NP} slot(s) x (${KV_MIB} MiB kv)"
echo "serve: kv=$CTK/$CTV  prompt-cache=${CRAM}MiB${SLOT_SAVE:+  disk-tier=$SLOT_SAVE}  swap-cap=${SWAPCAP}MiB  http://$HOST:$PORT..$((PORT + INSTANCES - 1))  logs: $ROOT/logs/serve.$PORT.log"

for i in $(seq 0 $((INSTANCES - 1))); do
    port=$((PORT + i))
    python3 "$ROOT/scripts/guard.py" --swap-cap-mib "$SWAPCAP" \
        --log "$ROOT/logs/serve.$port.log" -- \
        "$SERVER" \
        -m "$MODEL" -ngl "$NGL" -c "$TOTAL_CTX" -np "$NP" -t "$THREADS" \
        -lm "$LM" -b "$BATCH" -ub "$UBATCH" \
        -ctk "$CTK" -ctv "$CTV" -fa "$FA" --cache-ram "$CRAM" \
        ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} \
        ${DISK_ARGS[@]+"${DISK_ARGS[@]}"} \
        --host "$HOST" --port "$port" "$@" &
done

# Ctrl-C must reach every server: kill the whole group, which lets each guard
# forward the signal to its own child.
trap 'kill 0' INT TERM
wait
