#!/usr/bin/env bash
# quant_compare.sh — is a smaller quant actually faster on this machine?
#
# Three questions, in order of how much they decide:
#
#   1. How many layers fit?  A smaller quant has smaller layers, so it can
#      offload more before hitting the 5000 MiB cap. Measured as a per-quant
#      `-ngl` ladder, which yields the client-VRAM-per-layer slope directly.
#   2. Throughput at a common `-ngl`, shallow and at depth 4096 — so the quants
#      are compared at the same offload rather than at their own ceilings.
#   3. Perplexity on data/eval_corpus.txt. This is deterministic, so it is the
#      one part of the comparison that host load cannot corrupt.
#
# Usage: ./scripts/quant_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu"

QUANTS="IQ3_S Q2_K_XL Q3_K_XL Q4_K_M"
MODELS="models/Qwen3.8-27B-UD-IQ3_S.gguf,models/Qwen3.8-27B-UD-Q2_K_XL.gguf,models/Qwen3.8-27B-UD-Q3_K_XL.gguf,models/Qwen3.8-27B-UD-Q4_K_M.gguf"

echo "=== quant compare start $(date +%T)"
echo "--- ladder 8,10,12,14 at depth 0 (client VRAM per layer) $(date +%T)"
for q in $QUANTS; do
    ./scripts/bench.py --stage ngl --model "models/Qwen3.8-27B-UD-$q.gguf" \
        --ngl-ladder 8,10,12,14 --depth 0 --reps 1 --lm dio \
        --out results/qc_ladder.jsonl
done

echo "--- common -ngl 10 at depth 4096 $(date +%T)"
./scripts/bench.py --stage quant --models "$MODELS" --ngl 10 --depth 4096 \
    --reps 1 --lm dio --out results/qc_d4096.jsonl

echo "--- perplexity, data/eval_corpus.txt $(date +%T)"
for q in $QUANTS; do
    python3 scripts/guard.py --cap-mib 5000 --label "ppl-$q" \
        --log "logs/ppl-$q.log" -- \
        build/bin/llama-perplexity -m "models/Qwen3.8-27B-UD-$q.gguf" \
        -f data/eval_corpus.txt -c 512 --chunks 6 -ngl 10 -lm dio -t 8
done

echo "=== quant compare done $(date +%T)"
