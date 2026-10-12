#!/usr/bin/env bash
# quant_compare.sh — what does a bigger quant buy, and what does it cost?
#
# On the Intel host this compared *offload ceilings*: a smaller quant has
# smaller layers, so more of them fit under the 5000 MiB display cap and
# throughput went up. That whole framing is gone — every quant fits here, so
# `-ngl 99` for all of them and the question becomes purely:
#
#   1. Throughput at depth 0 and at depth 131072. Decode is bandwidth-bound, so
#      a bigger quant streams more bytes per token and should cost roughly
#      linearly in model size. This is the number that decides Q4 vs Q8.
#   2. Perplexity on data/eval_corpus.txt. Deterministic, so it is the one part
#      of the comparison that host load cannot corrupt — which matters more
#      here than it did there, because the shared GPU is the contended
#      resource and no counter can tell a slow arm from a busy one.
#
# Usage: ./scripts/quant_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUANTS="UD-Q4_K_S"
MODELS="models/Qwen3.8-27B-UD-Q4_K_S.gguf"
# Add more once scripts/dl_smaller.sh has fetched them, e.g.
#   QUANTS="UD-Q2_K_XL UD-Q3_K_XL UD-Q4_K_S UD-Q5_K_M"
CTX="${CTX:-131072}"

echo "=== quant compare start $(date +%T)  ctx=$CTX"
echo "--- throughput, depth 0 $(date +%T)"
./scripts/bench.py --stage quant --models "$MODELS" --ngl 99 --depth 0 \
    --reps 2 --out results/qc_d0.jsonl

echo "--- throughput at $CTX depth $(date +%T)"
./scripts/bench.py --stage quant --models "$MODELS" --ngl 99 --depth "$CTX" \
    --reps 1 --out results/qc_deep.jsonl

echo "--- perplexity, data/eval_corpus.txt $(date +%T)"
for q in $QUANTS; do
    python3 scripts/guard.py --label "ppl-$q" \
        --log "logs/ppl-$q.log" -- \
        build/bin/llama-perplexity -m "models/Qwen3.8-27B-$q.gguf" \
        -f data/eval_corpus.txt -c 512 --chunks 6 -ngl 99 -fa on -t 6
done

echo "=== quant compare done $(date +%T)"
