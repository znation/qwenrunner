#!/bin/bash
# Download the MTP draft head plus the smaller quants for the Pareto sweep.
# The draft head lives under MTP/ in the upstream repo, not at the top level.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
FILES=(
  "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
  "Qwen3.8-27B-UD-IQ3_S.gguf"
  "Qwen3.8-27B-UD-Q2_K_XL.gguf"
  "Qwen3.8-27B-UD-Q3_K_XL.gguf"
  "Qwen3.8-27B-UD-Q5_K_M.gguf"
)
for f in "${FILES[@]}"; do
  out="models/$(basename "$f")"
  echo "=== $(date -Is) start $f"
  curl -fL --retry 8 --retry-delay 5 -C - -o "$out" "$BASE/$f" || echo "=== FAILED $f rc=$?"
  ls -la "$out" 2>/dev/null
done
echo "=== $(date -Is) ALL DONE"
