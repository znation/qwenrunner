#!/bin/bash
# Sequential download of smaller quants for the quant Pareto sweep.
set -u
cd /home/zach/qwenrunner
BASE=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main
for f in Qwen3.8-27B-UD-IQ3_S.gguf Qwen3.8-27B-UD-Q2_K_XL.gguf Qwen3.8-27B-UD-Q3_K_XL.gguf; do
  echo "=== $(date -Is) start $f"
  curl -fL --retry 8 --retry-delay 5 -C - -o "models/$f" "$BASE/$f" || echo "=== FAILED $f rc=$?"
  ls -la "models/$f" 2>/dev/null
done
echo "=== $(date -Is) ALL DONE"
