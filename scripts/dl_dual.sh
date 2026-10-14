#!/bin/bash
# Download the default dual-GPU pair (unsloth GGUF, imatrix):
#   Qwen3.5-9B-Q4_K_M.gguf  — eGPU (GTX 1070) main agent, fully resident
#   Qwen3.5-4B-Q4_K_M.gguf  — display GPU (Vega 10) secondary, under the cap
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for spec in \
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf" \
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"; do
    f="models/$(basename "$spec")"
    echo "=== $(date -Is) start $f"
    curl -fL --retry 8 --retry-delay 5 -C - -o "$f" "$spec" || { echo "=== FAILED $f rc=$?"; exit 1; }
    ls -la "$f"
done
echo "=== $(date -Is) ALL DONE"
