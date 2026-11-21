#!/usr/bin/env bash
# bench_inkling.sh — server-based ubatch × moe-cache sweep for the Inkling path.
# Each arm: start llama-server with the arm's flags, warm up, then measure one
# ~512-token prefill + 128 generated tokens via the native /completion endpoint
# (which reports timings), sampling system VRAM/GTT while it runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
PORT=8082
OUT="$ROOT/results/inkling_ub_cache.jsonl"
mkdir -p "$ROOT/results" "$ROOT/logs"

PROMPT="$(head -c 4000 "$ROOT/data/ppl_corpus.txt" | tr '\n' ' ')"

vram() { awk '{print int($1/1024)}' /sys/class/drm/card0/device/mem_info_vram_used; }
gtt()  { awk '{print int($1/1024)}' /sys/class/drm/card0/device/mem_info_gtt_used; }

arm() {
  local label="$1" ub="$2" cache="$3"; shift 3
  local peak_v=0 peak_g=0
  "$ROOT/build/bin/llama-server" -m "$ROOT/models/Inkling-2-Mini-IQ4_NL.gguf" -a bench \
    -c 32768 -t 8 -np 1 -dev Vulkan0 -sm layer -ngl 99 -cmoe \
    --moe-cache-mib "$cache" --fit off -lm mmap -b 2048 -ub "$ub" \
    -ctk q4_0 -ctv q4_0 -fa on --jinja --host 127.0.0.1 --port $PORT "$@" \
    > "$ROOT/logs/bench_inkling_$label.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "$label: server failed"; tail -5 "$ROOT/logs/bench_inkling_$label.log"; kill $pid 2>/dev/null; return 1; }
  # warm-up: small request to fill the expert cache and page in mmap'd weights
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"2 + 2 =","n_predict":16,"temperature":0.0}' >/dev/null
  # measured run: ~512-token prefill + 128 gen, VRAM sampled in parallel
  local samp="$ROOT/logs/bench_inkling_$label.samples"
  : > "$samp"
  ( while kill -0 $pid 2>/dev/null; do echo "$(vram) $(gtt)" >> "$samp"; sleep 0.5; done ) &
  local sampler=$!
  RESP="$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': 128, 'temperature': 0.0}))" "$PROMPT")")"
  kill $sampler 2>/dev/null
  sleep 1
  peak_v=$(awk 'BEGIN{m=0} {if ($1+0 > m) m=$1+0} END{print m}' "$samp")
  peak_g=$(awk 'BEGIN{m=0} {if ($2+0 > m) m=$2+0} END{print m}' "$samp")
  echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
t = r['timings']
label, ub, cache, pv, pg = sys.argv[1:6]
row = {'label': label, 'ub': int(ub), 'cache_mib': int(cache),
       'prompt_ts': round(t['prompt_n'] / (t['prompt_ms'] / 1000), 2),
       'gen_ts': round(t['predicted_n'] / (t['predicted_ms'] / 1000), 2),
       'prompt_n': t['prompt_n'], 'peak_vram_mib': int(pv), 'peak_gtt_mib': int(pg)}
print(json.dumps(row))
open('$OUT', 'a').write(json.dumps(row) + '\n')
" "$label" "$ub" "$cache" "$peak_v" "$peak_g"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
  sleep 5   # let the DRM client drain before the next arm samples a baseline
}

for arm_args in "ub128_c0 128 0" "ub64_c0 64 0" "ub256_c0 256 0" \
                "ub128_c1024_lmnone 128 1024 -lm none" \
                "ub64_c1024_lmnone 64 1024 -lm none" \
                "ub128_c1024_share1_lmnone 128 1024 -lm none --moe-cache-cpu-share 1.0"; do
  set -- $arm_args
  arm "$1" "$2" "$3" || true
done
echo "=== results ==="
cat "$OUT"
