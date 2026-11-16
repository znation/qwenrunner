#!/usr/bin/env bash
# bench_ngl_spill.sh — -ngl ladder for the disk-spill path: how much GPU
# footprint and generation speed each -ngl costs when the non-expert weights
# are pulled back to the CPU. The spill model's fixed GPU footprint (~4.1 GiB
# at -ngl 99, 8k ctx) is too large for a display-safe cap on a typical
# desktop; -ngl is the lever that shrinks it.
#
# Per arm: load (CTX 8192, -cmoe, -ub 256), settle, idle system VRAM, one
# 5240-token chat request with 128 max_tokens (prefill + gen t/s from the
# response timings), peak system VRAM sampled at 1 s. Guard cap 7000 is a
# measurement ceiling, not the shipped cap. Appends to results/ngl_spill.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/iq3s/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"
PORT="${PORT:-8092}"
CTX="${CTX:-8192}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/data/long_prompt.txt}"
GEN_N="${GEN_N:-128}"
NGLS="${NGLS:-99 80 60 40}"

for NGL in $NGLS; do
  LOG="$ROOT/logs/ngl_spill_$NGL.log"
  echo "=== -ngl $NGL"
  "$ROOT/build/bin/llama-server" -m "$MODEL" -a "Swift-ngl$NGL" \
      -c "$CTX" -t 8 -np 1 -dev Vulkan0 -ngl "$NGL" -cmoe --fit off \
      -lm mmap -b 2048 -ub 256 -fa on -ctk q4_0 -ctv q4_0 \
      --no-mmproj-offload --host 127.0.0.1 --port "$PORT" \
      > "$LOG" 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 $SPID 2>/dev/null || { echo "ngl $NGL died at load"; tail -3 "$LOG"; exit 1; }
    sleep 5
  done
  sleep 12   # let the previous arm's DRM client release before the idle reading
  IDLE=$(cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null || echo 0)
  ( while kill -0 $SPID 2>/dev/null; do
      cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null || echo 0
      sleep 1
    done ) > "/tmp/ngl_vram_$NGL.txt" &
  SMPID=$!
  MEAS=$(python3 - "$PORT" "$PROMPT_FILE" "$GEN_N" <<'PY'
import json, sys, urllib.request
port, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = open(path).read()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": prompt}],
                     "max_tokens": n, "temperature": 0.0}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.load(r)
t = d.get("timings") or {}
print(json.dumps({
    "prompt_n": t.get("prompt_n"),
    "prompt_t_ps": round(t.get("prompt_n")/(t.get("prompt_ms", 1)/1000), 2),
    "predicted_n": t.get("predicted_n"),
    "gen_t_ps": round(t.get("predicted_n")/(t.get("predicted_ms", 1)/1000), 3),
}))
PY
) || { echo "ngl $NGL request failed"; kill $SPID 2>/dev/null || true; exit 1; }
  PEAK=$(awk '{if ($1 > m) m = $1} END {print int(m/1048576)}' "/tmp/ngl_vram_$NGL.txt")
  printf '{"ngl": %s, "ctx": %s, "idle_system_vram_mib": %s, "peak_system_vram_mib": %s, "meas": %s}\n' \
      "$NGL" "$CTX" "$((IDLE/1048576))" "$PEAK" "$MEAS" | tee -a "$ROOT/results/ngl_spill.jsonl"
  kill $SPID $SMPID 2>/dev/null || true
  wait $SPID 2>/dev/null || true
  sleep 10
done
echo "=== done — results/ngl_spill.jsonl"
