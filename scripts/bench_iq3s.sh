#!/usr/bin/env bash
# bench_iq3s.sh — measure the Swift-1.5-Qwen3.8-Flash-Next IQ3_S GSQ-RCO
# disk-spill path (83.74 GB, two shards) on this box.
#
# The model cannot fit in RAM+VRAM (~55 GB combined), so the routed experts
# stay on the CPU (-cmoe) as mmap'd file pages and are streamed from the NVMe.
# Arms:
#   A  -cmoe, no GPU expert cache      — pure demand paging (kernel page cache only)
#   B  -cmoe --moe-cache-mib N         — llama.cpp's MoE expert cache on the Vega
#   C  B + --moe-cache-cpu-share 1.0   — cache misses computed on the CPU
#
# Each arm: load, health, ONE /v1/chat/completions request over the long
# prompt with GEN_N max_tokens; prefill and generation timings come from the
# response's "timings" block. Peak system-VRAM sampled while it runs. Results
# append to results/iq3s_spill.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/tools/local/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/bin:$ROOT/tools/local/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-$ROOT/models/iq3s/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"
PORT="${PORT:-8091}"
THREADS="${THREADS:-8}"
CTX="${CTX:-8192}"
CACHE_MIB="${CACHE_MIB:-3072}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/data/long_prompt.txt}"
GEN_N="${GEN_N:-128}"
UB="${UB:-512}"
BATCH="${BATCH:-2048}"
ARMS="${ARMS:-A B C}"

run_arm() {
  local name="$1"; shift
  local log="$ROOT/logs/iq3s_$name.log"
  echo "=== arm $name: $*"
  "$ROOT/build/bin/llama-server" -m "$MODEL" -a "Swift-IQ3S-$name" \
      -c "$CTX" -t "$THREADS" -np 1 -dev Vulkan0 -ngl 99 -cmoe --fit off \
      -lm mmap -b "$BATCH" -ub "$UB" -fa on -ctk q4_0 -ctv q4_0 \
      --no-mmproj-offload --host 127.0.0.1 --port "$PORT" "$@" \
      > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 $pid 2>/dev/null || { echo "arm $name died at load"; tail -5 "$log"; return 1; }
    sleep 5
  done
  ( while kill -0 $pid 2>/dev/null; do
      cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null || echo 0
      sleep 1
    done ) > "/tmp/iq3s_vram_$name.txt" &
  local sampler=$!
  local meas
  meas=$(python3 - "$PORT" "$PROMPT_FILE" "$GEN_N" <<'PY'
import json, sys, time, urllib.request
port, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = open(path).read()
body = {"messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0.0, "stream": False}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    resp = json.load(r)
wall = time.time() - t0
t = resp.get("timings") or {}
out = {
    "wall_s": round(wall, 2),
    "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
    "prompt_t_ps": round(t.get("prompt_n") / (t.get("prompt_ms") / 1000), 3) if t.get("prompt_ms") else None,
    "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
    "gen_t_ps": round(t.get("predicted_n") / (t.get("predicted_ms") / 1000), 3) if t.get("predicted_ms") else None,
    "content_head": (resp["choices"][0]["message"].get("content") or "")[:80],
}
print(json.dumps(out))
PY
) || { echo "arm $name measurement failed"; kill $pid 2>/dev/null || true; return 1; }
  local peak_vram floor
  peak_vram=$(awk '{if ($1 > m) m = $1} END {print int(m / 1048576)}' "/tmp/iq3s_vram_$name.txt")
  floor=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  printf '{"arm": "%s", "flags": "%s", "meas": %s, "peak_system_vram_mib": %s, "mem_avail_after_mib": %s}\n' \
      "$name" "$*" "$meas" "$peak_vram" "$floor" | tee -a "$ROOT/results/iq3s_spill.jsonl"
  kill $pid 2>/dev/null || true
  kill $sampler 2>/dev/null || true
  wait $pid 2>/dev/null || true
  sleep 5
}

mkdir -p "$ROOT/results" "$ROOT/logs"
for arm in $ARMS; do
  case "$arm" in
    A) run_arm A ;;
    B) run_arm B --moe-cache-mib "$CACHE_MIB" ;;
    C) run_arm C --moe-cache-mib "$CACHE_MIB" --moe-cache-cpu-share 1.0 ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
done
echo "=== done — results/iq3s_spill.jsonl"
