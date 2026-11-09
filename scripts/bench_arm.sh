#!/usr/bin/env bash
# bench_arm.sh — one benchmark arm: start a server, wait until it answers a
# chat request, replay tumwater sessions against it, stop exactly the process
# it started.
#
#   scripts/bench_arm.sh LABEL PORT MODEL_ID -- <server command...>
#
# Env: FIXTURE (replay fixture), REPLAY_ARGS (extra args for tumwater_replay.py
# run), READY_TIMEOUT (s, default 1800).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="$1"; PORT="$2"; MODEL_ID="$3"; shift 3
[ "${1:-}" = "--" ] && shift
mkdir -p "$ROOT/logs" "$ROOT/results"
SLOG="$ROOT/logs/server_${LABEL}.log"
"$@" > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; sleep 3; kill -9 $SPID 2>/dev/null' EXIT
echo "arm $LABEL: server pid $SPID, log $SLOG"
deadline=$(( $(date +%s) + ${READY_TIMEOUT:-1800} ))
until curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | grep -q '"choices"'; do
  if ! kill -0 $SPID 2>/dev/null; then echo "arm $LABEL: server exited"; tail -20 "$SLOG"; exit 1; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "arm $LABEL: not ready"; tail -20 "$SLOG"; exit 1; fi
  sleep 5
done
echo "arm $LABEL: ready after $(( $(date +%s) - deadline + ${READY_TIMEOUT:-1800} ))s"
python3 "$ROOT/scripts/tumwater_replay.py" run --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_ID" --fixture "$FIXTURE" --label "$LABEL" -v \
  --out results/replay.jsonl ${REPLAY_ARGS:-} > "$ROOT/logs/replay_${LABEL}.log" 2>&1
rc=$?
tail -13 "$ROOT/logs/replay_${LABEL}.log"
exit $rc
