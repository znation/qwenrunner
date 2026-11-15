#!/usr/bin/env bash
# dl_parallel.sh URL OUT [PARTS] — parallel-range download with plain curl.
# HF throttles per connection (~25 MB/s single-stream); N range streams in
# parallel multiply that. Parts land as OUT.partNN and are concatenated once
# every part is complete, so an interrupted run resumes per part.
set -euo pipefail
URL="$1"; OUT="$2"; PARTS="${3:-12}"

SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:" {s=$2} END {print s}' | tr -d '\r')
[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] || { echo "dl: no content-length for $URL" >&2; exit 1; }
if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$SIZE" ]; then
  echo "dl: $OUT already complete"; exit 0
fi

CHUNK=$(( SIZE / PARTS ))
pids=()
for i in $(seq 0 $((PARTS-1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 )); [ $i -eq $((PARTS-1)) ] && END=$(( SIZE - 1 ))
  PART="$OUT.part$(printf '%02d' $i)"
  EXPECT=$(( END - START + 1 ))
  (
    HAVE=0
    [ -f "$PART" ] && HAVE=$(stat -c%s "$PART")
    if [ "$HAVE" -lt "$EXPECT" ]; then
      curl -fL -r "$(( START + HAVE ))-$END" -o - "$URL" >> "$PART"
    fi
    GOT=$(stat -c%s "$PART")
    [ "$GOT" = "$EXPECT" ] || { echo "dl: part $i size $GOT != $EXPECT" >&2; exit 1; }
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "dl: a part failed for $OUT" >&2; exit 1; }

cat "$OUT".part* > "$OUT" && rm -f "$OUT".part*
GOT=$(stat -c%s "$OUT")
[ "$GOT" = "$SIZE" ] || { echo "dl: $OUT size $GOT != $SIZE" >&2; exit 1; }
echo "dl: $OUT complete ($SIZE bytes)"
