#!/usr/bin/env python3
"""Summarise results/replay.jsonl: one row per label, reps side by side."""
import collections, json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "results/replay.jsonl"
rows = collections.OrderedDict()
for line in open(path):
    r = json.loads(line)
    if r.get("kind") == "replay_summary" and not r.get("errors") and r["scored_turns"] >= 32:
        rows.setdefault(r["label"].replace("-r2", ""), []).append(r)
def f(rs, k):
    return " / ".join("%g" % r[k] for r in rs)
print("| arm | turns/min | agg out t/s | stream t/s p50 | TTFT p50 | TTFT p90 | cached | peak GiB |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, rs in sorted(rows.items(), key=lambda kv: -max(r["turns_per_min"] for r in kv[1])):
    print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (label, f(rs, "turns_per_min"),
          f(rs, "agg_out_tps"), f(rs, "stream_tps_p50"), f(rs, "ttft_p50"), f(rs, "ttft_p90"),
          f(rs, "cached_frac"), f(rs, "peak_wired_compr_gib")))
