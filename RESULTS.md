# RESULTS — Apple M5 Max, 128 GB

Measured evidence for the `mlx-serve` configuration. Every number here was
produced on this machine with the harnesses in `scripts/`, on the weights named
below, one server at a time.

## Provenance

| | |
|---|---|
| host | MacBook Pro, Apple M5 Max, 18 cores (6 super + 12 performance), 128 GB unified |
| OS | macOS 27 |
| runtime | `mlx-serve` 26.11.2 (release binary), MLX 0.32.5 |
| model | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` — 4-bit affine, 16.95 GB, arch `qwen3_5_moe` (64 layers, 5120-dim, head_dim 256, 24 heads / 4 KV heads) |
| drafter | `drafter/` inside the pack — DFlash2, 1.01 GB, block size 16 |
| MTP | in-checkpoint head, depth 6, profile `g17_nax_q4_gs64` |
| power | AC, High Power mode (`powermode 2`) |

## Engine survey

The model has an MLX build, so the question was which runtime runs it fastest.
Published decode tok/s, same model, 4-bit class, M5 Max 128 GB:

| engine | t/s | source |
|---|---:|---|
| Kapitsa (KFlash) | 144 | `nace-ai/kapitsa` `docs/benchmarks.md` |
| **mlx-serve** (DFlash2) | **128** | same table; its own: 106-109 on M4 Max 128 GB, 243 on M5 Ultra |
| Splash (DFlash2) | 116 | same table |
| oMLX (DFlash2) | 92 | same table |
| MTPLX (MTP) | 73 | same table |
| mlx-vlm (MTP) | 56 | same table |
| llama.cpp (MTP) | 48 | same table |
| LM Studio (MLX runtime) | 33 | same table |
| Rapid-MLX | 37.9 (M5 Max **36 GB**) | `rapidmlx.com/compare` |

Two caveats, both load-bearing. The table is **vendor-run** (Kapitsa's own
repo), and the only independent cross-engine data found — an HF discussion
benchmarking Splash/oMLX/Ollama on an M4 Max 64 GB — puts those three within
~10 % of each other on long-context decode, which does not reproduce the
vendor spreads. So the ordering is directional, not a measurement. Kapitsa is
also **not an MLX engine**: it ships its own Metal kernels and its own weight
format, so it is out of scope for "the MLX version" even though it is the
fastest number in the table.

`mlx-serve` is the fastest engine that runs MLX-format weights, so it is the
one adopted and the only one measured here.

## One stream

`scripts/openai_stream_bench.py`, greedy, 384 max tokens, nonce at the front of
the prompt so nothing is cached, warmup discarded. Three depths:
`data/long_prompt.txt` (15 907 chars ≈ 4 k tokens), `data/prompt_30k.txt`
(92 000 chars ≈ 30 k tokens), `data/prompt_128k.txt` (547 284 chars,
130 747 tokens).

| arm | 4 k decode t/s | 4 k TTFT | 30 k decode t/s | 30 k TTFT | 128 k decode t/s | 128 k TTFT |
|---|---:|---:|---:|---:|---:|---:|
| **mlx-serve, default** (MTP + DFlash2 + PLD) | **64.0-75.2 (median 73.4)** | **6.5-7.1 s** | **61.7-82.0** | **41.7-42.2 s** | **46.9** | **283.5 s** |
| mlx-serve, `--no-mtp --no-pld --no-drafter` | 28.4-34.0 (median 31.2) | 5.2-6.9 s | 26.8-30.9 | 32.0-38.5 s | — | — |

The 128k row is the server's own log as well as the client's:
`prefill: 461.3 tok/s, decode: 46.9 tok/s` over 130 747 prompt tokens. The
hot-cache held the 130 716-token prefix in 9 366 MB, inside its 9 389 MB
budget — one session at the working context, which is the default.

## Agent shape

`scripts/turn_loop_bench.py`, four sessions, four turns each, 12 k chars of
initial context, 2.4 k chars of observation per turn, 256 max tokens. Sessions
run in their own threads and fire as soon as they are ready, so the server sees
up to four overlapping requests.

| `NP` | aggregate out t/s | prefill t/s | TTFT p50 | TTFT p90 | TTFT max | wall p50 | wall p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | **54.7** | 1 493 | 1.28 s | 15.8 s | 21.7 s | 10.4 s | 29.6 s |
| 4 | 49.9 | **1 773** | **1.26 s** | **14.9 s** | **20.7 s** | **6.3 s** | 28.6 s |

`NP=4` reads lower aggregate only because the model stopped early on more turns
(2 481 output tokens against 3 253). Its wall p50 is 6.3 s against 10.4 s, so 4
is the better fleet setting and 2 the RAM-first default.

## What the numbers say

* **The whole win is speculation, not the MLX kernels.** With MTP, DFlash2 and
  PLD off, the server decodes at 28.4-34.0 t/s at 4 k. Turning them on takes it
  to 64-75. The server logs the reason: `avg_per_round=4.12-4.19` accepted
  tokens at 27.5-27.9 % per-draft acceptance, i.e. ~4 tokens per verify pass.
* **Speculation keeps paying at depth.** 30 k reads 61.7-82.0 and 128 k reads
  46.9, against 26.8-30.9 with it off at 30 k. The drafter is a separate
  DFlash2 block model, not the trunk's own MTP head, and the server's gate
  re-enables speculation per request.
* **The 4 k and 30 k decode numbers are not ordered the way bandwidth says they
  should be** (73 at 4 k, 62-82 at 30 k). Acceptance rate, not depth, dominates
  once speculation is on: the 30 k answers were more predictable than the 4 k
  ones. Do not read these as a depth curve.
* **Prefill is the other half of the win.** 461 t/s at 128k means a cold
  130 k-token turn costs 283 s instead of the ~10 minutes a 176-231 t/s path
  would take.

## The retired path, for the record

The repo previously ran llama.cpp on a GGUF Q4_K_S build. Those numbers are
kept only as the baseline this switch was decided on; the implementation is
gone.

| | llama.cpp (best config) | mlx-serve |
|---|---:|---:|
| decode, 4k | 28.7 t/s | **73.4** |
| decode, 30k | 25.2-25.7 | **61.7-82.0** |
| decode, 128k | 14.3 | **46.9** |
| prefill, 128k | 176-231 t/s | **461** |
| 4 agent sessions, aggregate | 37.4 t/s | **49.9** |
| 4 agent sessions, TTFT p50 | 2.28 s | **1.26 s** |
| 4 agent sessions, wall p50 | 22.2 s | **6.3 s** |
| weights read per token | 14.29 GiB | 16.95 GB |

The llama.cpp column is its best measured config (`NP=4 CRAM=8192` for the
agent shape). At its shipped `NP=2 CRAM=4096` the same four sessions read
24.4 t/s at TTFT p50 24.0 s, because 1.6 GiB prompt-cache entries evicted each
other and every turn re-prefilled.

## Caveats

* **One quant, not a sweep.** 4-bit affine at 16.95 GB. No 6-bit or 8-bit
  comparison has been run on this machine, and the 4-bit pack is not
  bit-identical to the GGUF it replaced.
* **The 30 k default arm's first two reps ended early** (13 and 187 tokens);
  the 384-token reps read 78.3 and 82.0. The 4 k arm is four full-length reps.
* **The engine ordering is vendor-published.** Only `mlx-serve` was measured
  here; Kapitsa, Splash, oMLX, MTPLX and Rapid-MLX were not.
* **No 8-session fleet run.** The agent shape above is four sessions.
* **`mlx-serve` was run from the v26.11.2 release binary**, not built here.

## What was discarded

The llama.cpp path: its launcher, the vendored `third_party/llama.cpp` checkout,
`build/`, the GGUF weights, and the harnesses that only made sense against
`llama-server` (`bench.py`, `mtp_ab.py`, `parallel_ab.py`, `fleet_ab.py`,
`agent_cache_probe.py`, `gguf_head.py`, `guard.py`, `wait_idle.py`,
`build.sh`, `dl_smaller.sh`, `quant_compare.sh`, `patches/`). The two harnesses
that drive any OpenAI-compatible endpoint — `openai_stream_bench.py` and
`turn_loop_bench.py` — were kept and are what produced every number above.
