# RESULTS — Apple M5 Max, 128 GB

Measured evidence for this repo's configuration. Every number here was produced
on this machine with the harnesses in `scripts/`, one server at a time.

The default is **Splash 1.6.0 on `mlx-community/Qwen3.8-27B-4bit` with the
`incoai` DFlash2 drafter, 4 lanes** (2026-11-08). It was chosen by the
"Engine shoot-out" below, which replays real tumwater sessions. Everything after
that section is the earlier single-engine evidence for mlx-serve. It is kept
because its single-stream numbers still hold. They just do not predict the fleet.

## Engine shoot-out: replaying tumwater (2026-11-08)

### Why a replay

The previous choice of engine rested on single-stream decode, plus a synthetic
four-session loop with 3k-token contexts. tumwater's real load looks different.
Measured over 19 358 recorded turns in 508 pi sessions
(`~/tumwater/.tumwater/sessions`):

| | p10 | p50 | p90 |
|---|---:|---:|---:|
| request prompt, tokens | 13.5k | 39.5k | 81k |
| first prompt of a session | 6.3k | 7.4k | 8.7k |
| new tokens per turn | 180 | 747 | 3.2k |
| output tokens per turn | 110 | 333 | 2.2k |
| turns per session | 9 | 29 | 78 |

Hosted APIs served ~92 % of prompt tokens from cache on this load. Of the turns,
855 carried a `context_edit`, where tumwater's context-shake elides an old tool
output in place and invalidates the cached prefix from that point.

`scripts/tumwater_replay.py extract` takes 8 sessions (seed 7, distinct roles
first: perf, feature ×3, coverage, dedupe, clean, improve). Each must have at
least 28 requests and reach 25-90k tokens by the 28th. For every request point
it stores the OpenAI messages exactly as pi would have sent them: native
`tool_calls` and `tool` messages, thinking stripped from assistant history as
tumwater's strip-thinking extension does, and context edits applied where they
happened. The fixture is real transcript content, so it stays outside the repo.

`run` replays the first 4 sessions concurrently, one thread each, and sends the
next request as soon as the last one returns. The first 12 requests per session
build the prefix and are not scored. The next 16 are scored, at 25-55k tokens
(prompt p50 42.2k). `max_tokens` is the recorded output length clamped to
128-512. Sampling is temperature 0.6, top_p 0.95, top_k 20, seed 1234
(Qwen's recommended thinking settings). The server sees the same prompts, in the
same order and interleaving, on every arm.

### Flash-Next on Splash (2026-11-19)

Since the shoot-out, the server's default was moved from Qwen3.8-27B to
**Qwen3.8-Flash-Next** (the 125B-A6B `qwen4_exp` preview) at IQ3_S, on the same
Splash 1.7.0, same replay, two reps — the row at the top of the table. Compared
with the best 27B Splash arms:

* **Throughput wins.** ~+10% turns/min (17.7/16.6 vs 16.0-16.3), ~+40-50%
  aggregate output (92-101 vs 64-72 t/s), and ~4× per-stream decode (119-127 vs
  28-34 t/s). Both models hit the 512 cap on similar turn counts, so the t/s gap
  is not a cap artifact.
* **But TTFT is structurally worse**: p50 7-7.6 s vs 2.2-3.4 s, p90 20-21 s vs
  9-14 s. The server log shows each request pays 2-5 s of "claim wait" for a
  recurrent-state buffer under 4-way concurrency, and Flash-Next's KV lives on
  the SSD (585 MiB hot block cache) rather than resident like the 27B's. Net
  turn wall (p50 9.4 s) is still better because decode is ~4× faster.
* **Memory: ~2× the 27B's** — 93 GiB peak vs 48. Fits the 112 GiB working set
  with ~19 GiB headroom. The 26.8 GiB PLE n-gram table streams from SSD (237 MiB
  row cache); the 49.2 GiB of routed experts are resident; long-request KV
  spills to the 32 GiB SSD budget.
* **Why IQ3_S, not Q2_0.** IQ3_S (83.6 GB) is task-parity with the unquantised
  model (93.3 vs 93.1); Q2_0 (66.4 GB) costs ~4 points to shed ~20 GB this box
  does not need. The two lighter quants (IQ2_XS, IQ3_XXS) carry reported
  thinking-loop / fabrication quirks.
* **Quality was not measured here** — this harness scores speed only. The
  quality case for Flash-Next over the 27B rests on Qwen's published benchmarks
  (DeepSWE 58.7 vs 42.2, JobBench 55.7 vs 33.4, Toolathlon 73.5 vs 67.1).
* **Caveats.** Flash-Next is a `qwen4_exp` preview on the Qwen Community 1.0
  licence, not a stable release. Both reps ran at different host load
  (loadavg 6.8->2.4 and 2.4->11.2) and still agreed to ~7% on turns/min.

Command: `splash serve --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S
--max-context 128K --no-webui --idle-release off --language-only` (Splash 1.7.0;
DFlash2 `incoai/Qwen3.8-Flash-Next-DFlash2` auto-selected; PLE streamed, experts
resident, KV on SSD). Note the measured arm was `--language-only`; the shipped
`serve.sh` runs Flash-Next **with vision on** for tumwater, which it has not
been replay-scored with.

### Result

Two reps per arm, run on different hours, "/"-separated. `serve.sh-splash-default`
is the shipped `./serve.sh` itself, on :8090 with pi's model id: a third Splash
rep, and the end-to-end check. Generated by `scripts/replay_table.py`.

| arm | turns/min | agg out t/s | stream t/s p50 | TTFT p50 | TTFT p90 | cached | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| splash-fn-iq3s | 17.74 / 16.57 | 101.46 / 92.46 | 127.1 / 118.63 | 7 / 7.57 | 20.31 / 21.19 | 0.97 / 0.97 | 93 / 91.7 |
| splash-mxfp4 | 16.33 / 15.63 | 68.35 / 65.23 | 28.88 / 29.06 | 2.36 / 2.73 | 9.07 / 9.16 | 0.966 / 0.966 | 46.9 / 47.5 |
| serve.sh-splash-default | 16.02 | 66.78 | 33.68 | 3.35 | 13.81 | 0.966 | 48.3 |
| splash-gguf-udq4km | 14.74 / 15.71 | 65.26 / 69.58 | 27.39 / 29.71 | 2.26 / 2.46 | 11.97 / 12.29 | 0.966 / 0.966 | 47.9 / 48.1 |
| splash-mlx4 | 14.98 / 15.69 | 63.77 / 71.71 | 28.87 / 27.19 | 2.18 / 2.65 | 9.96 / 10.08 | 0.966 / 0.966 | 48.4 / 47.8 |
| mlxserve-np4-cache32g | 10.69 / 11.28 | 17.28 / 18.93 | 16 / 18.11 | 8.31 / 6.63 | 31.04 / 21.26 | 0.944 / 0.944 | 68.1 / 65.7 |
| kapitsa-27b-4slots | 10.65 / 11.16 | 45.35 / 46.96 | 14.79 / 15.2 | 2.05 / 1.74 | 4.67 / 4.58 | 0.967 / 0.968 | 39.2 / 37.1 |
| omlx073-mlx4-dflash2 | 6.61 | 26.08 | 9.23 | 5.67 | 12.82 | 0.964 | 97.2 |
| mlxserve-np4 | 3.89 | 6.25 | 17.76 | 11.46 | 127.89 | 0.763 | 54.2 |

* **turns/min** = scored turns ÷ wall time of the scored region. This is what
  tumwater feels.
* **agg out t/s** = all output tokens ÷ the same wall time.
* **stream t/s** = per-turn decode rate (tokens after the first ÷ time from
  first to last token), median over turns.
* **cached** = prompt tokens the server reported as cache hits ÷ prompt tokens.
* **peak GiB** = peak wired + compressor memory for the whole host (`vm_stat`).

### What the numbers say

* **Batched speculation is the whole ordering.** Splash keeps DFlash2 on with
  four requests decoding and verifies the four lanes' drafts in one pass, so each
  stream holds 27-34 t/s at 40k depth. Kapitsa batches its tree-verify too, and
  holds 15. oMLX 0.7.3 batches DFlash for qwen3_5 and gets 9. mlx-serve's
  MTP+DFlash2+PLD reaches 73 t/s on a lone stream, but 16-18 once sessions
  overlap.
* **The previous default's real problem was its prefix cache.** At `NP=4`,
  mlx-serve's budget (one 128k session, 9.4 GB) held two of the four sessions.
  The server log shows it evicting 4.6-4.8 GB entries every few turns, so only
  76 % of prompt tokens were cache hits, against 94-97 % for every correctly
  sized arm. Turns at 37-54k depth waited 113-410 s for their first token.
  `CACHE_MB=32768` alone took it from 3.9 to 10.7/11.3 turns/min.
* **Format is not a lever on Splash.** MLX affine 4-bit (16.1 GB), MLX mxfp4
  (15.2 GB) and GGUF UD-Q4_K_M (16.5 GB) all land at 14.7-16.3 turns/min, within
  rep-to-rep spread. The ~6 % byte saving of mxfp4 does not show through
  speculation and four-way batching. Affine 4-bit is shipped.
* **Kapitsa has the best tail.** TTFT p90 is 4.6 s against Splash's 9-14 s,
  with the lowest memory (37-39 GiB). It reaches about two-thirds of Splash's
  turn rate.
* **oMLX loses on memory as well as speed.** With the oQ4e model pinned
  beside the 4-bit one (its config pins it), the host peaked at 97 GiB.

### Arms and versions

| arm | command (port varies) |
|---|---|
| splash-fn-iq3s | `splash serve --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S --max-context 128K --no-webui --idle-release off --language-only` (Splash 1.7.0; DFlash2 `incoai/Qwen3.8-Flash-Next-DFlash2` auto-selected; PLE streamed, experts resident, KV on SSD) |
| splash-mlx4 | `splash serve --model mlx-community/Qwen3.8-27B-4bit --max-context 128K --no-webui --idle-release off --offline` (Splash 1.6.0; DFlash2 `incoai/Qwen3.8-27B-DFlash2` auto-selected; Neural Engine FFN split at share 0.24 for long prefills) |
| splash-mxfp4 | same, `--model mlx-community/Qwen3.8-27B-mxfp4` |
| splash-gguf-udq4km | same, `--model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --language-only` (no GGUF mmproj in that repo) |
| kapitsa-27b-4slots | `kapitsa serve --model 27b --private --slots 4 --context 128k` (Kapitsa 0.2.31; KFlash tree drafting on DFlash2; SSD prefix cache) |
| mlxserve-np4-cache32g | old `serve.sh` with `NP=4 CACHE_MB=32768` (mlx-serve 26.11.2, `ddalcu/Qwen3.8-27B-MLX-Serve-4bit`, MTP + DFlash2 + PLD) |
| mlxserve-np4 | old `serve.sh` with `NP=4`, default prefix-cache budget |
| omlx073-mlx4-dflash2 | `omlx serve` 0.7.3 (Homebrew) from an isolated `--base-path`, `--max-concurrent-requests 4 --hot-cache-max-size 32GB`, model `mlx-community/Qwen3.8-27B-4bit` with `dflash_enabled` and `incoai/Qwen3.8-27B-DFlash2` |

Not run: **llama.cpp**. Its multi-slot MTP has an open content-mixing bug
(ggml-org/llama.cpp#28286, closed as stale), and its plain path measured 37.4
t/s aggregate on four synthetic sessions before. **vLLM-Metal** and
**mlx-lm** don't speculate when batching hybrid models, so they top out near
mlx-serve's no-speculation rate. **MTPLX**'s batched MTP lane is
for the 35B only. **Rapid-MLX** and **vllm-mlx** were not tried.

### Caveats

* **Output lengths differ by engine.** Splash and Kapitsa hit the 512 cap on
  35-41 of 64 scored turns, mlx-serve on 3 and oMLX on 36. Turns/min favours
  engines that answer short and agg out t/s favours long ones. Splash leads
  both.
* **The host was shared.** tumwater was running on its API models throughout:
  pi processes and its test gate, with load average 1-11 at arm boundaries.
  Each summary records `loadavg_start/end`. The first mlx-serve 32 GB-cache rep
  overlapped a Homebrew install (load 19 at its end) and agrees with its quiet
  rep to 5 %.
* **Two reps, not five.** The spread between reps is up to ~12 % on agg out t/s
  and ~6 % on turns/min. The gaps between engines are 1.4× or more.
* **No quality comparison between quants**, and no 6/8-bit arm.

## The mlx-serve single-engine evidence (2026-11-07)

### Provenance

| | |
|---|---|
| host | MacBook Pro, Apple M5 Max, 18 cores (6 super + 12 performance), 128 GB unified |
| OS | macOS 27 |
| runtime | `mlx-serve` 26.11.2 (release binary), MLX 0.32.5 |
| model | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` — 4-bit affine, 16.95 GB, arch `qwen3_5_moe` (64 layers, 5120-dim, head_dim 256, 24 heads / 4 KV heads) |
| drafter | `drafter/` inside the pack — DFlash2, 1.01 GB, block size 16 |
| MTP | in-checkpoint head, depth 6, profile `g17_nax_q4_gs64` |
| power | AC, High Power mode (`powermode 2`) |

### Engine survey

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

### One stream

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

### Agent shape

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

### What the numbers say

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

### The retired path, for the record

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

### Caveats

* **One quant, not a sweep.** 4-bit affine at 16.95 GB. No 6-bit or 8-bit
  comparison has been run on this machine, and the 4-bit pack is not
  bit-identical to the GGUF it replaced.
* **The 30 k default arm's first two reps ended early** (13 and 187 tokens);
  the 384-token reps read 78.3 and 82.0. The 4 k arm is four full-length reps.
* **The engine ordering is vendor-published.** Only `mlx-serve` was measured
  here; Kapitsa, Splash, oMLX, MTPLX and Rapid-MLX were not.
* **No 8-session fleet run.** The agent shape above is four sessions.
* **`mlx-serve` was run from the v26.11.2 release binary**, not built here.

### What was discarded

The llama.cpp path: its launcher, the vendored `third_party/llama.cpp` checkout,
`build/`, the GGUF weights, and the harnesses that only made sense against
`llama-server` (`bench.py`, `mtp_ab.py`, `parallel_ab.py`, `fleet_ab.py`,
`agent_cache_probe.py`, `gguf_head.py`, `guard.py`, `wait_idle.py`,
`build.sh`, `dl_smaller.sh`, `quant_compare.sh`, `patches/`). The two harnesses
that drive any OpenAI-compatible endpoint — `openai_stream_bench.py` and
`turn_loop_bench.py` — were kept and are what produced every number above.
