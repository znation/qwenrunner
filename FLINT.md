# Flint on the M5 Max: where the time goes, and what actually makes it faster

[Flint](https://huggingface.co/RecursiveSI/flint) is the second model this repo
can host: 39.8 B total / ~2.5 B active decoder-only MoE (49 layers: 2 dense +
47 MoE, 256 routed experts with 6 selected per token plus 2 always-active
shared experts), Apache-2.0, and on this machine it is the *fast* one -- ~5x
the Qwen-3.8-27B operating point in `README.md`, for the arithmetic reason that
2.5 B parameters are read per token instead of 27 B.

Everything below was measured on the M5 Max described in `README.md`, one model
instance at a time, arms interleaved. Raw records are in `results/` and
`logs/`; the harnesses are `scripts/fleet_ab.py` (server configurations against
the turn loop, interleaved), `scripts/turn_loop_bench.py` (the fleet workload:
N appending transcripts), `scripts/ngram_ab.py` (speculative decoding A/B by
workload) and `scripts/openai_stream_bench.py` (wire-level streaming benchmark
that works against any OpenAI-compatible server, so its numbers are the
wire's, not llama.cpp's own timings).

**Text in, text out — there is no vision.** The card says so (`Input / output:
Text / text`), the org publishes no vision sibling, and the checkpoint carries
nothing to hook one to: 779 tensors with no vision tower, patch embedding or
projector among them, no image/vision keys in the metadata, no `mmproj` file in
the repository, and `src/models/flint.cpp` builds `build_inp_embd(tok_embd)` and
nothing else. A request that carries an image content part is refused outright
rather than mis-answered:

```
HTTP 500: image input is not supported - hint: if this is unexpected,
          you may need to provide the mmproj
```

So the `input: ["text"]` in the pi/omp catalog entries is a correct declaration,
not a conservative one, and image work has to route to one of the vision models
already in those catalogs.

## Slots and aggregate throughput

llama-server `-np N` puts up to N requests into one decode step, and a step
reads the weights once regardless of how many rows are in it, so extra rows
amortise the 1.56 GB/token weight read. Measured, one instance at a time, chat
endpoint, 256-token responses, temp 0, `SPEC=0`:

| config (llama.cpp) | shallow, per session | shallow, aggregate | 27k depth, per session | 27k depth, aggregate |
| :---- | ---: | ---: | ---: | ---: |
| `-np 1`, 4 concurrent | 121-123 | 105.7 | 86-96 | ~90 (one at a time) |
| `-np 2`, 4 concurrent | 82.6-94.5 | 161.3 | - | - |
| `-np 4`, 4 concurrent | 75.0-77.5 | **301-310** | 39.5 | **157.9** |
| `-np 4`, 4 concurrent, 4 *distinct* contents | 67-67 | 268.2 | - | - |
| `-np 8`, 4 concurrent | 75.9-77.6 | 304-310 (no gain) | - | - |
| `-np 8`, 8 concurrent | 47.4-61.6 | 376-488 | - | - |

Reading of that table:

* **Maximum aggregate: `-np N` with N = your real concurrency.** NP=4 gives
  301-310 t/s against 105.7 serialized and 123.5 for a single stream. NP=8 with
  only 4 concurrent requests gives exactly NP=4's numbers, i.e. slots beyond
  concurrency are wasted memory (each slot is 7 GiB of f16 KV at 262 144
  context, 0.44 GiB at 16 384).
* **The knee is the memory roofline.** At NP=4, 306 t/s x 1.56 GB/token is
  477 GB/s, i.e. ~92 % of the 510-530 GB/s this machine delivers to a streaming
  kernel. Four rows already saturate it at shallow depth. The NP=8/8-concurrent
  figure exceeds that roofline only because eight near-identical prompts pick
  the same experts, shrinking the per-step expert read -- with four *distinct*
  contents NP=4 drops from 306 to 268 t/s (418 GB/s), which is the honest
  shallow number.
* **At depth a slot is worth as much as the weights.** A session's step at 27k
  walks its own cache (28 672 B/token, no sharing across slots): 0.77 GB of KV
  against 1.56 GB of weights, so batching four rows costs 3.1 GB + 1.56 GB per
  step. Four sessions each producing 1 024 tokens from 27k of context measured
  42.6 t/s wall-rate on the first pass (prefills included) and **138.4 t/s** on
  the repeat pass: the slots still held the four conversations' KV, so the
  second pass skipped prefill entirely (29.6 s of wall for 4 092 tokens). That
  persistence is the whole case for slots over fresh processes, and why `NP` is
  the fleet's throughput knob.

Not measured: speculative decoding with `-np` > 1. The n-gram drafter in this
revision keeps one table shared across sequences ("shared across all
sequences", `common/speculative.cpp`), so multi-session acceptance can differ
from the single-stream numbers above; every number in this section is `SPEC=0`.

### Several harnesses at once

The sections above measure *decode-heavy, uniform, repeated* traffic. Real agent
harnesses are the opposite: every turn is a different, growing transcript with a
short output. That inverts which number matters, and it changes three things.

**1. Prefill does not batch across sessions; it queues.** Four sessions, four
*distinct* ~21k prompts, 64-token outputs, `-np 4`:

| | TTFT per session | wall |
| :---- | :---- | ---: |
| `-np 1` | 12.7, 32.2, 49.3, 66.6 s | 71.2 s |
| `-np 4` | 16.1, 29.1, 41.5, 51.9 s | 57.2 s |

Both configs move the same ~84k prompt tokens in ~55-70 s, i.e. ~1.6k t/s of
aggregate prefill either way: `-b`-sized chunks of the four prompts fill the
batch *in sequence* rather than side by side. So a harness's first-token
latency under load is roughly (its place in the queue) x (its prompt / prefill
rate) -- the fourth session waits ~3x the first, and `-np 4` buys only ~20 % of
wall. This is the metric that matters for agent traffic, and it is the one my
decode-focused tables do not show.

**2. Slot reuse collapses the per-turn cost, and it works at `-np 4`.** The same
four prompts sent a second time: TTFT 0.34, 0.47, 1.08, 1.15 s (llama.cpp found
each session's cached prefix in its slot), then all four decoded in parallel at
12.0-12.2 t/s each, 6.3 s of wall for 252 tokens. That is the appending-transcript
case: an agent turn costs its *delta* prefill plus decode, not its whole
transcript -- the same lever `README.md` documents for Qwen, and the reason
multi-harness use is viable at all. It holds only while each session's slot
keeps its KV, so the failure mode to watch is slot eviction: a session whose
transcript is displaced re-prefills from scratch at ~1.6k t/s.

**3. Distinct contents cost ~13 %.** Four sessions with identical prompts let
the batch share expert unions; four distinct excerpts dropped `-np 4` shallow
aggregate from 306 to 268 t/s. Any real multi-harness workload looks like the
second number.

**4. The turn loop, which is the workload that matters.** Each session appends
its own output plus a ~700-token tool observation to its own transcript (12k
characters of context at turn 0, ~24.5k by turn 5), 256-token outputs, turns
fired and measured per session -- `scripts/turn_loop_bench.py`, one server at a
time, all rows from a loaded window (see 5):

| sessions, window | TTFT p50 | TTFT p90 | TTFT max | turn wall p50 | aggregate out | aggregate prefill |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, `-np 1`, loaded | 0.49 s | 0.59 s | 0.60 s | 9.8 s | 26.6 t/s | 607 t/s |
| 4, `-np 4`, quiet box (load 9.7) | **1.19 s** | 4.78 s | 5.05 s | **5.5 s** | **171.6 t/s** | 4009 t/s |
| 4, `-np 4`, loaded (128-170) | 2.18 s | 6.98 s | 8.19 s | 23.7 s | 43.0 t/s | 1004 t/s |

The mechanism is **queueing, not re-prefill**. A slot reuses its own KV prefix
for an appending transcript, so a turn costs its delta prefill and TTFT stays
flat as the context grows: ~0.5 s at one session, ~2.2 s at four, and the four
sessions' per-turn wall stayed balanced at 23.70-23.82 s. What the slot count
changes is what happens when sessions *overlap*: at `-np 1` the four turns
serialize (TTFT 12.7, 32.2, 49.3, 66.6 s in item 1's table), at `-np 4` the four
rows decode together so each session keeps its own rate, and at `-np 8` with only
four sessions nothing improves -- slots beyond the real concurrency are wasted
memory, which is why `NP` is sized to the fleet.

**5. Absolute rates swing ~4x on this shared machine, and the cause is not the
obvious one.** Paired A-B-A-B arms, llama.cpp `-np 4`, same 4x4 turn loop:

| added load | arm A | arm B |
| :---- | ---: | ---: |
| 32 spinning CPU processes on 18 cores | 136.1, 128.1 t/s | 148.0, 133.7 t/s |
| 2 saturating GPU read loops (~490 GB/s each) | 129.1 t/s | 125.8 t/s |
| same, second pair | 179.8 t/s | 162.9 t/s |

Neither does anything: 32 CPU hogs (nearly 2x the core count, swap 0, pageouts
7, ~1M pages free) cost nothing measurable, and load that *saturates* the memory
system costs 3-10 %. Yet across this session the same configuration measured
43.0 t/s (when the machine's ambient load average was 128-170) and 171.6 t/s
(hour earlier, ambient 9.7) -- a 4x spread -- with a streaming Metal kernel
reading a normal 492 GB/s in the slow window and the GPU 26 % busy. Two
consecutive no-load arms in one session also differed 129 against 180 t/s (39 %).

So the observation is solid and the mechanism is **not**: it is not runnable CPU
threads, it is not memory bandwidth, and an earlier revision of this file
attributed it to "host-side scheduling" -- that is withdrawn, because the
spinner test was designed to confirm it and instead refuted it. What is
established is only that a shared machine moves these numbers by up to ~4x
between windows, which is larger than every optimisation in this file.

Consequences for anyone tuning here: pick the *structure* from the facts that
held in every window measured -- slots, slot reuse and cross-slot batching are
architectural -- and treat absolute t/s as a range. The one number that stayed
tight across all windows is first-token latency for llama.cpp `-np 4`: 1.19-2.18 s
p50 per turn, clean or loaded, which is what an agent's turn latency actually
depends on.

The two rows marked loaded come from windows with ambient load average 128-170
and are 4x below their quiet-box counterparts; that window's cause is unresolved
(item 5), and the quiet-box run of the same configuration (171.6 t/s against
43.0) is the control that says it is the window, not the configuration.

Memory is the binding constraint once both slots and transcripts are real: f16
KV is 28 672 B/token, so the KV floor is `NP x per-slot context x 28 KiB` --
four 100k-token transcripts are 11.2 GiB, four 262 144-token ones are 28 GiB,
on top of 24.21 GiB of weights. `-c` is the total (`CTX x NP`), and a session
that outgrows its share is truncated rather than borrowing from a neighbour.

## A fleet of harnesses: slots are the knob, CRAM is the rescue

The tables above are one instance at a time with prompts built for the test. The
shape this repo actually serves is a *fleet*: N harnesses, each holding its own
transcript, each firing a turn when its own tool call returns. That workload is
`scripts/turn_loop_bench.py` (N sessions, appending transcripts, 256-token
turns), and `scripts/fleet_ab.py` runs it against interleaved server
configurations -- one server at a time, arms alternating A-B-A-B -- so the arms
share a window instead of a load average. Measured 2026-10-29, two interleaved
reps per arm, ambient load 8-125 from other workloads on the box:

| server config | sessions | aggregate out | aggregate prefill | TTFT p50 | turn wall p50 | RSS |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: |
| `-np 4`, CTX=131072 | 4 | 123.3 / 117.2 t/s | 2882 / 2739 t/s | 1.88 / 1.42 s | 7.3 / 8.3 s | 42.1 GiB |
| `-np 8`, CTX=131072 | 8 | 143.4 / 140.0 t/s | 3349 / 3271 t/s | 2.00 / 1.90 s | 12.8 / 13.0 s | 59.3 GiB |
| `-np 4`, CTX=131072, CRAM=0 | 8 | 79.7 / 79.9 t/s | 1836 / 1883 t/s | 13.2 / 12.1 s | 20.8 / 20.9 s | 40.2 GiB |
| `-np 4`, CTX=131072, CRAM=4096 | 8 | 130.4 / 142.9 t/s | 3586 / 3425 t/s | 5.3 / 5.3 s | 9.4 / 9.6 s | 47.0 GiB |

The first two rows and the last two are separate interleaved rounds (6 turns and
4 turns respectively, two reps each); compare within a pair, and note that the
slot-starved pair ran in the quieter window (its `-np 4` control would sit near
the top of the first pair's spread, not at 79 t/s).

Reading of that table:

* **Slots >= concurrent sessions, or the fleet re-prefills every turn.** The
  third row is the failure: eight conversations on four slots with the prompt
  cache off, so each arriving turn takes a slot from a conversation whose KV is
  then dropped -- every turn pays a whole-transcript prefill, and TTFT p50 goes
  from ~2 s to 12-13 s while aggregate output *falls* to 80 t/s. This is not a
  slow arm, it is a broken configuration, and it is the one that looks like "we
  have four slots, let's point eight agents at it".
* **CRAM is the rescue, not a substitute.** `--cache-ram 4096` keeps the
  displaced conversations' KV in host RAM, and in the same window it turns the
  third row into the fourth: 79.8 -> 136.7 t/s aggregate output (+71 %), prefill
  1860 -> 3506 t/s (+88 %), TTFT p50 12.6 -> 5.3 s, turn wall 20.9 -> 9.5 s, for
  +7 GiB of RSS. What it does not do is beat giving the fleet its own slots --
  `-np 8` measured 141.7 t/s at TTFT p50 1.9 s for +10 GiB more -- because with
  eight conversations on four slots the decode step still carries four rows.
* **Eight sessions on eight slots is worth ~+12 % of total tokens** over four on
  four: pooled over three `-np 8` runs and two `-np 4` runs (the table above plus
  a second interleaved round at 4 turns), output aggregates 141.7 / 126.9 / 137.1
  against 127.3 / 127.1 t/s and prefill 3310 / 3210 / 3182 against 2972 / 2813 --
  so +6 % on output, +12 % on prefill, ~+12 % on the token count that matters,
  at the same per-turn TTFT, for +16 GiB of KV. That is the whole reason
  `serve_flint.sh` now defaults to `NP=8`: the fleet's tokens are ~96 % prefill
  (3200 t/s of prefill against 135 t/s of output), and both rates rise with the
  number of rows a step carries, so the slot count is the throughput knob.
* **The RAM-saving alternative is `-np 4` plus CRAM**: 47.0 GiB against 57.5-59.3
  for `-np 8`, at 136.7 against 141.7 t/s aggregate output, and the price is
  first-token latency (TTFT p50 5.3 s against 1.9 s, because half the fleet is
  waiting for a slot at any moment). When the machine has RAM to spare, slots
  buy latency and throughput together; when it does not, CRAM is the cheaper
  half of the same fix.
* **RAM, measured** (RSS of the server process, weights included): 41.3-42.1
  GiB at `-np 4`/CTX=131072, 57.5-59.3 GiB at `-np 8`/CTX=131072, 33.0 GiB at
  `-np 4`/CTX=65536, +7-8 GiB with CRAM=4096 in use. The weights are 24.21 GiB of
  that, so the KV is `NP x CTX x 28 KiB` plus a fixed ~35 MiB per slot for the
  sliding-window layers -- the CTX block in `serve_flint.sh` is the table to
  size from.

### Tuning the fleet defaults

Every knob a multi-harness fleet touches was then swept at the fleet shape itself
(8 slots, 8 appending sessions, 4 turns, card sampling, interleaved reps with the
arm order reversed each rep, one server at a time). One default changed:

| knob | default | fleet measurement |
| :---- | :---- | :---- |
| `-b` (BATCH) | **4096** (was 2048) | pooled over seven interleaved runs each, in three windows: median 141.6 against 124.2 t/s aggregate out, prefill 3591 against 3405; 4096 wins six of the seven paired comparisons and one window is level, so read it as +12-14 % aggregate; turn wall p50 10.3 against 11.3 s; TTFT +0.5 s; RSS unchanged |
| `-b 8192` | not taken | 143.1 t/s, no gain past 4096 |
| `-ub` 1024 with `-b` 4096 | not taken | 139.6 against 148.9 |
| `-t` (THREADS) | 8, unchanged | 12 measured 130.3 against 148.9 -- an artefact of arm order (see below) |
| CRAM | 0, unchanged | 4096 *costs* ~9 % at 8 sessions / 8 slots (109.9 against 131.3) and is worth 1.6-2x when conversations exceed slots |
| SPEC | 0 at NP > 1, unchanged | off 113.8 / 123.3; `ngram-map-k` 107.7 / 119.0; `ngram-map-k4v` 115.1 / 112.8; `ngram-mod` 109.7 / 99.5; `ngram-cache` 81.9 / 84.5 |

Two notes on method, because both cost a run to learn:

* **`-b` is invisible to llama-bench and to a single stream.** It caps the prompt
  tokens one decode may carry, and only a *wave* of turn deltas (NP x ~900
  tokens) is large enough to feel the cap; single-stream pp512 at d20000
  (1651.6 / 1661.9 at 2048 against 1682.1 / 1658.5 at 4096) and tg128 are
  identical. A batch default for a fleet cannot be inherited from the
  single-stream tables above, only measured with the fleet harness.
* **Arm order matters even when the arms interleave.** The first sweep ran the
  same arm order every rep while the box's ambient load fell over the run, which
  handed the last arm a 13 % "win" for `-t 12`; `scripts/fleet_ab.py` now
  reverses the order on every other rep and the effect disappeared. The `-b`
  result survives that control, and so does the CRAM result above.

### Where a prefill token goes, and what the kernels can do

Microbenchmarks (`test-backend-ops perf`, this revision, M5 Max) and the model's
own depth curve bracket what is left on the table, and they say the two halves of
a prefill token are limited by different things:

| shape (Q4_1 unless noted) | achieved | the same shape, other kernel |
| :---- | ---: | ---: |
| dense GEMM 2048x512x2048 | **27.5 TFLOPS** | - |
| MoE `mul_mat_id`, 256 experts / 6 used, same shape | **10.1 TFLOPS** | 27.5 dense |
| MoE `mul_mat_id` at n=2048 tokens | 9.4 TFLOPS | 8.6 (down-proj shape) |
| FA prefill, hsk=hsv=128, kv=16384, nb=512 | **70.7 TFLOPS** | 88.98 with GQA-8 |
| FA prefill, kv=32768, nb=4096 | 17.2 TFLOPS | 19.2 at kv=4096 |
| FA decode (vec), nb=1, kv=4096 | 0.74 TFLOPS | 0.51 at kv=32768 |

* **The MoE expert GEMM is the biggest gap: 10.1 TFLOPS against 27.5 for the
  dense path at the same dtype and shape.** In the model that is ~23 % of a
  prefill token at depth (the routed experts cost 1.77 GFLOP/token against 1.18
  for the two shared experts, which are plain dense matmuls and already run at
  the 27 TFLOPS above). The kernel is not bandwidth-bound there (237 GB/s of
  the ~510 available) and not row-padding-bound either -- it is the same
  10.1 TFLOPS at 12 rows per expert (ub 512) and at 48 rows per expert (ub
  2048) -- so the win would be in the tile/dispatch structure of
  `kernel_mul_mm_id`, not in a flag.
* **Prefill attention is fast when the KV is cache-resident and slow when it is
  not.** The same kernel reads 70.7 TFLOPS at nb=512/kv=16384 and 17.2 at
  nb=4096/kv=32768, and the model's own depth curve shows the same shape:
  pp512 holds ~1670 t/s from d2048 to d8192 and then falls linearly (1368 at
  d16384, 1090 at d32768), i.e. attention is flat while the per-layer KV fits in
  cache and DRAM-bound past ~8k. Flint runs 32 query heads over 4 KV heads, so
  each KV head's rows are read by eight query heads; the GQA-8 microbenchmark
  (same FLOPs, one KV head instead of eight) is 26 % faster, which is that
  sharing being partly served by cache and partly not.
* **The shipped flash-attention tile is already the right one.** Widening the
  query tile to 64 rows and halving the KV chunk (Q=64/C=32) measured 23 %
  *slower* (14.7 against 19.2 TFLOPS at kv=4096/nb=4096, 13.0 against 17.2 at
  kv=32768) and is reverted; the threadgroup-memory budget does not allow Q=64
  with the 64-row chunk, and the smaller chunk costs more than the wider tile
  wins.
* **The tensor API is not a switch left unflipped here, and only one of its two
  consumers is wrong.** In this revision `has_tensor` reaches exactly two places:
  `ggml_metal_library_get_pipeline_mul_mm` (the dense GEMM -- the two dense
  layers, the 94 shared-expert matrices, `output.weight`) and the
  flash-attention gate in `ggml-metal-ops.cpp`. `mul_mat_id` never consults it,
  which is why the MoE check passes either way. The whole-path cost is real --
  `GGML_METAL_TENSOR_DISABLE=1` measures pp512 at d20000 at 701-836 t/s against
  1431-1591 with it on, ~2x in that interleaved window. But the tensor-API
  *flash-attention* kernel is wrong for prompt batches:
  `test-backend-ops test -b MTL0 -o FLASH_ATTN_EXT` fails 41/558 at nb >= 64 with
  ERR = inf, and a model that reaches it decodes word salad from the prompt --
  while FA is worth only ~3.4 % of a prefill token at depth. The `ne11 % 64`
  condition that sends non-multiple-of-64 KV lengths to the half8x8 kernel is
  real in the microbenchmark (8.15 against 70.70 TFLOPS at nb=512) but costs
  only 3.4 % at the model level (d20000 against d20032). So the env var is the
  wrong lever: it spends the whole ~2x to avoid a path worth ~3.4 %.
  `scripts/patches/0001-metal-flash-attn-disable-tensor-path.patch` routes FA to
  the half8x8 kernels and leaves `mul_mm` on the tensor path, which is what
  `serve_flint.sh` now defaults to (`metal-tensor=1`, variable not exported).
  Measured paired A-B-A-B on a fleet-free box, r=3, pp512 @ d20000: **1337.19 ±
  8.76 and 1173.93 ± 47.25 t/s** patched against **876.77 ± 27.46 and 880.23 ±
  11.60** with the disable, i.e. +34 % to +52 % of prefill recovered. With the
  patch the filtered and full `FLASH_ATTN_EXT` suites and `MUL_MAT` all pass with
  the tensor path on, and a live patched server answers a 19-token question
  (Bogotá) and recovers a needle from 222- and 1073-token prompts. The earlier
  in-tree figures for tensor-on prefill (1589 t/s at 30k in the
  `serve_flint.sh` header) describe this patched shape, not the disabled one.
* **Backend switches are noise.** Paired pp512 at d20032 with
  `GGML_METAL_CONCURRENCY_DISABLE`, `GGML_METAL_FUSION_DISABLE`,
  `GGML_METAL_GRAPH_OPTIMIZE_DISABLE` and `GGML_OP_OFFLOAD_MIN_BATCH=1` all land
  inside the base arms' own spread (1443-1584 t/s); the only consistent signal
  is that concurrency *on* is 3.5-7 % better, which is the default.

So the honest summary of this round: the fleet-level knobs are measured and now
defaulted (8 slots, CRAM documented), the per-token knobs are at their best
values, and the remaining wins are kernel work with named targets -- the MoE
expert GEMM first (10.1 -> 27 TFLOPS would be ~1.2x on a depth-20k prefill
token), GQA-aware attention second (1.6x on the attention term at depth, but
only past the point where the KV stops fitting in cache).

## Where the time goes

**The bytes.** Per token, decode reads 1.56 GB of weights: 1.01 GB of always-on
tensors (attention, the 2 dense layers, the 94 shared-expert matrices, the
router, `output.weight`) plus 0.55 GB of the 6 routed experts actually selected
out of 47 x 3 x 256. `token_embd.weight` is F32 (1.15 GB in the file) but is a
gather, so it costs one row, not the tensor. KV adds 28 672 B/token (14
full-attention layers x 4 kv heads x 128 dims x 2 tensors x 2 B), so 100k of
context adds 2.87 GB/token.

**The roofline.** A streaming Metal kernel on this machine reads
**510-530 GB/s** (8 GiB buffer, 16 B loads, best of 10; the spec sheet says
614 GB/s, and an independent M5 Max Max-Studio measurement reports a 528-563
GB/s GEMV plateau). llama.cpp sustains 193 GB/s at depth 0 (1.56 GB x
123.5 t/s) and 213 GB/s at d100k (4.43 GB x 48.0 t/s) -- **38-42 % of what the
silicon delivers, flat across depth**. Flat is the interesting part: the
per-token cost tracks bytes/token, so decode is proportional to traffic but at
about a third of the achievable rate. That is a kernel-efficiency gap, not a
capacity or bandwidth gap, and it is why the levers below are about *how*
bytes are read rather than how many.

**Why llama.cpp leaves 60 % on the table.** In this revision the MoE expert
matvec has no tensor-op or MMA path. `mul_mat_id` only reaches the simdgroup
kernels when it has >= 32 rows (`ggml-metal-common.cpp:44-51`, and `ne21` is
the token count), and the few-row MMA path for ordinary matmuls starts at 2
rows and caps at 16 (`ggml-metal-common.cpp:113-153`). Plain decode is 1 row,
so every expert matvec goes to `kernel_mul_mv_id_<type>_f32` -- 47 layers x 6
experts x 3 matrices = 846 small matvecs per token, each 2048 x 512, plus
everything else in the graph. Anything that puts more rows in flight changes
the kernel family as well as amortising the weight read.

**Prefill is a batch problem, and the card's values are wrong for this machine.**
The GGUF card suggests `-b 256 -ub 128`. llama-server splits a prompt into
`-b`-sized chunks, so the card's values cost 85 % of prefill at 27k: TTFT
24.3-24.7 s against **13.2 s** at 2048/512, measured on the same prompt with the
same server (warm, cache-busted). At 100k the same change takes TTFT from
119.4-126.5 s to **71.8-82.7 s**. Paired llama-bench at d30000 agrees on the
mechanism: `-ub 128` 802-939 t/s prefill and 76.9-94.4 t/s decode against
`-ub 512` 1255-1618 and 94.6-95.0. Shallow it is a wash (pp512 ~3700 either
way). `serve_flint.sh` now defaults to 2048/512, which is also what
llama-bench's own defaults are -- the t/s in its header describe those, not the
card's.

That measurement also produced the sharpest lesson in this file about this
machine: it runs at load 16-18 from other workloads, and one unpaired sweep
measured decode 15-30 % low *across every configuration in it*, including ones
that cannot be affected by `-ub`. Unpaired numbers from a busy box are not
evidence; the paired runs above are.

## Speculative decoding: a big lossless win, but only on the right text

Flint ships no MTP head, and no EAGLE-3/Medusa head exists for it, so the only
speculation available is llama.cpp's model-free n-gram drafter (`--spec-type
ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod`). It is lossless: at temp 0
the spec arms produced byte-identical greedy output to base in every run
compared.

| workload | base | n-gram | acceptance | protocol |
| :---- | ---: | ---: | ---: | :---- |
| verbatim reproduction, 4.1k, temp 0 | 119.1 | **200.7** (+68 %) | 71-82 % | raw, 512 tok |
| verbatim reproduction, 4.1k, temp 1 | 118.0 | **156-203** (+30-72 %) | 0.39-0.77 | chat, 384 tok |
| markdown continuation, 12k, temp 0 | 77.2-86.2 | **119-163** (+50-91 %) | 37-53 % | raw, 384 tok |
| markdown continuation, 30k, temp 0, raw | 76.4-95.6 | **229-283** (+195-270 %) | 73-74 % | raw, 384 tok |
| markdown document, 30k, temp 0, chat | 80.0-95.5 | 72.7-96.4 (neutral) | no drafts | chat, 384 tok |
| agentic chat, 27k, temp 1 | 93.4 | **97.3** (+4 %) | 29 % | chat, 384 tok |
| agentic chat, 27k, temp 0 | 93.4 | 79.9 (-15 %) | 0 % | chat, 384 tok |
| agentic chat, 100k, temp 1 | 49.4-62.5 | 56.0-62.5 (neutral) | no drafts | chat, 384 tok |
| prose continuation, 12k, temp 0 | 118.8-121.5 | neutral | no drafts | raw, 384 tok |

The temp-1 copy row is a range on purpose: the same prompt, same server and
same 384-token response measured 203, 202, 200, 196 t/s in the interleaved A/B
(acceptance 0.66-0.77) and 187, 159, 156, 140 t/s in follow-up runs
(acceptance 0.39-0.61). The difference is the sampler: the drafter proposes
tokens from the context, so *anything* that changes the sampled text changes how
much of it is copyable, and llama.cpp's own documentation for this drafter
warns about exactly this bimodality. Sampler settings are therefore part of the
speculative result, not a detail behind it.

The gain is a property of the text, exactly as `RESULTS.md` found for MTP on
Qwen: when the output re-emits text that is already in the context (verbatim
reproduction, file-level edits, markdown continuation) the drafter hits
66-82 % and the win is 70-270 %; when the output is novel reasoning it ranges
from nothing to a small loss. **The endpoint matters as much as the prompt.**
The +195-270 % row is a *raw* continuation: the model keeps writing the
document it was fed, which is the ideal case for a context-copying drafter. Ask
the same server the same question through `/v1/chat/completions` -- which is
what every client does -- and the model answers *about* the document instead,
the drafter finds no 48-token run to copy, and the result is neutral
(72.7-96.4 against 80.0-95.5). The win survives in chat mode only when the
*task* is copy-heavy: 4.1k verbatim reproduction measured 156-203 against 118.0
through the chat endpoint.

`ngram-mod` refuses to draft shorter than
`--spec-ngram-mod-n-min` (default 48), which is what keeps the reasoning case
neutral; `ngram-simple` drafts short sequences, and that is the arm that
measured -15 % when the acceptance was zero. `serve_flint.sh` therefore
defaults to `SPEC=ngram-mod` with `NMIN=48`, and it is one env var to turn off.

Three measurement lessons, all of which cost a run to learn:

1. **Temperature changes the workload, not just the sampling.** At temp 0 this
   model's chat output collapses into terse, non-code reasoning ("We need
   answer user. Need modify file..."), acceptance 0 %, and speculative
   decoding is a loss. At the card's sampling (temp 1, top-p 0.95, top-k 20)
   the same prompt produces real reasoning and code, acceptance ~29 %, and
   speculation is a small win. The card's settings are the honest setting.
2. **`--spec-synth-len` is not an acceptance ceiling.** It needs a registered
   spec context, so the drafter still runs and proposes real tokens; pairing it
   with `ngram-simple` on copy-heavy text measured 54/77/112 t/s for synthetic
   acceptance lengths 2/4/8 against 119 t/s base -- slower than no speculation
   at all. Read a synthetic arm as "this drafter, acceptance forced to L".
3. **Only interleaved arms are trustworthy.** The same deterministic content
   (temp 0, same prompt, same 384 output tokens) measured 80.6 and 121.5 t/s in
   two runs of the *same* configuration minutes apart, i.e. more than the
   effects being chased. An A-B-A-B control is what separated the real
   +66-72 % from that drift, and it is why `ngram_ab.py` interleaves reps.

**KV dtype.** At d30k, quantizing the cache was *worse* even with long drafts
that take the >= 32-row path: base f16 76.4-95.6 against base q8_0 65.2-90.1;
with speculation, f16 229-283 against q8_0 188-232. At 30k the KV term is 0.86
GB against 1.56 GB of weights, so the dequantization cost exceeds what the
smaller format saves -- the same direction `README.md` measured for Qwen, and
the opposite of what the literature predicts for long drafts. f16 stays the
default.

## What was deliberately not done

* **Requantizing to a smaller or different format** (Q4_0, Q4_K, MXFP4, or the
  i-quant ladder unsloth publishes for this model). Decode is not byte-bound:
  38-42 % of the achievable rate, flat across depth, means removing 10-15 % of
  the bytes buys about that much at best, while requantizing a Q4_1 file is
  lossy (double quantization) and a clean conversion needs the 85 GB BF16
  source. `llama-quantize --allow-requantize --tensor-type 'regex=type'` is
  available if this is revisited.
* **A trained draft head** (EAGLE-3, Medusa, LayerSkip). This is the fix for
  the reasoning-heavy case that n-gram drafting cannot address, and it is a
  training project, not a flag: the BF16 weights exist, but the head does not.
* **A fused compressed-domain int4 KV attention kernel for Metal** (the
  approach in arXiv 2604.16957 and 2605.05699, both Apple-Silicon-specific).
  It removes the dequantize-then-attend pass that makes quantized KV slower
  here, and it is the only lever in this survey that would attack depth
  directly -- but it is a new kernel plus a quality gate, and at 100k the
  measured gap between f16 and q8_0 was not yet worth it.
* **Expert caching / offload** (`--moe-cache-mib`, `--n-cpu-moe`). The cache in
  this revision only engages for layers whose experts are *host*-resident
  (`llama-moe-cache.cpp:179-182`), and on a 128 GB unified-memory machine the
  whole 24.21 GiB model is resident; adding an upload leg would only add cost.
* **Lookahead decoding** (`llama-lookahead` is built). Same guess-and-verify
  family as the n-gram drafters, but it is an example binary with no batching,
  no prompt cache and no OpenAI surface, so it cannot be compared fairly
  against `llama-server`.

## Reproduce

```bash
./scripts/build_flint.sh                     # llama.cpp with Flint support (PR #30928)
                                             # + scripts/patches/*.patch, applied there
hf download RecursiveSI/flint-int4-gguf flint-int4-q4_1.gguf --local-dir models

# speculative A/B by workload (interleaved arms, one instance at a time)
./scripts/ngram_ab.py --prompt-file data/flint_copy_prompt.txt \
    --arms base,ngram-simple,ngram-mod --reps 2 --n-predict 384 --temperature -1
./scripts/ngram_ab.py --prompt-file data/flint_agentic_30k.txt --chat \
    --arms base,ngram-mod --reps 2 --n-predict 384 --temperature -1

# fleet A/B: interleaved server configurations against the turn loop
python3 scripts/fleet_ab.py --reps 2 --turns 6 \
    --arm "np4:NP=4,sessions=4" --arm "np8:NP=8,sessions=8" \
    --arm "np4cram:NP=4,sessions=8,CRAM=4096"

# the fleet-default sweep (order alternates every rep; card sampling)
python3 scripts/fleet_ab.py --reps 3 --turns 4 --temperature -1 \
    --arm "b2048:NP=8,sessions=8,BATCH=2048" --arm "b4096:NP=8,sessions=8,BATCH=4096" \
    --arm "b8192:NP=8,sessions=8,BATCH=8192" \
    --arm "b4096t12:NP=8,sessions=8,BATCH=4096,THREADS=12" \
    --arm "b4096u1k:NP=8,sessions=8,BATCH=4096,UBATCH=1024"

# per-kernel ceilings and the tensor-API A/B. Build the test binary first --
# note GGML_BUILD_TESTS must stay off at this revision (ggml/CMakeLists.txt
# adds a ggml/tests subdirectory that the tree does not carry):
#   cd third_party/llama-flint-build
#   cmake -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=ON . && ninja test-backend-ops
# the MoE shapes in the first command were added to tests/test-backend-ops.cpp
# (256 experts / 6 used, Q4_1, m=512/k=2048 and m=2048/k=512, n = 1..2048)
./third_party/llama-flint-build/bin/test-backend-ops perf -b MTL0 -o MUL_MAT_ID -p 'type_a=q4_1.*n_mats=256'
./third_party/llama-flint-build/bin/test-backend-ops perf -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=128,hsv=128'
GGML_METAL_TENSOR_DISABLE=1 ./third_party/llama-flint-build/bin/llama-bench \
    -m models/flint-int4-q4_1.gguf -ngl 99 -fa on -d 20000 -p 512 -n 0

# wire-level streaming benchmark against the served endpoint
./scripts/serve_flint.sh &                                     # llama.cpp on 8091
./scripts/openai_stream_bench.py --base-url http://127.0.0.1:8091/v1 \
    --label llama --prompt-file data/flint_agentic_30k.txt --reps 2 --max-tokens 384
```

Prompts added for these measurements: `data/flint_copy_prompt.txt` (verbatim
reproduction, the high-acceptance case), `data/flint_code_edit_prompt.txt`
(agentic edit, the 0 % case at temp 0), `data/flint_agentic_30k.txt` (27k of
this repo's own sources plus an edit instruction, the realistic case).

`llama-bench` remains the tool for non-speculative sweeps on the llama.cpp
side, and its numbers for Flint are in the `serve_flint.sh` header: pp512
3706 t/s / tg128 123.5 t/s at depth 0, 1589 / 90.9 at d30k, 604 / 48.0 at
d100k. It has no speculative support at this revision, which is why every
speculative number above went through `llama-server`.
