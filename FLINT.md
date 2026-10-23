# Flint on the M5 Max: where the time goes, and what actually makes it faster

[Flint](https://huggingface.co/RecursiveSI/flint) is the second model this repo
can host: 39.8 B total / ~2.5 B active decoder-only MoE (49 layers: 2 dense +
47 MoE, 256 routed experts with 6 selected per token plus 2 always-active
shared experts), Apache-2.0, and on this machine it is the *fast* one -- ~5x
the Qwen-3.8-27B operating point in `README.md`, for the arithmetic reason that
2.5 B parameters are read per token instead of 27 B.

Everything below was measured on the M5 Max described in `README.md`, one model
instance at a time, arms interleaved. Raw records are in `results/` and
`logs/`; the harnesses are `scripts/ngram_ab.py` (speculative decoding A/B by
workload) and `scripts/openai_stream_bench.py` (wire-level streaming benchmark
that works against any OpenAI-compatible server, so llama.cpp and MLX are
measured identically rather than through llama.cpp's own timings).

## Two runtimes, and the vendor's own engine wins

| workload (streamed, same harness, 384-token responses) | llama.cpp | llama.cpp + n-gram | `rec` (MLX) | `rec` vs llama.cpp |
| :---- | ---: | ---: | ---: | ---: |
| copy-heavy prompt, 4.1k, temp 0 | 121.6 | 200-203 | **226.8** | 1.87x |
| copy-heavy prompt, 4.1k, temp 1 | 118.0 | 156-203 | **213.7** | 1.81x |
| markdown document, 30k, temp 0, chat | 80.0-95.5 | 72.7-96.4 | **141.4-141.8** | 1.5-1.8x |
| agentic chat, 27k, temp 1 | 93.4 | 97.3 | **134.9** | 1.44x |
| markdown continuation, 100k, temp 1 | 49.4-62.5 | 56.0-62.5 | **75.8-76.6** | ~1.4x |
| prefill 27k, TTFT (warm, cache-busted) | 13.2 s | - | **9.4-9.5 s** | 1.4x |
| prefill 100k, TTFT (warm, cache-busted) | 71.8-82.7 s | - | **45.4 s** | 1.7x |

`rec` is Recursive's own engine for Apple Silicon
([recursive-oss/rec](https://github.com/recursive-oss/rec), installed to
`~/.local/bin/rec`): MLX plus Metal kernels (the runtime carries a Metal
"megakernel" and per-architecture kernels), running the 4-bit MLX checkpoint
(group-64 affine, 8-bit embeddings, bf16 KV) instead of the Q4_1 GGUF. Its
README claims 1.4-2.6x over llama.cpp for this model on an M4 Pro, which is the
same range measured independently here. Launch it with
`scripts/serve_flint_rec.sh`; the llama.cpp path stays as
`scripts/build_flint.sh` + `scripts/serve_flint.sh`.

Practical difference for clients: `rec` serves **only** `/v1/chat/completions`
(`/v1/completions` is a 404), so a raw-completion workload has to stay on
llama.cpp. Note also that `rec`'s numbers above were taken through the chat
endpoint, while llama.cpp's 30k/100k rows above are chat too -- the raw
comparison cannot be made on this runtime.

Quality caveat, not measured here: the two runtimes run *different* 4-bit
checkpoints. Third-party scoring of the MLX release puts it at mean KL 0.0481,
p99 0.730, 94.1 % top-1 against BF16 ([OsaurusAI card](https://huggingface.co/OsaurusAI/Flint-40B-A2.5B-JANGH2));
no equivalent number is published for `RecursiveSI/flint-int4-gguf`. The speed
comparison above is apples-to-apples; the quality comparison is not.

## Slots and aggregate throughput: the reason to keep llama.cpp

`rec` processes **one generation at a time** -- its docs say so ("overlapping
requests can receive HTTP 429") and it measured so here: four concurrent
requests came back in order, each at full engine speed (~220 t/s), for an
aggregate of **173-175 t/s**, which is just the single-stream rate. Its
aggregate is therefore *pinned* at whatever one stream does.

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
| `rec`, 4 concurrent | 217-226 (queued) | 173.5-175.2 | 124-140 (queued) | ~135 |

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
* **At depth the comparison narrows to a tie, and the reason is instructive.**
  A session's step at 27k walks its own cache (28 672 B/token, no sharing
  across slots): 0.77 GB of KV against 1.56 GB of weights, so batching four
  rows costs 3.1 GB + 1.56 GB per step. End to end, four sessions each
  producing 1 024 tokens from 27k of context:

  | | first pass (prefills included) | repeat pass |
  | :---- | ---: | ---: |
  | llama.cpp `-np 4` | 42.6 t/s wall-rate | **138.4** |
  | `rec` (queued) | 52.2 | 42.9 |

  The first pass is a wash. The repeat pass is not: llama.cpp's slots still
  held the four conversations' KV, so the second pass skipped prefill entirely
  (29.6 s of wall for 4 092 tokens), while rec re-prefilled all four (95.4 s).
  [INFERENCE] That reads as slot-KV persistence on one side and a cache policy
  that did not retain four 27k contexts on the other -- it is the measured wall
  time, not the mechanism, that is solid.
* **Runtime by concurrency.** One agent (even with appending turns): `rec` --
  55.1 against 26.6 t/s aggregate with comparable TTFT, and 220 against 123 t/s
  on a single stream. Two or more concurrent agents: llama.cpp with
  `-np N` = the number of agents, because rec queues (30.5 s against 2.2 s of
  TTFT at four sessions). At 27k/4 sessions with *fixed* prompts the two are
  within ~15 % on decode alone (llama 158 against rec's ~135), so it is the
  appending-transcript case and the concurrency that decide, not depth alone.

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

| | TTFT p50 | TTFT p90 | TTFT max | turn wall p50 | aggregate out | aggregate prefill |
| :---- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 session, llama.cpp `-np 1` | 0.49 s | 0.59 s | 0.60 s | 9.8 s | 26.6 t/s | 607 t/s |
| 1 session, `rec` | 0.76 s | 3.30 s | 3.33 s | **3.9 s** | **55.1 t/s** | 1256 t/s |
| 4 sessions, llama.cpp `-np 4` | **2.18 s** | 6.98 s | 8.19 s | 23.7 s | **43.0 t/s** | 1004 t/s |
| 4 sessions, `rec` | 30.5 s | 32.8 s | 34.7 s | 34.9 s | 29.4 t/s | 703 t/s |

The mechanism is **queueing, not re-prefill**. Both runtimes reuse prefixes for
an appending transcript -- llama.cpp in its slot, rec in its cache -- so a turn
costs its delta prefill and TTFT stays flat as context grows (llama.cpp ~0.5 s at
one session, ~2.2 s at four; rec ~0.8 s at one). What differs is what happens
when sessions overlap: llama.cpp decodes the four rows together, so each session
keeps its own rate; rec takes them one at a time, so a turn's first token waits
behind the other sessions' turns and the *p50 TTFT goes from 0.8 s to 30 s*.
Per-session balance follows: llama.cpp's four sessions averaged 23.70-23.82 s
per turn, rec's 29.1-34.0 s.

So the serving shape inverts between one agent and several: **one agent, rec
wins on throughput (55.1 against 26.6 t/s) with comparable first-token latency;
several agents, llama.cpp with `-np N` wins on both** (43.0 against 29.4 t/s,
and 2.2 s against 30.5 s of TTFT). An earlier version of this file attributed
rec's 4-session TTFT to re-prefilling; the 1-session run disproves that and the
number above is the correction.

**5. Both runtimes are hostage to *host* load, not GPU.** Mid-measurement this
machine went from load average 16 to 130-170 (the user's other workloads: a
fleet of node processes, Spotlight indexing the ~50 GB of checkpoints this work
downloaded), and decode fell from 123.5 to 21.6-26.9 t/s -- 4.6-5.7x -- while a
streaming Metal kernel still measured 492 GB/s of read bandwidth (was 510-530)
and the GPU sat at 26 % busy. So the GPU and its memory were idle and the
per-token cost was host-side waiting: each step needs the CPU to build, dispatch
and fence. rec degraded less in the same window (226 -> 59.8-69.9 t/s, 3.2-3.8x),
consistent with fewer host round-trips per token, but not enough to overcome the
slot-reuse and batching gap above. **This matters because it is the normal state
during agentic work**: a coding harness runs builds, tests, linters and git on
the same machine, so "several harnesses at once" *is* a loaded host. Absolute
t/s from a quiet machine do not describe it.

Caveat on the table in 4: both runs happened inside that loaded window, so the
absolute rates are ~4-5x below what the same configuration does on a quiet box
(that is why 43 t/s aggregate appears where the shallow table above shows
268-306). The load was slightly *higher* during the llama.cpp run (128-170)
than the rec run (130), so the comparison is if anything biased toward rec.

Memory is the binding constraint once both slots and transcripts are real: f16
KV is 28 672 B/token, so the KV floor is `NP x per-slot context x 28 KiB` --
four 100k-token transcripts are 11.2 GiB, four 262 144-token ones are 28 GiB,
on top of 24.21 GiB of weights. `-c` is the total (`CTX x NP`), and a session
that outgrows its share is truncated rather than borrowing from a neighbour.

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
* **MLX via mlx-lm.** Upstream mlx-lm has no Flint architecture; the MLX
  weights are only usable through `rec`, which is why the comparison above is
  against `rec` rather than against a hand-built MLX path.

## Reproduce

```bash
./scripts/build_flint.sh                     # llama.cpp with Flint support (PR #30928)
hf download RecursiveSI/flint-int4-gguf flint-int4-q4_1.gguf --local-dir models

# speculative A/B by workload (interleaved arms, one instance at a time)
./scripts/ngram_ab.py --prompt-file data/flint_copy_prompt.txt \
    --arms base,ngram-simple,ngram-mod --reps 2 --n-predict 384 --temperature -1
./scripts/ngram_ab.py --prompt-file data/flint_agentic_30k.txt --chat \
    --arms base,ngram-mod --reps 2 --n-predict 384 --temperature -1

# cross-runtime comparison through one harness (start one server, then the other)
./scripts/serve_flint.sh &                                     # llama.cpp on 8091
./scripts/openai_stream_bench.py --base-url http://127.0.0.1:8091/v1 \
    --label llama --prompt-file data/flint_agentic_30k.txt --reps 2 --max-tokens 384
./scripts/serve_flint_rec.sh &                                 # rec on 11500
./scripts/openai_stream_bench.py --base-url http://127.0.0.1:11500/v1 \
    --api-key "$(rec auth token)" --label rec \
    --prompt-file data/flint_agentic_30k.txt --reps 2 --max-tokens 384
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
