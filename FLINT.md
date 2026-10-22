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
