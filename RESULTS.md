# RESULTS — Apple M5 Max, 128 GB

Measured evidence for the `mbp_m5_max` configuration. Every number here comes
from a run recorded under `results/`; the table rows are the CSV exports of
`scripts/bench.py`, `scripts/mtp_ab.py` and `scripts/parallel_ab.py`.

> `results/` and `logs/` also still contain the Intel host's measurement files
> (`bench_*.jsonl`, `qc_*`, `mtp_q2.jsonl`, …). None of them were produced on
> this machine and none of them are referenced below; the files created by this
> branch are `bench_shallow.*`, `depth_f16.*`, `depth_q8.*`, `mtp_m5.jsonl`,
> `parallel.jsonl`, `parallel_hi.jsonl` and the `mtp_depth_*.jsonl` files added
> by the depth re-measurements below.

## Provenance

| | |
|---|---|
| host | MacBook Pro, Apple M5 Max, 18 cores (6 super + 12 performance), 128 GB unified |
| OS | macOS 27 |
| runtime | llama.cpp `aa5e0092f` (build 11487), backends `MTL,BLAS` |
| GPU | `MTL0 (Apple M5 Max)`, `MTLGPUFamilyMetal4` (5002), simdgroup matrix mul = true, **has tensor = true**, unified memory = true, `recommendedMaxWorkingSetSize` = 112.8 GiB |
| model | `Qwen3.8-27B-UD-Q4_K_S.gguf` — 14.29 GiB, 27.32 B params, arch `qwen35` (64 blocks, `full_attention_interval` 4, 4 KV heads x 256 dims, 1 `nextn` MTP block) |
| draft | `mtp-Qwen3.8-27B-Q4_0.gguf` (upstream `MTP/` subdirectory) |

Build:

```
./scripts/build.sh        # Metal + Accelerate + BLAS, Release, -j 18     ~1 min
```

The build produces one device. There is no `-dev`, no `-ts`, no `-sm`, and no
second backend to select between — the entire Intel design (Vulkan shaders
unpacked into `tools/local`, a SPIRV-Headers include hack, a `g++-13` pin for
CUDA, `-ngl` derived from a VRAM cap) is gone with the hardware.

## Where every layer lives (`-ngl`)

`llama-bench -p 512 -n 128 -r 2`, f16 KV, `-t 6`, default batch, shallow
context. `results/bench_shallow.csv`.

| `-ngl` | pp512 t/s | tg128 t/s | what is on the GPU |
|---|---|---|---|
| 0 | 21.28 | 1.32 | nothing — pure CPU, and this arm alone took 830 s |
| 8 | 25.31 | 6.18 | first blocks |
| 16 | 40.88 | 8.42 | the 4 full-attention blocks land |
| 32 | 57.50 | 9.86 | half the linear-attention blocks |
| 48 | 101.58 | 13.62 | |
| 64 | 416.20 | 17.09 | all 64 transformer blocks |
| **99** | **680.58** | **24.80** | `lm_head` + `output_norm` + the MTP block |

Two things are worth pulling out.

* Prefill is *19x* faster with the whole model on the GPU than with none of it
  (680 vs 21 t/s), and generation *19x* (24.80 vs 1.32). The Intel host was
  CPU-bound by arithmetic; the constraint here is bandwidth.
* The jump from 64 to 99 — two layers — is worth **+63 % prefill**. Those two
  tensors are the token embedding and `lm_head` over a 248 320-token vocab,
  which is ~1.27 B params, i.e. as large as several transformer blocks. A
  "everything fits, 99 is free" reading of the layer count would have
  understated this by a wide margin.
* `-ngl 64` at 17.09 t/s versus `-ngl 99` at 24.80: **the last layer is worth
  more than the previous 48 combined** in *relative* terms, so there is no
  reason ever to run this model partially offloaded on this machine.

## Flash attention

`results/bench_shallow.csv`, `-ngl 99`.

| `-fa` | pp512 t/s | tg128 t/s |
|---|---|---|
| off | 465.89 | 23.02 |
| auto | 639.38 | 25.11 |
| **on** | **705.07** | **24.36** |

`-fa on` is worth **+51 % prefill** over `off` and is free-to-slightly-better on
generation. `auto` picks something close to `on` but not the same thing, which
is why it is spelled out in `serve.sh` rather than left to the default.

## KV cache dtype

Shallow (`results/bench_shallow.csv`, `-ngl 99 -fa` default) and at 128k
(`results/depth_*.csv`).

| dtype | pp512 shallow | tg shallow | pp512 @128k | **tg @128k** | GPU footprint @128k |
|---|---|---|---|---|---|
| **f16/f16** | 583.0 | **27.89** | **230.96** | **14.34** | 24 689 MiB |
| bf16/bf16 | 609.8 | 25.68 | — | — | — |
| q8_0/f16 | 718.4 | 24.02 | — | — | — |
| q8_0/q8_0 | 717.8 | 21.90 | 221.81 | 12.894 | 20 763 MiB |
| q4_0/q4_0 | **723.3** | 22.91 | — | — | — |

The shallow numbers are misleading in both directions: at depth 0 the KV cache
is a few hundred KiB, so the dtype only changes which kernel gets selected
(quantized KV wins prefill by ~23 %, f16 wins generation by ~27 %). **At 128k,
where the cache is ~8 GiB, f16 wins both axes** — 14.34 vs 12.89 t/s generation
and 231 vs 222 t/s prefill — for 4 GiB more memory per instance.

So the default is f16, and `CTK=q8_0 CTV=q8_0` is the lever to pull when the
goal is more concurrent instances rather than a faster single stream.

**Read that with the speculative-decoding caveat in "KV dtype" below**: with
MTP on — which is how the server ships — the same comparison at 98.6k depth is
46.1 (`f16`), 30.7 (`q8_0`) and 26.2 (`q4_0`) t/s, and the shipped default is
`q4_0` for host memory rather than speed. The numbers above are unspeculated.

A quantized V cache requires flash attention; with `-fa on` that is satisfied
either way.

## Threads

`results/bench_shallow.csv`, `-ngl 99`, shallow.

| `-t` | pp512 t/s | tg128 t/s |
|---|---|---|
| 4 | **758.00** | 25.51 |
| 6 | 715.70 | 24.93 |
| 8 | 693.03 | 25.34 |
| 12 | 685.73 | 25.10 |
| 18 | 528.66 | 25.06 |

Generation is flat from 4 to 18 — the GPU does that work. Prefill is not:
`-t 18` (all cores, including the 12 slower ones) costs **30 %** versus `-t 4`.
The Intel result was the same shape (`-t 16` was ~8 % worse than 8); here the
penalty is larger and the optimum lower. `serve.sh` defaults to 4, which the
sweep says is marginally better than the super-core count; anywhere in 4–12 is
fine, 18 is not.

## Batch

`results/bench_shallow.csv`, `-ngl 99`, shallow. This is the one lever that
trades the two axes against each other.

| `-b`/`-ub` | pp512 t/s | tg128 t/s |
|---|---|---|
| 2048/512 (default) | 709.56 | 25.54 |
| 4096/1024 | 667.68 | 24.89 |
| 4096/2048 | 583.03 | 28.34 |
| 8192/2048 | 621.65 | 28.21 |
| 1024/256 | 633.32 | **28.76** |

Small ubatches win generation (+13 % at `-b 1024 -ub 256`) and lose prefill
(−11 %). The default `-b 2048 -ub 512` is the balanced choice and is what
`serve.sh` uses; use `BATCH=1024 UBATCH=256` for a generation-heavy workload.

## Context depth to 128k

`results/depth_f16.csv` (f16 KV) and `results/depth_q8.csv` (q8_0), `-ngl 99
-fa on`. Each row is one `llama-bench` invocation at that depth; llama-bench
also exercises the default `-p 512` / `-n 128` pair per invocation, so both are
listed.

| depth | f16 tg128 t/s | f16 pp128 t/s | q8_0 tg128 t/s | q8_0 pp128 t/s | GPU peak MiB (f16 / q8_0) |
|---|---|---|---|---|---|
| 0 | 22.0 / 22.6 | 439.3 | 25.8 / 26.2 | 490.8 | 15 994 / 15 999 |
| 4 096 | 24.3 / 25.0 | 465.4 | 19.3 / 21.6 | 466.2 | 16 091 / 16 135 |
| 16 384 | 26.4 / 26.0 | 453.9 | 17.7 / 18.6 | 422.0 | 17 036 / 16 554 |
| 32 768 | 17.4 / 18.4 | 363.0 | 20.4 / 21.5 | 367.0 | 18 143 / 17 548 |
| 65 536 | 16.1 / 16.2 | 264.2 | 15.2 / 15.4 | 261.2 | 20 306 / 18 484 |
| **131 072** | **14.3 / 14.0** | **175.8** | **12.9 / 12.5** | **172.6** | **24 689 / 20 763** |

**A 128k context runs at 14.3 t/s generation and 176–231 t/s prefill.** On the
Intel host the equivalent was 1.19 t/s generation and 29.8 t/s prefill with a
73-minute prefill. The gap is the whole point of this branch.

The mid-depth rows disagree with each other (f16 dips at 32 768 while q8_0
rises; both have ±30 % scatter that the shallow stages did not), so treat the
interior as a trend and the two endpoints as the measured values. Two candidates
for the scatter, neither confirmed: the desktop's own GPU tenant (see traps),
and the KV/weight split crossing a residency-set boundary as the cache grows
past ~4 GiB.

The memory column is the sum of `"In use system memory"` across accelerators at
its peak, which includes the mapped weights — it is a footprint, not a budget.
The per-instance marginal cost of 128k is the *difference* between one instance
and two, which is what `parallel_ab.py` measures.

## MTP speculation

`results/mtp_m5.jsonl`, `llama-server -ngl 99 -c 131072 -t 4`, one
`POST /completion` with `n_predict=64` against the ~4 k-token
`data/long_prompt.txt`. Each arm is SIGKILLed after the request, hence
`exit_code = -9` throughout.

| arm | prefill t/s | **generation t/s** | draft accepted | GPU peak MiB | load s |
|---|---|---|---|---|---|
| base | 507.36 | 21.06 | — | 24 337 | 7.2 |
| `--spec-type draft-mtp -md mtp-…Q4_0.gguf -ngld 99` | **579.71** | **40.40** | 47/48 | 27 720 | 6.1 |
| `--spec-type draft-mtp` (no `-md`) | 484.61 | 37.90 | 47/48 | 26 434 | 3.0 |

**MTP speculation (on this prompt — see the caveat below) nearly doubles
generation: 21.06 → 40.40 t/s (+92 %)**, and it does not
cost prefill here (579.7 vs 507.4 — the draft head shares the same offload).
That is a much larger win than the +29 % measured on the Intel two-GPU split,
and for the predicted reason: there the draft head ran on the CPU (`-ngld 0`)
because the 1070 had no room for it, and the CPU-resident draft was the
bottleneck. With unified memory the whole draft goes on the GPU (`-ngld 99`).

The third row corrects a claim inherited from the Intel harness. Its comment
said that `--spec-type draft-mtp` without `-md` leaves `ctx_dft` null and never
enables the driver; here it accepted 47 of 48 draft tokens and ran at 37.9 t/s.
The main GGUF carries its own `blk.64.nextn` block (`nextn_predict_layers = 1`),
and this build drives speculation from it. The sidecar is still preferred —
+7 % generation, and it keeps the draft separate — but `serve.sh` falls back to
the embedded tensors when the sidecar is missing, and that fallback is measured,
not assumed.

Both gaps this paragraph used to list — MTP at 128k *depth* (the table above is
at ~4k) and MTP with a quantized KV cache — are measured at the operating point
in "The shipped operating point, re-measured at 98k depth" below. The second one
turned out to be worth far more than its −10 % row said — see "KV dtype" there.

## The shipped operating point, re-measured at 98k depth

### MTP on/off and draft depth

Every MTP number above was taken at ~4k of prompt *and* with an f16 KV cache.
`serve.sh` serves neither: the workload is a ~100k-token transcript and the
shipped KV dtype is `q8_0`. `results/mtp_depth_ab.jsonl`: `data/prompt_100k.txt`
(the llama.cpp README, `llama-server` logs `prompt_n = 98591`), 128k context,
`-ngl 99 -t 4 -b 1024 -ub 256 -ctk q8_0 -ctv q8_0 -fa on --spec-type draft-mtp
-md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99`, `n_predict=128`, temperature 0,
one server per arm, two interleaved reps each:

| arm | tg t/s, per rep | mean | vs `MTP=0` | draft accepted | pp t/s |
|---|---|---|---|---|---|
| `MTP=0` | 15.69 14.73 | **15.21** | — | — | 365.7 / 331.3 |
| `--spec-draft-n-max 3` | 23.97 20.08 | **22.03** | +45 % | 95/95 | 347.1 / 333.6 |
| `--spec-draft-n-max 6` (shipped) | 26.99 29.39 | **28.19** | **+85 %** | 108/108 | 344.7 / 342.0 |
| `--spec-draft-n-max 9` | 28.17 24.28 | 26.23 | +72 % | 114/114 | 330.0 / 341.7 |

* **Speculation is worth more at depth, not less.** The unspeculated arm pays
  the full 128k attention read on every token and runs at 15.2 t/s against
  24.4 t/s shallow; `n_max=6` recovers most of that, +85 % over `MTP=0` here
  against +180 % on the same kind of text at shallow depth.
* **`n_max` 9 is not a depth win either.** Its reps, 28.17 and 24.28, straddle
  the 6-arm's 26.99 and 29.39, and at shallow depth 9 loses to 6 on acceptance
  (89 % against 97 %). The shipped depth is right at both ends of the context
  range.
* **Prefill is unaffected** (330–366 t/s across all four arms), as at shallow
  depth: the draft head shares the same offload.
* Same caveat as the table above: acceptance here is 100 % because the prompt
  is the llama.cpp README and the model continues markdown verbatim. This is
  the patterned-text upper bound, not an average. Acceptance is *identical*
  across every arm below, so differences between arms are kernel cost, not
  speculation quality.

### KV dtype: f16 is fastest, q4_0 is the RAM choice, and the verify batch decides

Five sweeps at this operating point, only `-ctk/-ctv` changed, every sample at
98.6k depth with MTP `n_max=6`:

| KV | tg t/s, every sample | mean | slot KV at `NP=2` | GPU peak MiB |
|---|---|---|---|---|
| `f16/f16` | 40.97 44.80 46.32 47.13 47.98 49.52 | **46.1** | 16 GiB | 28 244–29 784 |
| `q8_0/q8_0` | 25.27 26.99 28.04 29.39 31.20 32.01 32.11 33.03 34.08 35.16 | **30.7** | 8 GiB | 24 342–26 182 |
| `q4_0/q4_0` | 24.46 27.88 | **26.2** | 4 GiB | 23 166–23 941 |

`results/mtp_depth_kv.jsonl`, `mtp_depth_kv2.jsonl` (a 2×2 of K and V dtype,
two interleaved reps), `mtp_depth_kv3.jsonl` (three-way, arm order reversed on
the second rep). No two arms overlap, and every arm accepted exactly 108/108
draft tokens, so this is kernel cost on identical work, not speculation
quality. Prefill is unmoved by dtype (349–365 t/s throughout).

**The whole difference is decoding with a speculative verify batch.** The
unspeculated depth table earlier in this file credits `q8_0` with a −10 % cost
(12.89 against 14.34 t/s at 131k); with `n_max = 6` that becomes −33 %, and
`q4_0` is a further −15 %. Mechanism: the flash-attention vec kernel is chosen
when the batch has fewer than 20 query rows, and the `use_kv_f16` heuristic
(`ggml/src/ggml-metal/ggml-metal-ops.cpp`) *refuses* the dequantise-to-f16 path
for head sizes below 512 in exactly that case. So a speculative verify batch —
`ne01 = n_max + 1 = 7` rows — runs the quantized-KV kernel that dequantizes its
K/V inline, the same K/V being re-read and re-dequantized once per query row,
while without speculation the batch is one row and the same kernel is
competitive. q4_0 keeps that structure and pays slightly more dequantization per
element for half the bytes. [INFERENCE: the code path is read, the attribution
to that kernel is inferred from the shape; the measurements are direct.]

**What each one buys in host memory**, paired back to back in one session
(`NP=2`, 128k per slot, `CRAM=4096`, two requests served each, byte-identical
output and draft acceptance):

| KV | host `used` | free |
|---|---|---|
| `f16/f16` | 106.2 / 105.6 | 21.5–21.8 GiB |
| `q4_0/q4_0` | 94.1 / 94.0 | **33.9–34.0 GiB** |

**+12 GiB of free RAM for the same configuration**, reproduced in both pairs.
This is the only measurement in the file where the idle baseline is controlled:
the machine's idle `used` drifted from 68.7 to 88.6 GiB over the same afternoon,
so absolute figures taken hours apart are not comparable and the paired
before/after is what carries the result.

**Quality, measured at last.** `llama-perplexity`, this model, one chunk of
`data/ppl_corpus.txt`:

| KV | PPL @16 384 | PPL @65 536 |
|---|---|---|
| `f16/f16` | 1.9780 ± 0.0325 | 3.3809 ± 0.0355 |
| `q8_0/q8_0` | — | 3.3809 ± 0.0355 |
| `q4_0/q4_0` | 1.9793 ± 0.0325 | 3.3946 ± 0.0357 |

q4_0 is +0.07 % at 16k and **+0.4 % at 65k, inside one standard error**; q8_0 is
identical to f16 to four decimal places. Read it as "no gross damage", not as
"free": perplexity does not test long-context retrieval, which is where a
quantized V cache is expected to fail first, and the corpus is once-through
prose rather than the code this server is used for. 128k needs at least
2 × context tokens and this corpus has 174 869, so 65 536 is the deepest gate
these data support.

**Mixing K and V dtypes is a trap.** All four `q8_0`-K/`f16`-V and
`f16`-K/`q8_0`-V runs — both orders, both reps — *load* normally (4–5 s,
24–27 GiB) and then never answer a request: the 128-token generation blew the
1800 s request timeout in 4 of 4 attempts, against ~4 s for either matched
pair, with no error in the server log. Do not mix. [INFERENCE: whether the
mixed pair hits a degenerate kernel path or an unbounded dequant loop is not
established — nothing in the log says, and this was not bisected.]

**This changed the shipped default twice.** The default is now `NP=4 CTK=q4_0
CTV=q4_0 CRAM=4096`, because this machine shares its 128 GB with the user's
other workloads (local Kubernetes clusters) and the KV cache is the part of the
footprint that has to give: four q4_0 slots reserve 8 GiB of KV where four f16
slots reserved 32, and the cache ceiling drops from 24 GiB to 4. Paired in one
session at `CRAM=4096`: `NP=2` 32.4 GiB free, `NP=3` 28.1, `NP=4` 22.6 — so the
four-slot q4_0 default still leaves more headroom than the two-slot f16
configuration it replaced (21.5), and the cost is decode speed: 26.2 against
30.7 (`q8_0`) and 46.1 (`f16`) t/s at 98.6k. Per-slot cost is ~5 GiB all-in, not
the 2 GiB of KV, because each slot carries its own compute and draft buffers.
The f16 default that this file briefly carried was right about speed and wrong
about the machine: `CTK=f16 CTV=f16 CRAM=8192` is the speed configuration when
nothing else needs the RAM.


### Batch and threads at the operating point: the +31 % does not reproduce

The default `-b 1024 -ub 256` was chosen on a single rep per arm at 98.6k
depth. Re-measured with two interleaved reps per arm
(`results/mtp_depth_tune2.jsonl`, MTP `n_max=6`, `q8_0` KV, 98 591-token
prompt), the shipped configuration's own two reps differ by 39 % — 25.27 and
35.16 t/s — which is wider than any between-config difference in the sweep:

| config | tg t/s, per rep | mean | pp t/s, per rep | mean |
|---|---|---|---|---|
| `-b 1024 -ub 256 -t 4` (shipped) | 25.27 35.16 | 30.21 | 341.5 359.9 | 350.7 |
| `-b 1024 -ub 256 -t 8` | 27.00 30.11 | 28.56 | 346.6 354.9 | 350.7 |
| `-b 2048 -ub 512 -t 4` | 31.53 28.03 | 29.78 | 358.2 365.3 | 361.7 |
| `-b 1024 -ub 128 -t 4` | 35.74 26.51 | 31.12 | 286.1 287.4 | 286.7 |

So `-b 2048 -ub 512` is not slower at depth (29.8 against the shipped 30.2, and
28.5 in the original single-rep run), and the generation axis at 98k is
dominated by machine drift — ±20 % *within* a config, measured across a 40 min
sweep — not by the ubatch. The prefill axis does reproduce, and it points the
other way from the single-rep claim that the pair was "prefill level":
`-ub 128` costs 18 % prefill (286.7 against 350.7) and `-ub 512` is the best of
the three (361.7). `-t 8` is indistinguishable from `-t 4` on both axes, the
same shape as the 64k thread sweep.

The shipped pair is still defensible — shallow depth is where the small ubatch
is measured to win generation (28.76 against 25.54, and 21.64 against 17.76 at
64k) and at the operating point it is at worst tied — but its depth figure is
a single sample that the interleaved re-measurement does not support, and the
prefill half of that claim is contradicted.

## MTP's gain is a property of the text, not of the machine

This is the most important caveat in the file. Everything above about MTP was
measured on `data/long_prompt.txt` — the llama.cpp README, which is markdown
with badges, tables and repeated link structure. The draft head predicts that
kind of continuation almost perfectly, and prose it does not predict at all.
`results/mtp_shape.jsonl` and `results/mtp_prose.jsonl`, 128k context, two reps
each; structured = README continuation at 512 tokens, prose = a Project
Gutenberg chapter at 256 tokens.

| arm | structured t/s | acceptance | **prose t/s** | acceptance |
|---|---|---|---|---|
| base (`MTP=0`) | 26.9 | — | **24.6** | — |
| MTP `--spec-draft-n-max 3` | 58.1 | 99 % | 23.8 | 26 % |
| MTP `--spec-draft-n-max 4` | 68.0 | 98 % | 19.8 | 20 % |
| MTP `--spec-draft-n-max 6` | **75.3** | 94 % | 21.9 | 14 % |

Read the two columns together: **MTP is worth +180 % on patterned text and
−11 % on prose**, and the draft depth that maximises one is the one that hurts
the other. Deep speculation only pays when the draft is usually right; when
acceptance is 14 %, six proposed tokens cost six verifications and return
almost nothing. There is no adaptive escape: `--spec-draft-p-min 0.6` (which
does work — it cut prose drafting from 821 tokens to 224) also cut the
structured case from 75.3 to 57.5 t/s, because the confidence threshold that
suppresses useless drafts on prose suppresses useful ones on code.

For reference, the structured numbers are what the rest of this file uses, so
the headline MTP figures (+92 %, +40 %) are *upper bounds for predictable text*,
not averages. **`serve.sh` ships `NMAX=6`**, the setting that wins on the
structured column, because that column is the workload this repo exists for: an
agent client with tool schemas and a long transcript, emitting code and JSON.
The prose regression is real and is the price of that choice — `NMAX=3` (the
upstream default, never a regression either way) or `MTP=0` are the settings
for free-form prose, one env var away.

## Speculation depth — `--spec-draft-n-max`

The default is `n_max = 3`. How many tokens the draft head is allowed to
propose per step is the largest single knob left, and it is worth **+40 %**.
`results/mtp_nmax_ab.jsonl`, five interleaved reps each, 512-token generations,
128k context, draft on the GPU:

| `--spec-draft-n-max` | generation t/s, per rep | mean |
|---|---|---|
| 3 (default) | 47.2 42.3 44.9 58.0 48.4 | **48.2** |
| **6** | 67.6 58.3 86.6 63.8 61.3 | **67.5** |

Every `n_max=6` run beat every `n_max=3` run — the worst 6 (58.3) exceeds the
best 3 (58.0) — so this is a real effect and not the machine's drift.
`results/mtp_nmax_hi.jsonl`, four more interleaved reps:

| `--spec-draft-n-max` | t/s, per rep | mean | draft acceptance |
|---|---|---|---|
| 6 | 54.1 52.6 52.1 58.7 | **54.4** | 461/434 |
| 9 | 52.0 47.5 34.2 49.4 | 45.8 | 510/454 |

Going deeper than 6 loses: acceptance drops from ~97 % to ~89 % and the extra
draft work is no longer repaid. `serve.sh` therefore ships
`--spec-draft-n-max 6` (`NMAX=<n>` to override).

Two measurement facts worth keeping:

* **Acceptance is deterministic, timing is not.** All five `n_max=3` runs
  accepted exactly 382 of 385 drafts, and all five `n_max=6` runs exactly 434 of
  461 — temperature is 0 and the prompt is fixed — while throughput ranged
  42.3–58.0 t/s. The scatter is the machine (other tenants, thermals), not the
  measurement. It is why every comparison in this file is interleaved and why
  the absolute level moves: the *same* `n_max=6` configuration measured 67.5
  mean in one sweep and 54.4 in the next, 20 % apart, with nothing changed.
  Compare within a sweep, never across sweeps.

## Quant size — a smaller quant is *not* the lever it was

`results/quant_m5.jsonl` (shallow, no MTP, `reps 2`) and
`results/quant_mtp_m5.jsonl` (128k context, MTP, `--spec-draft-n-max 6`, two
interleaved reps).

| quant | weights | shallow pp512 / tg128 | 128k+MTP tg t/s, per rep | mean |
|---|---|---|---|---|
| **UD-Q4_K_S** | 14.29 GiB | 746.5 / 24.37 | 71.2, 93.2 | 82.2 |
| UD-Q3_K_XL | 13.15 GB | 627.7 / 25.52 | 76.5, 97.8 | 87.2 |
| UD-Q2_K_XL | 9.83 GB | 753.3 / 26.08 | 93.6, 97.4 | 95.5 |

A 36 % smaller weight file buys **7 %** generation at shallow depth and at most
**16 %** at the 128k operating point — and the two Q4 reps bracket the Q2 mean
(93.2 vs 95.5), so the high end is inside the noise. That is a completely
different result from the Intel host, where a smaller quant was worth +52 %
generation, and the difference is the *mechanism*: there, a smaller quant fit
more layers into 5 GB of VRAM, so it changed what ran on the GPU at all. Here
nothing is left on the CPU to move, and decode at `-ngl 99` is not purely
bandwidth-limited:

* implied bandwidth at Q4_K_S shallow = 14.29 GiB x 24.37 t/s = **374 GB/s**,
  roughly 70 % of the M5 Max's peak, so there is headroom left.
* Q2_K_XL achieves only 256 GB/s at the same point — fewer bits per weight
  means *more* dequantization work per byte, and the kernels get less efficient
  as the format gets smaller.

So `serve.sh` keeps Q4_K_S: the speed is nearly flat across quants and the
quality is not. `scripts/dl_smaller.sh` fetches the others if you want to
trade.

## Batch and threads at 64k depth

`results/batch_depth.jsonl`, `results/threads_depth.jsonl`; `-ngl 99 -fa on`,
f16 KV, `-d 65536 -p 2048 -n 128`, one rep each, `bg_cpu_pct` 15–25 %.

| `-b`/`-ub` | pp2048 t/s at depth | tg128 t/s at depth | (shallow pp / tg) |
|---|---|---|---|
| 2048/512 (default) | 289.17 | 17.76 | 709.56 / 25.54 |
| 4096/1024 | 284.86 | 16.21 | 667.68 / 24.89 |
| 4096/2048 | 259.99 | 16.12 | 583.03 / 28.34 |
| 8192/2048 | 267.77 | 18.79 | 621.65 / 28.21 |
| **1024/256** | **345.01** | **21.64** | 633.32 / 28.76 |

At depth the small-batch configuration wins **both** axes (+19 % prefill and
+22 % generation over the default), which is the opposite of the shallow
result, where `-b 2048 -ub 512` was the best prefill setting and `-b 1024 -ub
256` the best generation one. The likely mechanism is attention working set:
prefill at 65k depth is dominated by attention over the whole cache, and a
smaller ubatch keeps that kernel's working set inside the GPU's caches.

Threads at the same 64k depth disagree with the shallow sweep:

| `-t` | pp2048 t/s at depth | tg128 t/s at depth | (shallow pp) |
|---|---|---|---|
| 4 | 271.64 | 18.37 | **758.00** |
| 6 | 294.62 | 18.29 | 715.70 |
| 8 | 286.74 | 16.57 | 693.03 |
| 12 | **323.81** | 17.58 | 685.73 |
| 18 | 317.91 | 20.27 | 528.66 |

`-t 4` is the *worst* prefill setting at depth and the best at 0; `-t 18` is
the worst shallow and near-best at depth. Both spreads are ~15 %, against 15–25 %
background CPU load, so the honest reading is that **4–12 threads are
indistinguishable at depth and only `-t 18` is clearly bad at shallow depth**.
`serve.sh` keeps `-t 4`. Resolving this properly needs interleaved reps at 128k,
which was not run.

The same caveat applies to the batch table: it is one rep per arm at 64k depth.
The section below re-measures the two interesting configurations at ~98k depth,
which is what made `-b 1024 -ub 256` the default.

### The same pair at the real operating point

`results/batch128k_ab.jsonl`. A 98 591-token prompt (verified with
`llama-tokenize`) against a 131 072-token server, MTP on with
`--spec-draft-n-max 6`, `-t 4`, one rep each, back to back.

| `-b`/`-ub` | prefill t/s | **generation t/s** | GPU peak MiB |
|---|---|---|---|
| 2048/512 | 339.96 | 28.52 | 30 266 |
| **1024/256** | **349.01** | **37.45** | 29 706 |

**+31 % generation at depth**, prefill level, for a slightly smaller footprint.
That is the operating point `serve.sh` serves, so `-b 1024 -ub 256` is now its
default; the shallow-prefill preference for 2048/512 (−11 %) is the cost, and a
128k-context server spends its life generating rather than prefilling. The
mechanism proposed above — a smaller ubatch keeping the long-context attention
working set out of the GPU's cache thrash — is consistent with the win being
larger at deeper context and absent at shallow.

Caveat: one rep per arm. The direction is supported by both depth points (64k
and 98k) and by the generation axis at shallow depth; the prefill-at-128k number
specifically is a single measurement.

## Parallel instances

`results/parallel.jsonl`. MTP on, `-c 131072 -ngl 99 -t 4`, one
`n_predict=128` request per server, all fired at once. `sum(RSS)` is the total
resident set of every server process.

| instances | sum(RSS) GiB | single-stream t/s | aggregate t/s | per-stream t/s |
|---|---|---|---|---|
| 1 | 24.9 | 41.96 | 53.51 | 53.51 |
| 2 | 49.9 | 38.71 | 67.16 | 33.58 |
| 4 | 99.8 | 62.09 | 68.32 | 17.08 |

**Memory is not the constraint; the Metal working set is.** `sum(RSS)` grows
22.3 GiB per instance, which is an artefact — resident set counts shared
file-backed pages once per process. The physical cost was measured directly by
sampling `vm_stat` around each server start:

| | host used | file-backed pages |
|---|---|---|
| idle | 42.8 GiB | 35.1 GiB |
| + 1st server | **+15.1 GiB** | 21.1 GiB |
| + 2nd server | **+8.2 GiB** | 21.1 GiB |

The 14.29 GiB of weights are mapped once and shared: file-backed pages *fall*
when the first server starts (they become wired, and stay shared), and the
second server costs exactly its 128k f16 KV cache and nothing else. So at 128k
f16 the marginal cost of an instance is ~8.2 GiB, and the machine's 128 GB is
not what stops you.

What stops you is the GPU. Six instances at f16 failed during the *concurrent*
phase with

```
E ggml_metal_synchronize: error: command buffer 1 failed with status 5
E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

on 3 of the 6 servers — the host was not swapping (swap moved +16 MiB over the
whole experiment) and host usage peaked at ~107 GiB, but Metal's per-process
working-set budget (~112.8 GiB recommended, summed over processes) ran out.
**Four instances at 128k f16 is the largest configuration measured to work.**
The bisect above it, in `results/parallel_hi.jsonl`:

| instances | KV | outcome |
|---|---|---|
| 4 | f16 | all 4 streams served, aggregate 68.3 t/s |
| 5 | f16 | 3 of 5 streams failed; the survivors averaged 21.4 t/s |
| 6 | f16 | servers died during load with the Metal OOM above |
| 6 | **q8_0** (half the KV) | 4 of 6 streams still failed |

Halving the KV cache did not buy back the sixth instance, so the binding term
is not the KV cache alone — MTP adds ~3.4 GiB of draft/compute buffers per
instance (24 337 MiB → 27 720 MiB measured for the single-server case), and it
is the summed Metal working set that runs out. More instances with `MTP=0` is
plausible and unmeasured.

Throughput saturates much earlier than memory: aggregate generation is flat
from 2 instances (67.2) to 4 (68.3), while per-stream speed falls 33.6 → 17.1
t/s. One GPU with one memory-bandwidth budget serves them all, so **N instances
buy concurrency, not throughput.** The measured spread on identical work is
large — the single-stream figure for the *1-instance* configuration was 42.0
t/s in one measurement and 53.5 t/s in the next, a 27 % gap with nothing
changed — so treat these as ±25 %, and read the shape rather than the digits.

## Parallel prompts — `-np` slots vs N processes

Two mechanisms serve N prompts at once, and they are not equivalent.
`results/parallel_slots.jsonl`, measured on the current defaults
(`-b 1024 -ub 256`, `-ngl 99`, f16 KV, MTP `n_max=3`), 128 prompts each,
128k context *per slot*:

| N | mechanism | total RSS | GPU (summed) | load | single | aggregate | per-stream |
|---|---|---|---|---|---|---|---|
| 2 | `-np 2`, one process | 34.0 GiB | 37.9 GiB | 4 s | 53.05 | 59.53 | 29.77 |
| 4 | `-np 4`, one process | 52.2 GiB | 58.2 GiB | 5 s | 43.78 | 52.89 | 13.22 |
| **6** | `-np 6`, one process | **70.4 GiB** | 78.6 GiB | 11 s | 43.72 | 53.16 | 8.86 |
| 2 | 2 processes | 49.8 GiB | 53.8 GiB | 8 s | 49.20 | 55.48 | 27.74 |
| 4 | 4 processes | 99.7 GiB | 105.6 GiB | 13 s | 59.44 | 66.94 | 16.73 |
| 6 | 6 processes | — | — | — | — | Metal OOM | — |

**Slots are the cheap axis.** Six slots in one process run where six processes
die: one process holds one weight mapping and one set of compute buffers, so
each additional 128k slot costs ~9.4 GiB of resident memory (14.3 GiB weights +
6 x 8 GiB KV + compute = 70.4 GiB measured), against ~22 GiB of summed RSS per
extra process — and the summed *GPU* figure, which is what the Metal
out-of-memory was actually about, stays at 78.6 GiB instead of exceeding the
~112 GiB working-set budget. Loading is also 2-3x faster (4-11 s against
8-13 s) because the weights are mapped once. A single process verifies this
itself: `-np 2` logs `n_slots = 2, n_ctx_slot = 131072, kv_unified = 'false'`,
so `-c` must be scaled to `CTX * NP` to keep every slot at full context — which
is exactly what `serve.sh` now does.

Aggregate throughput, though, is a property of the *workload*, not of the
mechanism. On the README-continuation prompt used above — where MTP accepts
94-99 % of drafts and a single stream already runs at 53-59 t/s — parallelism
just divides: per-stream falls 53.0 -> 13.2 -> 8.9 as N goes 2 -> 4 -> 6 and
aggregate sits near 53-60 t/s. Two concurrent requests on a *code* prompt with
`NP=2` behaved differently:

```
A: tg 44.29 t/s  draft 100/146     (write a Python function to reverse a linked list)
B: tg 46.78 t/s  draft 102/139     (write a SQL query to find duplicate emails)
```

aggregate ~91 t/s against 44-73 t/s for one stream on the same kind of prompt.
Where the single stream is already GPU-saturated (high-acceptance structured
continuation) concurrency costs; where it is not (lower acceptance, more
compute per token) concurrency *adds* until the GPU saturates. Both are real,
and which one you get depends on the traffic.

**Recommendation.** `serve.sh` defaults to `NP=4 CTK=CTV=q4_0 CRAM=4096` and
`INSTANCES=1`: four concurrent agent sessions at the RAM-first dtype, which is
the configuration the user chose once q4_0 made the KV cheap. Note the per-slot
cost is *not* the 2 GiB of KV — each slot also carries Metal compute and draft
buffers, and the measured all-in cost is ~5 GiB per extra slot, paired in one
session with `CRAM=4096`: `NP=2` 32.4 GiB free, `NP=3` 28.1, `NP=4` 22.6. So
`NP=2 ./serve.sh` buys 10 GiB back at half the concurrency, and `NP=1
CRAM=8192 ./serve.sh` is the minimal-footprint configuration.
For N concurrent clients: `NP=N` (one port, one process, cheapest memory, most
slots) or `INSTANCES=N` (N ports, separate processes, marginally better measured
per-stream at N=4 but half the headroom and a hard ceiling at 4). Pick slots for
capacity, processes for isolation.

### The N=4 gap is real but small, and it is confounded

The first pass at this table was one sample per arm, and it showed four
processes 26 % ahead of four slots. Two interleaved reps (`results/slots_vs_procs.jsonl`)
halve that and make the direction reproducible:

| N=4 | rep 1 | rep 2 | mean | total RSS | GPU (summed) |
|---|---|---|---|---|---|
| `-np 4` | 61.7 | 60.1 | **60.9** | 52.2 GiB | 57.8 GiB |
| 4 processes | 66.4 | 65.4 | **65.9** | 99.7 GiB | 104.3 GiB |

**+8 % aggregate for processes, for twice the memory and a ceiling of 4 instead
of 6+.** Within-mode spread is now ±3 %, so the difference is real; the earlier
26 % was drift.

It is also *confounded*, which matters for how you use it: four processes each
get their own `-t 4` CPU thread pool (16 threads total), while `-np 4` shares
one `-t 4` pool across all four slots. The measured 8 % may be CPU scheduling
for the sampling and SSM steps rather than anything about batching. Raising
`THREADS` for the slot configuration is the obvious test and was not run.

## The prefill cache — the knob that matters most for agentic use

An agent loop re-sends a transcript that grows every turn, so the cost that
dominates is *prefill*, and whether it can be skipped. It can. There are three
layers, and `scripts/agent_cache_probe.py` measures them via
`timings.prompt_n` — how many tokens the server actually had to process:

1. **Slot prefix reuse.** A request that shares a prefix with the slot's
   current KV prefills only the suffix.
2. **The server prompt cache in host RAM** (`--cache-ram`, default 8192 MiB).
   Holds *evicted* slot states, so a conversation can come back after another
   displaced it. This is the layer with a budget, and therefore the one that
   fails silently.
3. **`--slot-save-path`** persists a state to disk. Not used here.

Measured with ~30 000-token prompts, `NP=1`:

| sequence | what it tests | prompt_n | wall |
|---|---|---|---|
| A, then A again | slot prefix reuse | 29 772 -> **4** | 55.5 s -> **0.24 s** |
| A, then A + 19 tokens | **the agent turn** (append and re-send) | **19** | **0.41 s** |
| A, B, A (default 8192) | eviction + restore by the RAM cache | 3rd = **4** | 0.58 s |
| A, B, A (`--cache-ram 2048`) | the budget too small to hold both | 3rd = **29 772** | **57.1 s** |
| A, B, A (`--cache-ram 24576`, the former default) | the budget the f16 default used | 3rd = **4** | 0.50 s |

**A growing transcript costs its delta.** The second row is the whole agentic
case: 19 new tokens, 0.41 s, against 62.9 s to prefill the same context cold —
a 150x difference on every turn of every conversation.

**And the default budget is too small for more than one conversation.** A cache
entry costs the context's KV, 64 KiB/token at f16, so a 128k conversation is
~8 GiB and the 8192 MiB default holds exactly one. The fourth row is the failure
mode: with two 30k conversations and a 2 GiB budget, the third request
re-prefilled all 29 772 tokens and nothing in the response says so — a cache
miss and a first turn look identical from outside. `serve.sh` therefore ships
`CRAM=24576`, sized by `conversations x context-KV`: three 128k f16
conversations (two live slots plus one displaced), or many more when sessions
are shorter, since an entry is cut to the conversation's actual length. It is
host RAM, which is the resource this machine has most of; the cache only fills
as conversations arrive, and `CRAM=0` disables it.

**It is also a ceiling, and the budget has to absorb it.** Measured with four
idle slots and no traffic:

| config | host `used` | free | server's own footprint |
|---|---|---|---|
| idle (other tenants + OS) | 39.3 GiB | 58.2 | — |
| `NP=4`, f16 KV | 89.6 GiB | 15.4 | 50.3 GiB |
| `NP=4`, q8_0 KV (the former default) | 78.4 GiB | 24.4 | 39.1 GiB |
| `NP=2`, q8_0 KV | 62.1 GiB | 42.1 | 22.8 GiB |

A 128k conversation costs 8 GiB of cache at f16, ~4.3 GiB at q8_0 and ~2.1 GiB
at q4_0 (64 KiB/token and down; the load footprints above confirm the f16 figure
— `NP=4` f16 is weights 14.3 + 4x8 KV + ~4 compute = 50.3). So `CRAM=65536` at
f16 would commit 50 + 64 = 114 GiB on top of the 39 GiB other tenants hold, i.e.
153 GiB on a 128 GB machine: it does not fit. **The shipped default is now `q4_0/q4_0` with
`CRAM=4096` at `NP=4`**, chosen by the user after the f16 default (`NP=2`, 128k,
`CRAM=24576`) measured 106.2 GiB used / 21.5 GiB free while serving and left
less room than local Kubernetes clusters need. The q4_0 four-slot default
measures 105.0 / 22.6, and drops to 95.9 / 32.4 at `NP=2` and 99.8 / 28.1 at
`NP=3`; the cache ceiling falls from 24 GiB to 4 at the same time. At q4_0 those
4 GiB still hold one full 128k conversation or about ten 30k ones, because
entries are cut to the conversation's actual length; the dtype change costs
decode speed (26.2 against 30.7 at `q8_0` and 46.1 at `f16` t/s at 98.6k), not
capacity, and the perplexity gate at 65 536 tokens is inside one standard error.
See "KV dtype" above.

**And `CRAM=24576` covers the fleet it is sized for.** Six *independent*
conversations (non-overlapping 42k-char slices of the corpus, ~13.5k tokens
each, so no shared prefixes) pushed through four slots:

| | first send | re-send |
|---|---|---|
| A (displaced twice) | 26 s, prompt_n 13 549 | **0.99 s, prompt_n 4** |
| B (displaced twice) | 27 s, prompt_n 14 081 | **0.68 s, prompt_n 4** |
| C, D (live slots) | 26-27 s | **1.12 s / 0.75 s, prompt_n 4** |
| E, F (live slots) | 25-27 s | **0.65 s / 0.75 s, prompt_n 4** |

Four live slots plus two displaced, all six warm; host usage grew 78.4 -> 82.5
GiB, i.e. the six entries cost ~4 GiB in total because entries are cut to the
conversation's actual length (13.5k tokens ~ 0.6 GiB at q8_0), not to the 128k
slot size.

Two things this also corrected:

* **Prefix reuse is partial, not exact.** A first attempt used *nested*
  truncations of one document, and the re-sends came back with prompt_n of
  3840, 3122, 2578, 1827, 1034, 4 — each exactly the tail beyond the resident
  prompt's shared prefix, i.e. the server reused whatever prefix the slot or
  cache happened to hold rather than falling back to a full prefill. That is
  the agent case working from the other direction: a turn costs its delta even
  when the slot holds a related-but-different transcript.
* The same run is why the "cache that is one entry short" warning above is
  about *independent* conversations, which is what N agent sessions are.

## Disk-backed slot states (SLOT_SAVE) — works, but does not pay off yet

The third cache tier persists a slot's KV to a file. `serve.sh` can pass
`--slot-save-path` via `SLOT_SAVE=<dir>`; it is **off by default** and should
stay off, because the round trip was measured and does not save a prefill.

A ~30k-token (1.11 GiB) conversation, `CRAM=0` so the RAM cache cannot help:

| step | result |
|---|---|
| `POST /slots/0?action=save` `{"filename":"a.bin"}` | n_saved 29 775, 1.11 GiB written, **346 ms** |
| `POST /slots/0?action=restore` `{"filename":"a.bin"}` | n_restored 29 775, **112 ms** |
| `GET /slots` immediately after the restore | `n_prompt_tokens = None` |
| re-sending that conversation (also with `id_slot: 0`) | **29 772 tokens prefilled, 63.4 s** |

Save and restore both work and are fast — 1.11 GiB round-tripped at ~3 GB/s, far
cheaper than the ~60 s re-prefill it is meant to replace. But the restored state
is not *matchable*: the slot reports no prompt after the restore, so the next
request reprocesses the whole transcript anyway. The KV buffers come back; the
slot's prompt bookkeeping does not. Also note the request must carry the
filename in the JSON *body* — the query string only takes `action=save|restore`,
and a GET returns 404.

Until that is resolved, the disk tier is plumbing, not a performance tier, and
`CRAM` is the only cache knob worth trusting.

## Concurrency and interference — the caveat behind the slot default

Measured with warm caches (so `prompt_n` is 4-123, i.e. no prefilling at all),
4k-token prompts, `NMAX=6`, decode-only:

| | per-stream | aggregate |
|---|---|---|
| 1 stream | 92.2 | 92.2 |
| 4 concurrent (4 slots) | 26.9, 26.9, 26.9, 25.9 | **106.6** |

**Concurrency added 16 % on code-like text.** That is the good case, and it is
the opposite of the README-continuation measurement earlier (where a single
stream was already GPU-saturated and 4 streams merely divided it). Which one you
get is a property of the workload, as with MTP.

The bad case is mixing a cold prefill with other slots' decode, and it is worse
than "shared bandwidth":

| N=2, one slot prefilling 30 000 tokens while the other decodes 128 | decode t/s |
|---|---|
| decode alone (no other traffic) | 93.6 |
| **sibling slot prefilling, same process (`-np 2`)** | **3.5** |
| sibling *process* prefilling (`INSTANCES=2`, measured solo 83.4) | 22.3 |

**A slot that prefills starves every other slot in the same process: 93.6 ->
3.5 t/s, 27x.** The mechanism is continuous batching — all slots' tokens go into
one batch per step, so a 30 000-token prefill makes every decode step cost a
prefill step. Separate processes share the GPU but not the batch, so the same
event costs 3.7x instead of 27x.

This is why `INSTANCES` is not only about fault isolation and per-port config:
**if N agent sessions send *cold* transcripts at overlapping times, processes
degrade far more gracefully than slots.** Two things soften it in practice:

* An agent turn that appends to its transcript prefills only the delta (19
  tokens measured), so the starvation window only opens on a cache *miss* —
  i.e. when `CRAM` is too small for the number of live conversations, or on the
  first turn of each session. The cache budget and this effect compound.
* The starving prefill is itself fast (30 000 tokens in ~60 s at ~490 t/s).

The defaults are `NP=2 CTK=CTV=q4_0 CRAM=4096` for the common agentic case (N
similar sessions, transcripts cached, decode-dominated turns) and `INSTANCES` is
the answer when prefills overlap: `INSTANCES=2 NP=2` is the middle ground — two
batches, four conversations, half the memory of four processes.

## What was discarded

* **A local re-sweep of the Metal FA-vec kernel config.** `build/bin/ggml-metal-tuning`
  (vendored as `third_party/llama.cpp/tools/tuning`) tunes the flash-attention
  vec `(Q, NE)` selection per (dtype, head size, KV depth, batch width), and the
  table it writes into (`ggml/src/ggml-metal/ggml-metal-tuning.cpp`) already
  carries an `Apple10 - M5_MAX` segment — including `dk=dv=256` at `q4_0`,
  `q8_0` and `f16`, which is exactly this model's shape at the KV dtypes the
  server has shipped. Re-running
  it for that shape (`build/bin/ggml-metal-tuning fa-vec --dtype q8_0 --dk 256`,
  7.4 min) emitted two rows, both identical to the rows already in the table
  (`{-1,0} -> {1,4}` and `{-1,1} -> {1,4}`), so the kernel the server actually
  uses is already the tuned one for this device. The sweep also shows why
  nothing finer is on offer: per-cell anchor drift on this box runs 11–53 %
  (desktop GPU tenant plus thermals), and the no-harm rule refuses a config that
  is not faster at *every* batch width in a bucket. The cell that would have
  mattered — the MTP verify batch, where `ne01 = n_max + 1` — lands in bucket
  `ne01_b = 4`, where `Q1 NE4` wins at `ne01` 5, 6, 8 and 16 and only `ne01 = 7`
  prefers `Q2 NE4`, by 9 % against a 21 % noise floor. Rejected, correctly.
* **`-ngl 0` took 830 s** for one arm (2 reps of pp512 + tg128 on 18 CPU
  cores, with the model streaming from page cache). It is kept in the table
  because it is the floor, but it dominates the wall clock of the `ngl` stage
  and nothing else in this file was measured at that speed.
* **The `depth` stage's `-p`/`-n` duplication.** `bench.py` applies the base
  `-p 512 -n 128` and the `depth` stage adds `-p 128 -n 128` to the same
  command line, and llama-bench treats repeated flags as a set of tests. Every
  depth row therefore contains four measurements, two of them duplicates. The
  data is consistent, the duplication is wasted time, and the fix is to pass
  `--pp 128 --tg 128` when sweeping depth.
* **`ioreg` GPU baselines are noisy.** Idle accelerator memory readings drifted
  18.2 → 498.2 → 831.3 MiB across three consecutive runs with nothing loaded —
  a 46x spread — so `bench.py`'s `gpu_mem_delta_mib` is reported but not used
  to make any decision. Peaks are stable; baselines are not.

## Traps this port had to solve

**1. `host_statistics(HOST_CPU_LOAD_INFO)` reports an idle machine as 100 %
busy.** The obvious `/proc/stat` replacement reads cumulative ticks per CPU
state and computes `busy = total - idle`. On this kernel the *idle* counter
never advances: sampled twice two seconds apart, user grew 1890 and system
1721 ticks while idle grew 0. A busy fraction computed from those counters is
1.0 by construction, and the first version of `guard.py` reported
`cpu_busy_pct = 100.0` while the box was asleep. `guard.py` now sums
`ps -o time=` across all processes — CPU *seconds*, no tick rate, no state
enumeration — and normalizes by `cores x wall`. A single spinning process
reads 1 core, which was verified against the process's own
`time.process_time()`.

**2. The contended resource is the GPU, and nothing about our process reveals
that.** With `-ngl 99` the CPU is nearly idle during generation; the 12-core
"saturation" test above still moved the needle only through prefill. What
actually steals throughput is the desktop compositor and anything else on the
GPU: `Device Utilization %` reads **24 % on an idle machine** with no model
loaded. So `wait_idle.py --gpu` gates on a rise *above* a sampled baseline
rather than an absolute threshold (an absolute 5 % threshold can never be met
here), and every bench record carries `gpu_busy_pct_peak`.

**3. Unified memory breaks the old accounting.** "In use system memory" counts
the model's mapped weights, the KV cache and the desktop's own allocations as
one number, and it lags process exit, so the Intel habit of reading a client
counter before and after is not portable. Memory decisions here are made from
*peaks* and from the *difference* between N instances, not from deltas against
an idle baseline.

**4. `ps` was trusted only after cross-checking.** `ps -o time=` agreeing with
a child's own `time.process_time()` (0.47 s vs 0.66 s on the same interval) is
what proved the CPU number was right and the process genuinely starved; the
mach-API result was the outlier, not `ps`.
