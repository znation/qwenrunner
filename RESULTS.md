# RESULTS — Apple M5 Max, 128 GB

Measured evidence for the `mbp_m5_max` configuration. Every number here comes
from a run recorded under `results/`; the table rows are the CSV exports of
`scripts/bench.py`, `scripts/mtp_ab.py` and `scripts/parallel_ab.py`.

> `results/` and `logs/` also still contain the Intel host's measurement files
> (`bench_*.jsonl`, `qc_*`, `mtp_q2.jsonl`, …). None of them were produced on
> this machine and none of them are referenced below; the files created by this
> branch are `bench_shallow.*`, `depth_f16.*`, `depth_q8.*`, `mtp_m5.jsonl`,
> `parallel.jsonl` and `parallel_hi.jsonl`.

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
penalty is larger and the optimum lower. `serve.sh` defaults to the super-core
count (6) and the sweep says 4 is marginally better; anywhere in 4–12 is fine,
18 is not.

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

Not measured: MTP at 128k *depth* (this row is at ~4k), and MTP with a
quantized KV cache.

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
not averages. `serve.sh` defaults to `NMAX=3` — the upstream default and the
setting that is never a regression on either workload — and `NMAX=6` is a
documented win for code/agent traffic, which is what this repo is built to
serve. `MTP=0` is the setting for prose.

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

**Recommendation.** `serve.sh` defaults to `NP=1` and `INSTANCES=1`, because
each slot reserves its full KV up front (8 GiB per 128k slot) whether or not a
client is connected — paying 32 GiB for four idle slots is the wrong default for
one user. For N concurrent clients: `NP=N` (one port, one process, cheapest
memory, most slots) or `INSTANCES=N` (N ports, separate processes, marginally
better measured per-stream at N=4 but half the headroom and a hard ceiling at
4). Pick slots for capacity, processes for isolation.

## What was discarded

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
