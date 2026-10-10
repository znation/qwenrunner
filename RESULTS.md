# qwenrunner — measured results

All numbers were measured on this host, serially, through `scripts/guard.py`
(hard VRAM cap enforced by measurement). Nothing is extrapolated unless it says
so.

Sections marked **[clean]** were taken while the host was idle
(`bg_cpu_pct` < 10 %). Anything measured under load is marked **[loaded]** and
its absolute rates should not be quoted — see "Method" for why that distinction
carries most of the weight in this file.

## Host

| | |
|---|---|
| CPU | Intel i9-9900K, 8c/16t |
| RAM | 39 GiB, **swap 0** |
| GPU | AMD Vega 10 [Radeon Pro Vega 48], `03:00.0`, `amdgpu` |
| GPU compute path | Vulkan / RADV (Mesa 25.0.7), apiVersion 1.4.305 — **no ROCm/HIP** |
| VRAM total | 8176 MiB (shared with the desktop compositor) |
| `RLIMIT_MEMLOCK` | 8 MiB soft **and** hard — cannot be raised unprivileged |

## Build

llama.cpp pinned at `aa5e0092fdb8980b9e89498dfb73049b4fe204c2`, built with
Vulkan, `GGML_NATIVE=ON`, `LLAMA_CURL=OFF`. `scripts/build.sh` reproduces it.
The pinned checkout is never patched; the one missing SPIRV include is worked
around caller-side with `-isystem tools/local/usr/include`.

`glslc` + `libshaderc.so.1` + `spirv-headers` were extracted from Debian
packages into `tools/local/`.

## Method: what makes a measurement here trustworthy

This host cannot be benchmarked naively, and learning that was most of the work.

* **Throughput is CPU-bound.** At `-ngl 10`, 54 of 64 layers still run on the
  CPU, so background load scales throughput directly. Every result carries
  `cpu_busy_pct` (all cores, benchmark included) and `bg_cpu_pct` (all cores
  *excluding* the guarded process tree).
* **Contended numbers here are wrong by 3–8x, not by percent.** Repeats of the
  *identical* configuration spanned 4.6x, and a background indexer holding
  3–10 of 16 cores depressed generation 6x. Six conclusions in earlier drafts of
  this file were load artefacts and are corrected below with the clean figure.
* **A just-exited arm's DRM client lingers for seconds.** Sampling the baseline
  in that window made a whole stage look already over the cap and `guard.py`
  killed every arm at 0 s. `bench.py` now settles VRAM before every *arm* and
  every *stage*, requiring the reading to be stable *and* inside the cap.
* **Gate on `python3 scripts/wait_idle.py --system --idle-cores 3 --hold 30`.**
  A process-name gate is not enough — the load moved from one named job to
  another during this work.
* **RAM, not VRAM, is the other constraint.** `/tmp` was a 20 GB tmpfs filled to
  19 GB by other sessions; four back-to-back model loads OOM-killed a run. Run
  one load at a time and prefer `-lm dio`, which peaks at ~300 MiB RSS where the
  mmap modes hold 13–14 GB.

### Corrections made from clean data

| earlier claim (loaded) | clean measurement |
|---|---|
| generation collapses with depth — 0.18 t/s at 8192 | **1.53 t/s**; the 0.18 was a 6x load artefact |
| `-t 12` halves generation | **1.99 t/s**, equal to `-t 8` |
| `-t 16` collapses to 0.14 t/s | **1.86 t/s** — ~8 % worse, not catastrophic |
| prompt saturates at `-ngl 8` | flat to rising: **74.41 at 10** vs 74.00 at 8 |
| `-lm` non-mmap modes are ~60 % faster | **~12 % faster** (74.55 vs 66.32) |
| `-b 1024 -ub 256` beats the default | **default 2048/512 wins** (70.26 vs 59.09) |
| `-nkvo 1` costs 28 % generation | **no measurable difference** |
| MTP +55.6 % / −12 % / +120 % | **+97.8 %**, from an idle interleaved A/B |

## VRAM policy

Two different numbers, never subtracted across scopes:

* **`vram_peak_mib` (system counter)** — the cap source of truth, must stay
  ≤ 5000 MiB. Includes the desktop, whose baseline drifts (measured 746–2188 MiB
  within one hour, traced to Firefox video and `claude-desktop`). Read the
  printed baseline before interpreting any peak.
* **`client_vram_peak_mib` (per-DRM-client sum)** — the model's own footprint.
  The guarded child has no DRM client before it execs, so this baseline is zero
  by construction and the peak *is* the footprint. **This is the number to use
  for the `-ngl` rule** — it is baseline-independent and reproduced to within
  5 MiB across many runs.

### The safe `-ngl` is a function of the live desktop baseline

Client peaks, measured at `-d 0`, `-p 512 -n 128`:

| `-ngl` | client MiB | fits while baseline <= |
|---|---|---|
| 2 | 1522.2 | ~3480 MiB |
| 4 | 2049.7 | ~2950 MiB |
| 6 | 2551.2 | ~2450 MiB |
| 8 | 3039.4 | ~1960 MiB |
| 10 | 3540.9 | ~1460 MiB |
| 12 | 4014.6 | ~985 MiB |

**~252 MiB of client VRAM per offloaded layer** from `-ngl` 2 up; the first two
layers are nearly free (1522.1 → 1522.2), so never extrapolate below 2.
`-ngl 14` (~4530 MiB client) cannot fit at any baseline observed here.

Verified both ways: `-ngl 12` ran clean at a 765 MiB baseline (peak 4824.6 MiB)
and breached at 1113.8 MiB (peak 5117.6 MiB, killed 5 s in). **Do not hard-code
an `-ngl`** — read the printed baseline and let `guard.py` be the backstop.

## Correctness smoke (CPU, `-ngl 0`)

```
build/bin/llama-completion -m models/Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 0 -no-cnv \
    -t 8 -c 512 --seed 42 -p "The capital of France is" -n 24
```
→ `The capital of France is Paris. The capital of Germany is Berlin. The
capital of the United States is Washington, D.C. The capital of`
(`exit_code 0`, 42.0 s wall, RSS peak 15.8 GiB, no cap violation).

## Model load mode **[clean]**

`results/cl_load.jsonl`, `-ngl 10 -d 0`, baseline 788.7 MiB.

| `-lm` | pp t/s | gen t/s | RSS peak MiB | client MiB |
|---|---|---|---|---|
| `auto` (default) | 67.06 | 1.99 | 12 861 | 3540.8 |
| `mmap` | 66.32 | 1.97 | 14 391 | 3540.8 |
| `none` | 74.35 | 1.97 | **296** | 3540.9 |
| `mlock` | — | — | 254 | ✗ abort, exit −6 |
| **`dio`** | **74.55** | 1.97 | **339** | 3540.9 |

Conclusions:

* **`-lm dio` is the best mode, `-lm none` a close second; both beat the default
  by ~12 % on prompt throughput.** They are also **~40x lighter**: a `-lm dio`
  arm peaks at ~300 MiB RSS and commits nothing (`MemAvailable` *rose* 422 MiB
  across a re-verified arm), where `auto`/`mmap` hold 12.9–14.4 GB resident.
* Generation is unaffected by load mode (1.97–1.99 across all four).
* **`-lm mlock` is unavailable and unnecessary.** `RLIMIT_MEMLOCK` is 8 MiB soft
  and hard; llama.cpp aborts rather than run unlocked. Raising it needs a
  privileged change and buys nothing — `mlock` pins a *mapping* so its pages
  cannot be reclaimed, but `none`/`dio` already avoid the reclaimable-mapping
  problem, and swap is 0 so there is nothing to swap to.

## `-ngl` **[clean]**

`results/dio9_ngl.jsonl`, two samples each, `-lm dio`, `-d 0 -p 512 -n 128 -r 1`,
baselines 785–789 MiB, `bg_cpu_pct` 1.1–1.7 %.

| `-ngl` | pp t/s (A / B) | gen t/s (A / B) | client MiB |
|---|---|---|---|
| 8 | 73.35 / 74.00 | 1.927 / 1.942 | 3039.4 |
| **10** | 73.67 / **74.41** | **1.996** / **2.008** | 3540.9 |

Conclusions:

* **`-ngl 10` beats 8 on both axes, 2 of 2** — ~1 % prompt, ~3.5 % generation.
  Take the largest `-ngl` the live baseline allows.
* **Offload does not saturate.** The low rungs, from the loaded ladder
  (`results/clean_ngl.jsonl`, `bg_cpu_pct` 24–51 %): `-ngl 0` 28.10 pp / 0.325
  gen, `-ngl 2` 41.19 / 1.313. `-ngl 0` is **not** a zero-VRAM configuration on
  Vulkan — 1522.1 MiB of client VRAM before a single layer is offloaded.
* The loaded ladder showed a non-monotonic prompt curve (peak at 4–6, *falling*
  at 8–10) that vanishes on an idle host. That curve was the artefact that
  motivated building the idle gate.

## Thread count **[clean]**

`results/cl_threads.jsonl`, `-ngl 10 -d 0 -p 512 -n 128 -r 1 -lm dio`,
baseline 792.8 MiB.

| `-t` | pp t/s | gen t/s |
|---|---|---|
| 4 | 74.44 | 2.03 |
| 6 | 74.25 | **2.10** |
| **8** | 74.18 | 1.98 |
| 12 | **74.44** | 1.99 |
| 16 | 68.32 | 1.86 |

Conclusions:

* **`-t 4` through `-t 12` are equivalent** (74.18–74.44 pp, 1.98–2.10 gen) with
  no consistent ordering. Keep the default `-t 8`.
* **`-t 16` is mildly worse** (−8 % prompt, −6 % generation). Do not raise the
  thread count above the physical core count.
* The earlier "`-t 12` halves generation / `-t 16` collapses to 0.14 t/s" was
  load, not configuration.

## KV cache dtype **[clean]**

`results/dio9_kv.jsonl` at depth 8192 and `results/cl_kv0.jsonl` at depth 0, both
`-ngl 10 -p 512 -n 128 -r 1 -lm dio`, baselines 785.5 / 794.8 MiB.

Depth 0 — **all five are equivalent** (spread ~3 %, within noise):

| ctk/ctv | pp t/s | gen t/s | client MiB |
|---|---|---|---|
| f16/f16 | 72.21 | 1.802 | 3540.9 |
| q8_0/f16 | 71.20 | 1.802 | 3540.2 |
| q8_0/q8_0 | 72.40 | 1.796 | 3541.2 |
| q4_0/q4_0 | 72.50 | 1.800 | 3538.2 |
| bf16/bf16 | 73.36 | 1.848 | 3540.9 |

Depth 8192 — **quantizing now pays**:

| ctk/ctv | pp t/s | gen t/s | client MiB |
|---|---|---|---|
| f16/f16 | 66.35 | 1.579 | 3605.3 |
| q8_0/f16 | 64.24 | 1.699 | 3589.4 |
| **q8_0/q8_0** | **66.72** | 1.685 | 3607.4 |
| **q4_0/q4_0** | 61.56 | **1.730** | 3556.4 |
| bf16/bf16 | 65.75 | 1.632 | 3605.3 |

Conclusions:

* **KV dtype is free at shallow context and worth ~9 % generation at depth.** At
  depth 0 the five dtypes are indistinguishable; at depth 8192 `q4_0/q4_0`
  generates 1.730 against f16's 1.579 (**+9.6 %**) for 7 % less prompt, and
  `q8_0/q8_0` is the balanced choice (best prompt, 1.685 gen).
* **For a generation-bound workload, use `q4_0/q4_0` with `-fa on`** — it costs
  nothing shallow and wins deep.
* **KV dtype barely moves VRAM** (±50 MiB across all ten arms). Only 16 of the 64
  layers carry a KV cache — the rest are Gated DeltaNet linear-attention layers —
  and only layers 3 and 7 fall inside `-ngl 10`.
* `-fa on` is **mandatory** with a quantized V cache; `-fa off` aborts at
  context creation with `quantized V cache requires flash_attn to be enabled`
  (exit 1, ~4 s, no rows).

## Flash attention at depth 8192 **[clean]**

`results/cl_fa.jsonl`, pinned to `q4_0/q4_0`, `-ngl 10 -d 8192 -lm dio`:
`-fa on` → 61.79 pp / 1.77 gen, client 3556.4, 214 s.
`-fa off` → exit 1 at 4 s (above), a flag-validation abort, not a crash.

## Generation cost vs context depth **[clean]**

`results/cl_depth.jsonl`, `-ngl 10 -r 1 -lm dio`, baseline 785.5 MiB.

| depth | pp t/s (p512) | pp t/s (p128) | gen t/s | client MiB |
|---|---|---|---|---|
| 0 | **74.46** | 47.65 | **2.00** | 3540.9 |
| 2048 | 71.76 | 45.24 | 1.92 | 3557.3 |
| 4096 | 69.97 | 44.07 | 1.83 | 3573.3 |
| 8192 | 66.29 | 42.07 | 1.53 | 3605.3 |
| 16384 | 59.17 | 38.18 | 1.20 | 3669.3 |
| 32768 | 50.70 | 33.49 | 0.85 | 3797.4 |

Conclusions:

* **Depth degrades throughput gracefully and monotonically** — generation
  2.00 → 1.53 at 8k (−24 %) → 1.20 at 16k (−40 %) → 0.85 at 32k (−58 %); prompt
  −32 % at 32k. There is no cliff.
* **`-d 32768` fits at `-ngl 10` with `-lm dio`** (peak 4639.2 MiB at a 785.5 MiB
  baseline, client 3797.4). Earlier drafts had it breaching, under a heavier
  desktop and the mmap load mode.
* **The KV cache is not the VRAM problem at depth.** Client VRAM grows only
  3540.9 → 3797.4 MiB from depth 0 → 32768 (+257 MiB) because only 16 layers
  have a KV cache.
* `-d/--n-depth` is the only context-size knob in `llama-bench`; `-o json` does
  not echo `n_ctx`, so read `n_depth` from the rows.

## KV offload and op offload **[clean]**

`results/dio8_nkvo.jsonl`, `results/dio8_nopo.jsonl`, `-ngl 8 -d 0 -lm dio`,
`bg_cpu_pct` 0.8–8.6 %.

| flag | pp t/s | gen t/s |
|---|---|---|
| `-nkvo 0` (default) | 74.12 | 1.773 |
| `-nkvo 1` | 74.02 | 1.748 |
| `-nopo 0` (default) | 74.02 | 1.879 |
| `-nopo 1` | **12.22** | 1.930 |

* **`-nkvo` makes no measurable difference** — keep the default. An earlier
  claim that `-nkvo 1` cost 28 % generation was load.
* **`-nopo 1` is catastrophic** — 12.22 t/s prompt, an 84 % collapse. Never use
  it.

## Batch size at depth 4096 **[clean]**

`results/dio9_batch.jsonl`, `-ngl 10 -d 4096 -lm dio`, baseline 785.5 MiB.

| `-b`/`-ub` | pp t/s | gen t/s | client MiB |
|---|---|---|---|
| **2048/512 (default)** | **70.26** | 1.819 | 3573.3 |
| 1024/256 | 59.09 | 1.816 | 3310.7 |
| 512/128 | 45.06 | 1.808 | 3180.6 |
| 256/64 | 29.48 | 1.837 | 3168.4 |

Conclusions:

* **The llama-server default is the fastest.** Prompt throughput falls
  monotonically with the micro-batch (70.26 → 29.48) while generation is flat
  ~1.82. Smaller micro-batches buy VRAM (3573 → 3168 MiB client) and nothing
  else — lower them only if the cap is tight.
* An earlier draft had 1024/256 winning; that was the default arm running at
  55.5 % background load against 33.8 % for its rival.

## Context-shape reference **[clean]**

`results/cl_ctx.jsonl`, `-ngl 10 -lm dio`, baseline 785.5 MiB.

| shape | pp t/s | gen t/s |
|---|---|---|
| pp512 / tg128 | 74.49 | 1.98 |
| pp2048 / tg128 | 73.27 | 1.95 |
| pp128 / tg512 | 47.55 | 1.95 |
| tg128 only | — | 1.99 |
| pp512 only | 74.53 | — |

Prompt throughput is flat in prompt *length* (74.49 at 512 vs 73.27 at 2048) and
generation is flat in generation *length*. What costs is accumulated context
depth — see the depth ladder above.

## Quant comparison **[clean]**

A smaller quant has smaller layers, so it can offload more before hitting the
cap — and this workload wants offload. Measured with `scripts/quant_compare.sh`,
all `-lm dio`, baselines ~800–850 MiB.

### Offload ceiling

Client VRAM per layer, from a `-ngl` ladder per quant
(`results/qc_ladder.jsonl`, `results/qc_hi.jsonl`):

| quant | file size | MiB per layer | max `-ngl` | client at max |
|---|---|---|---|---|
| UD-Q4_K_M | 16.46 GB | 243 | 12 | 4015.1 |
| UD-Q3_K_XL | 13.15 GB | 194 | 16 | 4114.6 |
| UD-IQ3_S | 12.04 GB | 182 | 16 | 3968.3 |
| UD-Q2_K_XL | 9.83 GB | **155** | **20** | 3929.3 |

### Throughput at each quant's own ceiling

`-d 0 -p 512 -n 128 -r 1 -lm dio`:

| quant | `-ngl` | pp t/s | gen t/s |
|---|---|---|---|
| UD-Q4_K_M | 12 | 74.73 | 1.88 |
| UD-Q3_K_XL | 16 | 77.93 | 2.14 |
| UD-IQ3_S | 16 | 78.16 | 2.18 |
| **UD-Q2_K_XL** | **20** | **80.79** | **2.86** |

For Q2_K_XL the three rungs give 80.62 / 2.65, 80.97 / 2.75 and 80.79 / **2.86**
at `-ngl` 16 / 18 / 20 — generation rises monotonically with offload. **A smaller
quant is worth +52 % generation**, which is not a marginal lever.

### Quality — and why the first corpus was not good enough

| quant | PPL on `data/ppl_corpus.txt` (184 k tokens) | Δ | PPL on `data/eval_corpus.txt` (3 k tokens) | Δ |
|---|---|---|---|---|
| UD-Q4_K_M | **3.9057** | — | 4.0136 | — |
| UD-Q3_K_XL | 4.4392 | **+13.7 %** | 4.1417 | +3.2 % |
| UD-IQ3_S | 4.8361 | **+23.8 %** | 4.1614 | +3.7 % |
| UD-Q2_K_XL | 6.2984 | **+61.3 %** | 4.3511 | +8.4 % |

`logs/ppl-*.log` (6 chunks) and `logs/pplbig-*.log` (40 chunks).

Conclusions:

* **UD-Q4_K_M is the right choice, but not for the reason first assumed.** The
  speed gain from a smaller quant is large (+52 % generation), not marginal — the
  quality cost is simply larger still. Q2_K_XL buys 52 % speed for **+61 %
  perplexity**.
* **`data/eval_corpus.txt` is the llama.cpp README**, and perplexity over 6
  chunks of it understated every gap by 3–7×: Q2_K_XL read as +8.4 % there and
  +61.3 % on 184 k tokens of prose. A ±3 % standard error cannot resolve a
  +3.2 % gap, and a narrow technical domain flatters every quant. Use the larger
  corpus — `data/ppl_corpus.txt` is Project Gutenberg 1342, sha256 `81300b79…`.
* **Perplexity is load-insensitive**, making these the most trustworthy numbers
  in this file: they were taken while other stages' throughput was still being
  depressed by host load.
* The intermediate quants are not compelling either: Q3_K_XL +13.7 % PPL for
  +14 % generation; IQ3_S +23.8 % for +16 %.

## MTP (multi-token prediction) speculation **[clean]**

`results/final_mtp.jsonl`, `-ngl 8 -c 8192 -t 8 -np 1`, 5189-token prompt,
`--n-predict 64`, `--server-extra "-lm dio -b 2048 -ub 512"`, interleaved
`--reps 2` so arm order cannot confound, baseline 788.7 MiB.

| arm | pp t/s | **gen t/s** | `draft_n` / accepted | client MiB | load s |
|---|---|---|---|---|---|
| `base` r0 | 67.61 | 1.7189 | — | 2793.9 | 6.0 |
| **`mtp-sidecar`** r0 | 62.42 | **3.3989** | 48 / 47 | 3291.6 | 7.0 |
| `base` r1 | 68.53 | 1.7189 | — | 2793.9 | 6.0 |
| **`mtp-sidecar`** r1 | 63.10 | **3.4018** | 48 / 47 | 3291.2 | 7.0 |

Conclusions:

* **MTP nearly doubles generation: 1.7189 → 3.4004 t/s, +97.8 %**, winning
  **2 of 2** interleaved pairs. `base` measured *bit-identical* (1.7189) across
  both reps, so this is the most trustworthy number in the file.
* **Cost: ~8 % prompt throughput, +498 MiB client VRAM, ~17 % longer load.**
  Worth it for generation-dominated workloads; not for long prompts.
* **`--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 0` is the
  configuration to use.** `-md` is *not* mandatory on qwen35 — `mtp-nomd`
  (`--spec-type draft-mtp` alone) also drafts, because UD-Q4_K_M already carries
  the four `blk.64.nextn.*` tensors — but the sidecar is the higher-precision
  replacement. `base`'s loader log lists all four as `unused tensor … ignoring`;
  the speculative arms do not.
* **`--ngld 0` is required.** The default `-1` (auto) offloads the 1.37 GB draft
  file to VRAM and breaches the cap. With `-ngld 0` the draft head runs on the
  CPU — fine at `-ngl 8`, and the reason the sidecar will **not** fit at
  `-ngl 10`: client 3787–3809 MiB, breached twice in `results/clean_mtp.jsonl`.
* **The default `--spec-draft-n-max` (3) is optimal.** Drafting further ahead
  costs more than it gains — acceptance falls and the draft head does more work:

  | `--spec-draft-n-max` | gen t/s | draft / accepted | acceptance |
  |---|---|---|---|
  | **3 (default)** | **3.4004** | 48 / 47 | 98 % |
  | 5 | 3.283 | 53 / 52 | 98 % |
  | 8 | 3.242 | 59 / 55 | 93 % |

  (`results/mtp_nmax.jsonl`; the `n-max 5` pair was unstable at 3.051/3.515, the
  others reproduced to <1 %.) Leave the flag alone.
* **MTP's benefit is configuration-dependent — it is not additive with a smaller
  quant.** The same interleaved A/B run against `UD-Q2_K_XL` at `-ngl 16` gives
  only **+3.6 %**, not ~2x:

  | trunk | `-ngl` | base gen t/s | sidecar gen t/s | delta |
  |---|---|---|---|---|
  | UD-Q4_K_M | 8 | 1.7189 | **3.4004** | **+97.8 %** |
  | UD-Q2_K_XL | 16 | 1.867 | 1.935 | +3.6 % |

  (`results/mtp_q2.jsonl`.) Speculation still works — 47/48 accepted — so this is
  not an acceptance failure. With `-ngld 0` the draft head runs on the CPU: at
  `-ngl 8` the target is so CPU-bound that drafting ahead pays, whereas at
  `-ngl 16` the target is fast enough that the CPU draft becomes the bottleneck
  and there is nothing left to win. **MTP is a `-ngl 8` / Q4_K_M win.**
* **The fastest generation configuration measured is therefore Q4_K_M + sidecar
  at `-ngl 8` (3.4004 t/s)** — ahead of Q2_K_XL at `-ngl 16` with or without MTP
  (1.867 / 1.935 t/s) at the same prompt shape.
* **History worth keeping:** across three earlier designs the figure came out as
  +56 %, −12 % and +120 %. Only the idle interleaved A/B is trustworthy.

## Serving (`llama-server`)

`scripts/serve.sh` runs `llama-server` under `guard.py`, which polls system-wide
VRAM every 50 ms and `SIGKILL`s the process group above `CAP` (default
5000 MiB), so the display GPU is protected by measurement rather than by
arithmetic. Defaults are the measured winners: `-ngl 8 -c 8192 -t 8 -lm dio
-b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on`. It forwards `SIGINT`/`SIGTERM` to
the child, so Ctrl-C actually stops the server.

```
./scripts/serve.sh                       # → http://127.0.0.1:8080
curl -sf localhost:8080/health           # ok after a 9 s load
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Smoke-tested end to end with that exact command line: `exit_code 0`,
`vram_violation: false`, peak **3553.3 MiB** of the 5000 MiB cap (client
2754.5 MiB, RSS 1189.4 MiB), and a clean teardown (`kill -TERM <guard>` leaves
`pgrep -c llama-server` at 0). The completion returns `predicted_n 16` with an
**empty `content`**: every token went to `reasoning_content`. Qwen-3.8-27B is a
thinking model — budget `max_tokens` for the reasoning trace or the visible
answer is truncated away.

`guard.py` forwarding signals is load-bearing: the child is started with
`start_new_session=True`, so before the fix a Ctrl-C on `serve.sh` killed only
the guard and **orphaned `llama-server` holding ~3 GB of the 5 GB budget**.

## Reproducing

```
./scripts/build.sh                                     # once
export PATH=$PWD/tools/local/usr/bin:$PATH
export LD_LIBRARY_PATH=$PWD/build/bin:$PWD/tools/local/usr/lib/x86_64-linux-gnu
python3 scripts/wait_idle.py --system --idle-cores 3 --hold 30
./scripts/bench.py --stage ngl     --ngl-ladder 8,10 --depth 0 --reps 1 --lm dio --out results/dio9_ngl.jsonl
./scripts/bench.py --stage kv      --ngl 10 --depth 8192 --reps 1 --lm dio --out results/dio9_kv.jsonl
./scripts/bench.py --stage threads --ngl 10 --depth 0 --reps 1 --lm dio --out results/cl_threads.jsonl
./scripts/bench.py --stage depth   --ngl 10 --reps 1 --lm dio --out results/cl_depth.jsonl
./scripts/bench.py --stage batch   --ngl 10 --depth 4096 --reps 1 --lm dio --out results/dio9_batch.jsonl
./scripts/bench.py --stage load    --ngl 10 --depth 0 --reps 1 --lm dio --out results/cl_load.jsonl
./scripts/mtp_ab.py --ngl 8 --ctx 8192 --n-predict 64 --arms base,mtp-sidecar \
    --ngld 0 --reps 2 --server-extra "-lm dio -b 2048 -ub 512" --out results/final_mtp.jsonl
./scripts/quant_compare.sh                              # quant ceiling, speed, perplexity
```

`bench.py` does not export `PATH`/`LD_LIBRARY_PATH` itself — set them on the same
command line, or the child `llama-bench` cannot find `libllama.so`.
`--out` is appended to, and `.csv`/`.md` siblings are derived from it — use a
distinct `--out` per stage, and delete stale ones first. `--dry-run` prints the
argv it would run; it does not load a model, so it cannot show the baseline.
`--ngl-ladder` overrides the built-in `NGL_LADDER` for `--stage ngl` (the rung
wins over a base `--ngl`, which that stage ignores).

**Measure on an idle host.** `bench.py` settles VRAM before every arm and stage,
and `wait_idle.py --system` gates on total host CPU, but neither can fix a busy
machine. Results taken under load are recorded with their `bg_cpu_pct` so they
can be discarded.

## Still open

* **`-lm mlock`** — blocked on `RLIMIT_MEMLOCK` (8 MiB, needs a privileged
  change) and argued to be redundant against `-lm dio`.
* **The `-ngl` 0–6 rungs** were last measured under load; their VRAM figures are
  sound but their rates are not.
