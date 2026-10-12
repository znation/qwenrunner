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
| MTP +29 % at the split (2.73 → 3.52) | **+38 %** with the draft on the Vega (4.509 → 6.224); MTP's win is content-dependent, +9 % to +132 % |
| compaction pays full price (a rewrite re-prefills everything) | **it resumes from a checkpoint, but only on the chat path** — the old probe used a raw `/completion` prompt, which carries no `message_delimiters`, so no mid-prompt checkpoint could exist. 6630 → **1943** tokens of re-prefill at `-cms 2048` on a chat-shaped transcript, and 6630 → 6647 (i.e. nothing) at the shipped `-cms 8192` on a 6.6k prompt |
| `ngram-simple` chained onto `draft-mtp` "adds nothing" | **it is a 22 % loss** (5.31 vs 6.81 t/s) on the content n-gram speculation exists for — output that is a verbatim copy of the context, 100 % of drafts accepted, 2 reps |
| MTP's upstream defaults (`-n-max 3`, `-p-min 0.00`) are neutral | **they cost 18 %** where drafts land; `-n-max 6 -p-min 0.5` is a strict win and `-n-max 14` is worth +29-45 % on high-acceptance content |

## VRAM policy

Two different numbers, never subtracted across scopes:

* **`vram_peak_mib` (system counter)** — the cap source of truth, must stay
  ≤ 5000 MiB. Includes the desktop, whose baseline drifts (measured 746–2188 MiB
  within one hour, traced to Firefox video and `claude-desktop`). Read the
  printed baseline before interpreting any peak. Attributed per DRM client at a
  1790 MiB baseline: **plasmashell 487, firefox-esr 331, claude-desktop 254,
  ghostty 250 MiB**, ~56 MiB of everything else. The split's display footprint
  moves in steps of ~343 MiB (one Vega layer — see "The split's display cost is a
  staircase"), so that is roughly one rung of headroom sitting in the compositor
  and two browsers. The desktop is a first-class tunable here, and `serve.sh`'s
  warning fires at 1216 MiB precisely because the shipped 28-layer config is
  sized against a light one.
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

### The split's display cost is a staircase, not per-layer

The `-ngl` 2-12 ladder above is the Vega-only path. With the eGPU split
(`-dev CUDA0,Vulkan0 -sm layer -ts 6,1`) the display footprint moves in *steps*,
because `-ts` hands the Vega whole layers at a 7:1 ratio, so the Vega holds about
`-ngl/7` of them. Measured at the shipped shape (`-c 131072 -np 2`, MTP with the
draft chain at 6, one run each, `results/ngl_staircase.jsonl`; the `safe baseline`
column is `5000 - client - 148`, where 148 MiB is the largest transient measured
above baseline+client when a real request runs):

| `-ngl` | client MiB | NVIDIA MiB | system peak | safe baseline ≤ |
|---|---|---|---|---|
| 24 | 2972.5 | 7169.3 | 4579.7 | ~1880 MiB |
| 26 | 2971.5 | 7689.3 | 4579.4 | ~1880 MiB |
| 28 | 3314.2 | 8013.3 | 4936.7 | ~1540 MiB |

* **24 and 26 cost the display exactly the same** (2972.5 vs 2971.5 MiB — both
  hold 3 trunk layers on the Vega) while 26 keeps two more layers on the 1070
  (7689 vs 7169 MiB). So when the display is tight **the rung to drop to is 26,
  not 24**: the same display relief for 2 layers offloaded instead of 4.
  `serve.sh`'s old advice, "Lower NGL= by 2-4", is wrong in one direction.
* **One Vega layer is worth ~343 MiB of the cap and the lever is 7 `-ngl` wide**,
  so the layer count is a blunt instrument for display pressure. The efficient
  dials are the KV pool (`-c`, `-np`), the draft's residency (1.3 GB on the Vega
  — see "MTP at the split") and the desktop itself (~1.4 GiB).
* **The default is safe only while the baseline is ≤ ~1540 MiB, and the observed
  drift is 746-2188 MiB.** At the 1483 MiB measured during this work it peaks
  4794.1 MiB — 206 MiB of margin — and at the 1808 MiB measured an hour earlier
  the same config would have peaked ~5120 and been killed. This is the failure
  `serve.sh` warns about rather than prevents.

### The draft chain spends the 1070's reserve

`n_outputs_max` scales with `--spec-draft-n-max`, so the chain length costs the
*1070*, not the display. At the shipped shape, one run each
(`results/ngl_staircase.jsonl`):

| `DNMAX` | client MiB | NVIDIA MiB | outcome |
|---|---|---|---|
| 3 (upstream) | 3282.7 | 7677.3 | loads; 28 layers is 2 under the 30-layer ceiling |
| **6 (shipped)** | 3314.2 | **8013.3** | loads; **179 MiB of 8192 left**, so 28 *is* the ceiling |
| 14 | — | — | **✗ fails to load** — `allocating 837.25 MiB on device 0: cudaMalloc failed` |

* **`-n-max 14` is a 32k option only.** It was measured at `-c 32768 -ngl 32`,
  where the compute buffer is smaller; at 128k it cannot allocate the pp compute
  buffer at all. The +29-45 % figure stands, but not at the shipped context.
* **At `DNMAX=6` the shipped 28 layers is the last startable rung, not two under
  it** — the chain consumed the reserve the formula was built to keep. That is
  why the safe baseline falls from ~1570 to ~1540 MiB even though the display
  client only grows by 31 MiB: the 1070 is the device that sets every ceiling
  here, and the chain ate 336 MiB of it.
* **What the reserve costs, measured** (`results/ngl_reserve_*.jsonl`, 128k,
  5189-token prompt, 64 tokens out, 2 reps each): `-ngl 26` measured pp 33.63 /
  tg 6.413 at `bg_cpu_pct` 1.4, `-ngl 28` measured pp 34.06 / tg 6.663 at 31.1.
  The two 26-layer reps bracket the background correction — 40.8 % background
  costs 4.3 % of generation — so 28 layers is worth roughly **+5 % prefill and
  +7 % generation**.
* **`-ngl 28` is a coin flip against the watchdog.** Two runs of the *same*
  command at the *same* baseline — 1457.4 and 1453.9 MiB, 3.5 MiB apart — peaked
  at **5053.3 MiB (`vram_violation`, killed at 0 s of generation)** and 4850.9.
  `-ngl 26` peaked 4529.1 and 4721.1: 280-470 MiB clear. And the shipped config
  is `-np 2`, which stacks a second slot's buffers on top of these `-np 1` arms,
  so its real margin is worse than the table above.
* **Hence 26 ships and 28 is opt-in** (`NGL=28 ./serve.sh`, on a light desktop).
  Verified at the new default: health 13 s, a completion returns, `exit_code 0`,
  `vram_violation: false`, peak **4428.5** of 5000, client 2971.5, NVIDIA 7689.3
  of 8192 — against 206 / 179 MiB of margin for the 28-layer version of the same
  config.

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

### The draft chain: `--spec-draft-n-max` and `--spec-draft-p-min` **[clean]**

llama.cpp's MTP defaults are `--spec-draft-n-max 3` and `--spec-draft-p-min
0.00` — the chain stops on *length*, never on confidence — and neither had been
swept on this host. Measured at `-ngl 32 -c 32768 -ts 6,1 -np 1` (a real
`serve.sh` rung, 4 layers under the measured 36-layer 32k ceiling; chosen over
the shipped 128k/28 because the desktop was holding ~1.6 GiB of the Vega and
would have left the shipped config ~16 MiB of cap margin), `-lm dio -b 2048 -ub
512 -ctk q4_0 -ctv q4_0 -fa on`, `--n-predict 64`, interleaved reps,
`results/draft_chain_*.jsonl` + `results/draft_chain2_*.jsonl`. Every row is
`gen t/s (bg_cpu_pct)`; `bg_cpu_pct` is quoted because it is what decides
whether two arms are comparable:

| arm | feature-list continuation (46/48 accepted) | mid-document (32/90 = 26 %) |
|---|---|---|
| `-n-max 3` (upstream default) | 5.98 (7.6) / 5.33 (9.5) | 3.41 (8.8) / 3.67 (3.9) |
| `-n-max 6` | 7.05 (1.4) | 2.64 (1.5) |
| `-n-max 6 -p-min 0.5` | 7.08 (2.8) / 7.21 (6.7) | 3.63 (1.4) |
| `-n-max 10 -p-min 0.5` | 6.07 (19.6) / 7.17 (12.9) | 2.95 (13.2) / 3.08 (9.3) |
| `-n-max 14 -p-min 0.5` | **7.71 (24.7) / 7.76 (9.2)** | not measured |

* **`-n-max 6 -p-min 0.5` is a strict win over the default: +18 % where drafts
  land, no loss where they do not.** It is what `serve.sh` now ships.
* **Chain length is the lever; `-p-min` is not an acceptance proxy.** On the
  26 %-acceptance prompt `-n-max 6` alone costs **−25 %** (2.64 vs 3.41/3.67) —
  the chain keeps drafting tokens the target then rejects, and on a pipelined
  split every rejected position is still a full pass. `-p-min 0.5` removes that
  loss (3.63) but cannot remove it *further*: at `-n-max 10` the same prompt
  still drafted 110 tokens for 38 accepted (35 %), so the top draft token's
  probability stayed above 0.5 throughout. **Confidence gating and acceptance
  are different things on this model**, which is why the chain length has to be
  chosen for the content and `-p-min` is only a safety net.
* **Where acceptance is high the chain wants to be long, and the default leaves
  a lot on the table.** `-n-max 14` measures **+29-45 %** over `-n-max 3`
  (7.71/7.76 against 5.98/5.33), reproducibly (0.6 % spread across reps, and the
  *higher*-background rep is the faster one). The marginal economics say why:
  one extra draft step costs ~76 ms for ~0.96 accepted tokens — a marginal
  ~12.6 t/s against the 7.1 t/s the arm already runs at, i.e. still worth buying
  at 14. It is not the shipped default because it costs **−13 %** on the
  low-acceptance prompt (2.95/3.08 against 3.63), and a coding agent's turns
  vary in acceptance turn to turn.
* **None of this changes the output.** Greedy drafting plus target verification
  is exactly target-greedy decoding; `-n-max`/`-p-min` only decide how much work
  is spent trying. The measured `draft_n_accepted / draft_n` stays 96-100 % on
  the high-acceptance prompt across every arm.
* **Cost is negligible:** client VRAM 3026.4 MiB at `-n-max 3` against 3035.8 at
  6 and 3077.4 at 14 (+51 MiB, ~2 layers' worth of nothing — the extra output
  rows, not weights).

### `ngram-simple` is a loss even on verbatim output **[clean]**

`RESULTS.md`'s earlier verdict on n-gram speculation — "the ngram path found no
usable match on this prompt" — was measured on *prose*, which is the one content
an n-gram drafter cannot help with. Retested on the case it exists for: a
18292-char prompt containing all of `scripts/bench.py` with "return the complete
file verbatim", whose output *is* the file (`.content` begins
`#!/usr/bin/env python3`), `--n-predict 96`, `results/ngram_code_*.jsonl`:

| arm | pp t/s | gen t/s | drafts accepted/generated | `bg_cpu_pct` |
|---|---|---|---|---|
| MTP (`-n-max 3`) | 40.36 / 40.38 | **6.769 / 6.848** | 71/71 | 1.4 / 0.6 |
| MTP + `ngram-simple` | 40.37 / 40.43 | 5.308 / 5.311 | 91/91 | 0.8 / 0.6 |
| `ngram-simple` alone | 43.00 / 42.93 | 5.054 / 5.053 | 86/86 | 0.6 / 0.5 |

* **Slower, at 100 % acceptance, with 0.5 % reproducibility.** So the earlier
  "adds nothing" was not a content artefact — chaining `ngram-simple` onto
  `draft-mtp` is a **22 % loss** and n-gram speculation on its own is a **26 %
  loss** against MTP alone, on the most favourable content that exists for it.
* **The reason is the shape of the drafts, not their accuracy.** n-gram drafted
  86 tokens over 96 emitted in 10 target steps (8.6 accepted per step) where MTP
  drafted 71 in 25 steps (2.8 per step): n-gram buys *long* steps and pays for
  them — 1.9 s per step against 0.56 s. On a split whose CPU stage owns 33 of 65
  layers, every verified position costs real time, so the extra ~5 accepted
  tokens per step do not cover a 3.4× step cost. MTP's short, high-confidence
  chains are the right shape for this host; long ones are not.
* Not a tuning problem in the way the split-mode gap is, but note it *is* the
  `--spec-ngram-simple-size-m` (default 48) that sets the step length — a much
  shorter `size_m` was not tried.

## Thunderbolt eGPU: one model across two GPUs **[clean]**

An NVIDIA GTX 1070 (GP104, 8 GB) in a Thunderbolt 3 enclosure was added, giving a
second, independent VRAM pool. Setup (driver, the 288 MB BAR1 window fix, the
CUDA build) is in the README. Both backends live in one binary:

```
CUDA0:   NVIDIA GeForce GTX 1070   (8106 MiB, 8022 MiB free)
Vulkan0: AMD Radeon Graphics (RADV VEGA10) (8176 MiB, 6933 MiB free)
```

One model instance spreads across both, with the remainder on CPU, via
`-dev CUDA0,Vulkan0 -sm layer -ts 4,1 -ngl N`.

### Measured at a realistic shape

5188-token prompt, `-c 8192`, `-n 32`, `-lm dio`, `-ctk q4_0 -ctv q4_0 -fa on`:

| config | layers offloaded | sys peak MiB | NVIDIA peak MiB | prompt t/s | gen t/s |
|---|---|---|---|---|---|
| Vega 10 only, `-ngl 8` | 8 | 4349 | 87 | **66.66** | 1.640 |
| Vega 10 only, `-ngl 10` | 10 | 4846 | 87 | **66.96** | 1.740 |
| 1070 only, `-ngl 24` | 24 | 1237 | 6967 | 36.74 | 2.160 |
| 1070 only, `-ngl 28` | 28 | 1228 | 7873 | 39.63 | 2.540 |
| 1070 only, `-ngl 30` | 30 | 1220 | 7445 | **✗ fail** | **✗ fail** |
| **split `-ts 4,1 -ngl 40`** | **40** | 4304 | 7633 | 50.60 | **3.240** |

Conclusions:

* **The split dominates on generation and is second on prompt.** 40 layers
  offloaded against 28 for the 1070 alone and 10 for the Vega 10; +28 %
  generation over the best 1070-only config.
* **The 1070-only configuration fails at this shape.** `-ngl 30` cannot allocate
  the compute buffer for a 5188-token batch alongside 30 layers of weights in
  8 GB. Splitting moves weights to a second device and frees the room — so the
  split is not just faster, it is the configuration that works.
* **The Vega 10 is the better prefill engine, the 1070 the better offloader.**
  Vega-only prompts at 66.7 t/s against the 1070's 39.6, because the Vega 48 has
  ~2× the FP32 throughput (3840 vs 1920 shaders) and prefill is compute-bound;
  the 1070 wins generation purely by holding more layers.
* The Thunderbolt link trains at **2.5 GT/s ×4** (PCIe 1.0, ~1 GB/s) where
  8.0 GT/s is available — and it does **not** matter: only ~20 KB crosses the
  CPU↔GPU boundary per token.

### Context length trades against layer count

At 128k the KV cache competes with the weights for the same VRAM. Only 16 of the
64 layers are full-attention, and with `q4_0` KV they cost ~18 KiB/token across
all 16, i.e. ~37 MiB per offloaded layer at 131 072 tokens:

| `-ts` | `-ngl` | `-c` | exit | sys peak MiB | NVIDIA peak MiB |
|---|---|---|---|---|---|
| 4,1 | 40 | 131072 | 1 | — | ✗ needs 1152 MiB KV with 7067 MiB of weights |
| 4,1 | 32 | 131072 | **0** | 3893 | 7839 |
| 4,1 | 36 | 131072 | 1 | 3765 | 6453 |
| 3,1 | 32 | 131072 | **0** | 4518 | 7213 |

So **128k costs ~8 layers of offload** (40 → 32). f16 KV is not an option here:
~8.6 GB across the 16 full-attention layers, more than either GPU has free.
`-ctk q4_0 -ctv q4_0 -fa on` is mandatory at this context.

### 128k measured end to end

A genuine 130 729-token prompt, `-dev CUDA0,Vulkan0 -sm layer -ts 4,1 -ngl 32
-ctk q4_0 -ctv q4_0 -fa on -c 131072`, on an idle host (`bg_cpu_pct` 2.9 %):

| | |
|---|---|
| exit | 0 |
| wall | 4413 s |
| prompt eval | 130 729 tokens in 4388 s = **29.79 t/s** |
| generation | 838.13 ms/token = **1.19 t/s** |
| system peak | 4314 MiB (cap 5000) |
| NVIDIA peak | 7945 of 8192 MiB |
| RSS peak | 9629 MiB |

So 128k is **usable**: 1.19 t/s generation, 29.8 t/s prompt, at the cost of a
73-minute prefill. Generation at this depth beats what the single-GPU depth ladder
would predict — extrapolating `-ngl 10` at 32k (0.85 t/s) suggested 0.3-0.4 t/s,
but offloading 32 layers across two GPUs more than compensates for the context.
The 1070 finishes at 7945 of 8192 MiB, so `-ngl 32` is at the edge; `-ngl 36`
does not fit.

### MTP at the split **[clean]**

128k context, `data/long_prompt.txt` (5246-token prompt), one `/completion` of 64
tokens, `-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on -np 1`, idle host
(`bg_cpu_pct` 1.4–1.8 %). `results/opt_ab_long.jsonl`,
`results/opt_ab_confirm.jsonl`, `results/opt_ngld.jsonl`. The last four rows are
two interleaved reps each; spread < 1 %.

| arm | `-ts` | `-ngl` | prompt t/s | generation t/s | client MiB | NVIDIA peak MiB |
|---|---|---|---|---|---|---|
| base | 4,1 | 28 | 39.21 | 2.681 | 2174 | 7127 |
| MTP, draft on CPU (`-ngld 0`) | 4,1 | 26 | 34.85 | 4.509 | 2528 | 7459 |
| MTP, draft on CPU | 4,1 | 28 | 36.17 | 4.640 | 2528 | 8049 |
| MTP, draft on the Vega (`-ngld 1`) | 6,1 | 26 | 35.37 | 5.42 | 2835 | 7217 |
| MTP, draft on the Vega (`-ngld 1`) | 6,1 | 28 | 36.79 | 5.67 | 3151 | 7541 |
| MTP, draft on the Vega (`-ngld 1`) | 8,1 | 28 | 36.72 | 5.63 | 2835 | 7803 |
| **MTP, draft on the Vega (`-ngld 99`)** | **6,1** | **28** | **36.90** | **6.22** | **3384** | 7541 |

Conclusions:

* **The draft head belongs on the Vega 10, not the CPU** — that is the change
  `serve.sh` now ships. At the same `-ngl 28`, moving the draft from CPU to
  `-devd Vulkan0` is 4.640 → 6.22 t/s (**+34 %**); against the previous default
  (CPU draft, `-ngl 26`) it is 4.509 → 6.22 (**+38 %**), for +6 % prompt and
  ~856 MiB more of the display GPU (client 2528 → 3384). The MTP block's
  forward pass is *serial* with the target's, so once 26+ trunk layers are
  offloaded the CPU draft is the bottleneck; the 1070 has no room for the draft
  and the Vega does.
* **`-ngld 99`, not `-ngld 1`.** The draft declares
  `qwen35.nextn_predict_layers = 1`, and offloading exactly that layer still
  leaves the draft's output projection (151k × 5120, a matmul per draft step) on
  the CPU. A count above the layer count moves the non-repeating tensors too:
  measured **+10 %** generation (5.66 → 6.22 t/s) for +234 MiB of the Vega
  (client 3151 → 3384), 2 of 2 interleaved pairs.
* **`-ts 6,1`, not 4,1.** The draft costs ~1.5 GB of the display GPU, so the
  trunk's share there must shrink: `-ts 4,1` puts the draft's weights on top of
  ~6 trunk layers and reaches client 3644 MiB, breaching the cap under a normal
  desktop baseline. 6,1 measures 3384. `-ts 8,1` is a viable fallback when the
  desktop is heavy — at `-ngld 1` it measured 5.63 t/s against 5.67 and client
  2835 against 3151, i.e. ~1 % of generation for 316 MiB of headroom.
* **`-ctkd q4_0 -ctvd q4_0` is not optional at 128k.** The draft's own KV is f16
  by default: 4 kv heads × 256 × 2 × 2 B = 4 KiB/token, ~537 MiB at 131072
  tokens, against ~134 MiB at q4_0. Measured on the Vega draft at 128k, the
  quantized draft cache is worth 367 MiB of client VRAM.
* **MTP costs the 1070 enough that the un-speculated layer count no longer
  fits.** At 128k, `-ngl 32` with MTP aborts before load:
  `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 901.25 MiB on device
  0: cudaMalloc failed: out of memory` → `failed to allocate compute pp
  buffers`; the CPU-draft variant fails identically. At 8k, `-ngl 40` exits 1
  before health while 38 starts and completes clean. See "Layer ceiling with the
  draft on the Vega" for the rungs `serve.sh` derives from.
* **Superseded:** the earlier version of this section reported +29 % (2.73 →
  3.52 t/s) for the CPU draft at `-ngl 26` and concluded `-ngld 0` was the
  configuration to use. Both are now wrong — see the content-dependence finding
  below for why that particular 3.52 was low, and the draft-on-Vega row for why
  `-ngld 0` is not the fastest. The old `-ngld 0` claim that "the 1070 has no
  room for it" was true and remains the reason the draft goes to the *Vega*.

#### Layer ceiling with the draft on the Vega **[clean]**

`scripts/split_fit.py` at the shipped flags (`-dev CUDA0,Vulkan0 -sm layer -ts
6,1 -np 1`, `-devd Vulkan0 -ngld 99 -ctkd q4_0 -ctvd q4_0 -lm dio -b 2048 -ub
512 -fa on`), which loads the server and then issues one real completion —
`results/split_fit.jsonl`, `results/split_fit_long.jsonl`:

| context | `-ngl` | NVIDIA peak MiB | client MiB | prompt t/s | generation t/s | verdict |
|---|---|---|---|---|---|---|
| 8192 | 36 | 7503 | 3188 | 21.15 | 5.39 | fits |
| 8192 | **38** | 7927 | 3188 | 22.38 | 5.30 | **last startable** |
| 8192 | 40 | 7883 | 2093 | — | — | exit 1, no health |
| 32768 | 34 | 7639 | 3025 | 20.12 | 5.07 | fits |
| 32768 | **36** | 7849 | 3290 | 21.14 | 5.38 | **last tested** |
| 131072 | 28 | 7525 | 3383 | 17.37 | 4.39 | fits |
| 131072 | 30 | 7963 | 3383 | 18.12 | 4.51 | fits |
| 131072 | 32 | 7361 | 1851 | — | — | exit 1, no health |

And the same two 128k rungs with the full 5246-token prompt
(`--prompt-chars 0`, which is what the depth caveat below is about):

| context | `-ngl` | NVIDIA peak MiB | client MiB | prompt t/s | generation t/s |
|---|---|---|---|---|---|
| 131072 | **28** | 7541 | 3384 | 36.95 | **6.18** |
| 131072 | 30 | 7979 | 3384 | 38.54 | 6.42 |

* **The ceiling is the 1070, not the display cap.** At 128k `-ngl 32` does not
  even load: `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 901.25 MiB
  on device 0: cudaMalloc failed: out of memory` → `failed to allocate compute
  pp buffers` (`logs/split_fit-32.log`). At 8k `-ngl 40` exits 1 before health.
  The Vega sits ~1.2 GB under its cap at every point above.
* **`-ngl 30` at 128k is startable and 4 % faster than 28, and `serve.sh` still
  ships 28.** These probes peak at depth 5246; the flash-attention workspace
  grows with context, and the 1070 aborts somewhere around 8100–8189 MiB — it
  did at `-ngl 30 -ts 4,1` with the CPU draft, and at `-np 4` even for
  `-ngl 28` (SIGABRT, 8189 of 8192). 28 leaves ~650 MiB of the 1070 free to
  reach 131072 tokens; 30 leaves 213.
* **Client VRAM is not monotonic in `-ngl`** (8192/36 and 8192/38 both 3188;
  32768/32 and 34 both 3025) because `-ts` rounding decides how many whole
  layers the Vega takes and extra layers land on CUDA0. Read the NVIDIA
  counter, not the client one, when tuning up.
* `serve.sh` derives `-ngl` as `32 - (CTX - 32768)/16384`, which is two layers
  under the last startable rung at the 32k and 128k ends, and deliberately
  conservative below 32k. Those rungs (38/36/30) were measured with upstream's
  3-token draft chain; the shipped 6-token chain moves the 128k ceiling to 28,
  which is why the formula's constant is 32 rather than 34 — see "The draft
  chain spends the 1070's reserve".

#### Threads at the split **[clean]**

`-t` matters more here than on the single-GPU path, where 4–12 measured
equivalent. Shipped config (`-ngl 28 -ts 6,1 -np 1`, 128k context, 4500-character
mid-document prompt, 64 tokens out), `results/opt_threads_split.json`:

| `-t` | prompt t/s | generation t/s | `bg_cpu_pct` |
|---|---|---|---|
| **8** | 34.06 | **3.01** | 1.8 |
| 12 | 34.05 | 2.92 | 3.0 |
| 6 | 34.03 | 2.61 | 2.0 |
| 4 | 33.78 | 2.08 | 2.5 |

* **8 physical cores, not 16.** `-t 12` costs 3 % and `-t 6` costs 13 %; the
  prompt rate is flat across all four. `-t 8` reproduced at 3.008 in the earlier
  content-dependence run (`bg_cpu_pct` 1.7), so the ranking is not a single
  sample.
* One `-t 8` repeat landed at 2.217 t/s with `bg_cpu_pct` 14.1 — the background
  load the method section warns about, not a thread effect. Discarded.
* Fewer CPU-resident trunk layers than the single-GPU path does **not** mean
  fewer threads help: the draft and the target's CPU layers are serial, so the
  CPU work that remains is on the critical path.

The two other thread knobs are batch threads and CPU affinity, and neither helps
(`results/cpu_pin_fit.jsonl`, and the `-tb` table under "Prefill-side levers"
below). Pinning the 8 threads to the 8 physical cores (`--cpu-strict 1 -Cr 0-7`)
is neutral on prompt (36.92 vs 36.95) and −3 % on generation (6.054 vs 6.234);
pinning across the whole 8c/16t package (`-Cr 0-15`) costs **38 %** of
generation (3.867 t/s). llama.cpp does not document how the mask maps threads to
CPUs, so the mechanism is unestablished — but the default, un-pinned placement
is the fastest of the three, so both knobs are best left at their defaults.

#### MTP's benefit is content-dependent **[clean]**

The same configuration on a *different* continuation of the same corpus —
`data/long_prompt.txt` truncated to its first 4500 characters, so generation
starts mid-document instead of at the end — `-ngl 28 -ts 6,1`, two interleaved
reps, `results/opt_midprompt.json`:

| arm | prompt t/s | generation t/s | draft accepted |
|---|---|---|---|
| base | 36.23 | 2.746 | — |
| MTP, draft on the Vega | 34.09 | 2.997 | **27 / 104 (26 %)** |

* **+9 % generation here against +132 % on the feature-list continuation**
  (2.681 → 6.224, 46/48 accepted). Acceptance per draft step is what sets the
  win, and it ranges from 26 % to 98 % across two prompts from the *same*
  document. Quote MTP's benefit with its acceptance rate or not at all.
* The Vega draft helps **most when acceptance is lowest**: on this prompt the
  CPU-draft variant at `-ngl 26 -ts 4,1` measured 1.894 t/s (31/93 accepted)
  where the Vega draft at `-ngl 28 -ts 6,1` gives 2.997 — the draft's own cost
  dominates exactly when few drafts land, so moving it off the CPU is worth the
  most in the worst case.
* MTP never measured as a *loss* at the shipped configuration, but +9 % is the
  honest floor for a generation-bound workload.

#### Speculation variants that change nothing **[clean]**

The same mid-document continuation as the content-dependence row above
(`data/long_prompt.txt` truncated to 4500 chars, `-ngl 28 -ts 6,1 -np 1`,
`--n-predict 64`, `-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on`),
`results/spec_ab.jsonl`:

| arm | prompt t/s | generation t/s | draft / accepted |
|---|---|---|---|
| shipped MTP (`--spec-type draft-mtp -md … -devd Vulkan0 -ngld 99`) | 34.11 | 2.855 | 27 / 104 |
| + `--spec-draft-sampling probabilistic` | 34.00 | 2.946 | 27 / 104 |
| + `ngram-simple` (`--spec-type draft-mtp,ngram-simple`) | 33.94 | 2.957 | **27 / 104** |

* **Both variants draft exactly the same 104 tokens and land 27 of them**, so
  the ≤3.6 % spread is single-sample noise, not an effect. Probabilistic draft
  sampling does not move acceptance here, and chaining `ngram-simple` onto
  `draft-mtp` adds nothing — the ngram path found no usable match on this
  prompt. Neither is worth carrying. **Note the prompt:** this is prose, which an
  n-gram drafter cannot help with by construction. Retested on the content it
  exists for — output that is a verbatim copy of the context, 100 % of drafts
  accepted — `ngram-simple` is not merely useless but a **22 % loss** chained
  onto MTP and 26 % alone; see "`ngram-simple` is a loss even on verbatim
  output".
* The run also reproduced the content-dependence figure exactly (27/104, 26 %),
  which is a good check on that section.

### Prefill-side levers that are dead ends **[clean]**

Prompt throughput at depth is the expensive number here (29.79 t/s on a real
130 729-token prefill, ~73 min), so three ways to raise it were tried at the
shipped split (`-ngl 28 -ts 6,1` + MTP on the Vega, 15 907-char / 5246-token
prompt, `--n-predict 32`).

**Batch threads do nothing** — `results/tb8_fit.jsonl`, `tb12_fit.jsonl`,
`tb16_fit.jsonl`:

| `-tb` | prompt t/s | generation t/s |
|---|---|---|
| 8 (= `-t`, default) | 36.95 | 6.234 |
| 12 | 36.96 | 6.264 |
| 16 | 36.24 | 6.139 |

All three within 2 %, i.e. noise. This is the same finding as the thread sweep
seen from the prefill side: the CPU work that remains is on the serial path, so
handing prompt processing more threads than `-t` buys nothing.

**Larger batches are forbidden by the cap** — `results/batch_hi_fit.jsonl`:

| `-b` / `-ub` | result |
|---|---|
| 4096 / 512 | 37.04 prompt / 6.241 gen — identical to the default 2048/512 |
| 2048 / 1024 | **✗ cap breach** — system peak 5319 MiB, `SIGKILL` before health |
| 4096 / 1024 | **✗ cap breach** — same 5319 MiB, client 2509 / NVIDIA 7745 |

The `-b 4096` half is free (the logical batch does not drive allocation), but
`-ub 1024` grows the compute buffer enough to breach the 5000 MiB cap even
though the trunk is unchanged — so 512 is the largest usable micro-batch at this
configuration. Combined with the depth-4096 sweep (where the default 2048/512
beat every smaller micro-batch), **`-b 2048 -ub 512` is pinned from both sides**.

**A smaller ubatch does not buy layers** — `results/ub256_fit.jsonl`. The 1070
refuses `-ngl 32` at 128k with `allocating 901.25 MiB on device 0` (compute pp
buffers), so `-ub 256` was tried as a way to shrink that buffer and offload more:

| `-ngl` | `-ub` | outcome |
|---|---|---|
| 30 | 256 | ok — client 3158, NVIDIA 7861, 4.591 t/s |
| 32 | 256 | ✗ `allocating 799.25 MiB on device 0: cudaMalloc failed` |
| 34 | 256 | ✗ same |

Halving the micro-batch shrank the pp compute buffer by only 11 % (901 → 799
MiB) because it is dominated by the flash-attention workspace, which scales with
`n_ctx`, not with the ubatch. The `-ngl` ceiling is therefore unchanged, and
trading prefill rate for it is pointless.

### Split modes other than `-sm layer` are unavailable **[source]**

A parallelized split (`-sm row`, `-sm tensor`) would have put *both* GPUs on
every layer — the one structural change that could have attacked the 73-minute
prefill, which is pipelined and therefore single-GPU at a time. It cannot be
used on this build:

```
$ llama-completion … -dev CUDA0,Vulkan0 -sm row -ts 6,1 -ngl 20 -c 8192 …
E llama_model_load: error loading model: device CUDA0 does not support split buffers
```

(`logs/split-mode-row.log`, exit 1 after 1.2 s, no health.)

Row/tensor split requires each backend to export
`ggml_backend_split_buffer_type`; `llama-model.cpp:1175` throws exactly this
message when `ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type")`
returns nothing. Of the in-tree backends **only SYCL implements it**
(`ggml-sycl.cpp:28` → `ggml_backend_sycl_split_buffer_type`; zero occurrences in
`ggml-cuda.cu` and `ggml-vulkan.cpp`). `--fit` refuses the mode too
(`llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`). Not a
tuning problem — there is no code path.

### `--fit`: the runtime derives the layer budget itself **[clean]**

llama.cpp carries its own fitter (`common/fit.cpp`), on by default, which
projects per-device weight + KV + compute + *draft* memory and picks
`n_gpu_layers` and the tensor split — provided neither is set by the user, in
which case it aborts (`n_gpu_layers already set by user to 32, abort`, the
warning every `split_fit` rung prints). The display cap can be encoded as a
per-device margin: `-fitt` is "leave at least this much free", so
`vram_total - CAP` on the Vega expresses the cap *baseline-independently*,
because the fit already sees the desktop's usage in `free`.

`scripts/split_fit.py --autofit` drives it (omits `-ngl`/`-ts`, passes
`--fit on -fitt`). At 128k + MTP, `-fitt 256,3176`:

| | fit chose | client MiB | NVIDIA MiB | short-prompt gen t/s |
|---|---|---|---|---|
| formula (shipped) | 28 (25/4) | 3384 | 7541 | 4.39 |
| `--fit` | **28** (25/4) | **3383** | **7525** | 4.42 |
| `--fit`, rerun | **30** (25/5) | 3648 | 7479 | 4.63 |

* **It derives the shipped configuration.** Same layer count, same split, peaks
  within 2 MiB of client VRAM, and its projection tracked measurement closely
  where it can be checked: it predicted CUDA0 25 layers / 7504 MiB against a
  measured NVIDIA peak of 7479, and Vulkan0 5 layers / 3640 against a measured
  client 3648. (The fit is *pessimistic* on the 1070 and accurate on the Vega.)
  The full fit trace — targets, the per-device search, and the final assignment —
  is in `logs/split_fit-auto.log` under `-lv 4`.
* **It is adaptive.** Sweeping the margin — i.e. simulating a heavier desktop —
  the fit gives up layers instead of breaching the cap, which is the one thing
  the formula cannot do (`results/autofit_sweep.jsonl`, all `violation false`):

  | `-fitt` (1070, Vega) | Vega allowance | fit chose | client MiB | NVIDIA MiB |
  |---|---|---|---|---|
  | 256,3176 | 5000 | 28 | 3383 | 7525 |
  | 256,3176 (rerun) | 5000 | 30 | 3648 | 7479 |
  | 256,3688 | 4488 | 26 | 3068 | 7553 |
  | 256,3900 | 4276 | 25 | 2467 | 7615 |
  | 256,4176 | 4000 | 26 | 2467 | 7615 |
  | 256,5176 | 3000 | 20 | 1123 | 7501 |

* **But it is discrete and jittery, and at 30 layers it spends the reserve.**
  Two runs with identical margins and a 10 MiB difference in free Vega memory
  (7059 vs 7069) chose 28 and 30 layers; the margin sweep is non-monotonic
  (3900 → 25 layers, 4176 → 26). And the extra two layers cost the drift
  headroom the formula keeps: client 3648 leaves ~350 MiB of desktop tolerance
  against the cap where the formula's 3384 leaves ~620. That is why `serve.sh`
  keeps the deterministic formula as its default and exposes this as `FIT=1`.
* **Failure mode, if the fit cannot satisfy the margins:** it logs a warning and
  leaves `n_gpu_layers` at its default, which is `-1` = *all* layers
  (`llama_model::n_gpu_layers()` → `hparams.n_layer_all + 1`). The load then
  fails loudly in `cudaMalloc` rather than silently over-committing the display
  GPU — but it is a load failure, not a graceful degrade.

### Traps

* **A `models.yml` custom provider without `apiKey` disables *all* custom
  providers.** Adding the `llama.cpp` override produced `Warning: models.yml
  validation failed — custom providers disabled` and `Validate(models) error:
  Provider llama.cpp: "apiKey" is required when defining custom models`, which
  silently dropped the `huggingface` provider's 140 models as well — including
  the then-active default role. `apiKey: llama.cpp` (any literal; llama-server
  ignores `Authorization`) fixes it. Check with `omp models --json`, which prints
  the effective window per model.
* **`llama-bench` cannot measure a combined split.** Its `-dev` lists devices to
  benchmark *separately*: it loaded the model twice, once CUDA-only and once
  Vulkan-only (two independent 5172 MiB buffers). A `-ngl 44` attempt "OOM'd"
  because it was 44 layers on one card. Use `llama-cli`/`llama-completion`/
  `llama-server`, where `-dev` is a real device list.
* **`-ts` is not honoured as written.** `-ts 2,1` breached the display cap while
  `-ts 4,1` fit, and a requested `-ts 6,1` at `-ngl 32` assigned 28 layers to
  CUDA0 and 4 to Vulkan0 — 7:1, not 6:1 (`load_tensors: layer N assigned to
  device`, `-lv 5`). Non-layer tensors appear to land on the primary device.
  Tune against the two VRAM counters, do not trust the ratio.
* **llama-server's own `-np` default is 4, and it costs ~570 MiB on the 1070.**
  Measured at 128k, `-ts 6,1 -ngl 28` with the Vega draft: NVIDIA peak 7541 MiB
  at `-np 1`, 7701 at 2, **8117 at 4** — and at 4 a real `/v1/chat/completions`
  request aborted the server in `ggml_cuda_pool_vmm::alloc` inside
  `launch_fattn` (SIGABRT, exit −6, NVIDIA peak 8189 of 8192). `-np 4` at
  `-ngl 26` is 7793 and at 24 is 7259. **Every measurement in this file is
  `-np 1`** — the 128k-context figures are *one* 128k slot, not the two 64k slots
  `serve.sh` now defaults to, and they are the reason `-np` is passed explicitly
  rather than left to llama-server's own default of 4.
* **`guard.py` needed extending.** It read only `card0` sysfs, so it was blind to
  the 1070. It now reads NVIDIA VRAM through NVML via `ctypes` (microseconds;
  `nvidia-smi` costs ~150 ms per spawn) and records `nvidia_vram_peak_mib`. The
  5000 MiB cap still governs the display GPU only.
* **`llama-fit-params` cannot project an MTP draft.** It is the cheap way to ask
  the fitter what it would pick (it loads with `no_alloc`, ~5 s, no VRAM), but
  its argument set excludes the speculative flags: `error: invalid argument:
  --spec-type`. Use `scripts/split_fit.py --autofit` instead, which goes through
  `llama-server` and therefore fits the trunk and the draft together.
* **`-sm row`/`-sm tensor` fail on the first device, not the second.** The error
  names `CUDA0`, which reads like a CUDA problem; it is a backend-capability
  check, and Vulkan has the same gap (see "Split modes other than `-sm layer`").

## Serving (`llama-server`)

`serve.sh` runs `llama-server` under `guard.py`, which polls system-wide
VRAM every 50 ms and `SIGKILL`s the process group above `CAP` (default
5000 MiB), so the display GPU is protected by measurement rather than by
arithmetic. It serves one model split across both GPUs — `-dev CUDA0,Vulkan0
-sm layer` — and **requires the eGPU**: without `/dev/nvidia0` or a CUDA build
it exits 1 instead of degrading to a configuration the watchdog would kill.

Defaults: `-a Qwen3.8-27B -ts 6,1 -ngl 26 -np 2 -t 8 -lm dio -b 2048 -ub 512 -ctk q4_0
-ctv q4_0 -fa on`, plus MTP with the draft head **on the Vega 10**
(`--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -devd Vulkan0
-ngld 99 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 6 --spec-draft-p-min 0.5`). `-c`
is the shared pool — `CTX=131072` with
`-np 2` is **65536 per slot**, so a client's `contextWindow` must say 65536
(pi's `~/.pi/agent/models.json` does) — and `-ngl` is derived from the pool
(`32 - (CTX-32768)/16384` with MTP, `40 - (CTX-8192)/15360` without) with `-ts`
from whether the draft is resident; `MTP=0` disables speculation, `NGL=`,
`TS=`, `NP=`, `DEVD=`, `NGLD=`, `DNMAX=`, `DPMIN=` override. `FIT=1` replaces the formula with
llama.cpp's own `--fit` (`-fitt "$FITT_CUDA,$(vram_total - CAP)"`, default
`256,3176`), which re-derives `-ngl` and the split from live free VRAM — 28-30
layers at a 128k pool with `-np 1`, **26 with `-np 2`** (it accounts for the
second sequence's compute buffers: CUDA0 6876 MiB used, 472 free, client 3284) —
at the cost of the formula's reserve (see "`--fit`: the runtime derives the layer
budget itself"). Setting `NGL=` or `TS=` forces the fit off, because it aborts
rather than honour either. `serve.sh` prints the live display baseline and warns
when it is above the ~1216 MiB this config was measured to survive, and forwards
`SIGINT`/`SIGTERM` to the child, so Ctrl-C actually stops the server.

```
./serve.sh                       # → http://127.0.0.1:8080, 128k context + MTP
curl -sf localhost:8080/health           # ok after a ~13 s load
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":64}'
```

Smoke-tested end to end at the shipped defaults (Vega draft, `-ts 6,1 -ngl 28
-np 1`, 128k): `/health` ok after a 12.4 s load, the completion returns
`content: "pong"` (`predicted_per_second` 4.97, `draft_n_accepted` 20 of 33),
`exit_code 0`, `vram_violation: false`, peak **4523.1 MiB** of the 5000 MiB cap
(client 3383.1 MiB, RSS 11 354.2 MiB, NVIDIA 7519.3 of 8192 MiB), and a clean
teardown (`SIGTERM` to the guard leaves `pgrep -c llama-server` at 0). Re-run
after the `FIT` refactor it reproduced to the MiB: health 12.2 s, `pong`,
4.996 t/s, 20/33, peak 4480.6, client 3383.1, NVIDIA 7519.3, no violation.
Qwen-3.8-27B is a thinking model, so a `max_tokens` smaller than the reasoning
trace returns an **empty `content`** with everything in `reasoning_content` —
budget for the trace or the visible answer is truncated away.

`FIT=1` on the same host instead picked **30 layers** (25/5): health 16.1 s,
`pong`, 4.634 t/s, 20/33 accepted, peak **4738.3 MiB**, client 3648.2, NVIDIA
7479.3, no violation. Faster, and 262 MiB closer to the cap — the trade the
section above measures.

Smoke-tested again at the shipped `-np 2` default, this time through the real
client rather than curl: `/health` ok after 12.4 s, `/props` reports
`n_ctx = 65536, total_slots = 2`, and `pi -p 'Reply with exactly: pong'` — pi
being the agent configured against `127.0.0.1:8080`, with its full system prompt
and tool schemas — exits 0 after 122.7 s with `pong`. Server `exit 0`,
`vram_violation: false`, peak **4551.5 MiB** (client 3285.5, NVIDIA 7701.3 of
8192). The same client run is what fixed the number in pi's `models.json`:
`contextWindow` 65536, since the server rejects anything above the slot with
`input (N tokens) is larger than the max context size`.

Without an explicit `-np` — i.e. llama-server's own default of 4 — the server
aborted on that same request at a 128k pool and `-ngl 28`: `exit_code -6`
(SIGABRT), NVIDIA peak 8189 of 8192 MiB, `ggml_cuda_pool_vmm::alloc` inside
`launch_fattn`. That is why `serve.sh` passes `-np` explicitly; see the `-np`
trap above.

`guard.py` forwarding signals is load-bearing: the child is started with
`start_new_session=True`, so before the fix a Ctrl-C on `serve.sh` killed only
the guard and **orphaned `llama-server` holding ~3 GB of the 5 GB budget**.

## Agentic-coding profile: growth, compaction, slots **[clean]**

At the shipped defaults (128k, `-ngl 28 -ts 6,1 -np 1`, MTP on the Vega), driving
`/completion` with `cache_prompt: true` on `data/long_prompt.txt` (4000 chars ≈
1546 tokens), `n_predict` 4:

| request | `prompt_n` | `prompt_ms` |
|---|---|---|
| A (1546 tokens) | 1546 | 41 432 |
| **A + 172 tokens appended** | **172** | **11 519** |
| A again (slot holds A+ext) | 4 | 983 |
| A with 2000 chars rewritten in the middle | 1556 | 42 644 |
| the same rewrite, `--cache-reuse 256` | **1556** | 42 644 |

* **Growth is nearly free.** Prefix caching is on by default (`--cache-prompt`
  default enabled), so an append-only transcript pays for the new tokens, not the
  history — 11.5 s instead of a 41.4 s re-prefill, and re-sending the same prompt
  costs 1 s. The marginal cost of a turn is set by the depth-dependent prefill
  rate (36.95 t/s at 5.2k depth, 29.79 t/s at 130k), i.e. ~30-70 s per 1-2k-token
  tool result at the deep end.
* **Compaction resumes from a checkpoint — but only on the chat path, and only
  as far back as `-cms` allows.** The earlier claim here ("compaction pays full
  price") was an artefact of the probe, not a property of the model; see
  "Context checkpoints need `message_delimiters`" below. What is true: the KV
  shift path is dead (`--cache-reuse 256` measured *identical* to reuse off), and
  nothing persists to disk (`--slot-save-path` unset), so a server restart is a
  cold prefill.
* **The context ceiling is enforced, not silent.** A prompt longer than the
  slot's context is rejected — `input (N tokens) is larger than the max context
  size (M tokens)`, `ERROR_TYPE_EXCEED_CONTEXT_SIZE` — and with
  `--context-shift` disabled (the default) running out mid-generation sets
  `truncated: true` and stops. A client that only grows must compact on its own
  schedule; it will not be quietly truncated.
* **Parallel agents are the trade the default now makes: 2 slots, 64k each.**
  `-c` is a shared pool divided by `-np`, and the extra sequences need compute
  buffers the 1070 does not have:

  | `-np` | `n_ctx_slot` at `-c 131072` | two concurrent 1135-token requests | per-request gen t/s |
  |---|---|---|---|
  | 1 | 131072 | 91.4 s wall — the second waited 44 s behind the first | 2.90 / 4.05 |
  | **2 (shipped)** | **65536** | 100.0 s wall, both progressing | 1.39 / 1.56 |
  | 4 | **32768** | — (NVIDIA 8093 of 8192 MiB at load; a real request has aborted in `ggml_cuda_pool_vmm::alloc`) | — |

  Slots buy *fairness*, not throughput: the two requests took the same wall time
  together either way (91.4 s serialized vs 100.0 s concurrent), because the box
  is CPU/GPU-bound on one model. The pool cannot be grown to compensate —
  `-c 262144 -np 2`, i.e. two slots at the full 128k, **does not load**
  (`results/np2_fit.jsonl`: health timeout, exit 1 at `-ngl 28`) — so the choice
  is 2 x 64k, 1 x 128k or 4 x 32k, and the *client* has to be told which: pi's
  `~/.pi/agent/models.json` carries `contextWindow: 65536` for exactly this
  reason, since a request above the slot is rejected outright. Under `FIT=1` the
  fitter also gives up two layers for the second sequence (26 rather than 28).
* **Subagents are the worst case for the cache.** Each child's transcript
  diverges early from the parent's, so the shared prefix is small and every child
  pays a near-full prefill; alternating parent and child across two slots thrashes
  both slots' KV. `--cache-ram` (8192 MiB of host RAM, default) plus
  `--cache-idle-slots` exist to restore *idle* slots, but `/slots` reported
  `n_prompt_tokens_cache: 0` in every probe above, including at `-np 2` — nothing
  was held in RAM.
* **The local model is configured for 58k of the 64k slot in both clients**,
  which is what keeps a turn from overrunning the slot. In both, its *entry* is
  pinned at 58k; whether it is the default differs — pi defaults to the cloud
  model and selects the local one with `--model qwenrunner/Qwen3.8-27B`, while
  omp's `llama.cpp` provider is the local model outright:
  * **pi** — `~/.pi/agent/models.json`: `contextWindow: 59392`, i.e. 58k with 6k
    left under the slot for tool-call output.
  * **omp** — auto-discovers the server at `http://127.0.0.1:8080`
    (`LLAMA_CPP_BASE_URL` overrides) as provider `llama.cpp`, and reports the
    server's raw per-slot context as **both** `contextWindow` and `maxTokens`
    (65536/65536, i.e. no output budget at all). `~/.omp/agent/models.yml`
    overrides that one model to 59392/16384. Measured end to end:
    `omp -p 'Reply with exactly: pong' --model llama.cpp/Qwen3.8-27B` → exit 0,
    `pong`, a 7766-token prompt prefilled at 35.66 t/s (216 s) then 23 tokens
    generated at 3.28 t/s with 0.89 draft acceptance, `truncated = 0`.
  * **tumwater** — the two fleets on this box name the local model as their free
    `fallback` (`"fallback": "qwenrunner/Qwen3.8-27B"`) with no primary pinned, so
    role ticks run pi's default and fall back to the local server when it fails
    with provider-class errors. Only a pair pi's catalog prices at zero is
    accepted as a fallback, and this one is priced 0/0/0/0.
* **`serve.sh` passes `-a/--alias Qwen3.8-27B`.** Without it the API advertises
  the .gguf *path* as the model id, which is what omp's discovery then shows as
  the selector (`llama.cpp//home/zach/qwenrunner/models/Qwen3.8-27B-UD-Q4_K_M.gguf`).
  The alias is advisory — llama-server ignores the request's `model` field, so a
  client still sending an older id gets 200 (measured both ways), not an error.

### Context checkpoints need `message_delimiters` **[clean]**

The agentic probe above re-prefilled a mid-transcript rewrite in full and the
conclusion drawn from it was that compaction pays full price. That is wrong for
the path the clients actually use. `results/ckpt_probe.jsonl` — a raw
`/completion` prompt, `cache_prompt: true`, 50000 chars (~11.5k tokens) with a
2000-char block rewritten at char 47000 (~11.1k tokens), one server per arm:

| arm | req1 `prompt_n` | req2 `prompt_n` | req2 ms |
|---|---|---|---|
| `-ctxcp 32 -cms 8192` (shipped) | 11557 | **11559** | 285785 |
| `-ctxcp 32 -cms 2048` | 11557 | **11559** | 286070 |
| `-ctxcp 0` (control) | 11557 | **11559** | 302284 |

All three identical, and `-lv 4` says why: the only checkpoints that existed
were the pair the server *always* makes near the end of the prompt
(`server-context.cpp:3838`, offsets `4+n_ubatch` and `4`) at `[11040,11040]` and
`[11552,11552]` — both at or past the divergence at 10812, so both were
correctly rejected and it printed `forcing full prompt re-processing due to lack
of cache data (likely due to SWA or hybrid/recurrent memory)`.

**Mid-prompt checkpoints are gated on `spans.is_user_start(...)`
(`server-context.cpp:3828`, `:3885`), and `spans` is built from the request's
`message_delimiters` field.** A raw `/completion` prompt carries none. Every
chat request does: the differential autoparser derives them from the template's
`assistant_start`/`user_start` (`common/chat.cpp:1363-1370`), so pi and omp —
which post to `/v1/chat/completions` — are on a different path from the probe
that produced the old conclusion.

Retested with the delimiters sent explicitly on `/completion`
(`results/ckpt_chat_probe.jsonl`; a 10-turn Qwen-format transcript, 6630 tokens,
the *user* content of turn 8 rewritten — `prompt_n` is a deterministic token
count, so single reps are decisive here):

| arm | req1 `prompt_n` | req2 `prompt_n` | req2 ms | trace |
|---|---|---|---|---|
| `-ctxcp 32 -cms 8192` (shipped) | 6630 | 6647 | 186706 | checkpoints at `[11]`, `[6078]`, `[6142]`, `[6654]` |
| `-ctxcp 32 -cms 2048` | 6630 | **1943** | **63880** | `restored context checkpoint (pos_min = 4715, n_tokens = 4716, n_past = 4716)` |
| `-ctxcp 0` (control) | 6630 | 6659 | 169676 | `forcing full prompt re-processing` |

* **The mechanism works, and it is worth 3.4× on a 6.6k-token transcript**
  (6630 → 1943 tokens of re-prefill, 170 s → 64 s). The checkpoint is created at
  a user-message start, so this is a *transcript-shaped* win: it fires where a
  coding agent rewrites the middle of its history.
* **The shipped default is too coarse to help at this depth, and that is a
  spacing artefact, not a mechanism failure.** With `-cms 8192` and user turns
  ~2.85k tokens apart, the nearest checkpoint before the divergence was at token
  **11** — so the "resume" was a full re-prefill with extra bookkeeping. With
  `-cms 2048` the checkpoint at 4715 was reachable. `-cms` therefore sets the
  *bound* on compaction cost: the resume point is at most `-cms` tokens before
  the divergence.

The same probe at **12.2k depth** (`results/ckpt_depth_probe.jsonl`, 1.23k-token
turns so that turn boundaries fall past the 8192 spacing, rewrite in turn 9 of
10) shows the shipped default working, because now there *is* a checkpoint
before the divergence:

| arm | req1 `prompt_n` | req2 `prompt_n` | req2 ms |
|---|---|---|---|
| `-ctxcp 32 -cms 8192` (shipped) | 12214 | **3952** | 111020 |

* **3.1× off the re-prefill at the shipped setting** — 12214 → 3952 tokens,
  318 s → 111 s, resuming from the checkpoint at 8262. `-cms 8192` was never the
  problem; a 6.6k prompt was.
* **At the shipped 128k depth this bound is the whole story.** A compaction at
  depth re-prefills at most `-cms` tokens instead of 130 729 (73 min) — ~8k
  tokens, ~3.5 min, at the default. **Still unmeasured at 130k** (that is a
  73-minute run), but the mechanism is now measured at both 6.6k and 12.2k and
  the bound follows from the checkpoint spacing, not from the depth.
* **Cost is host RAM, not VRAM.** Each checkpoint is a `std::vector<uint8_t>` of
  the sequence's recurrent state — measured `size = 149.6-157.1 MiB` for this
  model — so `-ctxcp 32` caps the bill at ~5 GiB of the 39 GiB box, and
  `-cms 2048` fills that cap sooner than `-cms 8192` does. VRAM peaks were
  4738-4893 MiB in every arm, i.e. unchanged.
* **`-cms 2048` is a lever for shallow transcripts, not the depth case.** The
  shipped `-cms 8192` already bounds a deep compaction to ~8k tokens of
  re-prefill; `-cms 2048` only helps when the transcript is short enough that
  8k spacing leaves no checkpoint between the last user message and the
  divergence — measured 3.4× at 6.6k depth, where the default gave nothing. It
  costs RAM (the `-ctxcp 32` cap fills sooner) and `serve.sh` does not set it;
  pass `-cms 2048` as an extra argument if the workload compacts shallow
  transcripts rather than deep ones.

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

# MTP draft placement on the split: base vs CPU draft vs draft on the Vega.
# --server-extra is global to a run, so one -ts per invocation -- the CPU-draft
# rows in "MTP at the split" came from a -ts 4,1 run of the same harness.
./scripts/mtp_ab.py --ngl 28 --ctx 131072 --n-predict 64 \
    --arms base,mtp-sidecar,mtp-draftgpu --ngld 0 --ngld-gpu 99 --draft-dev Vulkan0 \
    --reps 2 --out results/opt_draftgpu.jsonl \
    --server-extra "-dev CUDA0,Vulkan0 -sm layer -ts 6,1 -np 1 -lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on"
# Fit anchors behind serve.sh's -ngl derivation (short prompt: cheap rungs)
./scripts/split_fit.py --ctx 8192   --ngl-ladder 34,36,38,40 --ts 6,1 --out results/split_fit.jsonl
./scripts/split_fit.py --ctx 32768  --ngl-ladder 32,34,36    --ts 6,1 --out results/split_fit.jsonl
./scripts/split_fit.py --ctx 131072 --ngl-ladder 28,30       --ts 6,1 --out results/split_fit.jsonl
# ...and the same at the shipped config with a realistic prompt (slow rungs)
./scripts/split_fit.py --ctx 131072 --ngl-ladder 28,30 --ts 6,1 \
    --prompt-chars 0 --n-predict 32 --out results/split_fit_long.jsonl
# --fit as the layer-budget source (--autofit omits -ngl/-ts; single rung), and
# the margin sweep behind the adaptivity table
./scripts/split_fit.py --autofit --ctx 131072 --prompt-chars 600 --out results/autofit.jsonl
for m in 3176 3688 3900 4176 5176; do
    ./scripts/split_fit.py --autofit --fitt 256,$m --ctx 131072 --prompt-chars 600 \
        --out results/autofit_sweep.jsonl
done
# prefill-side dead ends: batch threads, larger batches, smaller ubatch
./scripts/split_fit.py --ngl-ladder 28 --ts 6,1 --prompt-chars 0 --n-predict 32 \
    --out results/tb16_fit.jsonl \
    --server-extra "-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on -tb 16"
./scripts/split_fit.py --ngl-ladder 28 --ts 6,1 --prompt-chars 0 --n-predict 32 \
    --out results/batch_hi_fit.jsonl \
    --server-extra "-lm dio -b 2048 -ub 1024 -ctk q4_0 -ctv q4_0 -fa on"
./scripts/split_fit.py --ngl-ladder 30,32,34 --ts 6,1 --out results/ub256_fit.jsonl \
    --server-extra "-lm dio -b 2048 -ub 256 -ctk q4_0 -ctv q4_0 -fa on"
# CPU affinity at the shipped config (un-pinned is the fastest of the three)
./scripts/split_fit.py --ngl-ladder 28 --ts 6,1 --prompt-chars 0 --n-predict 32 \
    --out results/cpu_pin_fit.jsonl \
    --server-extra "-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on --cpu-strict 1 -Cr 0-15"
# speculation variants: --no-mtp + the spec flags in --server-extra
./scripts/split_fit.py --no-mtp --ngl-ladder 28 --ts 6,1 --prompt-chars 4500 --n-predict 64 \
    --out results/spec_ab.jsonl --server-extra "-lm dio -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -fa on \
    --spec-type draft-mtp,ngram-simple -md models/mtp-Qwen3.8-27B-Q4_0.gguf -devd Vulkan0 -ngld 99 -ctkd q4_0 -ctvd q4_0"
# agentic-coding profile: prefix reuse on growth, full re-prefill on a rewrite,
# and the per-slot context cost of extra slots (see "Agentic-coding profile")
./serve.sh &                       # then, against a running server:
P="$(head -c 4000 data/long_prompt.txt)"
curl -s localhost:8080/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$P\",\"n_predict\":4,\"cache_prompt\":true}"      # read .timings.prompt_n
#   ...repeat with ~200 chars appended (prompt_n ≈ 200), with the same prompt
#   again (≈ 4), and with 2000 chars rewritten mid-prompt (prompt_n ≈ full).
./serve.sh &                       # /props -> n_ctx = 65536, total_slots = 2 (the default)
NP=1 ./serve.sh &                  # /props -> n_ctx = 131072, total_slots = 1
# ...and the pool cannot be grown instead: --ctx 262144 with --np 2 does not load
./scripts/split_fit.py --np 2 --ctx 262144 --ngl-ladder 28 --ts 6,1 --out results/np2_fit.jsonl
./scripts/split_fit.py --autofit --np 2 --ctx 131072 --prompt-chars 600 --out results/autofit_np2.jsonl
pi -p 'Reply with exactly: pong'   # the real client, against a running server
omp models --json                  # effective window per model (llama.cpp -> 59392)
omp -p 'Reply with exactly: pong' --model llama.cpp/Qwen3.8-27B --no-session
./scripts/quant_compare.sh                              # quant ceiling, speed, perplexity

# one model across both GPUs, remainder on CPU (see "Thunderbolt eGPU")
build/bin/llama-completion -m models/Qwen3.8-27B-UD-Q4_K_M.gguf \
    -dev CUDA0,Vulkan0 -sm layer -ts 4,1 -ngl 40 -lm dio \
    -ctk q4_0 -ctv q4_0 -fa on -no-cnv -t 8 -c 8192 \
    --seed 42 -f data/long_prompt.txt -n 32

# MTP draft chain (see "The draft chain"). One invocation per arm, because
# --server-extra is global to a run; alternate the arms by hand for reps.
# SP is data/long_prompt.txt (feature-list, 46/48 accepted) or its first 4500
# chars (mid-document, 32/90). Substitute "--spec-draft-n-max 3" for the
# upstream default and "--spec-draft-n-max 14" for the aggressive arm.
./scripts/mtp_ab.py --ngl 32 --ctx 32768 --threads 8 --arms base --n-predict 64 \
    --prompt-file "$SP" --out results/draft_chain_P1_nmax6p50_r1.jsonl \
    --server-extra "-dev CUDA0,Vulkan0 -sm layer -ts 6,1 -lm dio -b 2048 -ub 512 \
    -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
    -md models/mtp-Qwen3.8-27B-Q4_0.gguf -devd Vulkan0 -ngld 99 -ctkd q4_0 -ctvd q4_0 \
    --spec-draft-n-max 6 --spec-draft-p-min 0.5"

# n-gram speculation on the content it exists for: all of scripts/bench.py in the
# prompt with "return the complete file verbatim", whose output is the file.
python3 -c "open('/tmp/code_edit.txt','w').write('Here is the complete contents of a Python file from a project.\n\nChange the default \`--reps\` value from 1 to 2 and return the complete file, verbatim in every other respect. Output only the file, with no commentary, no markdown fences, and no summary.\n\n----- BEGIN bench.py -----\n' + open('scripts/bench.py').read() + '\n----- END bench.py -----\n')"
./scripts/mtp_ab.py --ngl 32 --ctx 32768 --threads 8 --arms base --n-predict 96 \
    --prompt-file /tmp/code_edit.txt --out results/ngram_code_C1_mtpngram_r1.jsonl \
    --server-extra "-dev CUDA0,Vulkan0 -sm layer -ts 6,1 -lm dio -b 2048 -ub 512 \
    -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp,ngram-simple \
    -md models/mtp-Qwen3.8-27B-Q4_0.gguf -devd Vulkan0 -ngld 99 -ctkd q4_0 -ctvd q4_0"
# ...and "--spec-type ngram-simple" alone, and the plain draft-mtp control.

# Context checkpoints / compaction (see "Context checkpoints need
# message_delimiters"). --mode raw is the probe that produced the old
# "compaction pays full price" conclusion; --mode chat sends the delimiters the
# differential autoparser sends for every /v1/chat/completions request.
./scripts/ckpt_probe.py --mode raw  --out results/ckpt_probe.jsonl
./scripts/ckpt_probe.py --mode chat --out results/ckpt_chat_probe.jsonl
# ...and the same at ~12.5k depth, where the shipped -cms 8192 spacing can show
# itself: --chunk 5200 --rewrite-turn 9
./scripts/ckpt_probe.py --mode chat --chunk 5200 --rewrite-turn 9 \
    --out results/ckpt_depth_probe.jsonl
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
* **The `-ngl` 30 rung at 128k is verified only to depth 5246**, which is why
  `serve.sh` ships 28: the ceiling was set from probes that peak at 5246 tokens
  and the flash-attention workspace was assumed to grow with context. Depth
  added only +16 MiB to the 1070's peak at fixed `-ngl` (600-char vs 5246-token
  prompt), and the fit models the pp compute buffer at the full `n_ctx`, so the
  expectation is that 30 survives a real 130k prefill — unmeasured, and a
  73-minute run to settle. `FIT=1` picks 30 on a light desktop and takes that
  bet.
* **Row/tensor split would be the next structural lever** (both GPUs on every
  layer, including prefill), and is blocked on the backends, not on tuning: it
  needs a SYCL build. See "Split modes other than `-sm layer` are unavailable".
* **MTP at 130k depth.** The 6.22 t/s figure is at depth 5246; the 1.19 t/s
  end-to-end number is the base config at 130 729 tokens. The draft's benefit at
  full depth is unmeasured, and its acceptance rate is the thing that decides it
  — see "MTP's benefit is content-dependent".
* **The draft's own context length.** It is created at the target's `n_ctx`
  (131072) and needs a KV cache for it, ~134 MiB at q4_0. The MTP head only ever
  looks at the last few tokens, but there is no flag to shrink its window.
* **`-t` at other contexts.** Verified at the shipped 128k split (8 beats 12, 6
  and 4); the 32k/8k splits were not swept, though the CPU work there is
  smaller still.
* **The draft chain at the shipped 128k/28.** `-n-max` was swept at `-c 32768
  -ngl 32`, because the desktop was holding ~1.6-1.8 GiB of the Vega and left
  the shipped config ~16 MiB of cap margin. The transfer is expected to hold or
  improve — the shipped config leaves more CPU work per target step, so a draft
  step is relatively cheaper — but it is an inference. `serve.sh`'s `DNMAX=`
  makes it a one-flag re-measure on a light desktop.
* **`--spec-draft-n-max` above 14.** The marginal cost of a draft step (76 ms
  for ~0.96 accepted tokens) was still below the arm's running rate at 14, so
  the high-acceptance optimum has not been found — only bounded from below.
* **A short n-gram draft.** `--spec-ngram-simple-size-m` defaults to 48 and it
  is the *length* of the drafts that made `ngram-simple` lose (8.6 accepted
  tokens per step at 1.9 s/step against MTP's 2.8 at 0.56 s). A much shorter
  `size_m` was not tried.
* **`--poll` / `-bs`.** The threadpool spin level (default 50) and backend
  sampling are the last untested CPU-side flags; a sweep was written and then
  dropped to shorten the critical path, so they are untested rather than
  negative.
* **Compaction at 130k.** The checkpoint bound is measured at 6.6k and 12.2k
  depth; the 130k case follows from the `-cms` spacing rather than from depth,
  but a 73-minute prefill is what it would take to confirm.
* **A `--fit`-derived `-ngl` at the current desktop baseline.** With the desktop
  at ~1.8 GiB the shipped 128k/28 config peaks ~5.3 GiB and is killed by the
  watchdog; `FIT=1` would give up layers to hold the cap, and the resulting
  throughput on a *heavy* desktop has not been measured.
