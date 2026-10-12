qwenrunner

A local LLM inference stack optimized specifically to run the following
combination as fast as possible:

* A **Qwen-3.8-27B** model (hybrid attention: 64 blocks, only 16 of them full
  attention, the rest linear-attention/SSM)
* An **Apple M5 Max** MacBook Pro with **128 GB of unified memory**

Every number in this README is measured on this machine and reproduced in
[`RESULTS.md`](RESULTS.md), which also records which measurements were
discarded and why.

> This branch re-targets the project from its original host (a 2019 Intel iMac
> with a Radeon Pro Vega 48 and a Thunderbolt GTX 1070 eGPU, where the whole
> design problem was *fitting layers into 5 GB of display VRAM*). `main` still
> carries that configuration. On Apple Silicon the problem is different and
> much smaller: one Metal device, one memory pool, and the entire model fits.

## Result

| lever | setting | evidence |
|---|---|---|
| runtime | llama.cpp @ `aa5e0092`, **Metal** + Accelerate/BLAS | one Metal device (`MTLGPUFamilyMetal4`, `has tensor = true`); no Vulkan/CUDA path exists here |
| model | `models/Qwen3.8-27B-UD-Q4_K_S.gguf`, 14.29 GiB, 27.32 B params | whole model fits *plus* 128k of KV. **Smaller quants are not faster here**: Q2_K_XL (9.83 GB, 36 % less weight traffic) buys just +7 % shallow and ≤+16 % at 128k, inside the run-to-run spread — decode at `-ngl 99` is not purely bandwidth-bound, so keep the quality |
| `-ngl` | **99 — every layer, including `lm_head` and the MTP head** | pp512 680 t/s vs 416 at 64, 58 at 32, 21 at 0; tg 24.8 vs 17.1 at 64, 1.32 at 0. The last 3 layers are the two biggest matmuls in the model (vocab 248320 embedding/lm_head), worth +63 % prefill on their own |
| `-fa` | **`on`** | pp512 705 vs 466 off (**+51 %**); generation equal at shallow depth (24.4 vs 23.0) |
| KV cache | **`f16/f16` for a single 128k server**, `q8_0/q8_0` when running several | at 128k, f16 generates **14.3 t/s vs 12.9** for q8_0 and prefills faster (231 vs 222 t/s) but costs 4 GiB more per instance |
| threads | **`-t 4`** | pp 758 vs 716 (t6), 693 (t8), 686 (t12), 529 (t18); generation flat from 4–18. The 12 performance cores cost more in contention than they return |
| batch | **`-b 1024 -ub 256`** at depth, `-b 2048 -ub 512` if you only ever prefill shallow | at 98.6k depth with MTP: generation **37.45 vs 28.52 t/s (+31 %)** against 2048/512, prefill level (349 vs 340); the same pair wins both axes at 64k. Only shallow prefill prefers 2048/512 (709.6 vs 633.3). See `RESULTS.md` |
| MTP | **on**, `--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99`, `NMAX=3` | draft on the GPU. **The gain is a property of the text, not the machine**: +180 % on patterned text (markdown/code/JSON, 99 % draft acceptance) and **−11 % on free-form prose** (14 %). `NMAX=6` is a further +40 % on structured/code traffic — the agent case this repo serves — but costs more on prose; `MTP=0` is the prose setting. Numbers in `RESULTS.md` |
| context | **128k** (`-c 131072`) | 14.3 t/s generation and 176–231 t/s prefill *at* depth 131072; costs ~8 GiB of KV with f16 |
| instances | **`INSTANCES=N`** — independent servers, weights shared | each extra 128k server costs **+8.2 GiB physical**, not 22 GiB (RSS double-counts shared pages); 4 measured working, 6 dies of a Metal `Insufficient Memory`; aggregate saturates at ~68 t/s from N=2 while per-stream falls 33.6 → 17.1 t/s |

**Offload is the lever that no longer exists, and context is the one that
stopped being expensive.** On the Intel host the whole design was built around
8–12 of 64 layers fitting in 5 GB of display VRAM; here `-ngl 99` is a 19x
generation improvement over the CPU-only path and there is nothing left to
offload. 128k of context, which cost a 73-minute prefill there, runs at 14.3
t/s generation and 176–231 t/s prefill because only 16 of this model's 64
blocks keep a growing cache. MTP speculation, the largest single win on the
Intel host, is larger still here — up to +180 % on patterned text, where the
draft head's acceptance is 99 %, because the draft fits on the same GPU — but
it is *harmful* on prose, which is why its knob is documented as a workload
choice rather than a constant.

Because 128 GB of unified memory holds this model many times over, the
interesting questions moved from "what fits" to "how many servers can run and
what does context cost".

## Hardware

| | |
|---|---|
| CPU | Apple M5 Max, 18 cores (6 super + 12 performance) |
| GPU | integrated, `MTLGPUFamilyMetal4`, simdgroup matrix mul + tensor ops, unified memory |
| RAM | 128 GB unified (`recommendedMaxWorkingSetSize` ≈ 112 GiB) |
| OS | macOS 27 |

**Unified memory is the whole story.** There is no VRAM, no `-ts` split, no
layer budget, no display-GPU cap and no watchdog kill: the 14.29 GiB of weights
live in the same pool as the OS, the KV cache and every other application. The
model's *entire* cost is 14.29 GiB + KV + ~2.5 GiB of compute buffers.

The KV cache is unusually cheap for this model: only 16 of 64 blocks are full
attention, and those use 4 KV heads of 256 dims, so a token costs
16 x 4 x 256 x 2 x 2 B = **64 KiB** at f16. 128k of context is therefore ~8 GiB,
not the ~40 GiB an all-attention 27B would need. The other 48 blocks carry a
constant ~10 MiB of SSM/conv state per sequence, regardless of context.

## Prerequisites

Two things are deliberately not committed (see `.gitignore`): the model weights
and the vendored llama.cpp checkout.

```
# 1. pinned llama.cpp
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout aa5e0092fdb8980b9e89498dfb73049b4fe204c2

# 2. cmake (the only build dependency; Xcode provides the Metal toolchain)
brew install cmake ninja

# 3. models
mkdir -p models
BASE=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main
curl -fL -o models/Qwen3.8-27B-UD-Q4_K_S.gguf "$BASE/Qwen3.8-27B-UD-Q4_K_S.gguf"
curl -fL -o models/mtp-Qwen3.8-27B-Q4_0.gguf  "$BASE/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
./scripts/dl_smaller.sh                          # optional: other quants
```

The MTP draft head lives under `MTP/` upstream, not at the repo root.

## Build

```
./scripts/build.sh
```

Metal + Accelerate + BLAS, Release, `-j 18`. The Intel build needed Vulkan
shader tooling unpacked into `tools/local`, a SPIRV-Headers include hack and a
`g++-13` pin for CUDA; none of that exists anymore. The build takes about a
minute.

## Run

```
./serve.sh                       # one server, 128k context, MTP on
INSTANCES=4 ./serve.sh           # four 128k servers on ports 8080-8083
INSTANCES=4 CTK=q8_0 CTV=q8_0 ./serve.sh   # halve the per-instance KV
MTP=0 ./serve.sh                 # no speculative decoding
```

`serve.sh` defaults: `-ngl 99 -c 131072 -t 4 -b 2048 -ub 512 -ctk f16 -ctv f16
-fa on`, plus `--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf
-ngld 99`. Every knob is an env var (`MODEL CTX INSTANCES PORT THREADS NGL LM
BATCH UBATCH CTK CTV FA MTP MD NGLD SWAPCAP`), and extra arguments are appended
verbatim to each server.

Servers run under `scripts/guard.py`, which forwards Ctrl-C to the whole
process group and records memory/CPU/GPU statistics for every run. The one
thing it can still enforce is swap: `SWAPCAP=<MiB>` SIGKILLs on new swap, off
by default because nothing here should swap.

### Running more than one server

Each server is a complete 128k-context instance. The weights are *shared*:
the default `-lm mmap` maps the GGUF once, and the file-backed pages are the
same physical pages for every process. Measured, at 128k with f16 KV:

| | host memory | file-backed pages |
|---|---|---|
| idle | 42.8 GiB | 35.1 GiB |
| + 1st server | **+15.1 GiB** | 21.1 GiB |
| + 2nd server | **+8.2 GiB** | 21.1 GiB |

So an extra instance costs its KV cache (8 GiB at 128k f16) and a little
compute — not another 14.29 GiB. `INSTANCES=4` is the largest configuration
measured to work; 6 fails with a Metal `Insufficient Memory` even though the
host is nowhere near its 128 GB. `CTK=q8_0 CTV=q8_0` halves the KV and is the
lever to pull if you need more instances rather than a faster stream.

Throughput, measured, MTP on:

| instances | aggregate t/s | per-stream t/s |
|---|---|---|
| 1 | 53.5 | 53.5 |
| 2 | 67.2 | 33.6 |
| 4 | 68.3 | 17.1 |

**Aggregate generation saturates at two instances.** Decode is
bandwidth-bound — the whole weight set is streamed per token — and one GPU has
one bandwidth budget, so N servers buy concurrency (N idle-most-of-the-time
agent sessions), not total throughput. Spread on identical work is ±25 %, so
read the shape rather than the digits.

## Harness

| script | purpose |
|---|---|
| `scripts/build.sh` | configure + build llama.cpp with Metal, pinned revision |
| `serve.sh` | launch N `llama-server` instances under the guard |
| `scripts/guard.py` | guarded runner + measurement: RSS, swap, host/GPU utilization, `bg_cpu_pct`, `gpu_busy_pct_peak` |
| `scripts/bench.py` | `llama-bench` sweeps: `ngl`, `fa`, `kv`, `depth`, `ctx`, `threads`, `batch`, `load`, `nkvo`, `nopo`, `quant` |
| `scripts/mtp_ab.py` | MTP A/B over HTTP, with `--server-extra` for arbitrary flags |
| `scripts/parallel_ab.py` | how many 128k servers fit, and what N instances does to throughput |
| `scripts/quant_compare.sh` | throughput + perplexity across quants |
| `scripts/wait_idle.py` | block until the GPU — or the host CPU — goes quiet |
| `scripts/gguf_head.py` | read GGUF metadata from a local file or a remote URL |

Four measurement traps this harness exists to catch. The first two are ports of
the Intel list, the second two are new here:

* **The contended resource moved from the CPU to the GPU.** With `-ngl 99` the
  Metal device does everything; a browser or a video call steals the same
  `Device Utilization %` our kernels need. Every result carries
  `gpu_busy_pct_peak`, and `wait_idle.py --gpu` gates on it.
* **A just-exited arm's Metal buffers linger.** `bench.py` settles accelerator
  memory before every arm and every stage, or the next arm reads a phantom
  baseline — the bug that cost the Intel harness an entire stage.
* **`host_statistics(HOST_CPU_LOAD_INFO)` is unusable on this kernel.** Its idle
  tick counter never advances, so the obvious `/proc/stat` analogue reports a
  machine sitting at idle as 100 % busy. `guard.py` computes CPU time from
  `ps -o time=` deltas instead, which needs no tick rate at all — see
  "frozen idle ticks" in `RESULTS.md`.
* **Unified memory makes "VRAM" mean nothing.** `ioreg`'s "In use system
  memory" counts the model's mapped weights, the KV cache and the desktop's own
  allocations in one number, and it does not fall back to baseline the instant
  a process exits. Absolute figures are only meaningful against a baseline
  sampled the same way.
