qwenrunner

A local LLM inference stack optimized specifically to run the following combination as fast as possible:

* A Qwen-3.8-27B model
* A 2019 Intel iMac with 8-core (16-hyperthread) Core i9 processor and AMD Vega 10 (Radeon Pro Vega 48) Graphics with 8 GB total

Usage of the GPU should never, ever exceed 5 GB since that is also the display GPU.

We should plan and run experiments to determine the fastest possible way to run Qwen-3.8-27B on this hardware.

## Result

Every number below is measured on this machine and reproduced in
[`RESULTS.md`](RESULTS.md), which also records the runs that failed and why —
including six conclusions that earlier, load-contaminated measurements got wrong.

| lever | setting | evidence |
|---|---|---|
| runtime | llama.cpp @ `aa5e0092`, Vulkan/RADV | no ROCm/HIP on Vega 10 — Vulkan is the only GPU compute path |
| model | **`models/Qwen3.8-27B-UD-Q4_K_M.gguf`** | 16.46 GB. Smaller quants are *faster* — Q2_K_XL offloads 20 layers vs 12 and gains +52 % generation — but cost +61 % perplexity. See `RESULTS.md` |
| `-lm` | **`dio`** | 74.55 vs 66.32 t/s prompt against the mmap default, and ~40× lighter RSS |
| `-ngl` | **largest the live baseline allows** (10 here) | ~252 MiB client VRAM per layer; 10 beats 8 on both axes, 2/2 |
| `-b`/`-ub` | **2048/512 (default)** | 70.26 vs 59.09/45.06/29.48 t/s prompt; smaller batches buy VRAM, not speed |
| KV cache | **`q4_0/q4_0` + `-fa on`** | free at depth 0 (all five dtypes within 3 %), **+9.6 % generation** at depth 8192; `q8_0/q8_0` is the balanced alternative |
| flash attn | `-fa on` with a quantized V cache | `-fa off` aborts: `quantized V cache requires flash_attn to be enabled` |
| threads | `-t 8` | 4–12 are equivalent; 16 is ~8 % worse |
| MTP | `--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 0` | **+97.8 % generation** (1.719 → 3.400 t/s, 2/2 interleaved pairs) for −8 % prompt and +498 MiB |

**MTP speculation is the one large win — it nearly doubles generation
throughput.** Everything else is worth single digits to tens of percent. The
binding constraint is that **only 8–12 of 64 layers fit in VRAM**, so the model
stays CPU-bound and desktop CPU load directly sets throughput.

Best measured figures: **74.41 t/s prompt and 2.01 t/s generation** (`-ngl 10`,
`-lm dio`, shallow context), rising to **3.40 t/s generation** with MTP at
`-ngl 8`.

## Prerequisites

Three things are deliberately not committed (see `.gitignore`): the model weights
(50 GB), the vendored llama.cpp checkout (which carries its own `.git`), and the
unpacked shader tooling.

```
# 1. pinned llama.cpp
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout aa5e0092fdb8980b9e89498dfb73049b4fe204c2

# 2. Vulkan shader tooling, without root — no glslc/libshaderc system-wide
mkdir -p tools/pkg tools/local && cd tools/pkg
apt-get download glslc libshaderc1 spirv-headers
for d in *.deb; do dpkg-deb -x "$d" ../local/; done
cd ../..

# 3. models
mkdir -p models
BASE=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main
curl -fL -o models/Qwen3.8-27B-UD-Q4_K_M.gguf "$BASE/Qwen3.8-27B-UD-Q4_K_M.gguf"
curl -fL -o models/mtp-Qwen3.8-27B-Q4_0.gguf  "$BASE/mtp-Qwen3.8-27B-Q4_0.gguf"
./scripts/dl_smaller.sh                          # optional: the other three quants
```

## Build

```
./scripts/build.sh
```

`third_party/llama.cpp` is never patched; the one missing SPIRV include is worked
around caller-side with `-isystem tools/local/usr/include`.

## Run

```
./scripts/serve.sh                                  # OpenAI-compatible API on 127.0.0.1:8080
NGL=12 CTX=16384 ./scripts/serve.sh                 # overrides; NGL must fit the baseline
LM=mmap ./scripts/serve.sh                          # fall back if the FS lacks O_DIRECT
./scripts/serve.sh --spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 0
```

`serve.sh` runs `llama-server` under `scripts/guard.py`, which polls system-wide
VRAM every 50 ms and `SIGKILL`s the process group if it exceeds `CAP`
(default 5000 MiB), so the display GPU is protected by measurement rather than
by arithmetic. It forwards `SIGINT`/`SIGTERM` to the child, so Ctrl-C actually
stops the server.

**Read the printed baseline before trusting `-ngl`.** The desktop is the other
tenant of this GPU and its footprint drifts (measured 746–2188 MiB within one
hour). `-ngl 12` fits at 765 MiB and breaches at 1114; 10 fits at 785 and
breaches at 1592; 8 survives a loaded desktop. Pick from the measured client
peak, not from a constant.

**MTP needs `-ngl 8` or lower.** The draft head adds ~498 MiB of client VRAM and
will breach the cap at `-ngl 10` (measured twice). Its win is also
configuration-dependent: **+97.8 %** generation on Q4_K_M at `-ngl 8`, but only
**+3.6 %** on Q2_K_XL at `-ngl 16`, where the target is fast enough that the
CPU-resident draft head becomes the bottleneck.

## Harness

| script | purpose |
|---|---|
| `scripts/build.sh` | configure + build llama.cpp with Vulkan, pinned revision |
| `scripts/serve.sh` | launch `llama-server` under the VRAM watchdog |
| `scripts/guard.py` | VRAM watchdog + measurement: system VRAM, per-DRM-client VRAM, RSS, `cpu_busy_pct`, `bg_cpu_pct` |
| `scripts/bench.py` | `llama-bench` sweeps: `ngl`, `kv`, `fa`, `depth`, `ctx`, `threads`, `batch`, `load`, `nkvo`, `nopo`, `quant` |
| `scripts/mtp_ab.py` | MTP A/B over HTTP, with `--server-extra` for arbitrary flags |
| `scripts/quant_compare.sh` | quant offload ceiling + throughput + perplexity |
| `scripts/wait_idle.py` | block until a workload's CPU — or the whole host's — goes quiet |
| `scripts/gguf_head.py` | read GGUF metadata from a local file or a remote URL without downloading it |

Four measurement traps this harness exists to catch, each of which produced a
wrong answer before it was caught:

* **Throughput here is CPU-bound** (54 of 64 layers run on the CPU), so
  background load scales it directly. Every result carries `bg_cpu_pct` — the
  CPU busy % of all cores *excluding* the guarded tree.
* **Contended numbers are wrong by 3–8×, not by percent.** Repeats of the
  identical `-ngl 8` arm spanned 4.6×, and a background indexer depressed
  generation 6×. Six claims in earlier drafts were load artefacts — including
  "generation collapses with depth" and "`-t 12` halves generation". Interleave
  A/B arms and gate on `wait_idle.py --system`.
* **A just-exited arm's DRM client lingers for seconds.** `bench.py` settles
  VRAM before every *arm* and every *stage* and requires it to be inside the cap,
  or the next stage reads a phantom baseline and `guard.py` kills all of its arms
  at 0 s.
* **RAM is the binding constraint, not VRAM.** `/tmp` was a 20 GB tmpfs filled to
  19 GB by other sessions — non-evictable memory on a 39 GiB box with no swap.
  Run one model load at a time and check `MemAvailable` first. `-lm dio` peaks at
  ~300 MiB RSS where the mmap modes hold 13–14 GB resident.
