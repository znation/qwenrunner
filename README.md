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

**With the Thunderbolt eGPU attached, `serve.sh` splits one model across both
GPUs** — 40 layers at 8k context, 32 at 128k — which is +28 % generation over the
1070 alone and roughly triples the offloadable layer count (40 vs 12). See
[`RESULTS.md`](RESULTS.md).

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
./scripts/build.sh                  # Vulkan only (the Vega 10)
GGML_CUDA=ON ./scripts/build.sh     # add the NVIDIA backend for the eGPU
```

`third_party/llama.cpp` is never patched; the one missing SPIRV include is worked
around caller-side with `-isystem tools/local/usr/include`.

With `GGML_CUDA=ON` the host compiler is pinned to `g++-13`, because CUDA 12.4
rejects gcc 14 — `nvidia-cuda-toolkit` already depends on `g++-13`, so it is
present. Both backends end up in one binary and `-dev` selects between them.

## Run

```
./serve.sh                       # picks the best config for the hardware present
CTX=131072 ./serve.sh            # 128k context (fewer layers fit — see below)
NGL=40 TS=4,1 ./serve.sh         # override the split
DEV=Vulkan0 NGL=10 ./serve.sh    # force the Vega 10 alone
./serve.sh --spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 0
```

`serve.sh` chooses between two configurations at startup and prints which:

* **eGPU present** — one model instance split across the GTX 1070 (CUDA) and the
  Vega 10 (Vulkan), with the remainder on CPU: `-dev CUDA0,Vulkan0 -sm layer
  -ts 4,1`. The layer count is derived from the context, because the KV cache
  competes with the weights for the same VRAM — **40 layers at 8k, 32 at 128k**,
  both measured.
* **no eGPU** — the Vega 10 alone, `-ngl` from the client-VRAM-per-layer rule.

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

## Thunderbolt eGPU (optional)

An NVIDIA eGPU over Thunderbolt works, and roughly **triples** the offloadable
layer count (40 vs 12). Three steps were needed, in order.

**1. Driver + CUDA** (root):

```
sudo apt-get install -y nvidia-kernel-dkms libcuda1 nvidia-cuda-toolkit
```

This deliberately omits `nvidia-driver-libs`, which depends on
`libgl1-nvidia-glvnd-glx` and conflicts with the glvnd packages — installing it
would repoint GLX/EGL at NVIDIA and break the KDE session running on the Vega 10.
`nvidia-kernel-dkms` brings `firmware-nvidia-gsp`, so the nouveau firmware
package is not needed on this path.

**2. Enlarge the Thunderbolt bridge windows** — without this the card cannot be
driven at all. The kernel reserves only a **2 MB** hotplug window by default,
while a GTX 1070 needs 288 MB (256 MB BAR1 + 32 MB BAR3). The window never had a
chance, BAR1 was silently dropped, and the driver refused to initialise:

```
NVRM: BAR1 is 0M @ 0x0 (PCI:0000:46:00.0)
pci 0000:45:01.0: bridge window [mem size 0x12000000 64bit pref]: can't assign; no space
```

In `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet pci=realloc=on,hpmmiosize=64M,hpmmioprefsize=512M,hpbussize=32 iommu=pt"
```

then `sudo update-grub && sudo reboot`. The 256 MB BAR1 then lands in the 64-bit
window above 4 GB. Verify: `nvidia-smi` lists the card with 8192 MiB, and
`llama-bench --list-devices` lists both `CUDA0` and `Vulkan0`.

**3. Build with CUDA** — `GGML_CUDA=ON ./scripts/build.sh`.

### Gotchas worth remembering

* **`llama-bench` cannot measure a combined split.** Its `-dev` is a list of
  devices to benchmark *separately* — it loads the model once per device (two
  separate 5172 MiB buffers). Use `llama-cli` / `llama-completion` /
  `llama-server`, where `-dev` is a real device list. This cost two misleading
  attempts, including an apparent OOM that was just 44 layers on one card.
* **The taint message is not a fault.** `nvidia: module license 'NVIDIA' taints
  kernel` is the kernel flagging a proprietary out-of-tree module — it says
  nothing about the failure. The real errors were the `NVRM: BAR…` lines.
* **`-ts` does not behave as documented.** Requested `4,1`, measured layer counts
  land nearer 2:1, because non-layer tensors go to the primary device. Tune it
  against the two VRAM counters rather than trusting the ratio.
* **The Thunderbolt link trains at 2.5 GT/s ×4** (PCIe 1.0, ~1 GB/s) where 8.0
  GT/s is available. This is *not* the bottleneck — per token only ~20 KB crosses
  the CPU↔GPU boundary. The 1070's weaker prefill is simply that the Vega 48 has
  ~2× its FP32 throughput.

## Harness

| script | purpose |
|---|---|
| `scripts/build.sh` | configure + build llama.cpp with Vulkan, pinned revision |
| `serve.sh` | launch `llama-server` under the VRAM watchdog |
| `scripts/guard.py` | VRAM watchdog + measurement: display-GPU VRAM, per-DRM-client VRAM, NVIDIA VRAM via NVML, RSS, `cpu_busy_pct`, `bg_cpu_pct` |
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
