qwenrunner

A local LLM inference stack for agentic coding on a 2019 Intel iMac (8-core /
16-thread Core i9, AMD Vega 10 / Radeon Pro Vega 48 with 8 GB — the display
GPU).

Usage of the display GPU should never, ever exceed 5 GB since that is also the
desktop.

The default (`./serve.sh`) runs one **Qwen3.8-27B** instance split across the
GPUs and CPU, fitted by llama.cpp's own `--fit` with a `--fit-target` of
`vram_total - fit-cap` on the display GPU, so the fit budget holds regardless
of desktop drift. The device list depends on whether the Thunderbolt eGPU is
attached:

* **eGPU present** — `-dev CUDA0,Vulkan0 -sm layer`: spread across the GTX 1070
  (eGPU), the Vega 10 (built-in display GPU), and the CPU for the rest.
* **eGPU absent** — `-dev Vulkan0 -sm layer`: the Vega 10 plus the CPU. `--fit`
  offloads nothing here (it counts the host buffer the CPU-side weights land in
  as device memory), so the Vega runs only the MTP draft.

The built-in display GPU always respects two guards, deliberately different:
`--fit` packs the model into `FIT_CAP` (default 4600 MiB) via a `-fitt` margin
of `vram_total - FIT_CAP`, and `scripts/guard.py` only `SIGKILL`s at the higher
`CAP` (default 5500 MiB). The eGPU drives no display, so it is not capped —
only a small fit margin (`FITT_CUDA`, default 256 MiB) keeps it from filling
the card. `guard.py` enforces a second limit on RAM — `MemAvailable` above
`MIN_AVAIL` (default 2048 MiB) — because there is no swap, no cgroup limit and
no systemd-oomd here, so the kernel OOM killer is otherwise the only backstop
and it takes the model. MTP speculation is on by default, its draft head on the
Vega 10. The 27B is CPU-bound (only a fraction of its 64 layers fit in VRAM),
so generation is 2-6 t/s — slow, but the 4B that used to be the default was
hopeless on quality, so the 27B is the model. Everything below is this config's
measurement history.

## Result

Every number below is measured on this machine and reproduced in
[`RESULTS.md`](RESULTS.md), which also records the runs that failed and why —
including six conclusions that earlier, load-contaminated measurements got wrong.

| lever | setting | evidence |
|---|---|---|
| runtime | llama.cpp @ `aa5e0092`, Vulkan/RADV | no ROCm/HIP on Vega 10 — Vulkan is the only GPU compute path |
| model | **`models/Qwen3.8-27B-UD-Q4_K_M.gguf`** | 16.46 GB. Smaller quants are *faster* — Q2_K_XL offloads 20 layers vs 12 and gains +52 % generation — but cost +61 % perplexity. See `RESULTS.md` |
| `-lm` | **`mmap`** | `dio` is 46.10 vs 36.15 t/s prompt at 128k, but it puts the CPU-side weights in the Vulkan *host* buffer type (GTT, unreclaimable system RAM): **22.8 GB vs 6.6 GB of `MemAvailable` drop**, 2 interleaved reps each. The old "~40× lighter RSS" reading measured RSS, which cannot see GTT. `LM=dio` restores the prompt throughput when RAM is free |
| `-ngl` | **fit-derived** (was 10 on the Vega-only path) | ~252 MiB client VRAM per layer; 10 beats 8 on both axes, 2/2. `--fit` picks it from live free VRAM now — but only when there is a second device to fit against: on the Vega-only path it converges to 0 layers, because it counts the Vulkan host buffer the CPU-side weights land in as device memory |
| `--fit` | **on (always)** | llama.cpp's own fitter re-derives `-ngl`/`-ts` from live free VRAM with the cap as a `-fitt` margin: 28-30 layers at 128k, i.e. the shipped config to 2 MiB, or two layers (+4 %) faster — but it spends the ~1120 MiB drift reserve and its choice moves with desktop noise. That trade is why the measured formula used to be the default; with the eGPU optional (and no formula for the Vega-only path) `--fit` is the only strategy that works for both. `NGL=`/`TS=` pins the layer count and turns `--fit` off |
| `-b`/`-ub` | **2048/512 (default)** | 70.26 vs 59.09/45.06/29.48 t/s prompt; smaller batches buy VRAM, not speed. At 128k the cap pins it from above too: `-ub 1024` breaches (killed before health), `-b 4096` alone changes nothing (37.04 vs 36.95 t/s) |
| `-tb` | `= -t` (default) | batch threads 8/12/16 measure 36.95/36.96/36.24 t/s prompt at 128k — equivalent, like `-t` itself |
| CPU affinity | off (default) | `--cpu-strict 1 -Cr 0-7` is −3 % generation, `-Cr 0-15` −38 %; un-pinned placement is fastest |
| KV cache | **`q4_0/q4_0` + `-fa on`** | free at depth 0 (all five dtypes within 3 %), **+9.6 % generation** at depth 8192; `q8_0/q8_0` is the balanced alternative |
| flash attn | `-fa on` with a quantized V cache | `-fa off` aborts: `quantized V cache requires flash_attn to be enabled` |
| threads | `-t 8` | Vega-only: 4–12 equivalent, 16 is ~8 % worse. Split: **8** beats 12 by 3 % and 6 by 13 % — the CPU work left is on the serial path, so fewer threads do not help |
| MTP draft chain | `--spec-draft-n-max 6 --spec-draft-p-min 0.5` | llama.cpp defaults to `3` and `0.00`, which stops the chain on length and never on confidence. Sweeping them is worth **+18 % generation** where drafts land and nothing where they do not (2 interleaved reps, `bg_cpu_pct` ≤ 10). `-n-max 14` measures **+29–45 %** on high-acceptance content but **−13 %** on low, and **does not load at 128k at all** — it is a 32k option. It also costs the 1070 VRAM (`n_outputs_max` scales with the chain): 7677 → **8013 of 8192 MiB**, which is why the shipped layer count dropped 28 → 26 to keep the reserve. `DNMAX=`/`DPMIN=` override |
| MTP | `--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -devd Vulkan0 -ngld 99 -ctkd q4_0 -ctvd q4_0` | **+97.8 % generation** on the Vega-only path (1.719 → 3.400 t/s), and **+38 %** on the two-GPU 128k split once the draft head runs on the **Vega 10** rather than the CPU (4.509 → 6.224 t/s). The largest lever on this host. Its win is content-dependent — +9 % to +132 % — because it is set by draft acceptance (26 % to 98 % on two prompts from the same document). Now the `serve.sh` default |
| n-gram speculation | **off** — `ngram-simple` is a loss | retested on the content it exists for (output is a verbatim copy of a file in the context, 100 % of drafts accepted, 2 reps): MTP + `ngram-simple` **5.31** t/s and `ngram-simple` alone **5.05** against MTP alone **6.81**. Long drafts buy 8.6 accepted tokens per step at 1.9 s/step where MTP buys 2.8 at 0.56 s — the shape is wrong for a split whose CPU stage owns 33 of 65 layers |
| context checkpoints | **on — shipped default `-ctxcp 32 -cms 8192`** | a mid-transcript rewrite resumes from a checkpoint on the chat path: at 12.2k depth it re-prefilled **3952 of 12214** tokens (3.1×, 318 s → 111 s) at the shipped setting, and `-cms 2048` is 3.4× on a 6.6k transcript where the default's 8k spacing leaves no checkpoint before the divergence. Only fires when the request carries chat message delimiters — `/v1/chat/completions` does, a raw `/completion` does not, which is why the older "compaction pays full price" measurement was wrong |
| slots | **`-np 2`** | a second agent (or subagent) makes progress instead of queueing: two concurrent 1135-token requests take 100.0 s wall together against 91.4 s serialized — fairness, not throughput. Costs +160 MiB of the 1070's compute buffers (NVIDIA 7701 vs 7541) and **halves the per-slot context** (`-c` is a shared pool: 131072 / 2 = 65536, which is what pi's `contextWindow` must say). `-np 4` fits at load (NVIDIA 8093) but leaves 32768/slot and has aborted a real request in `ggml_cuda_pool_vmm::alloc` |

**MTP speculation is the one large win — it more than doubles generation
throughput, and where its draft head runs decides how much.** Everything else is
worth single digits to tens of percent. Its *chain length* is the last thing on
that path that was untuned: upstream stops at 3 drafts per step and never on
confidence, and 6 with a 0.5 confidence floor is +18 % where drafts land. The
binding constraint is that **only a few of the 64 layers fit in VRAM** — 28-38
with the eGPU attached, and none at all on the Vega-only path, where `--fit`
converges to `offloaded 0/66` — so the model stays CPU-bound and desktop CPU
load directly sets throughput.

**A second, exhaustive pass confirmed the shipped configuration is the
fastest measured on this host** — see RESULTS.md "A second pass". The step time
is ~900 ms of target verification plus a cheap ~27 ms draft; the verification
scales with the verified batch but *not* with CPU frequency (forcing the CPU's
EPP to `performance` raised the generation-phase clock 3.64 → 4.10 GHz with
flat throughput), so the CPU is memory/latency-bound, not clock-bound. That
rules out the privileged levers (RAPL PL1 raise, EPP) as well as every draft
variant tried (Q2_K/Q4_0 outputs, moving the draft to the 1070, longer chains
at `-ngl 24`) — each lost to the shipped q3_K-output draft on the Vega. The
Vega runs pinned at its 852 MHz base clock during inference (bursty load never
ramps it); forcing `power_dpm_force_performance_level = high` is a small,
privileged, optional gain. A smaller trunk quant (Q2_K_XL) at the split buys a
large prefill win (pp 59 vs 34 t/s) but *loses* generation (4.96 vs 6.38 t/s):
the CPU's Q2_K verify kernel costs more per layer than the byte savings gain.
The verification pass is the whole story — it scales with the verified batch
and nothing else moves it.

Two serving-path results are worth knowing before tuning anything else. **The
desktop is a first-class tunable** — it currently holds ~1.4 GiB of the Vega
(plasmashell 487, firefox 331, claude-desktop 254, ghostty 250 MiB), and the
split's display footprint moves in *steps* of ~343 MiB (one Vega layer, i.e.
7 `-ngl`), so that is roughly one rung of headroom sitting in the compositor and
two browsers. It is also why `--fit` is the default: the fit sees the desktop
in `free` and gives up layers to hold the cap, so the config is
baseline-independent rather than under-provisioned. When the display is tight
the rung to drop to is `-ngl 26`, not 24: measured, 24 and 26 cost the display
the same and 26 keeps two more layers on the 1070. And **compaction
does resume from a context checkpoint, on the chat path only**: mid-prompt
checkpoints are gated on `message_delimiters`, which the differential autoparser
supplies for every `/v1/chat/completions` request and which a raw `/completion`
prompt lacks, so the earlier "compaction pays full price" measurement described
the probe rather than the model. Measured, the shipped `-cms 8192` already takes
a mid-transcript rewrite at 12.2k depth from a 12214-token re-prefill to **3952**
(3.1×), and the bound it sets — `-cms` tokens back from the divergence — is what
turns a 73-minute 130k re-prefill into ~3.5 minutes.

Best measured figures: **74.41 t/s prompt and 2.01 t/s generation** (`-ngl 10`,
`-lm dio`, shallow context), rising to **3.40 t/s generation** with MTP at
`-ngl 8`.

**With the Thunderbolt eGPU attached, `serve.sh` splits one model across both
GPUs and runs MTP speculation on top, with the draft head on the Vega 10** — 28
layers at a 128k pool and 38 at 8k with MTP on (`MTP=0` restores the 32/40
split); the default `-np 2` divides that pool into two 64k slots — which is +28 %
generation over the 1070 alone and roughly triples the offloadable layer count
(40 vs 12). The eGPU is optional: without it `serve.sh` falls back to the Vega
10 alone, which offloads **no trunk layers at all** — `--fit` counts the
CPU-side weights, which llama.cpp allocates in the Vulkan host buffer type, as
device memory, so it converges to `offloaded 0/66` at both 128k and 32k and the
Vega runs only the MTP draft. Pinning `-ngl` does offload, but at 128k with MTP
on the budget is thin: 2 layers peak at 4579 MiB of system VRAM, 4 at 5301, and
6 at 5876 — over the 5500 MiB guard cap. `NGL=` pins it anyway.

The draft head is the newest lever: it is a *serial* forward pass, so with 26+
trunk layers offloaded the CPU is the bottleneck and the 1070 has no room to
take it — but the Vega 10 does. Measured at 128k, `data/long_prompt.txt` (5246
tokens), 64 tokens out, two interleaved reps each:

| arm | prompt t/s | generation t/s | client MiB |
|---|---|---|---|
| no speculation | 39.21 | 2.681 | 2174 |
| MTP, draft on the CPU (`-ngld 0`), `-ngl 28` | 36.17 | 4.640 | 2528 |
| MTP, draft on the Vega, `-ngld 1` | 36.79 | 5.67 | 3151 |
| **MTP, draft on the Vega, `-ngld 99`** | **36.90** | **6.22** | 3384 |

A genuine 128k context runs at **1.19 t/s generation and 29.8 t/s prompt**
(73-minute prefill) without MTP, inside the display-GPU cap. MTP's benefit is
**content-dependent — +9 % to +132 %** on this host, set by how often the draft
is accepted (26 % to 98 % across two prompts from the same document), and it is
unmeasured at 130k depth. See [`RESULTS.md`](RESULTS.md).

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
./scripts/dl_smaller.sh                          # Qwen3.8-27B Q4_K_M (the default) + optional extra quants + the vision projector (mmproj)
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
./serve.sh                       # Qwen3.8-27B split across the GPUs + CPU (:8080)
CTX=32768 NP=2 ./serve.sh        # two 16k slots instead of two 64k (parallel subagents)
```

Qwen3.8-27B is a VL-class model: the text GGUF carries no vision encoder — it
ships as a separate projector (`models/mmproj-Qwen3.8-27B-F16.gguf`, fetched by
`dl_smaller.sh`), and `serve.sh` serves it by default with the encoder on the
CPU (`--no-mmproj-offload`). Text-only traffic therefore stays 100 % on the GPU;
only image input pays a CPU vision-encode. Set `MMPROJ_OFFLOAD=1` to move the
encoder to the GPU instead. Image input works through the OpenAI-compatible
`/v1/chat/completions` (`image_url` content parts); set `MMPROJ=` to serve
text-only.

The server always runs llama.cpp's own `--fit on` with a `--fit-target`
(`-fitt`) — the layer count is derived from live free VRAM, never by hand. Two
guards, deliberately different: the fit packs the model into `FIT_CAP` (default
4600 MiB), so the display GPU leaves `vram_total - FIT_CAP` free (`FITT`,
3576 MiB here) — because the fit sees the desktop's usage in `free`, that holds
the fit budget regardless of desktop drift — while `scripts/guard.py` only
`SIGKILL`s at the higher `CAP` (default 5500 MiB). The 900 MiB gap is
**desktop-drift tolerance, not model headroom**: the model footprint is fixed at
load (KV and compute buffers are preallocated; measured **+34 MiB** transient
under a real 8045-token prefill), so the gap is how far the desktop may grow
after load before the guard fires. The eGPU drives no display, so it is not
capped — only a small fit margin (`FITT_CUDA`, default 256 MiB) keeps it from
filling the card.

`guard.py` enforces a second, independent limit: **`MemAvailable` must stay above
`MIN_AVAIL` (default 2048 MiB)**, or it kills the server itself. There is no
swap, no cgroup limit and no systemd-oomd on this box, so the kernel OOM killer
is otherwise the only backstop — and it takes the largest process, which is
always the model. `serve.sh` also refuses to start when `MemAvailable` is below
`MIN_AVAIL + 8192` (mmap) or `+ 24576` (dio), rather than discovering it two
minutes into a 16 GB read. Both numbers are env-overridable (`MIN_AVAIL=`,
`NEED_MIB=`).

The load mode is the RAM lever, and it is not a throughput knob: `-lm mmap`
(default) leaves the CPU-side weights as a reclaimable file mapping, while
`-lm dio` reads them into the Vulkan host buffer type — GTT, unreclaimable
system RAM. Measured at 128k, `dio` drops `MemAvailable` 22.8 GB against
`mmap`'s 6.6 GB, for ~21 % more prompt throughput. `LM=dio ./serve.sh` when the
box has room.

The default is `CTX=131072` with `-np 2` → **two 65536-slots** — the same
per-slot context the clients already expect (pi `contextWindow` 59392). The 27B
is CPU-bound (only a fraction of its 64 layers fit in VRAM), so generation is
2-6 t/s; the desktop baseline sets how many layers the fit can keep on the Vega,
and the eGPU roughly triples the offloadable layer count (40 vs 12). `NP=2`
halves the pool per slot but buys fairness for concurrent agents.

The local model's entry in both clients is set to **58k of that 64k slot**,
leaving room for tool-call output: pi in `~/.pi/agent/models.json`
(`contextWindow: 59392`), and omp in `~/.omp/agent/models.yml`, which needs an
explicit override because its auto-discovery of `http://127.0.0.1:8080`
otherwise takes the server's raw per-slot context as both the window *and* the
output budget (65536/65536). pi's default is the local model
(`defaultProvider: "qwenrunner"`, `defaultModel: "Qwen3.8-27B"` in
`~/.pi/agent/settings.json`); the cloud model
(`huggingface/deepseek-ai/DeepSeek-V4.1-Flash:deepinfra`) remains available via
`--model`. The two tumwater fleets on this box name it as their free
`fallback`, which is the pair the docs' backend guide recommends (a budgeted API
primary, the cost-n/a local model behind it).
`serve.sh` passes `-a Qwen3.8-27B` so clients see a stable model id instead of
the .gguf path; the alias is advisory — llama-server ignores the request's
`model` field, so a client still sending an older id works.

`serve.sh` prints the configuration it picked, including whether the eGPU was
detected. The eGPU is optional: when `/dev/nvidia0` is present and the binary
was built with CUDA, `serve.sh` splits across `CUDA0,Vulkan0`; otherwise it
falls back to `Vulkan0` alone (the Vega 10 plus the CPU).

The layer count is never chosen by hand: `serve.sh` always runs llama.cpp's
own `--fit`, which derives `-ngl` and the tensor split from live free VRAM per
device, with the display cap encoded as a `-fitt` margin (`vram_total -
FIT_CAP` on the Vega, so it is baseline-independent). Measured at 128k it picks
28-30 layers on the two-GPU split — the same config the old measured formula
(`32 - (CTX-32768)/16384` with MTP) reached to 2 MiB, or two layers faster —
but it spends the ~1070 MiB drift reserve and its choice moves with a few MiB
of desktop noise. That trade is why the formula used to be the default; with
the eGPU now optional (and no measured formula for the Vega-only path), `--fit`
is the only strategy that works for both. Setting `NGL=`/`TS=` pins the layer
count and turns `--fit` off (llama.cpp aborts rather than honour either). See
[`RESULTS.md`](RESULTS.md) for both tables.

**With `NGL=`/`TS=` set, read the printed baseline before trusting `-ngl`.**
The desktop is the other tenant of this GPU and its footprint drifts (measured
746–2188 MiB within one hour). `-ngl 12` fits at 765 MiB and breaches at 1114;
10 fits at 785 and breaches at 1592; 8 survives a loaded desktop. `--fit` (the
default) gives up layers automatically to hold the cap, so it needs none of
this arithmetic.

**MTP costs the 1070 compute buffer, so it trades against `-ngl`.** Measured:
with MTP on, `-ngl 32` at 128k aborts before load in `cudaMalloc` (`allocating
901.25 MiB on device 0`) and at 8k `-ngl 40` exits before health; the last
startable rungs are **30 at 128k, 36 at 32k, 38 at 8k**, all 1070-limited.
`serve.sh` derives two layers under those, because the probes peak at 5.2k depth
and the flash-attention workspace grows with context. On the Vega 10 alone MTP
was stricter still — it needed `-ngl 8` or lower and breached the display cap at
`-ngl 10` (measured twice) — which is why the Vega-only path (no eGPU) is slow
and `--fit` is what keeps it under the cap. Its win is also
configuration-dependent: **+97.8 %**
generation on Q4_K_M at `-ngl 8`, **+38 %** on the two-GPU 128k split with the
draft on the Vega, **+3.6 %** on Q2_K_XL at `-ngl 16` where the target is fast
enough that the draft becomes the bottleneck, and **+9 %** on a mid-document
continuation where only 26 % of drafts are accepted. `MTP=0` turns it off.

## Thunderbolt eGPU (optional)

The eGPU is optional: `serve.sh` uses it when present (splitting across
`CUDA0,Vulkan0`) and falls back to the Vega 10 alone otherwise. Three steps
were needed, in order.

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
* **`-sm row`/`-sm tensor` are unavailable on this build.** A parallelized split
  would put both GPUs on every layer, prefill included — the one structural
  answer to the pipelined split's single-GPU-at-a-time prefill. It needs each
  backend to export `ggml_backend_split_buffer_type`, which only the SYCL backend
  does: `llama-completion -sm row` dies with `device CUDA0 does not support split
  buffers` before loading. Not a tuning problem.
* **The Thunderbolt link trains at 2.5 GT/s ×4** (PCIe 1.0, ~1 GB/s) where 8.0
  GT/s is available. This is *not* the bottleneck — per token only ~20 KB crosses
  the CPU↔GPU boundary. The 1070's weaker prefill is simply that the Vega 48 has
  ~2× its FP32 throughput.

## Harness

| script | purpose |
|---|---|
| `scripts/build.sh` | configure + build llama.cpp with Vulkan, pinned revision |
| `serve.sh` | launch `llama-server` under the VRAM cap and the RAM floor |
| `scripts/guard.py` | VRAM/RAM watchdog + measurement: display-GPU VRAM, per-DRM-client VRAM, NVIDIA VRAM via NVML, `MemAvailable` (the only counter that sees the weights), RSS, `cpu_busy_pct`, `bg_cpu_pct` |
| `scripts/bench.py` | `llama-bench` sweeps: `ngl`, `kv`, `fa`, `depth`, `ctx`, `threads`, `batch`, `load`, `nkvo`, `nopo`, `quant` |
| `scripts/mtp_ab.py` | MTP A/B over HTTP, with `--server-extra` for arbitrary flags. Arms: `base`, `mtp-sidecar` (CPU draft), `mtp-sidecar-qkv`, `mtp-draftgpu` (`--draft-dev`/`--ngld-gpu`, the Vega-draft arm), `mtp-nomd` |
| `scripts/split_fit.py` | fit probe for the two-GPU split — `llama-bench` cannot do it. Sweeps `-ngl`/`-ts`, loads, issues one real completion, reports both GPUs' peaks. `--autofit` omits `-ngl`/`-ts` and drives llama.cpp's own `--fit` (`-fitt` margins) instead |
| `scripts/quant_compare.sh` | quant offload ceiling + throughput + perplexity |
| `scripts/wait_idle.py` | block until a workload's CPU — or the whole host's — goes quiet |
| `scripts/gguf_head.py` | read GGUF metadata from a local file or a remote URL without downloading it |
| `scripts/ckpt_probe.py` | context-checkpoint/compaction probe — one prompt, then the same prompt with a block rewritten, and the `prompt_n` that comes back; `--mode chat` sends the `message_delimiters` a chat request carries, `--mode raw` is the bare `/completion` probe that made the old "compaction pays full price" measurement look like a model property |

Five measurement traps this harness exists to catch, each of which produced a
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
  Run one model load at a time and check `MemAvailable` first. **`MemAvailable`
  is the only honest counter**: the CPU-side weights live in the Vulkan *host*
  buffer type (GTT) under `-lm dio` and in a file mapping under `-lm mmap`, so
  neither RSS nor the VRAM counters see them. Measured at 128k, Q4_K_M, `--fit`,
  full flag set: `dio` drops `MemAvailable` 22.8 GB with RSS 2.6 GB; `mmap`
  drops 6.6 GB. The old "`dio` peaks at ~300 MiB RSS" reading was measuring the
  one number that cannot see the weights.
* **The probe's API surface picks the code path.** Mid-prompt context
  checkpoints are gated on `message_delimiters`, which only a chat request
  carries, so a raw `/completion` prompt can never exercise them — the
  "compaction pays full price" measurement described the probe, not the model.
  The clients post to `/v1/chat/completions`; `scripts/ckpt_probe.py --mode chat`
  is the probe that matches them.
