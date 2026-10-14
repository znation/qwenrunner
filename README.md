qwenrunner

A local LLM inference stack for agentic coding on a 2019 Intel iMac (8-core /
16-thread Core i9, AMD Vega 10 / Radeon Pro Vega 48 with 8 GB — the display
GPU).

Usage of the display GPU should never, ever exceed 5 GB since that is also the
desktop.

The default (`./serve.sh`) runs one fully-resident instance on the Vega 10
only — no eGPU involved — fitted by llama.cpp's own `--fit` with a
`--fit-target` of `vram_total - cap`, so the display cap holds regardless of
desktop drift:

* **Qwen3.5-4B Q4_K_M** on the Vega 10 (`:8080`), under the 5000 MiB watchdog

The 9B cannot fit under the display cap (its Q4_K_M weights alone exceed 5000
MiB), so the 4B is the default model. Fully resident and memory-bandwidth-bound:
~29 t/s generation on the Vega. With the eGPU attached, `EGPU=1 ./serve.sh`
adds a **Qwen3.5-9B** main agent on the GTX 1070 (`:8080`) and moves the 4B to
`:8081` as a parallel secondary (~15.8 t/s / ~35.9 t/s). The previous default —
one Qwen3.8-27B split across both GPUs, most layers on CPU — ran at 2-6 t/s;
everything below is that config's measurement history, and it survives as
`LEGACY=1 ./serve.sh`.

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
| `--fit` (serve.sh `FIT=1`) | **off**, auto-fallback on a heavy desktop | llama.cpp's own fitter re-derives `-ngl`/`-ts` from live free VRAM with the cap as a `-fitt` margin: 28-30 layers at 128k, i.e. the shipped config to 2 MiB, or two layers (+4 %) faster — but it spends the ~620 MiB drift reserve and its choice moves with desktop noise. The formula is the default; when the live display baseline exceeds ~1200 MiB `serve.sh` warns and auto-falls back to `--fit` (the only path that gives up layers to hold the cap), unless `FIT=0` or `NGL=`/`TS=` is set |
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
binding constraint is that **only 8–12 of 64 layers fit in VRAM**, so the model
stays CPU-bound and desktop CPU load directly sets throughput.

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
two browsers. It is also the reason the shipped default is only safe while the
desktop baseline is **≤ ~1540 MiB** — a mid-range value of the 746-2188 MiB
drift, so the default is under-provisioned rather than broken, and `serve.sh`
warns instead of preventing it. When the display is tight the rung to drop to is
`-ngl 26`, not 24: measured, 24 and 26 cost the display the same and 26 keeps two
more layers on the 1070. And **compaction
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
(40 vs 12). The eGPU is now required: `serve.sh` exits rather than fall back to
the Vega 10 alone, which cannot hold the 128k pool under the cap.

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
./scripts/dl_dual.sh                             # the default pair: Qwen3.5-9B + Qwen3.5-4B
./scripts/dl_smaller.sh                          # optional: extra quants of the legacy 27B
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
./serve.sh                       # Qwen3.5-4B on the Vega 10 (:8080) — no eGPU needed
CTX=32768 NP=2 ./serve.sh        # two 16k slots instead of one 64k (parallel subagents)
EGPU=1 ./serve.sh                # with the eGPU: Qwen3.5-9B on the 1070 (:8080) + 4B on the Vega (:8081)
LEGACY=1 ./serve.sh              # the old Qwen3.8-27B split across both GPUs
```

The server always runs llama.cpp's own `--fit on` with a `--fit-target`
(`-fitt`) — the layer count is derived from live free VRAM, never by hand. The
Vega instance leaves `vram_total - CAP` free (`FITT`, 3176 MiB here), which —
because the fit sees the desktop's usage in `free` — holds the display cap
regardless of desktop drift. At the default context the 4B is fully resident,
so there is no CPU inference path and no MTP (pointless at memory-bandwidth
speeds). `scripts/guard.py` is the backstop.

The default is `CTX=65536` with `-np 1` → **one 65536-slot** — the same per-slot
context the clients already expect (pi `contextWindow` 59392), at a ~3.3 GiB
model footprint that leaves ~1.7 GiB of display-cap headroom at a typical
desktop baseline. `NP=2` halves the pool per slot but buys fairness for
concurrent agents.

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
`serve.sh` passes `-a Qwen3.5-9B` / `-a Qwen3.5-4B` so clients see stable model
ids instead of the .gguf paths; the alias is advisory — llama-server ignores the
request's `model` field, so a client still sending an older id works. The Vega
secondary on `:8081` is a second, fully parallel server — point a subagent
fleet at it for a second lane of work that never contends with the main agent.

`serve.sh` prints the configuration it picked. The default needs no eGPU at
all — it can be powered off; `EGPU=1` and `LEGACY=1` require it and error out
if `/dev/nvidia0` is absent or the binary was built without CUDA.

### EGPU=1 — the two-GPU mode

`EGPU=1 ./serve.sh` runs Qwen3.5-9B on the GTX 1070 (`:8080`, `MAIN_*` env) and
Qwen3.5-4B on the Vega (`:8081`, `VEGA_*` env), both fully resident, each
fitted by `--fit` with its own `--fit-target` (256 MiB on the eGPU, vram_total
− cap on the display GPU). The 9B's hybrid DeltaNet keeps its KV cache to 8 of
32 layers (~1.1 GiB at the 131072 pool). Measured: ~15.8 t/s on the 1070 and
~35.9 t/s on the Vega.

### LEGACY=1 — the Qwen3.8-27B split (the previous default)

`LEGACY=1 ./serve.sh` restores the old single model split across both GPUs:
one Qwen3.8-27B instance on `-dev CUDA0,Vulkan0 -sm layer`, the remainder on
CPU, with MTP speculation. Everything below this point is that config's
documentation — the measured layer formula, the `--fit` fallback, the draft
chain — and applies to it, not to the default above.

`serve.sh` runs `llama-server` under `scripts/guard.py`, which polls system-wide
VRAM every 50 ms and `SIGKILL`s the process group if it exceeds `CAP`
(default 5000 MiB), so the display GPU is protected by measurement rather than
by arithmetic. It prints the live display baseline and warns when it is above the
~1216 MiB the default config was measured to survive. It forwards
`SIGINT`/`SIGTERM` to the child, so Ctrl-C actually stops the server.

The layer count comes from the measured formula (`32 - (CTX-32768)/16384` with
MTP — 32, not 34, because the shipped 6-token draft chain moves the 128k ceiling
from 30 to 28; see RESULTS.md "The draft chain spends the 1070's reserve"), two
layers under the last startable rung, which leaves ~570 MiB of the cap clear at a
1457 MiB baseline — safe while the desktop baseline is ≤ ~1880 MiB, i.e. most but
not all of the 746-2188 MiB range it drifts over (the top of that range needs
`FIT=1` or fewer layers). `FIT=1` instead hands `-ngl` and the tensor
split to llama.cpp's own `--fit`, with the cap encoded as a `-fitt` margin
(`vram_total - CAP` on the Vega, so it is baseline-independent); measured at
128k it picks 28-30 layers — the same config to 2 MiB, or two layers faster —
but it spends that reserve and its choice moves with a few MiB of desktop
noise, so it is opt-in. `serve.sh` auto-falls back to `--fit` (with a warning)
when the live display baseline exceeds ~1200 MiB, where the fixed `-ngl` would
otherwise breach the cap; an explicit `FIT=0`, or setting `NGL=`/`TS=`, keeps
the formula. See [`RESULTS.md`](RESULTS.md) for both tables.

**Read the printed baseline before trusting `-ngl`.** The desktop is the other
tenant of this GPU and its footprint drifts (measured 746–2188 MiB within one
hour). `-ngl 12` fits at 765 MiB and breaches at 1114; 10 fits at 785 and
breaches at 1592; 8 survives a loaded desktop. Pick from the measured client
peak, not from a constant.

**MTP costs the 1070 compute buffer, so it trades against `-ngl`.** Measured:
with MTP on, `-ngl 32` at 128k aborts before load in `cudaMalloc` (`allocating
901.25 MiB on device 0`) and at 8k `-ngl 40` exits before health; the last
startable rungs are **30 at 128k, 36 at 32k, 38 at 8k**, all 1070-limited.
`serve.sh` derives two layers under those, because the probes peak at 5.2k depth
and the flash-attention workspace grows with context. On the Vega 10 alone MTP
was stricter still — it needed `-ngl 8` or lower and breached the display cap at
`-ngl 10` (measured twice) — and that path is no longer a `serve.sh` mode now
that the eGPU is required. Its win is also configuration-dependent: **+97.8 %**
generation on Q4_K_M at `-ngl 8`, **+38 %** on the two-GPU 128k split with the
draft on the Vega, **+3.6 %** on Q2_K_XL at `-ngl 16` where the target is fast
enough that the draft becomes the bottleneck, and **+9 %** on a mid-document
continuation where only 26 % of drafts are accepted. `MTP=0` turns it off.

## Thunderbolt eGPU (optional)

Needed only for the `EGPU=1` and `LEGACY=1` modes; the default runs on the
Vega 10 alone, so the eGPU can stay powered off. Three steps were needed, in
order.

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
| `serve.sh` | launch `llama-server` under the VRAM watchdog |
| `scripts/guard.py` | VRAM watchdog + measurement: display-GPU VRAM, per-DRM-client VRAM, NVIDIA VRAM via NVML, RSS, `cpu_busy_pct`, `bg_cpu_pct` |
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
  Run one model load at a time and check `MemAvailable` first. `-lm dio` peaks at
  ~300 MiB RSS where the mmap modes hold 13–14 GB resident.
* **The probe's API surface picks the code path.** Mid-prompt context
  checkpoints are gated on `message_delimiters`, which only a chat request
  carries, so a raw `/completion` prompt can never exercise them — the
  "compaction pays full price" measurement described the probe, not the model.
  The clients post to `/v1/chat/completions`; `scripts/ckpt_probe.py --mode chat`
  is the probe that matches them.
