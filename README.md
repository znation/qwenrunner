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
| KV cache | **`q4_0/q4_0`** (default): 2 GiB per 128k slot, **+12 GiB of free RAM** against f16 | this machine shares its 128 GB with other workloads (local Kubernetes clusters), so the cache is sized by what has to be left over. Paired measurement, same session, `NP=2`/128k/`CRAM=4096`, two requests served: **33.4 GiB free** at q4_0 against 21.5–21.9 at f16, byte-identical output. It is the **slowest** of the three at depth — 26.2 t/s against 30.5 (`q8_0`) and 47.2 (`f16`) at 98.6k with MTP — because the 7-row verify batch dequantizes a quantized cache once per row. Quality gate: perplexity at 65 536 tokens of context is **3.3946 ± 0.0357 against f16's 3.3809 ± 0.0355** (+0.4 %, inside one SE), and 16k agrees (1.9793 vs 1.9780). `CTK=f16 CTV=f16 CRAM=8192` is the speed configuration. Never mix K and V dtypes: all four mixed runs hung past a 1800 s request timeout. See `RESULTS.md`, "KV dtype" |
| threads | **`-t 4`** | pp 758 vs 716 (t6), 693 (t8), 686 (t12), 529 (t18); generation flat from 4–18. The 12 performance cores cost more in contention than they return |
| batch | **`-b 1024 -ub 256`** — kept, but its depth justification did not reproduce | the depth case for the small ubatch (generation 37.45 vs 28.52 t/s at 98.6k, one rep per arm) does **not** survive interleaved reps: 30.2 vs 29.8 t/s mean, with ±20 % scatter *inside* a single config. What does reproduce: prefill at depth prefers the **larger** ubatch (−18 % at `-ub 128`; `-ub 512` best), shallow prefill prefers `-b 2048 -ub 512` (709.6 vs 633.3), and shallow generation prefers the small ubatch (28.76 vs 25.54). See `RESULTS.md`, "Batch and threads at the operating point" |
| MTP | **off** by default; `MTP=1 NMAX=6 ./serve.sh` for one interactive stream | draft on the GPU. **The gain is a property of the text, not the machine**: +180 % on patterned text (markdown/code/JSON, 99 % draft acceptance) and **−11 % on free-form prose** (14 %). It is also a property of the *concurrency*: at the eight slots the fleet shape runs on, the draft head runs per sequence on a GPU the eight rows already saturate, and it measures **46.6/48.0 t/s off against 33.2/32.2 with `NMAX=6`**, at +16.6 GiB of draft/verify buffers (`RESULTS.md`, "The fleet shape"); at the two slots the server now defaults to that case is weaker (MTP is a per-sequence lever). The draft depth is 6 because the repo serves agent/code traffic, where it is worth +40 % over upstream's 3; prose wants `NMAX=3`. At 98.6k depth, one stream: 28.2 t/s (`NMAX=6`) against 15.2 with `MTP=0` |
| context | **128k** (`-c 131072`) | 14.3 t/s generation and 176–231 t/s prefill *at* depth 131072; costs ~8 GiB of KV with f16 |
| prefill cache | **`--cache-ram` ≥ (sessions − slots) × ~1 GiB** (host RAM) | for agentic use this is the biggest lever there is. A growing transcript re-sent each turn prefills **only the delta**: 19 new tokens cost 0.41 s against 62.9 s cold. Upstream's 8 GiB budget holds exactly *one* 128k f16 conversation — with two, the second evicts the first and every turn silently re-prefills 29 772 tokens (measured: 57.1 s). An entry is **not** a conversation's KV: llama.cpp logs 637-820 MiB for a 2.6k-token conversation and 762-1 150 MiB for 7-9k, so budget ~0.65-1.2 GiB each. At one slot per harness nothing is displaced and the cache stays empty; at the default `NP=2` with up to 4 sessions it carries the two sessions beyond the slots (4096 ≈ two displaced 128k entries or four 30k-token ones at `q4_0`); a session-starved shape needs more: eight sessions on four slots thrash at 4096 (measured 2026-10-30: 21.9 against 20.0 t/s for `CRAM=0`, and 31.7 at `8192` — throughput back, latency not) |
| parallel prompts | **one process, `NP` = the concurrency you serve**, beats **`INSTANCES`** (processes) | 128k per slot: `-np 6` fits in **70.4 GiB** where 6 processes die of a Metal OOM, and loads 2-3x faster. At the fleet shape measured 2026-10-30 (8 harnesses, one appending transcript each, 4 turns): one `-np 8` process **37.7-48.0 t/s at TTFT p50 2.2 s and 39 GiB**, `INSTANCES=2 -np 4` 34.8-39.9 t/s at 3.6-3.9 s and 54-56 GiB, `INSTANCES=4 -np 2` 27.6-28.7 t/s at 6.1-7.0 s and 83-85 GiB. Slots buy concurrency and latency, not throughput |

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
not the ~40 GiB an all-attention 27B would need. The other 48 blocks carry
per-sequence state instead of a growing cache — and it is not the ~10 MiB the
layer shapes suggest: **a serialized sequence state measures 637-820 MiB for a
2.6k-token conversation and 762-1 150 MiB for 7-9k ones**, i.e. ~0.6 GiB that is
not that conversation's KV at all (`RESULTS.md`, "The fleet shape"). It is part
of what a slot costs beyond its KV, and it is what a prompt-cache entry costs, so
budget ~1 GiB per displaced conversation rather than `conversation_tokens x
16 KiB`.

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
./serve.sh                       # the default: 2 slots, MTP off, q4_0, 4 GiB cache
NP=8 ./serve.sh                  # eight slots: the measured fleet-shape optimum
CTK=f16 CTV=f16 CRAM=8192 ./serve.sh  # +20 t/s decode at depth, -12 GiB of free RAM
NP=1 MTP=1 ./serve.sh            # one interactive stream, speculation on
MTP=1 NMAX=3 ./serve.sh          # prose stream: speculation pays little, 3 deep
INSTANCES=4 NP=2 ./serve.sh      # four servers, ports 8090-8093
```

Never mix `CTK` and `CTV`: every mixed-dtype run hung past a 1800 s request
timeout on this machine (`RESULTS.md`, "KV dtype").

`serve.sh` defaults: `-ngl 99 -np 2 -c 262144 (2 slots x 128k) -t 4 -b 1024 -ub
256 -ctk q4_0 -ctv q4_0 -fa on --cache-ram 4096`, with MTP off (`MTP=1` adds
`--spec-type draft-mtp -md models/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99
--spec-draft-n-max 6`). Every knob is an env var (`MODEL CTX NP INSTANCES PORT
THREADS NGL LM BATCH UBATCH CTK CTV FA MTP MD NGLD NMAX CRAM SWAPCAP`), and extra
arguments are appended verbatim to each server.

**The default is sized for agentic coding on a shared machine**: two slots at
`q4_0` reserve 4 GiB of KV, and the loaded server measured **32.4 GiB free** here
— more than the four-slot q4_0 configuration's 22.6 and the two-slot f16 one's
21.5. The 4 GiB prompt cache keeps two more conversations warm, so a four-session
fleet stays cached without a re-prefill. `NP=8` buys the throughput optimum back
(`-np 8` measures 37.7-48.0 t/s at TTFT p50 2.2 s at the eight-harness shape) for
~10 GiB of RAM; `NP=3` sits at 28.1 free. See the ceiling note in `RESULTS.md`
if you plan to fill the prompt cache.

Servers run under `scripts/guard.py`, which forwards Ctrl-C to the whole
process group and records memory/CPU/GPU statistics for every run. The one
thing it can still enforce is swap: `SWAPCAP=<MiB>` SIGKILLs on new swap, off
by default because nothing here should swap.

### Running more than one prompt at once

Two mechanisms, and they are not equivalent. `NP=N` runs N parallel slots
inside **one** server (one port, one weight mapping, one set of compute
buffers); `INSTANCES=N` runs N separate servers on N ports.

```
NP=2 ./serve.sh                  # one server, two parallel 128k slots (the default)
NP=8 ./serve.sh                  # eight slots: the measured fleet-shape optimum
INSTANCES=4 NP=2 ./serve.sh      # four servers, ports 8090-8093
```

Measured, 128k context per slot, MTP on, current defaults:

| N | mechanism | total RSS | load | aggregate t/s | per-stream t/s |
|---|---|---|---|---|---|
| 2 | `-np 2` | 34.0 GiB | 4 s | 59.5 | 29.8 |
| 4 | `-np 4` | 52.2 GiB | 5 s | 52.9 | 13.2 |
| **6** | `-np 6` | **70.4 GiB** | 11 s | 53.2 | 8.9 |
| 2 | 2 processes | 49.8 GiB | 8 s | 55.5 | 27.7 |
| 4 | 4 processes | 99.7 GiB | 13 s | 66.9 | 16.7 |
| 6 | 6 processes | — | — | Metal OOM | — |

**Slots are the cheap axis: `-np 6` fits where 6 processes die of a Metal
out-of-memory**, at 70 GiB instead of beyond the ~112 GiB working-set budget,
and it loads 2-3x faster. `serve.sh` scales `-c` to `CTX x NP` so every slot
keeps the full 128k — verified in the server log as
`n_slots = 6, n_ctx_slot = 131072`.

What parallelism does to speed depends on the workload, not the mechanism:

* On text the draft head predicts well, a single stream is already near the
  GPU's limit and N streams just divide it (per-stream 29.8 -> 13.2 -> 8.9 for
  N = 2 -> 4 -> 6, aggregate flat at ~53-60 t/s).
* On code, where acceptance is lower and each token costs more compute, two
  concurrent requests measured **44.3 and 46.8 t/s — ~91 t/s aggregate**, above
  any single-stream figure for the same kind of prompt. Concurrency *added*.

Neither is the wrong answer: N slots buy concurrency. Use them when N clients
each idle most of the time; do not expect N times the throughput of one.

`serve.sh` defaults to `NP=2 CTK=CTV=q4_0 MTP=0 CRAM=4096` and `INSTANCES=1`:
two slots at the RAM-first dtype, speculation off, with the prompt cache sized
to keep four concurrent sessions warm (two resident, two cached). A slot's KV is
only 2 GiB at 128k here, but it also carries its own Metal compute buffers and
the model's per-sequence state, and the measured all-in cost is **~3 GiB per
extra slot at 64k/slot and ~4 at 128k** — `-np 8` measures 39.4 GiB of summed RSS
at 64k/slot and 41.5 GiB at `CTX=131072` with 22k-token transcripts. `NP=8
./serve.sh` buys the larger fleet's throughput back (37.7-48.0 t/s, TTFT p50
2.2 s, 39 GiB at the eight-harness shape) for ~10 GiB of RAM, and `NP=1 MTP=1
./serve.sh` is the single interactive stream.

#### For agentic coding

The workload is a transcript that grows every turn, so the config that matters
is the one that keeps prefill near zero and gives each concurrent session its
own cached conversation:

`./serve.sh` with no arguments *is* this configuration — `NP=2 CTK=q4_0 CTV=q4_0
MTP=0 CRAM=4096` are the defaults. What each one buys:

* `MTP=0` — off by default. The measurement that sets it is the eight-slot
  shape: there the draft head runs *per sequence* while the main batch already
  carries eight rows, and the same interleaved window measures **46.6 / 48.0 t/s
  with `MTP=0` against 33.2 / 32.2 with `NMAX=6`**, at +16.6 GiB of draft/verify
  buffers. At the two slots this server now defaults to that case is weaker —
  MTP is a per-sequence lever and there is spare batch — so `NP=2 MTP=1` is
  worth measuring before treating this default as settled at the new shape. It
  remains a single-stream lever either way: `NP=1 MTP=1 ./serve.sh` measured
  +180 % on patterned text (markdown/code/JSON, 99 % draft acceptance), −11 % on
  free-form prose, and 28.2 against 15.2 t/s at 98.6k depth — and `NMAX=6` is
  the depth for code (prose wants 3).
* `NP=2` — two live slots, with the prompt cache carrying the rest of the fleet,
  so up to four concurrent sessions stay warm (two resident, two cached). Slots
  are the cheap axis and more of them is what a large fleet wants: one slot per
  session at tumwater's `maxConcurrent: 8` measured 37.7-48.0 t/s at TTFT p50
  2.2 s and 39 GiB. Too few slots re-prefills in full — eight appending
  transcripts on four slots measure TTFT p50 49.7-62.6 s against 2.2 s and 22
  against 46 t/s aggregate — while slots beyond the fleet buy nothing (`-np 12`
  at 8 sessions measures 30.0/32.1 against 33.2/32.2 t/s for +10 GiB). Cost is
  ~3 GiB per extra slot at 64k/slot and ~4 at 128k with MTP off (`-np 8`
  measured 38.5-39.4 GiB summed), ~5.6 with MTP on (`RESULTS.md`, "The fleet
  shape").
* `CTK=CTV=q4_0` — the RAM decision: 2 GiB per 128k slot against 8 at f16, worth
  **+12 GiB of free memory** to everything else on the machine, at the cost of
  the slowest decode of the three (26.2 against 30.7 at `q8_0` and 46.1 at f16,
  t/s at 98.6k). Perplexity at 65 536 tokens of context is unchanged within one
  standard error (`RESULTS.md`, "KV dtype").
* `CRAM=4096` — the prompt cache, sized as `(sessions − slots) × ~1 GiB`; at the
  default (`NP=2`, up to 4 sessions) two conversations are displaced, and 4096
  covers them with headroom for the ~2.1 GiB a 128k `q4_0` entry costs. Its
  entries are not a conversation's KV: llama.cpp logs **637-820 MiB for a
  2.6k-token conversation and 762-1 150 MiB for 7-9k**, so `conversation_tokens
  × 16 KiB` is the wrong budget. A session-starved shape needs it bigger: eight
  sessions on four slots thrash at `4096` (17 evictions in one 4-turn run, 21.9
  against 20.0 t/s for `CRAM=0`), and `8192` is that shape's floor — twice the
  displaced conversations, twice the cache. It recovers 69 % of
  one-slot-per-harness throughput (31.7 t/s) but not its latency (TTFT p50
  24.2 s against 2.2 s). A cache that is one entry short does not degrade
  gracefully: it silently re-prefills the whole transcript on every turn.

Send requests with `"cache_prompt": true` (the default for the OpenAI-compatible
endpoints) so the server can match the prefix; `scripts/agent_cache_probe.py`
measures whether it actually did.

There is a third, disk-backed tier (`SLOT_SAVE=<dir>` → `--slot-save-path`, with
`POST /slots/{id}?action=save|restore`). It round-trips a 30k conversation's
1.11 GiB state in **346 ms / 112 ms**, but a restored slot reports no prompt
(`n_prompt_tokens = None`) and the next request re-prefilled all 29 772 tokens
anyway. It is off by default and documented as unproven, not as a tier —
`RESULTS.md`, "Disk-backed slot states".

Two things to know about this default:

* **It is not free.** Two q4_0 slots reserve 4 GiB of KV at 128k, and a slot
  costs ~3 GiB all-in at 64k/slot (~4 at 128k), not its 2 GiB of KV — at large
  `N` that adds up: `-np 8` measures **39.4 GiB of summed RSS** at 64k/slot and
  41.5 GiB at `CTX=131072` with 22k-token transcripts. `CRAM` is a ceiling on top
  of that and is empty until a displaced conversation needs it. Raising `NP` to
  8 costs ~10 GiB of RAM here; `NP=1 MTP=1 ./serve.sh` is one interactive
  session with speculation on.
* **Overlapping cold prefills hurt.** Slots share one batch, so while one
  session prefills an uncached transcript the others' generation falls to ~3.5
  t/s (from 93.6). Cached turns prefill only their delta and are unaffected, and
  with one slot per harness that is every turn; the effect bites when sessions
  outnumber slots, which is what `CRAM` and `INSTANCES` are for.

#### Which one to use

`NP` and `INSTANCES` differ on three axes, and only one of them is speed.

| | `NP=N` (slots, one process) | `INSTANCES=N` (N processes) |
|---|---|---|
| memory for 4 x 128k | **52 GiB** | 100 GiB |
| largest measured N at 128k | **6+** | 4 (6 = Metal out-of-memory) |
| load time | **4-5 s** | 9-13 s |
| aggregate t/s at N=4, 128k slots, one prompt shape | 60.9 | **65.9** (+8 %) |
| **8 harnesses, one appending transcript each (the fleet shape)** | **42.5 t/s, TTFT p50 2.2 s, 39 GiB** (`-np 8`) | 37.7 t/s, 3.6-3.9 s, 54-56 GiB as 2 × `-np 4`; 28.0 t/s, 6.5 s, 84 GiB as 4 × `-np 2` |
| failure domain | one process — a crash or OOM drops **all** slots | per process — the others keep serving |
| restart / upgrade one worker | no | **yes** |
| per-port configuration | no | **yes** (see below) |
| CPU threads | one `-t` pool shared by all slots | each process gets its own `-t` pool |
| one session prefilling a cold transcript | **starves the others: 93.6 -> 3.5 t/s** (27x) | degrades: 83.4 -> 22.3 t/s (3.7x) |
| ports | one | N |

**Use `NP=N`** when you just want N concurrent clients against one endpoint and
care about memory: it is the difference between 6 parallel 128k conversations
and 4, and it starts in a third of the time. On the fleet shape it is also the
fastest of the three splits, at the lowest latency and the least RAM — see the
row above and `RESULTS.md`, "The fleet shape".

**Use `INSTANCES=N`** when you need any of the things one process cannot give
you: fault isolation (an OOM in one worker does not kill the rest), the ability
to restart or upgrade one port while the others serve, separate logs,
**different settings per port** (see below) — or **overlapping cold prefills**.
Slots share one batch, so a slot prefilling a 30k-token transcript drops its
neighbours' decode from 93.6 to 3.5 t/s; separate processes share only the GPU
and degrade to 22.3 instead. For a fleet of eight, `INSTANCES=2 NP=4` is that
middle ground and it was measured: 37.7 t/s at TTFT p50 3.8 s and 54-56 GiB,
against 44.4 t/s at 2.2 s and 39 GiB for one `-np 8` process — the isolation
costs ~15 % of throughput, 1.7 s of TTFT and 16 GiB. With one
`serve.sh` every server gets the same flags, but nothing stops you running two:

```
PORT=8090 NP=1 MTP=1 NMAX=6 ./serve.sh &   # one interactive stream: speculation pays
PORT=8091 ./serve.sh &                      # the fleet default: 2 slots, MTP off
```

That is not expressible with `-np`, and it follows directly from the MTP
measurement above — the two workloads want opposite settings. Note each process
takes the whole weight mapping as its own GPU working set, so keep the process
count low (4 measured) and prefer slots when you only need more clients.

The +8 % is confounded: four processes also get four `-t 4` CPU thread pools
(16 threads) against one shared pool for slots. If you run `NP=4` and want that
back, `THREADS=8` is the untested knob.

## Clients — pi, omp, tumwater

`serve.sh` advertises the model as **`qwen3.8-27b`** (llama-server's `--alias`,
overridable with `ALIAS`). Without an alias the server reports the GGUF's own
name, which changes with the file, so every client would have to track it — both
catalogs below name this string.

The API is OpenAI-compatible at `http://127.0.0.1:8090/v1`; the API key is a
placeholder that llama-server ignores.

The port is **8090, not llama-server's 8080**. Rancher Desktop's Lima port
forwarder holds a wildcard `*:8080` on this host, so a request that arrives
while the server is down is answered by the forwarder — a bare Go
`404 page not found` — instead of being refused: the failure names the wrong
service, and a liveness probe on that port cannot tell "up" from "forwarded".
While the server runs, its specific `127.0.0.1` bind wins and requests reach it,
so nothing was broken by 8080; the move makes the down case honest. Any port the
local cluster does not publish works — `PORT` overrides it, and the two
catalogs below must move with it (tumwater names no URL of its own: it routes
through pi's `models.json`).

| client | file | entry |
|---|---|---|
| pi | `~/.pi/agent/models.json` | provider `qwenrunner`, model `qwen3.8-27b`, cost 0 |
| omp | `~/.omp/agent/models.yml` | same provider and model |

Two properties are load-bearing:

* **It must price at zero.** tumwater's budget gate refuses a priced fallback
  (it would spend past the cap), so the catalog entries carry
  `cost: {input: 0, output: 0, ...}` like the other local providers. Verified
  with the harness's own doctor while this pair was tumwater's fallback:
  `ok fallback — qwenrunner/qwen3.8-27b — priced at zero (cost n/a), serving not verified`.
  "Serving not verified" is the doctor's normal wording for a free pair: only the
  fallback's own ticks can prove the backend answers, and **the server must be
  running** for any client to use it (`./serve.sh`).
* **`reasoning: true`** — the model emits `reasoning_content` before its answer
  (verified on `/v1/chat/completions`), so a small `max_tokens` can be consumed
  entirely by thinking and return empty `content`.

## Harness

| script | purpose |
|---|---|
| `scripts/build.sh` | configure + build llama.cpp with Metal, pinned revision |
| `serve.sh` | launch the server(s) under the guard: `NP` slots in one process, `INSTANCES` processes |
| `scripts/guard.py` | guarded runner + measurement: RSS, swap, host/GPU utilization, `bg_cpu_pct`, `gpu_busy_pct_peak` |
| `scripts/bench.py` | `llama-bench` sweeps: `ngl`, `fa`, `kv`, `depth`, `ctx`, `threads`, `batch`, `load`, `nkvo`, `nopo`, `quant` |
| `scripts/mtp_ab.py` | MTP A/B over HTTP, with `--server-extra` for arbitrary flags |
| `scripts/parallel_ab.py` | `-np N` slots vs N processes: how many 128k slots fit, and what N does to throughput |
| `scripts/fleet_ab.py` | the fleet shape: interleaved A/B of `serve.sh` configurations against the turn loop (`NP`/`INSTANCES`/`MTP`/`CRAM`), one server at a time, arm order reversed every rep |
| `scripts/turn_loop_bench.py` | the fleet workload: N sessions, each with a growing transcript, fired concurrently (`--base-urls` spreads them over `INSTANCES` ports) |
| `scripts/agent_cache_probe.py` | does the prefill cache actually fire? Reports `prompt_n` for a re-send, an A/B/A eviction, and a growing transcript |
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
* **macOS ships bash 3.2**, where `"${ARR[@]}"` on an *empty* array is an
  unbound-variable error under `set -u`. `MTP=0 ./serve.sh` therefore died on
  `line 161: MTP_ARGS[@]: unbound variable` — a flag the README had advertised
  since before this branch. The portable form is
  `${ARR[@]+"${ARR[@]}"}`, which is what `serve.sh` uses now. Any script here
  that builds an argv array conditionally needs the same guard.
