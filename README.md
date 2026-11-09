# qwenrunner

A local inference stack for **Qwen-3.8-27B** on an **Apple M5 Max with 128 GB**
of unified memory, tuned for one workload: the [tumwater](../tumwater) agent
fleet, **four pi sessions at once** against one server. It runs
[Splash](https://github.com/incoai/splash) on MLX 4-bit weights with a DFlash2
drafter.

The model is a hybrid: 64 blocks, only 16 of them full attention, the rest
linear-attention (Gated DeltaNet). That is why 128k of context is cheap here:
the KV cache grows in 16 layers, not 64.

## Result

Measured by replaying real tumwater sessions against each engine:
`scripts/tumwater_replay.py`, 4 sessions at once, 16 scored turns each at
25-55k tokens of depth. Two reps per arm (Splash's default has three). Full
table and method in [`RESULTS.md`](RESULTS.md), "Engine shoot-out".

| engine (weights) | turns/min | aggregate out t/s | per-stream t/s | TTFT p50 / p90 |
|---|---:|---:|---:|---:|
| **Splash 1.6.0 (MLX 4-bit + DFlash2)**, `./serve.sh` | **15.0-16.0** | **64-72** | 27-34 | 2.2-3.4 / 10-14 s |
| Kapitsa 0.2.31 (`27b` 4-bit + DFlash2, 4 slots) | 10.7-11.2 | 45-47 | 15 | 1.7-2.1 / 4.6 s |
| mlx-serve 26.11.2, `NP=4`, 32 GB prefix cache | 10.7-11.3 | 17-19 | 16-18 | 6.6-8.3 / 21-31 s |
| oMLX 0.7.3 (MLX 4-bit + DFlash2) | 6.6 | 26 | 9 | 5.7 / 12.8 s |
| mlx-serve 26.11.2, as previously shipped (`NP=2` cache budget) | 3.9 | 6 | 18 | 11.5 / 128 s |

**This is 4.1× the previous default's turn rate and ~10× its output
throughput** on the same machine and the same model.

| lever | setting | evidence |
|---|---|---|
| runtime | **Splash 1.6.0** (`ENGINE=splash`, default) | the only engine measured here whose speculative decoding stays on while four requests decode: its DFlash2 draft/verify cycle runs batched across all four lanes, so each stream keeps ~29 t/s at 40k depth beside three others. mlx-serve's speculation is a single-stream win (73 t/s alone) that falls to ~16 t/s per stream once sessions overlap; oMLX's batched DFlash runs at ~9 |
| weights | **`mlx-community/Qwen3.8-27B-4bit`** (affine 4-bit, 16.1 GB) | mxfp4 (15.2 GB) and GGUF UD-Q4_K_M measure the same on Splash: 16.3/15.6 and 14.7/15.7 turns/min against 15.0/15.7. Format is not a lever on this engine, so the affine build is kept |
| drafter | `incoai/Qwen3.8-27B-DFlash2` | picked and downloaded by Splash itself |
| concurrency | **4** (`NP=4`) | tumwater's fleet size. Splash decodes 4 lanes at once by design; a fifth request queues |
| context | **128k** per session | tumwater's runs peak around 64-80k; recorded request p90 is 81k |
| prefix cache | Splash's own, plus a 32 GB SSD tier | 96.6 % of prompt tokens served from cache on the replay, the same as every other engine once it is sized for four sessions |
| idle release | **off** | Splash unwires the weights after 10 idle minutes by default, so the next turn after a quiet spell would pay a reload |

**The previous default lost most of its speed to its prefix cache, not its
kernels.** mlx-serve sized the cache for *one* 128k session (9.4 GB). Four
tumwater sessions at ~55k tokens are ~4.8 GB each, so they evicted each other,
and a quarter of all prompt tokens were re-prefilled: turns at 40-55k depth waited
100-410 s for a first token. Raising the budget to 32 GB alone took it from 3.9 to
10.7 turns/min. `ENGINE=mlx-serve` keeps that engine runnable, with the cache
now sized from `NP`.

## Hardware

| | |
|---|---|
| CPU | Apple M5 Max, 18 cores (6 super + 12 performance) |
| GPU | integrated, 40 cores, `MTLGPUFamilyMetal4`, unified memory |
| RAM | 128 GB unified (`recommendedMaxWorkingSetSize` ≈ 112 GiB) |
| OS | macOS 27 |

## Install

```bash
brew tap incoai/tap
brew install incoai/tap/splash
```

On its first start `serve.sh` downloads the weights (16.1 GB) and the DFlash2
drafter (3.8 GB) into Splash's own store
(`~/Library/Application Support/Splash/models`).

The previous engine stays available as `ENGINE=mlx-serve`. To install it, use
its release binary; the Homebrew tap builds from source, which takes over 30
minutes:

```bash
mkdir -p ~/opt/mlx-serve && cd ~/opt/mlx-serve
curl -sL https://github.com/ddalcu/mlx-serve/releases/latest/download/mlx-serve-bin-macos-arm64.tar.gz | tar xz
~/opt/mlx-serve/mlx-serve-macos-arm64/mlx-serve pull ddalcu/Qwen3.8-27B-MLX-Serve-4bit
```

## Run

```bash
./serve.sh                       # Splash, 4 sessions, 128k, port 8090
ENGINE=mlx-serve ./serve.sh      # the previous engine (NP=4, 32 GB prefix cache)
MODEL=mlx-community/Qwen3.8-27B-mxfp4 ./serve.sh
```

OpenAI-compatible at `http://127.0.0.1:8090/v1`. Logs go to
`logs/<engine>.8090.log`.

Env: `ENGINE MODEL CTX NP PORT HOST ALIAS LOG_DIR`, plus `CACHE_MB MTP` for
mlx-serve.

## Clients

`pi` and `omp` are pointed at this server from their own configs:
`~/.pi/agent/models.json` and `~/.omp/agent/models.yml`, provider `qwenrunner`,
model id `Qwen3.8-27B-MLX-Serve-4bit`. The switch to Splash needs **no client
change**. `serve.sh` announces the model as `Qwen3.8-27B` and also accepts the
old id. This matters because Splash rejects a model name it does not serve, with
`model_not_found`.

Checked against Splash with the old id (`scripts/toolcall_smoke.py`):

| field | value |
|---|---|
| tool calls | streamed `delta.tool_calls` with JSON arguments; parallel calls work; `finish_reason: tool_calls` |
| reasoning | in a separate `reasoning_content` field, never inline `<think>`, so pi's strip-thinking sees it |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max` are all accepted; `none` turns thinking off |
| context | 131 072 per conversation (`context_length` at `GET /v1/models`) |
| input | text, image, PDF |

For tumwater, the fleet's `maxConcurrent` plus the director should not exceed
the server's 4 lanes. A fifth request queues behind a whole turn rather than
joining the batch.

## Measured

### The tumwater shape

Workload, from 19 358 recorded tumwater turns (508 pi sessions):

| | p50 | p90 |
|---|---:|---:|
| request prompt | 39.5k tokens | 81k |
| new tokens per turn | 747 | 3.2k |
| output tokens per turn | 333 | 2.2k |
| turns per session | 29 | 78 |

About one turn in 23 carries a context edit: tumwater's context-shake elides an
old tool output in place, which invalidates the cached prefix from that point
on. The replay reproduces those edits where they happened.

So a turn decodes a few hundred tokens at 20-80k depth after prefilling ~750
new ones, with four sessions at once. What decides it is batched decode at
depth, and a prefix cache that holds all four sessions.

### Single stream (previous engine, for reference)

`mlx-serve` alone, one request, nothing cached (`scripts/openai_stream_bench.py`):
73.4 t/s decode at 4k, 61.7-82.0 at 30k, 46.9 at 128k, and 461 t/s prefill at
128k. That single-stream lead is real. It just does not survive four
overlapping sessions.

## Harnesses

All stdlib-only, and all drive any OpenAI-compatible endpoint:

```bash
# tumwater replay: extract a fixture from the local pi session store (keep it
# out of the repo: it is real transcript content), then replay it
python3 scripts/tumwater_replay.py extract --out /tmp/tumwater_replay.json
python3 scripts/tumwater_replay.py run --base-url http://127.0.0.1:8090/v1 \
    --model Qwen3.8-27B --fixture /tmp/tumwater_replay.json --label my-arm -v

# one arm end to end: start a server, wait for it, replay, stop that process
FIXTURE=/tmp/tumwater_replay.json scripts/bench_arm.sh my-arm 8100 \
    mlx-community/Qwen3.8-27B-4bit -- splash serve --model mlx-community/Qwen3.8-27B-4bit \
    --port 8100 --max-context 128K --no-webui --idle-release off --offline

python3 scripts/replay_table.py                 # results/replay.jsonl -> table
python3 scripts/toolcall_smoke.py --base-url http://127.0.0.1:8090/v1 --model Qwen3.8-27B

python3 scripts/openai_stream_bench.py ...      # one stream at a fixed depth
python3 scripts/turn_loop_bench.py ...          # synthetic N-session loop
```

## Caveats

* **The replay is real input but not real output.** Each turn sends the
  recorded transcript (thinking stripped, as tumwater does), not the model's
  own previous answers. Output lengths differ by engine: Splash and Kapitsa ran
  to the 512-token cap on ~36-41 of 64 turns, mlx-serve on 3. The two headline
  metrics disagree on which way that cuts. Turns/min favours short answers,
  and Splash still leads it. Output t/s favours long answers, and it leads
  that too, by 1.4-4×.
* **The host was not quiet.** tumwater itself was running (on its API models)
  throughout, with load average 1-11. Every arm records `loadavg_start/end`;
  the reps that agree to ~5 % were taken at different loads.
* **No quality comparison between quants.** Affine 4-bit is the default
  because mxfp4 and GGUF bought no speed on Splash, not because they were
  measured worse.
* **Splash is closed-source** (a Homebrew-distributed binary), with a fixed lane
  count of 4. More than four concurrent clients queue, and that has not been
  measured.
* **Kapitsa has the best tail latency** (TTFT p90 4.6 s vs Splash's 10-14 s) at
  two-thirds of Splash's throughput. A fleet that cares more about worst-case
  first-token time than turns per minute could prefer it. It is not wired into
  `serve.sh`.
