# qwenrunner

A local inference stack for **Qwen-3.8-27B** on an **Apple M5 Max with 128 GB**
of unified memory, built around [`mlx-serve`](https://github.com/ddalcu/mlx-serve)
on MLX weights.

The model is a hybrid: 64 blocks, only 16 of them full attention, the rest
linear-attention/SSM. That is why 128k of context is cheap here — the KV cache
grows in 16 layers, not 64 — and why the numbers below hold up at depth.

## Result

| lever | setting | evidence |
|---|---|---|
| runtime | **`mlx-serve` 26.11.2** — `./serve.sh` | Zig server linking MLX + mlx-lm, no Python. Fastest engine measured that runs MLX-format weights; see `RESULTS.md`, "Engine survey" |
| model | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` — 4-bit affine, 16.95 GB | ships a DFlash2 drafter in its `drafter/` subdirectory, which the server loads with the model. The `mlx-community` 4-bit build has no drafter and no MTP head and measures at the no-speculation rate |
| speculation | **on** — MTP (depth 6) + DFlash2 + PLD | the entire win over a plain decode path: **73.4 t/s at 4k and 76-82 at 30k** with it, **28.4-34.0 and 26.8-30.9** without. 4.1 tokens accepted per verify pass at 27.5 % per-draft. `MTP=0` turns all three off |
| context | **128k** (`CTX=131072`) | 461 t/s prefill and 46.9 t/s decode *at* 130 747 tokens; the hot-cache holds that prefix in 9 366 MB, inside its 9 389 MB default budget. The model's own limit is 256k |
| concurrency | **`NP=2`** default, `NP=4` for a fleet | continuous batching, not fixed slots: `NP` is a cap and KV is taken as requests arrive. Four agent sessions: 54.7 t/s aggregate at TTFT p50 1.28 s with `NP=2`, 49.9 at 1.26 s and wall p50 6.3 s (against 10.4 s) with `NP=4` |
| KV cache | **f16** (`KVQ=off`) | `KVQ=8` halves it at ~10 % decode; `KVQ=4` quarters it. Not needed at 128 GB |
| prefix cache | **on**, budget = one session at the working context | what makes an appending agent transcript cost 1.3 s of TTFT instead of a re-prefill. `CACHE_MB` raises the budget to hold more live conversations |

## Hardware

| | |
|---|---|
| CPU | Apple M5 Max, 18 cores (6 super + 12 performance) |
| GPU | integrated, `MTLGPUFamilyMetal4`, simdgroup matrix mul + tensor ops, unified memory |
| RAM | 128 GB unified (`recommendedMaxWorkingSetSize` ≈ 112 GiB) |
| OS | macOS 27 |

## Install

`mlx-serve` is not in Homebrew core, and the tap builds from source (Zig + MLX +
llama.cpp, over 30 minutes here). Use the release binary — it is what every
number in this repo was measured on:

```bash
mkdir -p ~/opt/mlx-serve && cd ~/opt/mlx-serve
curl -sL https://github.com/ddalcu/mlx-serve/releases/latest/download/mlx-serve-bin-macos-arm64.tar.gz | tar xz
```

Then pull the weights (16.95 GB, into `~/.mlx-serve/models`):

```bash
~/opt/mlx-serve/mlx-serve-macos-arm64/mlx-serve pull ddalcu/Qwen3.8-27B-MLX-Serve-4bit
```

`serve.sh` finds the binary at `~/opt/mlx-serve/...` or on `PATH`, and
`MLX_SERVE_BIN` overrides both.

## Run

```bash
./serve.sh                  # 2 concurrent, speculation on, 128k, port 8090
NP=4 ./serve.sh             # the measured fleet setting
MTP=0 ./serve.sh            # speculation off (the apples-to-apples arm)
KVQ=8 CACHE_MB=16384 ./serve.sh
```

OpenAI-compatible at `http://127.0.0.1:8090/v1`; Anthropic Messages at
`http://127.0.0.1:8090`. Logs go to `logs/mlx-serve.8090.log`.

Env: `MODEL CTX NP PORT HOST MTP KVQ CACHE_MB LOG_DIR`.

## Measured

One stream, greedy, 384 max tokens, nonce at the front of the prompt so nothing
is cached, warmup discarded (`scripts/openai_stream_bench.py`):

| prompt | decode t/s | TTFT |
|---|---:|---:|
| 4k (`data/long_prompt.txt`) | 64.0-75.2 (median 73.4) | 6.5-7.1 s |
| 30k (`data/prompt_30k.txt`) | 61.7-82.0 | 41.7-42.2 s |
| 128k (`data/prompt_128k.txt`) | 46.9 | 283.5 s |

Four agent sessions, four turns each, 256 max tokens
(`scripts/turn_loop_bench.py`):

| `NP` | aggregate out t/s | prefill t/s | TTFT p50 | TTFT p90 | wall p50 |
|---:|---:|---:|---:|---:|---:|
| 2 | 54.7 | 1 493 | 1.28 s | 15.8 s | 10.4 s |
| 4 | 49.9 | 1 773 | 1.26 s | 14.9 s | 6.3 s |

`NP=4` reads lower aggregate only because the model stopped early on more turns
(2 481 output tokens against 3 253); its wall p50 is 6.3 s against 10.4 s, so 4
is the better fleet setting and 2 the RAM-first default.

## Harnesses

Both are stdlib-only and drive any OpenAI-compatible endpoint, so they work
against this server unchanged:

```bash
python3 scripts/openai_stream_bench.py --base-url http://127.0.0.1:8090/v1 \
    --model Qwen3.8-27B-MLX-Serve-4bit --prompt-file data/prompt_30k.txt \
    --max-tokens 384 --reps 4 --nonce --warmup

python3 scripts/turn_loop_bench.py --base-url http://127.0.0.1:8090/v1 \
    --model Qwen3.8-27B-MLX-Serve-4bit --sessions 4 --turns 4 --max-tokens 256
```

## Caveats

* **The win is speculation, not the MLX kernels.** With `MTP=0` the server
  decodes at 28.4-34.0 t/s at 4k — parity with a plain llama.cpp Metal path.
  Everything above the parity line comes from MTP + DFlash2 + PLD.
* **One quant, not a sweep.** 4-bit affine at 16.95 GB. No 6-bit or 8-bit
  comparison has been run on this machine.
* **The engine ordering is vendor-published.** `mlx-serve` was chosen from a
  cross-engine table its own author did not write, but that table is
  vendor-run; only `mlx-serve` itself was measured here. `RESULTS.md`,
  "Engine survey", has the sources and the caveat.
* **No 8-session fleet run.** The agent shape above is four sessions. The
  previous llama.cpp work measured eight; that has not been repeated here.
