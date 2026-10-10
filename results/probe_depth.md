# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quant | Qwen3.8-27B-UD-Q2_K_XL.gguf | 8 | -1 | f16/f16 | 8 | 2048/512 | auto | 64 | 0 | 8192 | 23.17 |  | 3726.6 | 2272.8 | yes |
| quant | Qwen3.8-27B-UD-Q2_K_XL.gguf | 8 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 8 | 8192 |  | 1.04 | 3726.6 | 2272.8 | yes |
