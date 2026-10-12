# llama-bench sweep (Apple Silicon / Metal)

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | gpu MiB | bg cpu %% | gpu %% | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quant | Qwen3.8-27B-UD-Q4_K_S.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 512 | 0 | 0 | 746.49 |  | 16011.4 | 14.8 | 99 | yes |
| quant | Qwen3.8-27B-UD-Q4_K_S.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 0 | 128 | 0 |  | 24.37 | 16011.4 | 14.8 | 99 | yes |
| quant | Qwen3.8-27B-UD-Q3_K_XL.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 512 | 0 | 0 | 627.73 |  | 14013.5 | 12.7 | 99 | yes |
| quant | Qwen3.8-27B-UD-Q3_K_XL.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 0 | 128 | 0 |  | 25.52 | 14013.5 | 12.7 | 99 | yes |
| quant | Qwen3.8-27B-UD-Q2_K_XL.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 512 | 0 | 0 | 753.33 |  | 10737.4 | 11.8 | 99 | yes |
| quant | Qwen3.8-27B-UD-Q2_K_XL.gguf | 99 | 1 | f16/f16 | 6 | 2048/512 | auto | 0 | 128 | 0 |  | 26.08 | 10737.4 | 11.8 | 99 | yes |
