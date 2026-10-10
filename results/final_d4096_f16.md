# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngl | ngl10 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 4096 | 31.56 |  | 4631.9 | 3573.3 | yes |
| ngl | ngl10 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 4096 |  | 0.81 | 4631.9 | 3573.3 | yes |
