# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngl | ngl12 | 12 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 0 | 41.89 |  | 4856.3 | 4014.6 | yes |
| ngl | ngl12 | 12 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 0 |  | 1.21 | 4856.3 | 4014.6 | yes |
