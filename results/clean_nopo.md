# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nopo | nopo=0 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 0 | 59.44 |  | 4476.2 | 3540.9 | yes |
| nopo | nopo=0 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 0 |  | 1.54 | 4476.2 | 3540.9 | yes |
| nopo | nopo=1 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 0 | 6.20 |  | 4431.8 | 3503.4 | yes |
| nopo | nopo=1 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 0 |  | 0.93 | 4431.8 | 3503.4 | yes |
