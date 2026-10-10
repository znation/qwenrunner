# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nopo | nopo=0 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 512 | 0 | 0 | 74.02 |  | 3835.7 | 3039.4 | yes |
| nopo | nopo=0 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 0 | 128 | 0 |  | 1.88 | 3835.7 | 3039.4 | yes |
| nopo | nopo=1 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 512 | 0 | 0 | 12.22 |  | 3797.9 | 3003.9 | yes |
| nopo | nopo=1 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 0 | 128 | 0 |  | 1.93 | 3797.9 | 3003.9 | yes |
