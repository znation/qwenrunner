# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nkvo | nkvo=0 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 512 | 0 | 0 | 74.12 |  | 3969.8 | 3039.4 | yes |
| nkvo | nkvo=0 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 0 | 128 | 0 |  | 1.77 | 3969.8 | 3039.4 | yes |
| nkvo | nkvo=1 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 512 | 0 | 0 | 74.02 |  | 3809.5 | 3018.6 | yes |
| nkvo | nkvo=1 | 8 | -1 | f16/f16 | 8 | 2048/512 | dio | 0 | 128 | 0 |  | 1.75 | 3809.5 | 3018.6 | yes |
