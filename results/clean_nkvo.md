# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nkvo | nkvo=0 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 0 | 56.37 |  | 4528.1 | 3540.8 | yes |
| nkvo | nkvo=0 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 0 |  | 1.50 | 4528.1 | 3540.8 | yes |
| nkvo | nkvo=1 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 0 | 58.54 |  | 4439.1 | 3511.8 | yes |
| nkvo | nkvo=1 | 10 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 0 |  | 1.08 | 4439.1 | 3511.8 | yes |
