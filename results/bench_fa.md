# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fa | fa=off | | | | | | | | | | | | 3884.0 | 2489.9 | NO |
| fa | fa=on | 8 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 512 | 0 | 8192 | 24.62 |  | 4575.0 | 3054.9 | yes |
| fa | fa=on | 8 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 0 | 128 | 8192 |  | 0.23 | 4575.0 | 3054.9 | yes |
