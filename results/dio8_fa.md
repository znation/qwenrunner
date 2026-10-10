# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fa | fa=off | | | | | | | | | | | | 3287.8 | 2490.9 | NO |
| fa | fa=on | 8 | 1 | q4_0/q4_0 | 8 | 2048/512 | dio | 512 | 0 | 8192 | 61.01 |  | 3853.8 | 3054.9 | yes |
| fa | fa=on | 8 | 1 | q4_0/q4_0 | 8 | 2048/512 | dio | 0 | 128 | 8192 |  | 1.68 | 3853.8 | 3054.9 | yes |
