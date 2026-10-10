# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fa | fa=off | | | | | | | | | | | | 3769.6 | 2984.1 | NO |
| fa | fa=on | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | dio | 512 | 0 | 8192 | 61.79 |  | 4342.0 | 3556.4 | yes |
| fa | fa=on | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | dio | 0 | 128 | 8192 |  | 1.77 | 4342.0 | 3556.4 | yes |
