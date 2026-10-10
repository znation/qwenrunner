# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fa | fa=off | | | | | | | | | | | | 4190.3 | 2983.1 | NO |
| fa | fa=on | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 512 | 0 | 8192 | 47.36 |  | 4842.6 | 3556.4 | yes |
| fa | fa=on | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 0 | 128 | 8192 |  | 1.31 | 4842.6 | 3556.4 | yes |
