# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngl | ngl10 | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 512 | 0 | 4096 | 45.27 |  | 4699.8 | 3547.4 | yes |
| ngl | ngl10 | 10 | 1 | q4_0/q4_0 | 8 | 2048/512 | auto | 0 | 128 | 4096 |  | 0.97 | 4699.8 | 3547.4 | yes |
