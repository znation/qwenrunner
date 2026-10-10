# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngl | ngl8 | 8 | -1 | f16/f16 | 8 | 2048/512 | auto | 512 | 0 | 4096 | 51.92 |  | 4805.8 | 3071.8 | yes |
| ngl | ngl8 | 8 | -1 | f16/f16 | 8 | 2048/512 | auto | 0 | 128 | 4096 |  | 1.61 | 4805.8 | 3071.8 | yes |
| ngl | ngl8 | | | | | | | | | | | | 5176.2 | 3045.9 | NO |
