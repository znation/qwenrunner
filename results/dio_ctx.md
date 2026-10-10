# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx | pp512/tg128 | | | | | | | | | | | | 5127.2 | 3003.2 | NO |
| ctx | pp2048/tg128 | | | | | | | | | | | | 5111.8 | 0.0 | NO |
| ctx | pp128/tg512 | | | | | | | | | | | | 5128.1 | 3157.7 | NO |
| ctx | tg128 | | | | | | | | | | | | 5116.5 | 0.0 | NO |
| ctx | pp512 | | | | | | | | | | | | 5116.5 | 0.0 | NO |
