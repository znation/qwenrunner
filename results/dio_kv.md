# llama-bench sweep

| stage | label | ngl | fa | k/v | t | b/ub | lm | pp | tg | d | pp t/s | tg t/s | VRAM peak MiB | client MiB | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kv | kv=f16/f16 | | | | | | | | | | | | 5194.0 | 3004.6 | NO |
| kv | kv=q8_0/f16 | | | | | | | | | | | | 5173.8 | 0.0 | NO |
| kv | kv=q8_0/q8_0 | | | | | | | | | | | | 5173.8 | 0.0 | NO |
| kv | kv=q4_0/q4_0 | | | | | | | | | | | | 5134.6 | 2993.9 | NO |
| kv | kv=bf16/bf16 | | | | | | | | | | | | 5124.9 | 0.0 | NO |
