# Comparison vs llama.cpp

Mac Studio M2 Max — 38 GPU cores, 96 GB unified memory.

## Throughput

Method:
- Same GGUF files, same prompt (`Hello`)
- 100 generated tokens, context size `256`, 5-trial median
- Orome via `tools/benchmark.py --skip-quality-check`
- llama.cpp via `llama-completion` with greedy settings and `--no-warmup`
- Orome `bf77939`, llama.cpp `c46758d`

| Model | Orome tok/s | llama.cpp tok/s |
| --- | ---: | ---: |
| Qwen3.5-9B-Q8_0 | 35.32 | 31.22 |
| Qwen3.5-27B-Q4_K_M | 17.59 | 14.77 |
| Qwen3.5-35B-A3B-Q4_K_S | 65.15 | 51.34 |

Reproduce:

```bash
python3 tools/compare_orome_llama.py --models 9B 27B 35B --tokens 100 --trials 5 --json
```

## Completion Quality

Method:
- Same GGUF files, greedy completion, one engine at a time
- Prompts: `capital`, `opposite`, `sky`
- Orome `bf77939`, llama.cpp `c46758d`

| Model | Orome | llama.cpp | Notes |
| --- | --- | --- | --- |
| Qwen3.5-9B-Q8_0 | 3/3 pass | 3/3 pass | both coherent on the default suite |
| Qwen3.5-27B-Q4_K_M | 3/3 pass | 3/3 pass | both coherent on the default suite |
| Qwen3.5-35B-A3B-Q4_K_S | 3/3 pass | 3/3 pass | both coherent on the default suite |

Reproduce:

```bash
python3 tools/compare_orome_llama_quality.py --models 9B 27B 35B --json
```
