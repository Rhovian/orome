# Orome

Inference engine for Apple Silicon, currently focused on GGUF Qwen3.5 models.

**Hardware**: Mac Studio M2 Max — 38 GPU cores, 96 GB unified memory, NVMe SSD.

## Current Focus

- Supported autoresearch models: `Qwen3.5-9B-Q8_0.gguf`, `Qwen3.5-27B-Q4_K_M.gguf`, `Qwen3.5-35B-A3B-Q4_K_S.gguf`
- Experiment logs: `inference/experiments/`

## vs llama.cpp

Same GGUF files, same hardware. [Full methodology and quality results.](docs/llama-comparison.md)

| Model | Orome tok/s | llama.cpp tok/s | Quality |
| --- | ---: | ---: | --- |
| Qwen3.5-9B-Q8_0 | 35.32 | 31.22 | 3/3 both |
| Qwen3.5-27B-Q4_K_M | 17.59 | 14.77 | 3/3 both |
| Qwen3.5-35B-A3B-Q4_K_S | 65.15 | 51.34 | 3/3 both |

## Quick Start

```bash
# Build
make

# Run inference
./orome --model /path/to/model.gguf --prompt "Hello" --tokens 100

# Serve (OpenAI-compatible API)
make serve MODEL=/path/to/model.gguf

# Chat (terminal client)
make chat
```

## How It Works

The current engine is GGUF-only. The model fits in unified memory, so the hot path is a Metal forward pass that keeps hidden state on GPU, resolves tensors through the GGUF cache, and spends its time on dequant matvec, attention, expert routing, expert compute, and dispatch/barrier overhead.

## Architecture

```text
inference/
  include/orome.h      — Types, ModelConfig, TensorRef, Engine interfaces
  src/                  — Objective-C engine, Metal shaders, GGUF loader, HTTP server
  vendor/              — Third-party (linenoise, tokenizer)
  tools/               — Benchmarking, comparison, chat client, stress test
  experiments/         — Per-model optimization logs and configs
scripts/               — Experiment runner
docs/                  — Detailed comparison data, model notes, research
```

## Docs

- [Qwen3.5 family notes](docs/qwen-family.md)
- [llama.cpp comparison details](docs/llama-comparison.md)
- [Research & experiments](docs/research.md)
