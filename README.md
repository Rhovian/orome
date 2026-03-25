# Orome

Multi-model MoE inference engine for Apple Silicon, built with Obj-C and Metal.

Bootstrapped from [flash-moe](../flash-moe/) (MacBook Air M4, 16GB) and autonomously optimized via an autoresearch-style experiment loop.

## Hardware Target

- **Machine**: Mac Studio (M2 Max, 2023)
- **Chip**: 12-core CPU (8P + 4E), 38-core GPU
- **Memory**: 96 GB unified (~400 GB/s bandwidth)
- **Storage**: SSD (Apple Fabric)

## Quick Start

```bash
# 1. Download model weights
hf download Qwen/Qwen3.5-35B-A3B --local-dir ~/models/Qwen3.5-35B-A3B

# 2. Extract weights for the engine
uv run tools/extract_weights.py --model ~/models/Qwen3.5-35B-A3B --output .
uv run tools/repack_experts.py --model ~/models/Qwen3.5-35B-A3B --output packed_experts
uv run tools/export_tokenizer.py --model ~/models/Qwen3.5-35B-A3B --output vocab.bin

# 3. Build
make

# 4. Run inference
./orome --prompt "Hello" --tokens 20 --k 8

# 5. Benchmark
uv run tools/benchmark.py --trials 3
```

## Project Structure

```
include/orome.h        — Shared types, ModelConfig, interfaces
src/
  main.m               — CLI entry point
  engine.m             — Forward pass orchestration
  metal.m              — Metal GPU context & dispatch
  attention.m          — Full (GQA) + linear (GatedDeltaNet) attention
  moe.m                — Expert routing, I/O, forward pass
  weights.m            — Tensor manifest, mmap, model config
  kernels.m            — CPU compute kernels
  tokenizer.m          — BPE encode/decode
  server.m             — HTTP/SSE server (OpenAI-compatible)
  shaders.metal        — Metal GPU kernels
vendor/                — Vendored third-party (tokenizer.h, linenoise)
tools/                 — Python tooling (weight extraction, benchmarks, viz)
```

## Multi-Model

The engine is parameterized by `ModelConfig` — no hardcoded model dimensions. Swap between Qwen3.5-35B and 397B (or future models) via config, not code. Config is loaded from HF `config.json` or falls back to built-in presets.

## Autonomous Optimization

```bash
./run_experiments.sh
```

Launches Claude Code sessions in a loop. Each session reads `status.md` for handoff context, runs experiments (modify source, build, benchmark), logs to `results.tsv`, writes status back. See `program.md` for the full protocol.

Visualize results: `uv run tools/progress.py`
