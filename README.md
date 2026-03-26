# Orome

MoE inference engine for Apple Silicon. Metal GPU compute + SSD streaming.

| | Qwen3.5-35B-A3B | Qwen3.5-397B-A17B |
|---|---|---|
| **Parameters** | 35B (3B active) | 397B (17B active) |
| **Expert weights** | 18 GB (GPU-resident) | 121 GB 2-bit (SSD-streamed) |
| **Baseline** | 1.04 tok/s | 4.08 tok/s |
| **Optimized** | **62.53 tok/s** (60x) | **7.87 tok/s** (+93%) |
| **Bottleneck** | GPU dispatch | SSD I/O |
| **Experiments** | 73 | 49 |

**Hardware**: Mac Studio M2 Max — 38 GPU cores, 96 GB unified memory, NVMe SSD.

## Quick Start

```bash
# Build
make

# Run inference
./orome --model /path/to/model --prompt "Hello" --tokens 100

# Serve (OpenAI-compatible API)
make serve MODEL=/path/to/model

# Chat (terminal client)
make chat
```

For 2-bit expert quantization (397B):
```bash
uv run tools/repack_experts_2bit.py --model /path/to/model --output packed_experts_2bit
./orome --model /path/to/model --prompt "Hello" --tokens 100 --2bit
```

## How It Works

### Models That Fit in RAM (35B)

All expert weights are mlock'd into physical memory and wrapped as Metal shared buffers. The GPU reads directly from unified memory. The forward pass is a fused Metal pipeline — norm, projections, attention, routing, expert forward, and combine are batched into per-layer command buffers with concurrent dispatch. One GPU round-trip per layer, ~40 total per token.

### Models That Don't Fit (397B)

Expert weights live on SSD as packed binary files (one per layer). Each token triggers ~360 pread calls to load the active experts. The OS page cache serves ~90% of reads in <1ms; the remaining 10% hit SSD at 1-5ms. A per-layer expert cache (K+2 Metal shared buffers, 2.7 GB) tracks recently-used experts and skips the pread entirely on cache hit (~40% hit rate).

The serial dependency chain — attention → routing → I/O → expert compute — cannot be pipelined because expert identity is unknown until attention completes. This chain sets the performance floor.

## Architecture

```
include/orome.h        — Shared types, ModelConfig, interfaces
src/
  engine.m             — Forward pass orchestration
  metal.m              — Metal GPU context & dispatch
  attention.m          — Full (GQA) + linear (GatedDeltaNet) attention
  moe.m                — Expert routing, I/O, forward pass
  weights.m            — Tensor manifest, mmap, model config
  kernels.m            — CPU compute kernels
  tokenizer.m          — BPE encode/decode
  server.m             — HTTP/SSE server (OpenAI-compatible)
  shaders.metal        — Metal GPU kernels
tools/                 — Weight extraction, benchmarking, chat client
```

The engine is parameterized by `ModelConfig` — no hardcoded dimensions. Swap between 35B and 397B via config, not code.

## Optimization Campaigns

Two models, two completely different bottlenecks, two different optimization stories:

- **[Qwen3.5-35B-A3B](docs/qwen-35b-a3b.md)** — GPU dispatch campaign. 1.04 → 62.53 tok/s (60x). Everything fits in RAM; the entire story is reducing GPU synchronization overhead.
- **[Qwen3.5-397B-A17B](docs/qwen-397b-a17b.md)** — I/O streaming campaign. 4.08 → 7.87 tok/s (+93%). Experts don't fit in RAM; 49 experiments proving that pread + page cache + modest application cache is the optimal I/O strategy, and that the serial dependency chain sets a hard ceiling.

## Autonomous Research

```bash
./run_experiments.sh qwen35-397B --sessions 3
./run_experiments.sh qwen35-397B --agent codex --sessions 3
```

Launches Claude Code or Codex sessions that autonomously run experiments: modify source, build, benchmark, log results, commit or revert. Each session reads `status.md` for handoff, follows `program.md` for protocol, appends to `results.tsv`. 122 experiments logged across both campaigns.
