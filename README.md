# Orome

Inference engine for Apple Silicon, currently focused on GGUF Qwen3.5 models.

**Hardware**: Mac Studio M2 Max — 38 GPU cores, 96 GB unified memory, NVMe SSD.

## Current Focus

- Supported autoresearch models: `Qwen3.5-9B-Q8_0.gguf`, `Qwen3.5-27B-Q4_K_M.gguf`, `Qwen3.5-35B-A3B-Q4_K_S.gguf`
- Primary model: `Qwen3.5-35B-A3B-Q4_K_S.gguf`
- Best GGUF result: **68.91 tok/s** at `041860b`
- Historical packed-format peak: `62.39-62.53 tok/s`
- Bottleneck: GPU dispatch and kernel efficiency, not SSD streaming
- Campaign status: current 35B GGUF campaign closed out near the wall
- Experiment logs: `experiments/qwen35-9B/`, `experiments/qwen35-27B/`, `experiments/qwen35-35B/`

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
include/orome.h        — Shared types, ModelConfig, TensorRef, Engine interfaces
src/
  main.m               — CLI parsing, GGUF loading, engine startup
  engine.m             — Forward-pass orchestration
  metal.m              — Metal GPU context & dispatch
  format.m             — GGUF tensor cache build and format dispatch
  gguf.m               — GGUF parser and metadata extraction
  kernels.m            — Timing and sampling helpers
  tokenizer.m          — BPE encode/decode
  server.m             — HTTP/SSE server (OpenAI-compatible)
  shaders.metal        — Metal GPU kernels
tools/                 — Benchmarking, plotting, chat client
```

The engine is parameterized by `ModelConfig` and GGUF metadata rather than hardcoded dimensions.

## Qwen3.5 Dense Notes

Orome now distinguishes MoE vs dense FFN models from GGUF tensor names. The 35B-A3B path remains routed-MoE; Qwen3.5-9B and Qwen3.5-27B use the dense SwiGLU FFN path.

If your GGUF download does not include Orome's legacy `vocab.bin`, generate one from the official tokenizer assets:

```bash
python3 tools/build_vocab_bin.py /path/to/qwen-tokenizer-dir
```

Place the resulting `vocab.bin` next to the GGUF file or in the tokenizer asset directory that Orome can read.

## Research Notes

- **[Qwen3.5-35B-A3B](docs/qwen-35b-a3b.md)** — historical packed-format dispatch campaign notes. Useful for hypotheses, but the current source tree is GGUF-only and runs at different absolute numbers.
- **[experiments/qwen35-9B/program.md](experiments/qwen35-9B/program.md)** — autonomous optimization brief for the 9B dense GGUF path.
- **[experiments/qwen35-27B/program.md](experiments/qwen35-27B/program.md)** — autonomous optimization brief for the 27B dense GGUF path.
- **[experiments/qwen35-35B/program.md](experiments/qwen35-35B/program.md)** — current autonomous optimization brief for the GGUF-era codebase.

## Autonomous Research

```bash
./run_experiments.sh qwen35-9B --agent codex --sessions 3
./run_experiments.sh qwen35-27B --agent codex --sessions 3
./run_experiments.sh qwen35-35B --sessions 3
./run_experiments.sh qwen35-35B --agent codex --sessions 3
```

Launches Claude Code or Codex sessions that autonomously run experiments: modify source, build, benchmark, log results, commit or revert. Each session reads `status.md` for handoff, follows `program.md` for protocol, and appends to `results.tsv`.

The benchmark harness now includes a chat-quality canary, and the runner reverts
session commits if either throughput or visible output quality regresses. Each
experiment target must define a `cross_check.json`, and the runner uses those
configs to run a self-check plus the other supported models as cross-model
regression gates before it keeps a session's commits.
