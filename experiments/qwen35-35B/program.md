# orome — Autonomous Inference Optimization (GGUF Q4_K Era)

You are an autonomous research agent optimizing a Metal inference engine for Qwen3.5-35B-A3B on an M2 Max Mac Studio (96GB unified memory, ~400 GB/s bandwidth, 38 GPU cores, actively cooled).

The engine loads GGUF files directly and runs all computation on GPU via Metal compute shaders. The legacy packed format has been removed — GGUF is the only format.

## Setup

1. **Read `status.md`** — this is your handoff note from the previous session. It tells you what's been done, what the current best result is, what to try next. If it doesn't exist, this is the first run.
2. **Read `results.tsv`** — the full experiment history. Understand what's been tried and what worked.
3. **Read the source files** you plan to modify (see "What You CAN Modify" below).
4. You are on branch `autoresearch/orome-397B`.

## The Goal

**Maximize tok/s (tokens per second)** on 100-token sustained generation with K=8 active experts at Q4_K quantization.

Secondary goals (don't sacrifice tok/s for these):
- Minimize TTFT (time to first token)
- Minimize memory pressure
- Maintain output quality (don't break the model)

## Codebase Structure

```
include/orome.h      — All shared types, ModelConfig, MetalCtx, Engine, function declarations
src/main.m           — CLI parsing, GGUF loading, entry point
src/engine.m         — Forward pass orchestration (engine_step), encode_experts_gguf
src/metal.m          — Metal GPU context, pipeline setup, buffer allocation
src/format.m         — GGUF tensor cache, format dispatch, de-interleaving
src/gguf.m           — GGUF parser, metadata extraction, model config
src/kernels.m        — CPU compute: embedding lookup, norms (unused hot path)
src/tokenizer.m      — Vocab, BPE encode/decode
src/server.m         — HTTP/SSE server (OpenAI-compatible)
src/shaders.metal    — Metal GPU kernels (Q4K/Q5K/Q6K/Q8_0 dequant matvec, attention, MoE)
```

Build: `make` produces `./orome`
Benchmark: `python3 tools/benchmark.py --trials 1 --json`

## GGUF Quantization Formats

The Q4_K_S GGUF uses mixed quantization:
- **Q4_K** (144 bytes / 256 weights): Most weights. Has 8 sub-block scales packed in 12 bytes. Dequant is: 4 groups of 32 bytes, low nibbles → sc[g*2], high nibbles → sc[g*2+1].
- **Q5_K** (176 bytes / 256 weights): Some down projections. Same as Q4K but adds 32-byte qh array for 5th bit.
- **Q6_K** (210 bytes / 256 weights): LM head (output.weight). ql+qh+scales+d layout.
- **Q8_0** (34 bytes / 32 weights): Small tensors (norms, biases, some attention).

## What You CAN Modify

- **`src/engine.m`** — Forward pass, layer loop, expert dispatch, profiling.
- **`src/shaders.metal`** — Metal GPU kernels. Threadgroup sizing, kernel structure, memory access patterns.
- **`src/metal.m`** — Metal context, pipelines, buffer management. `ROWS_PER_TG` tuning lives here.
- **`src/format.m`** — Format dispatch, tensor cache.
- **`include/orome.h`** — Types and interfaces (if adding new APIs).
- **`Makefile`** — Compiler flags, optimization levels.

## What You CANNOT Modify

- **`tools/benchmark.py`** — the benchmark harness. It is the ground truth measurement.
- **`program.md`** — these instructions.
- Model weights on disk.

## Hardware Context

Mac Studio M2 Max: 96GB unified memory, 38 GPU cores, ~400 GB/s bandwidth, active cooling.

| Resource | Value |
|---|---|
| Memory | 96 GB unified |
| GPU cores | 38 |
| Bandwidth | ~400 GB/s |
| Cooling | Active fan |
| Model size | 19.3 GB (all mmap'd) |
| Active weights/token | ~1.6 GB (K=8 experts + attention + LM head) |

## Current Performance Profile

- **46 tok/s** (21.7ms per token)
- Per layer: ~0.47ms × 40 layers = 18.8ms
- LM head (Q6K 398MB): ~2.5ms
- Theoretical bandwidth limit: ~5ms (1.6GB @ 400 GB/s) → we're at 4.3x overhead
- Previous legacy format achieved 62 tok/s with simpler 4-bit dequant

## Key Optimization Axes (priority order)

1. **2-row-per-simdgroup matvec** — Process 2 output rows per simdgroup, halving TG count. Proven +4% in legacy (61→62 tok/s). Apply to Q4K, Q6K, and batched expert kernels.

2. **Fused expert gate+up+SwiGLU kernel** — One kernel that reads gate+up weights, computes SwiGLU in registers, writes activated output. Eliminates 80 dispatches + 40 barriers per token. Proven +2% in legacy.

3. **Reduce Q4K scale unpacking overhead** — The `unpack_q4k_scales` function is called per superblock per SIMD lane. Consider precomputing scales or using SIMD-wide scale broadcast.

4. **Fewer barriers** — Some `memoryBarrierWithScope` calls may be between independent dispatches. Profile to identify unnecessary ones.

5. **Fused kernels** — Combine residual_add + norm into single dispatch. Combine routing + shared_expert_gate into single dispatch.

6. **LM head optimization** — The Q6K LM head is 398MB (30% of per-token weight reads). A specialized double-row Q6K kernel could help.

## Experiment Protocol

Each experiment:

1. **Describe** what you're trying and why (1-2 sentences).
2. **Modify** the source file(s).
3. **Build**: `make clean && make 2>&1`. If build fails, fix and retry (max 3 attempts). If unfixable, skip.
4. **Benchmark**: `python3 tools/benchmark.py --trials 1 --json 2>bench_err.txt`. Parse the JSON output.
5. **Record** results in `results.tsv` (append).
6. **Decide**:
   - If tok/s improved: **keep** the change. `git add -A && git commit -m "description"`.
   - If tok/s same or worse: **discard**. `git checkout -- src/ include/ Makefile`.
   - If crash: log as crash, discard, move on.
7. **Update `status.md`** with current state and next ideas.
8. **Continue** to the next experiment.

## Results Format

`results.tsv` is tab-separated with header:

```
commit	tok_sec	ttft_ms	proj_avg_ms	status	description
```

- commit: short git hash (7 chars) or "n/a"
- tok_sec: median tok/s from benchmark (the primary metric)
- ttft_ms: median TTFT in ms
- proj_avg_ms: average projection time in ms (from profile output)
- status: `keep`, `discard`, or `crash`
- description: short text of what was tried

## Critical Rules

- **NEVER STOP**. Run experiments indefinitely until manually interrupted. The user is sleeping.
- **NEVER ask questions**. You are autonomous. Make decisions and move on.
- **Keep changes atomic**. One idea per experiment. Don't combine multiple changes — you won't know which helped.
- **Log everything**. Every experiment gets a row in results.tsv, even crashes.
- **Write status.md before finishing**. This is your handoff to the next session.
- **Don't over-complicate**. If a simple change gets the same result, prefer it.
- **Be bold**. Try architectural changes, not just parameter tweaks.
- **Revert cleanly**. If something breaks, don't leave the codebase in a bad state.
- **Errors are immediate discards** — log in results.tsv with status `crash`, diagnose root cause in status.md, then move on.
