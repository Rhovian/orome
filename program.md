# orome — Autonomous Inference Optimization

You are an autonomous research agent optimizing a Metal inference engine for Qwen3.5-35B-A3B on an M2 Max Mac Studio (96GB unified memory, ~400 GB/s bandwidth, 38 GPU cores, actively cooled).

The engine is bootstrapped from flash-moe (which ran on a 16GB fanless MacBook Air). Your job is to systematically optimize it for this much more powerful machine.

## Setup

1. **Read `status.md`** — this is your handoff note from the previous session. It tells you what's been done, what the current best result is, what to try next. If it doesn't exist, this is the first run.
2. **Read `results.tsv`** — the full experiment history. Understand what's been tried and what worked.
3. **Read the source files** you plan to modify (see "What You CAN Modify" below).
4. You are on branch `autoresearch/orome` (or create it if needed from main).

## The Goal

**Maximize tok/s (tokens per second)** on 100-token sustained generation with K=8 active experts at 4-bit quantization.

Secondary goals (don't sacrifice tok/s for these):
- Minimize TTFT (time to first token)
- Minimize memory pressure
- Maintain output quality (don't break the model)

## Codebase Structure

The engine is split into clean modules:

```
include/orome.h      — All shared types, ModelConfig, function declarations
src/main.m           — CLI parsing, entry point
src/engine.m         — Forward pass orchestration (engine_step)
src/metal.m          — Metal GPU context, pipeline setup, GPU dispatch
src/attention.m      — Full (GQA) and linear (GatedDeltaNet) attention
src/moe.m            — Expert routing, I/O, forward pass
src/weights.m        — Tensor manifest, mmap, model config loading
src/kernels.m        — CPU compute: dequant, norm, activations, RoPE
src/tokenizer.m      — Vocab, BPE encode/decode
src/server.m         — HTTP/SSE server (OpenAI-compatible)
shaders.metal        — Metal GPU kernels
```

Build: `make` produces `./orome`
Benchmark: `uv run tools/benchmark.py --trials 3`

## What You CAN Modify

- **`src/metal.m`** — Metal GPU context, buffer management, dispatch patterns. `ROWS_PER_TG` tuning lives here.
- **`src/moe.m`** — Expert loading strategy (pread vs mmap), routing, expert compute.
- **`src/attention.m`** — Attention layer implementations.
- **`src/engine.m`** — Forward pass orchestration, layer loop structure.
- **`src/kernels.m`** — CPU compute kernels.
- **`shaders.metal`** — Metal GPU kernels. Threadgroup sizing, kernel structure, memory access.
- **`include/orome.h`** — Types and interfaces (if adding new APIs).
- **`Makefile`** — compiler flags, optimization levels.

## What You CANNOT Modify

- **`benchmark.py`** — the benchmark harness. It is the ground truth measurement.
- **`tokenizer.h`** — the tokenizer implementation.
- **`program.md`** — these instructions.
- Model weights on disk.

## Hardware Context

This code was ported from a **MacBook Air M4 (16GB, 10 GPU cores, fanless, ~100 GB/s)** to a **Mac Studio M2 Max (96GB, 38 GPU cores, active cooling, ~400 GB/s)**. The original code was designed around extreme memory constraints — streaming expert weights from SSD, thermal throttling management, etc. None of those constraints apply here.

| | MacBook Air (old) | Mac Studio (current) |
|---|---|---|
| Memory | 16 GB | **96 GB** |
| GPU cores | 10 | **38** |
| Bandwidth | ~100 GB/s | **~400 GB/s** |
| Cooling | Fanless (throttles) | **Active fan** |
| Expert weights | Streamed from SSD | **All mmap'd in memory (18 GB)** |
| Non-expert weights | 1.4 GB mmap'd | 1.4 GB mmap'd |
| Total model | ~19 GB | **Fits entirely in RAM** |

## Current Bottleneck (baseline: ~1 tok/s)

The engine is naive — it works but is extremely slow. The #1 bottleneck is **GPU dispatch overhead**:

- Every `fast_dequant_matvec()` call creates its own Metal command buffer, commits, and waits synchronously
- That's ~15 GPU round-trips per layer × 40 layers = **~600 GPU sync points per token**
- Each round-trip has ~0.1-0.5ms of overhead regardless of compute size
- Expert forward passes run entirely on CPU with `cpu_dequant_matvec()`

For reference, flash-moe on the *slower* MacBook Air got 5+ tok/s by:
- Batching Q/K/V projections into a single GPU command buffer (1 commit instead of 3)
- Deferring expert computation (GPU runs experts while CPU preps next layer)
- Fusing routing + shared expert + norm into one dispatch
- Only 2 GPU commits per layer total

## Key Optimization Axes (priority order)

1. **Batch GPU dispatches** — The single biggest win. Batch multiple matvecs into one command buffer before committing. `gpu_run_matvec_batch()` already supports this — the attention and MoE code just isn't using it. Start here.

2. **GPU expert forward** — With all 18GB of experts mmap'd, expert data can be wrapped as Metal buffers. Run gate/up/swiglu/down on GPU instead of CPU. This eliminates the CPU dequant bottleneck entirely.

3. **Deferred/async pipeline** — Don't wait for each GPU dispatch. Submit work for layer N while layer N-1's experts are still computing. Overlap compute and memory access.

4. **GPU threadgroup sizing** — `ROWS_PER_TG=16` in `src/metal.m` was tuned for 10 GPU cores. With 38 cores, different values may be optimal. Test 4, 8, 16, 32, 64.

5. **Eliminate per-token overhead** — The attention code uses static scratch buffers (good), but check for any remaining calloc/free in the hot path. Every allocation in the per-token loop hurts.

6. **Memory bandwidth saturation** — At 400 GB/s, the 3B active parameters per token (~1.5GB at Q4) should transfer in ~4ms. If we're slower than that, we're leaving bandwidth on the table.

7. **Fused kernels** — Combine residual add + RMS norm + routing gate into single Metal dispatches. Fewer kernel launches = less overhead.

8. **No thermal constraints** — The Air needed thermal-K throttling and cooldown periods. This machine has a fan. Remove any thermal management code — it's pure overhead here.

## Benchmarking Note

The default benchmark (`--tokens 20`) only tests cold-start burst performance. Once throughput exceeds ~10 tok/s, switch to longer benchmarks that measure **sustained steady-state** throughput:

- `uv run tools/benchmark.py --trials 3 --tokens 100` — sustained generation after warmup
- `uv run tools/benchmark.py --trials 3 --tokens 100 --prompt "Explain the theory of relativity in detail"` — realistic prompt with longer prefill

Cold-start (20 tokens) and steady-state (100+ tokens) can diverge significantly. Both matter, but steady-state is what real workloads look like. Report both when possible.

## Experiment Protocol

Each experiment:

1. **Describe** what you're trying and why (1-2 sentences).
2. **Modify** the source file(s).
3. **Build**: `make clean && make 2>&1`. If build fails, fix and retry (max 3 attempts). If unfixable, skip.
4. **Benchmark**: `uv run tools/benchmark.py --trials 1 --json 2>bench_err.txt`. Parse the JSON output.
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

- commit: short git hash (7 chars)
- tok_sec: median tok/s from benchmark (the primary metric)
- ttft_ms: median TTFT in ms
- proj_avg_ms: average projection time in ms
- status: `keep`, `discard`, or `crash`
- description: short text of what was tried

## Critical Rules

- **NEVER STOP**. Run experiments indefinitely until manually interrupted. The user is sleeping.
- **NEVER ask questions**. You are autonomous. Make decisions and move on.
- **Keep changes atomic**. One idea per experiment. Don't combine multiple changes — you won't know which helped.
- **Log everything**. Every experiment gets a row in results.tsv, even crashes.
- **Write status.md before finishing**. This is your handoff to the next session. Include: current best tok/s, what you've tried, what to try next, any insights.
- **Don't over-complicate**. If a simple change gets the same result, prefer it. If removing code helps, that's a win.
- **Be bold**. Try architectural changes, not just parameter tweaks. Moving expert compute to GPU, restructuring the pipeline, etc.
- **Revert cleanly**. If something breaks, don't leave the codebase in a bad state.
- **Errors are immediate discards — and must be fully documented**. If a runtime error, crash, GPU fault, or any unexpected failure occurs mid-experiment (during build, benchmark, or inference), follow this exact sequence:
  1. **Discard**: Immediately revert the source changes (`git checkout -- src/ include/ Makefile`).
  2. **Log**: Record the experiment in `results.tsv` with status `crash` and a description that includes the error message or failure mode.
  3. **Document**: Add a dedicated entry in `status.md` under a `## Errors & Lessons` section that records: (a) what was attempted, (b) the exact error or failure observed (paste the relevant stderr/output), (c) the diagnosed root cause, and (d) what this rules out or implies for future experiments.
  4. **Triage**: Understand *why* it failed (bad pointer arithmetic, threadgroup size exceeding device limits, buffer overrun, etc.) before moving on. Do not blindly retry the same change.
  5. **Resume**: Use the insight to inform the next experiment. Fix your mental model, then continue the experiment loop.

## First Run Checklist

If `status.md` doesn't exist:

1. Verify the build: `make clean && make`
2. Verify inference works: `./orome --prompt "Hello" --tokens 5 --k 8`
3. Run baseline benchmark: `uv run tools/benchmark.py --trials 3 --json`
4. Create `results.tsv` with header and baseline row
5. Create `status.md` with baseline results
6. Begin experiment loop