# orome — Autonomous Inference Optimization (Qwen3.5-397B-A17B)

You are an autonomous research agent optimizing a Metal inference engine for Qwen3.5-397B-A17B on an M2 Max Mac Studio (96GB unified memory, ~400 GB/s bandwidth, 38 GPU cores, actively cooled).

This is the larger sibling of the 35B model. The 35B optimization campaign achieved 62.53 tok/s with all experts mlock'd in RAM. On this machine, the 397B model's packed expert layers are about **217 GB on disk** (60 x ~3.62 GB), so they categorically do not fit as a fully resident expert set. This campaign therefore focuses on **adaptive memory strategies** — pread, tiered quantization, thermal-aware K, and hardware-aware hybrid residency — while maintaining the GPU dispatch optimizations already proven on 35B.

## Setup

1. **Read `status.md`** — this is your handoff note from the previous session. It tells you what's been done, what the current best result is, what to try next. If it doesn't exist, this is the first run.
2. **Read `results.tsv`** — the full experiment history. Understand what's been tried and what worked.
3. **Read `adaptive_memory_plan.md`** in the project root — the implementation plan for adaptive memory.
4. **Read the source files** you plan to modify (see "What You CAN Modify" below).
5. You are on branch `autoresearch/orome-397B` (or create it if needed from the current HEAD).

## The Goal

**Maximize tok/s (tokens per second)** on 100-token sustained generation with K=10 active experts (397B uses 10 routed + 1 shared).

The 397B model architecture:
- 60 layers (vs 40 for 35B)
- hidden_dim: 4096 (vs 2048)
- num_experts: 512 (vs 256)
- num_experts_per_tok: 10 routed + 1 shared (vs 8 + 1)
- moe_intermediate: 1024 (vs 512)
- Expert size (4-bit): ~2.36 MB each (vs ~590 KB)
- Total expert weight: ~72.5 GB at 4-bit

Memory budget on the current Mac Studio: the machine reports ~103.1 GB physical RAM, so 0.8 x RAM is ~82.5 GB. `model_weights.bin` is ~5.52 GB, leaving roughly ~69-73 GB of safe resident budget for experts after runtime headroom. The packed expert footprint is ~217.4 GB. **This model requires a streaming/pread-aware path; do not treat it as globally GPU-resident.**

Secondary goals (don't sacrifice tok/s for these):
- Minimize TTFT (time to first token)
- Maintain output quality (don't break the model)
- Keep the mlock fast-path working for 35B (don't regress smaller models)

## Priority: Correctness Before Speed

**Do not optimize tok/s until the model produces correct output.** If the model outputs garbage, NaN, or nonsensical text, that is a correctness bug and must be fixed first. Speed optimization on broken output is wasted work.

To verify correctness: run `./orome --model /Users/j/models/Qwen3.5-397B-A17B-4bit --prompt "Explain quantum computing" --tokens 20 --k 10` and check that the output is coherent English. Use `--profile-experts` to enable NaN checks — any `NAN_GATE` or `NAN_HIDDEN` lines on stderr indicate corruption.

Once the model produces correct output, switch to the optimization axes below.

## Key Optimization Axes (priority order)

1. **Output correctness** — Fix any remaining NaN, garbage output, or dimension mismatches. Use `--profile-experts` for NaN diagnostics. Common issues: wrong buffer sizes for 397B dimensions, numeric overflow in attention recurrence, weight offset misalignment. Test with short prompts and verify output is coherent before benchmarking.

2. **OS page cache tuning** — With ~23 GB available for page cache (96 GB - 73 GB experts), ~32% of experts can be cached. Experiment with read-ahead, access patterns, and madvise hints.

3. **Tiered quantization** — Profile expert activation frequencies, generate hot_experts_25pct.json, run hot experts at 4-bit and cold at 2-bit. This shrinks the effective expert footprint and improves page cache hit rate.

4. **Thermal-aware K** — EMA-based projection latency tracking. Reduce K from 10 to a lower value when sustained generation causes GPU pressure. Less critical on the actively cooled Mac Studio but still useful under memory pressure.

5. **Hybrid mlock/pread** — Instead of pure pread for all 60 layers, keep as many expert layers resident as fit inside the measured resident budget on this machine, and pread the rest. At ~3.62 GB/layer and ~69-73 GB safe resident budget, expect roughly 19-20 resident layers while the remaining ~40 stream from SSD. This eliminates pread latency for a significant fraction of layers. Implement in `moe.m` expert loading — make the decision from actual layer file sizes and measured hardware budget, not from nominal model specs. Must not regress 35B (which already fits entirely in mlock).

6. **Pread I/O optimization** — Tune GCD queue priorities, experiment with readahead patterns, async pread-ahead for the next layer while current layer computes.

7. **GPU dispatch efficiency** — All the 35B optimizations (batched dispatch, fused experts, deferred execution) should carry over. Verify they work correctly with the larger dimensions.

## Codebase Structure

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
Benchmark: `uv run tools/benchmark.py --trials 3 --model-dir /Users/j/models/Qwen3.5-397B-A17B-4bit`

## What You CAN Modify

- **`src/metal.m`** — Metal GPU context, buffer management, dispatch patterns.
- **`src/moe.m`** — Expert loading strategy (pread vs mmap), routing, expert compute. **Primary target for this campaign.**
- **`src/attention.m`** — Attention layer implementations.
- **`src/engine.m`** — Forward pass orchestration, layer loop structure.
- **`src/kernels.m`** — CPU compute kernels.
- **`shaders.metal`** — Metal GPU kernels.
- **`include/orome.h`** — Types and interfaces (if adding new APIs).
- **`Makefile`** — compiler flags, optimization levels.

## What You CANNOT Modify

- **`benchmark.py`** — the benchmark harness. It is the ground truth measurement.
- **`tokenizer.h`** — the tokenizer implementation.
- **`program.md`** — these instructions.
- Model weights on disk.

## Hardware Context

| Resource | Value |
|---|---|
| Memory | ~103.1 GB physical unified (machine-reported) |
| GPU cores | 38 |
| Bandwidth | ~400 GB/s |
| Cooling | Active fan |
| Expert weights (packed 4-bit layers) | ~217.4 GB total |
| Expert weights (2-bit) | ~36.3 GB (fits comfortably) |
| Expert weights (tiered 25% hot) | ~45.4 GB |
| Non-expert weights | ~5.52 GB mmap'd |
| Safe resident expert budget | ~69-73 GB on this machine |
| Memory for page cache / OS headroom | whatever remains after resident budget; do not consume it with "virtual fit" mappings |

## Experiment Protocol

Each experiment:

1. **Describe** what you're trying and why (1-2 sentences).
2. **Modify** the source file(s).
3. **Build**: `make clean && make 2>&1`. If build fails, fix and retry (max 3 attempts). If unfixable, skip.
4. **Benchmark**: `uv run tools/benchmark.py --trials 1 --json 2>bench_err.txt`. Parse the JSON output.
5. **Record** results in `results.tsv` (append).
6. **Decide**:
   - If tok/s improved: **keep** the change. `git add -A && git commit -m "description"`.
   - If tok/s same or worse: **discard**. `git checkout -- src/ include/ Makefile shaders.metal`.
   - If crash: log as crash, discard, move on.
7. **Cross-model check** (after keep only): Run a quick smoke test on the 35B model to verify no regression: `./orome --model /Users/j/models/Qwen3.5-35B-A3B-4bit --prompt "Hello" --tokens 5 --k 8`. It should produce >55 tok/s. If it regresses, revert and try a different approach.
8. **Update `status.md`** with current state and next ideas.
9. **Continue** to the next experiment.

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
- **Be hardware-aware**. Every memory-path experiment must budget against the actual machine-reported RAM and the actual on-disk model files on this machine, not nominal parameter-count estimates.
- **Never confuse virtual mapping with safe residency**. A successful `mmap` of oversized expert files does not mean the model fits in unified memory. Do not disable streaming or enable the fused global GPU-resident path unless the full expert footprint fits inside the computed resident budget.
- **Keep changes atomic**. One idea per experiment. Don't combine multiple changes — you won't know which helped.
- **Log everything**. Every experiment gets a row in results.tsv, even crashes.
- **Write status.md before finishing**. This is your handoff to the next session. Include: current best tok/s, what you've tried, what to try next, any insights.
- **Don't break 35B**. All changes must be backward-compatible. The mlock path must still work for models that fit in memory.
- **Don't over-complicate**. If a simple change gets the same result, prefer it.
- **Be bold**. Try architectural changes, not just parameter tweaks.
- **Revert cleanly**. If something breaks, don't leave the codebase in a bad state.
- **Errors are immediate discards — and must be fully documented**. If a runtime error, crash, GPU fault, or any unexpected failure occurs mid-experiment (during build, benchmark, or inference), follow this exact sequence:
  1. **Discard**: Immediately revert the source changes (`git checkout -- src/ include/ Makefile shaders.metal`).
  2. **Log**: Record the experiment in `results.tsv` with status `crash` and a description that includes the error message or failure mode.
  3. **Document**: Add a dedicated entry in `status.md` under a `## Errors & Lessons` section that records: (a) what was attempted, (b) the exact error or failure observed (paste the relevant stderr/output), (c) the diagnosed root cause, and (d) what this rules out or implies for future experiments.
  4. **Triage**: Understand *why* it failed before moving on. Do not blindly retry the same change.
  5. **Resume**: Use the insight to inform the next experiment.

## First Run Checklist

If `status.md` doesn't exist:

1. Verify the build: `make clean && make`
2. Verify inference works with 397B: `./orome --model /Users/j/models/Qwen3.5-397B-A17B-4bit --prompt "Hello" --tokens 5 --k 10`
3. If the model doesn't load (pread path not yet implemented), implement Phase 1 from `adaptive_memory_plan.md` first
4. Run baseline benchmark: `uv run tools/benchmark.py --trials 3 --json --model-dir /Users/j/models/Qwen3.5-397B-A17B-4bit`
5. Create `results.tsv` with header and baseline row
6. Create `status.md` with baseline results
7. Begin experiment loop
