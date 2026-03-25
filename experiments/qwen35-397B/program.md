# orome — Autonomous Inference Optimization (Qwen3.5-397B-A17B)

You are an autonomous research agent optimizing a Metal inference engine for Qwen3.5-397B-A17B on an M2 Max Mac Studio (96GB unified memory, ~400 GB/s bandwidth, 38 GPU cores, actively cooled).

This is the larger sibling of the 35B model. The 35B optimization campaign achieved 62.53 tok/s with all experts mlock'd in RAM. On this machine, the 397B model's packed expert layers are about **217 GB on disk** (60 x ~3.62 GB), so they categorically do not fit as a fully resident expert set. This campaign therefore focuses on **adaptive memory strategies** — pread, tiered quantization, thermal-aware K, and hardware-aware hybrid residency — while maintaining the GPU dispatch optimizations already proven on 35B.

## Setup

1. **Read `status.md`** — this is your handoff note from the previous session. It tells you what's been done, what the current best result is, what to try next. If it doesn't exist, this is the first run.
2. **Read `results.tsv`** — the full experiment history. Understand what's been tried and what worked.
3. **Read `io_instrumentation_plan.md`** in this directory — the I/O instrumentation and hybrid memory research plan.
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

Memory budget on the current Mac Studio: the machine reports ~103.1 GB physical RAM, so 0.8 x RAM is ~82.5 GB. `model_weights.bin` is ~5.52 GB, leaving roughly ~69-73 GB of safe resident budget for experts after runtime headroom. The packed expert footprint is ~217.4 GB (4-bit) or ~120.8 GB (2-bit). Even at 2-bit, experts don't fit resident. **This model requires a streaming/pread-aware path; do not treat it as globally GPU-resident.** The per-layer expert cache uses ~2.7 GB of Metal shared memory — this comes out of the page cache budget.

Secondary goals (don't sacrifice tok/s for these):
- Minimize TTFT (time to first token)
- Maintain output quality (don't break the model)
- Keep the mlock fast-path working for 35B (don't regress smaller models)

## Priority: Correctness Before Speed

**Do not optimize tok/s until the model produces correct output.** If the model outputs garbage, NaN, or nonsensical text, that is a correctness bug and must be fixed first. Speed optimization on broken output is wasted work.

To verify correctness: run `./orome --model /Users/j/models/Qwen3.5-397B-A17B-4bit --prompt "Explain quantum computing" --tokens 20 --k 10` and check that the output is coherent English. Use `--profile-experts` to enable NaN checks — any `NAN_GATE` or `NAN_HIDDEN` lines on stderr indicate corruption.

Once the model produces correct output, switch to the optimization axes below.

## Phase 1: SSD/pread optimization [COMPLETE — diminishing returns]

Phase 1 ran 46 experiments optimizing the pread I/O path. Results: 4.08 → 7.87 tok/s (+93%). The last 4 experiments were all within noise or regression. The pread path is at its ceiling.

**What worked**: 2-bit quantization (-44% I/O bytes), per-layer expert cache (K+2 slots, ~40% hit rate), concurrent GPU dispatch.

**Current bottleneck breakdown** at 7.87 tok/s (~127 ms/tok):
- MoE I/O (pread): ~56ms (44%) — 90% page cache hits, 10% SSD misses
- Attention (GPU): ~35ms (28%) — 15 full + 45 linear attention layers
- MoE GPU (expert forward): ~32ms (25%) — GPU expert wait avg 0.53ms/layer (31.6ms total for 60 layers)
- LM head + other: ~4ms (3%)

**Serial dependency chain per layer prevents I/O-compute overlap**:
```
GPU attention → routing readback → pread I/O → GPU expert forward → CPU combine
```
Each step depends on the previous. Pipelining (overlap layer N's pread with layer N-1's GPU) requires knowing routing before attention completes, which is impossible.

**Memory budget is balanced**: ~83 GB page cache covers 69% of 120.8 GB 2-bit experts. The ~38 GB gap causes the 10% SSD miss rate. K+4 cache proved that shifting even 450 MB from page cache to app cache regresses. No free memory to reclaim.

**Expert routing frequency is input-dependent** and we lack representative workload data for statistical significance. Static hot expert sets would overfit to profiling prompts. The adaptive K+2 cache already captures temporal locality without this problem.

## Phase 2: Compute optimization + tiered quantization [ACTIVE]

### Key Optimization Axes (priority order)

1. **CMD merge (~32ms MoE GPU, 25%)** — Combine the GPU expert forward command buffer with the next layer's attention command buffer into a single CMD. Currently each of 60 layers does a separate commit/wait for expert forward. Merging saves ~60 commit/wait cycles and lets the GPU pipeline expert compute into the next layer's attention without a CPU round-trip. This is the only untried approach on the compute side. Expected: 1-5ms savings from eliminating dispatch overhead.

2. **Attention optimization (~35ms, 28%)** — Attention is the largest compute component. A 30% improvement saves ~10ms → ~8.5 tok/s. Possible approaches: kernel fusion within attention layers, reduced-precision attention state, or architectural shortcuts for linear attention (GatedDeltaNet). Do NOT spend more than 2-3 experiments before moving on.

3. **Tiered quantization** — First attempt regressed (6.83 tok/s, row 49) because cache slots grew to 6.75 MB (4-bit sized) → 4860 MB total, causing page cache pressure. Fix: mixed-size cache management (2-bit slots for cold, separate 4-bit buffers for hot). This is primarily a quality play, not a speed play.

4. **Token-level batching** — Process 2 tokens simultaneously through the layer loop. Expert data loaded once, used for both tokens. Expected: significant improvement if routing overlap is high. Risk: massive code change (attention, MoE routing, buffer management all need batch support). Consider only after simpler approaches are exhausted.

5. **Output correctness (guard rail)** — Model outputs correctly at 7.87 tok/s. Use `--profile-experts` for NaN checks if any code change is suspected of breaking numerics.

### What NOT to pursue (and why)

- **Further I/O scheduling**: 46 experiments exhausted this. Every pipelining, prefetch, and scheduling approach regressed from overhead or contention.
- **Expert cache expansion**: K+2 is the sweet spot. K+3 within noise, K+4 regressed from page cache pressure.
- **GPU dispatch optimization**: Batched kernels, GPU combine, -O3, GPU norm+LM head+argmax all within noise. GPU dispatch is not the bottleneck.
- **Fused 2-bit gate+up+SwiGLU kernel**: Crashed with all-NaN. Even if fixed, GPU dispatch overhead is proven within noise.
- **Tiered quantization with uniform cache slots**: Regressed (6.83 tok/s). Must use mixed-size cache if revisiting.
- **Sub-2-bit quantization**: Degrades quality too much.
- **Partial mlock or mmap**: Proven harmful in multiple experiments. The OS page cache with pread is the best I/O strategy for this memory/data ratio.
- **Static expert frequency analysis without representative workload**: Input-dependent routing makes offline profiling unreliable at this stage.
- **Naive expert prefetch heuristics**: 5 attempts all regressed (rows 4, 10, 37, 41, 46). However, a *learned* pre-attention expert prediction model (trained to predict routing from hidden state before attention) has been demonstrated externally at 93-97% accuracy on NVIDIA GB10, enabling async prefetch during the ~35ms attention window. Our heuristics failed because prediction accuracy was too low — wasted bandwidth on wrong experts. A learned predictor is a research project (collect routing training data, train small model, integrate async prefetch) but could break the serial dependency chain if accuracy exceeds ~90%. Consider after simpler optimizations are exhausted.

## Lessons Learned (do NOT repeat these)

- **Partial mlock starves page cache**: 19 layers mlock'd (69 GB) -> 3.61 tok/s (was 4.08). Pinning RAM steals from the page cache that serves the other 41 layers.
- **F_RDADVISE prefetch**: massive regression (1.81 tok/s) from I/O contention.
- **Speculative expert prefetch** (prev-token indices): 2.91 tok/s regression.
- **Pipelined pread+GPU** (split K=10 into 2 groups): slight regression (3.99), 2-cmd-buffer overhead ate the overlap.
- **mmap without mlock + MADV_RANDOM**: catastrophic (0.95 tok/s), page fault overhead.
- **mmap with default hints**: equally catastrophic (0.95 tok/s), memcpy from mmap no better than pread.
- **fcntl(F_RDAHEAD,0)**: regression (2.92 tok/s), read-ahead helps page cache.
- **Sorted expert indices**: within noise, GCD concurrent anyway.
- **Static GCD queue reuse**: within noise.
- **GPU matvec_4bit_2row for pread path**: within noise, GPU not bottleneck.
- **CPU-side expert cache (malloc)**: page cache regression (5.49 tok/s), Metal shared buffers behave differently.
- **K+4 cache slots (3.15 GB)**: page cache regression from memory pressure. Sweet spot is K+2.
- **F_RDADVISE pre-warming during attention**: I/O contention (6.89 tok/s).
- **2-bit batched expert kernels** (4 dispatches vs 31): within noise (7.96 tok/s). GPU dispatch overhead is not the bottleneck.
- **GPU combine for pread path**: extra barrier+kernel negated readback savings (7.64 tok/s). Within noise.
- **-O3 compiler optimization**: within noise of -O2 (7.89 tok/s). CPU code paths not bottleneck.
- **Pipelined pread+GPU via MTLSharedEvent**: 2-encoder+event overhead (>0.5ms/layer) negated any overlap savings (6.98 tok/s).
- **Fused 2-bit gate+up+SwiGLU kernel**: All-NaN output at layer 0. The kernel's 2-bit unpacking loop may have incorrect offset computation when reading from per-layer cache buffers. GPU dispatch overhead was already proven not to be the bottleneck (rows 43, 44, 45).
- **GPU norm+LM head+argmax for pread path**: Correct but within noise (~0.5ms savings out of 127ms). The 993KB logits readback + CPU argmax is negligible.
- **Tiered quantization with uniform cache sizing**: Cache slots sized for 4-bit (6.75 MB each) doubled cache from 2700 MB to 4860 MB, starving page cache. Tiered quant requires mixed-size cache management to be viable.

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
3. If the model doesn't load, check that the pread path is working in `moe.m`
4. Run baseline benchmark: `uv run tools/benchmark.py --trials 3 --json --model-dir /Users/j/models/Qwen3.5-397B-A17B-4bit`
5. Create `results.tsv` with header and baseline row
6. Create `status.md` with baseline results
7. Begin experiment loop
