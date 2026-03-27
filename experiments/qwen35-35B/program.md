# orome — Autonomous 35B GGUF Optimization

You are an autonomous research agent optimizing the current GGUF-only Metal inference engine for Qwen3.5-35B-A3B on a Mac Studio M2 Max (96 GB unified memory, 38 GPU cores, active cooling).

This is a continuation of the older 35B campaign, not a blank slate. The repo contains many historical experiments. Some are still highly relevant, but many of the best old numbers came from a pre-GGUF packed-format era that no longer exists in the source tree. Use the history as a source of hypotheses, not as a literal reproduction target.

## Setup

1. Read `status.md` first. It is your handoff note.
2. Read `results.tsv` next. It is the current GGUF campaign log.
3. Read `results.historical.tsv` and `docs/qwen-35b-a3b.md` for old 35B campaign context and prior wins when useful.
4. Read the source files you plan to modify before changing anything.
5. The runner will place you on branch `autoresearch/orome`.

## The Goal

Maximize sustained throughput on the current GGUF path:

- Model: `Qwen3.5-35B-A3B-Q4_K_S.gguf`
- Benchmark target: 100 generated tokens
- Active experts: `K=8`
- Primary metric: tok/s

Secondary goals:

- Keep TTFT low
- Preserve output correctness

The benchmark harness now includes a chat quality canary. A run is not valid if
the visible answer is empty, garbled, or fails the canary keyword check, even if
tok/s improves.

## Current Baseline

Treat the current HEAD baseline as approximately:

- `58.4 tok/s`
- `1.9-2.0 s` TTFT
- `~1.1 ms` average projection timing from benchmark output

Use the benchmark harness as the source of truth:

```bash
python3 tools/benchmark.py --trials 1 --warmup-runs 1 --cooldown-sec 0 --json
```

The command above now exits non-zero if the quality canary fails. Treat that as
a correctness regression, not a benchmark success.

The old `62.53 tok/s` result is historically important, but it came from the previous 35B format/layout. It is not the current baseline for this GGUF-only codebase.

## Model And Runtime Facts

These are the current facts that matter for optimization:

- 40 layers total
- 10 full-attention layers + 30 linear-attention layers in the current GGUF path
- Hidden dim `2048`
- 256 experts, with 8 routed experts per token plus 1 shared expert
- Mixed GGUF quantization:
  - Q4_K for routed expert projections and most large live weights
  - Q8_0 for shared expert gate/up/down tensors
  - Q6_K for LM head
  - F32 for routing_gate and shared_expert_gate
  - F32 / BF16 for smaller tensors and norms

There are no live Q5_K hot-path tensors in the current GGUF. Old Q5_K-specific wins are historical context, not current baseline assumptions.

The model fits in unified memory, so this is still fundamentally a GPU dispatch and kernel-efficiency problem, not an SSD streaming problem.

## Current Architecture

The current codebase is organized around GGUF + format abstraction:

```text
include/orome.h      shared types, ModelConfig, MetalCtx, TensorRef, Engine
src/main.m           CLI parsing, GGUF metadata loading, engine startup
src/engine.m         forward-pass orchestration, per-layer execution, profiling
src/metal.m          Metal setup, pipeline creation, buffer allocation
src/format.m         GGUF tensor cache build, deinterleaving, format dispatch
src/gguf.m           GGUF parser and metadata extraction
src/kernels.m        CPU timing and sampling helpers
src/tokenizer.m      BPE tokenizer wrapper
src/server.m         HTTP/SSE server
src/shaders.metal    Metal kernels for attention, MoE, and GGUF dequant matvec
```

Important runtime facts:

- The engine is GGUF-only now.
- The forward pass uses a concurrent compute encoder.
- Tensor metadata is pre-resolved into `TensorRef`, `LayerTensorCache`, and `ExpertLayerRef`.
- CPU fallback hot paths from the older system are gone.

## How To Read The Old Experiment History

`results.tsv` is now the live GGUF campaign only. Older packed-format results live in `results.historical.tsv`. Use that historical file intelligently.

Still relevant or likely relevant:

- 2-row-per-simdgroup ideas for Q4_K, Q6_K, and expert kernels
- Fusing expert gate + up + SwiGLU work
- GPU routing / softmax-topk / argmax to avoid CPU readbacks
- Concurrent dispatch and overlap of independent GPU work
- Precomputing offsets / removing per-token metadata or lookup overhead
- Reducing dispatch count and barriers when dependency analysis supports it
- Specialized LM-head optimization for Q6_K

Potentially obsolete or much less relevant:

- Explicit 35B `mlock` / prefault strategies from the old campaign
- Legacy packed-format assumptions or layouts
- Experiments that depend on removed CPU fallback paths
- Exact tok/s comparisons to the old 62 tok/s era

Do not blindly repeat old discarded experiments. Only revisit them if the code structure has materially changed and you can state clearly why the old result may no longer apply.

## Historical Signals Worth Carrying Forward

The older 35B campaign established some strong priors:

- `ROWS_PER_TG=16` was consistently best in prior testing
- 2-row matvec helped meaningfully in the old path
- 4-row matvec lost to register pressure
- Concurrent dispatch helped
- Over-aggressive barrier removal hurt when it destroyed overlap
- Whole-forward single-command-buffer ideas were not consistently wins
- CPU-side alternatives for hot-path dequant / attention work were bad
- Gate+up+SwiGLU fusion helped

Use those as priors, not commandments.

## First-Session Guidance

Because the codebase has changed quickly, start carefully:

1. Re-benchmark current HEAD.
2. If `results.tsv` does not already contain a clearly labeled baseline near the current `58.x tok/s`, append one before trying new optimizations.
3. Identify one concrete hypothesis from `results.historical.tsv` or the recent GGUF log that still matches the current architecture.
4. Run exactly one experiment.

Good first candidates:

- Re-apply 2-row ideas specifically to current GGUF kernels that do not already have them
- Reduce repeated Q4_K scale unpacking overhead in the hottest kernels
- Remove unnecessary barriers only when the producer/consumer relationship is understood
- Revisit expert fusion opportunities that are still split across dispatches in the GGUF path

## What You CAN Modify

Primary hot-path files:

- `src/engine.m`
- `src/shaders.metal`
- `src/metal.m`
- `src/format.m`
- `include/orome.h`
- `Makefile`

Secondary files if needed for a specific optimization:

- `src/main.m`
- `src/gguf.m`

## What You Should Not Modify During Experiment Runs

- `tools/benchmark.py`
- `experiments/qwen35-35B/program.md`
- Model weights on disk

## Experiment Protocol

Each experiment must be atomic.

1. State the hypothesis briefly in your own notes.
2. Modify only the files required for that idea.
3. Build with:

```bash
make clean && make 2>&1
```

4. Benchmark with:

```bash
python3 tools/benchmark.py --trials 1 --warmup-runs 1 --cooldown-sec 0 --json \
  2> experiments/qwen35-35B/bench_err.txt
```

5. If the benchmark command fails its quality gate, treat the experiment as a
   discard even if tok/s appears higher in the partial output.
6. Append a row to `results.tsv`.
7. Decide:
   - Better tok/s: keep it, `git add -A`, commit it.
   - Same or worse: discard it and revert source changes cleanly.
   - Quality regression: discard it and revert source changes cleanly.
   - Crash or build failure after reasonable fix attempts: log `crash`, revert, move on.
8. Update `status.md` with:
   - current best result
   - what you tried
   - what worked / failed
   - the next 2-4 best ideas
9. Continue until interrupted.

## Results Format

`results.tsv` is tab-separated:

```text
commit	tok_sec	ttft_ms	proj_avg_ms	status	description
```

Field meanings:

- `commit`: short git hash or `n/a`
- `tok_sec`: primary performance metric
- `ttft_ms`: time to first token
- `proj_avg_ms`: average projection timing from stderr/profile output when available
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line description of the experiment

## Critical Rules

- Never stop until interrupted by the runner.
- Never ask the user questions during the run.
- One idea per experiment.
- Log every experiment, including crashes and failed builds.
- Keep the repo buildable after every experiment.
- Write `status.md` before the session ends.
- Prefer simple changes when they perform the same.
- Be bold, but not random: every experiment should connect to a concrete bottleneck or a historical signal.
- Do not leave the tree dirty after a discarded experiment.
