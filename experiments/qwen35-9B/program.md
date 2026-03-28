# orome — Autonomous Qwen3.5-9B Dense Optimization

You are an autonomous research agent optimizing the current GGUF-only Metal inference engine for Qwen3.5-9B on a Mac Studio M2 Max (96 GB unified memory, 38 GPU cores, active cooling).

This is a dense-model campaign. Preserve generic logic and keep the other supported Qwen3.5 models healthy: the runner will self-check 9B and then cross-check 27B dense plus 35B MoE after each successful session.

## Setup

1. Read `status.md` first. It is your handoff note.
2. Read `results.tsv` next. It is the live 9B campaign log.
3. Read the source files you plan to modify before changing anything.
4. The runner will place you on branch `autoresearch/orome-9B`.

## The Goal

Maximize sustained throughput on the current dense GGUF path:

- Model: `Qwen3.5-9B-Q8_0.gguf`
- Benchmark target: 100 generated tokens
- Active experts: `K=0`
- Primary metric: tok/s

Secondary goals:

- Keep TTFT low
- Preserve output correctness
- Do not regress 27B dense or 35B MoE

The benchmark harness includes a chat quality canary. A run is not valid if the visible answer is empty, garbled, or otherwise fails the quality gate, even if tok/s improves.

## Current Baseline

Treat the current HEAD baseline as approximately:

- `35-36 tok/s`
- `~1.0 s` TTFT on the short smoke path

Use the benchmark harness as the source of truth:

```bash
python3 tools/benchmark.py \
  --model /Users/j/Code/lllm/models/Qwen3.5-9B-Q8_0.gguf \
  --prompt Hello \
  --tokens 100 \
  --k 0 \
  --trials 1 \
  --warmup-runs 1 \
  --cooldown-sec 0 \
  --quality-config experiments/qwen35-9B/cross_check.json \
  --json
```

The command above exits non-zero if the quality canary fails. Treat that as a correctness regression, not a benchmark success.

## Architecture Guidance

The current codebase is organized around GGUF + format abstraction:

- `include/orome.h`
- `src/main.m`
- `src/engine.m`
- `src/metal.m`
- `src/format.m`
- `src/gguf.m`
- `src/shaders.metal`

Important rules for this dense campaign:

- Keep logic generic and driven by `ModelConfig` + GGUF metadata.
- Do not reintroduce hardcoded 9B-shaped tensor assumptions into shared paths.
- Dense-path wins must preserve 27B dense correctness and 35B MoE performance.

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
- `experiments/qwen35-9B/program.md`
- Model weights on disk

## Experiment Protocol

Each experiment must be atomic.

1. Re-benchmark current HEAD if the baseline is unclear.
2. State one concrete hypothesis.
3. Modify only the files required for that idea.
4. Build with:

```bash
make clean && make 2>&1
```

5. Benchmark with:

```bash
python3 tools/benchmark.py \
  --model /Users/j/Code/lllm/models/Qwen3.5-9B-Q8_0.gguf \
  --prompt Hello \
  --tokens 100 \
  --k 0 \
  --trials 1 \
  --warmup-runs 1 \
  --cooldown-sec 0 \
  --quality-config experiments/qwen35-9B/cross_check.json \
  --json \
  2> experiments/qwen35-9B/bench_err.txt
```

6. If the benchmark command fails its quality gate, treat the experiment as a discard even if tok/s appears higher in partial output.
7. Append a row to `results.tsv`.
8. Decide:
   - Better tok/s: keep it, `git add -A`, commit it.
   - Same or worse: discard it and revert source changes cleanly.
   - Quality regression: discard it and revert source changes cleanly.
   - Crash or build failure after reasonable fix attempts: log `crash`, revert, move on.
9. Update `status.md` with:
   - current best result
   - what you tried
   - what worked / failed
   - the next 2-4 best ideas
10. Continue until interrupted.

The runner owns the final `## Runner Validation` section in `status.md` and normalizes the retained `keep` row in `results.tsv` to the final commit hash after post-session checks. Do not hand-edit that runner-managed section.

## Results Format

`results.tsv` is tab-separated:

```text
commit	tok_sec	ttft_ms	proj_avg_ms	status	description
```

## Critical Rules

- Never stop until interrupted by the runner.
- Never ask the user questions during the run.
- One idea per experiment.
- Log every experiment, including crashes and failed builds.
- Keep the repo buildable after every experiment.
- Write `status.md` before the session ends.
- Prefer simple changes when they perform the same.
- Do not leave the tree dirty after a discarded experiment.
