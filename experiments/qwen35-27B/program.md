# orome — Autonomous Qwen3.5-27B Dense Optimization

You are an autonomous research agent optimizing the current GGUF-only Metal inference engine for Qwen3.5-27B on a Mac Studio M2 Max (96 GB unified memory, 38 GPU cores, active cooling).

This is a dense-model campaign. Preserve generic logic and keep the other supported Qwen3.5 models healthy: the runner will self-check 27B and then cross-check 9B dense plus 35B MoE after each successful session.

## Setup

1. Read `status.md` first. It is your handoff note.
2. Read `results.tsv` next. It is the live 27B campaign log.
3. Read the source files you plan to modify before changing anything.
4. The runner will place you on branch `autoresearch/orome-27B`.
5. Use local `../llama.cpp` as the reference implementation for parity work on this model.

## The Goal

Improve 27B output quality and stability from the current parity-winning backend without materially giving back throughput:

- Model: `Qwen3.5-27B-Q4_K_M.gguf`
- Benchmark target: 100 generated tokens
- Active experts: `K=0`
- Throughput floor: keep Orome in the current `~17.8 tok/s` band
- Quality target: make the semantic suite more consistently pass and eliminate raw think/punctuation collapse behavior

Secondary goals:

- Keep TTFT low
- Preserve or improve the current lead over local `llama.cpp`
- Do not regress 9B dense or 35B MoE

The parity target for this campaign is local `llama.cpp` on the same machine, not just Orome's past numbers. The runner now validates successful 27B sessions against both Orome-only checks and Orome-vs-llama compare harnesses.

The benchmark harness includes a multi-case quality suite. For 27B, the runner now uses the same three semantic prompts we use for local quality comparison (`capital`, `opposite`, `sky`) instead of a bare `Hello` canary. The current Orome-only floor is intentionally set to the observed baseline: at least 2 of those 3 cases must pass. A run is not valid if it drops below `2/3`, even if tok/s improves.

## Current Baseline

Treat the current HEAD baseline as approximately:

- `17.8-18.0 tok/s`

Use the benchmark harness as the source of truth for the local Orome-only floor:

```bash
python3 tools/benchmark.py \
  --model /Users/j/Code/lllm/models/Qwen3.5-27B-Q4_K_M.gguf \
  --prompt Hello \
  --tokens 100 \
  --k 0 \
  --trials 1 \
  --warmup-runs 1 \
  --cooldown-sec 0 \
  --quality-config experiments/qwen35-27B/cross_check.json \
  --json
```

The command above exits non-zero if the 27B quality suite falls below its configured case-pass floor. Treat that as a correctness regression, not a benchmark success.

The runner also performs local `llama.cpp` parity checks after a kept session. The current parity floors are:

- Orome throughput must be at least `1.0x` local `llama.cpp` on the fixed 100-token compare
- Orome quality must hit at least `2/3` cases in at least `2` of `3` repeated compare runs

This means the optimization priority is now:

1. Keep Orome ahead of local `llama.cpp` on throughput
2. Improve repeated quality behavior on the semantic suite
3. Only then chase additional tok/s

For the next batch of work, be explicit about what NOT to do:

- Do not spend sessions broadly expanding `Q4_K` coverage just because earlier wins came from that line of work.
- Do not treat “more llama-style quant kernels” as a default hypothesis.
- Only touch `Q4_K` kernel coverage if the hypothesis is directly tied to reducing think leakage, punctuation collapse, or quality variance.

The preferred hypothesis classes right now are:

1. suppress raw `<think>` leakage in Orome outputs
2. reduce punctuation-collapse / repeated-`!` variance
3. preserve the current `~17.8 tok/s` band and local `llama.cpp` lead while improving repeated quality

## Local llama.cpp Reference

Use the local reference repo at:

- `/Users/j/Code/lllm/llama.cpp`

Useful local comparison commands:

```bash
python3 tools/compare_orome_llama.py --models 27B --json
python3 tools/compare_orome_llama_quality.py --models 27B --json
```

Start from these local llama.cpp files before guessing:

- `/Users/j/Code/lllm/llama.cpp/src/models/qwen35.cpp`
- `/Users/j/Code/lllm/llama.cpp/src/models/delta-net-base.cpp`
- `/Users/j/Code/lllm/llama.cpp/src/llama-context.cpp`
- `/Users/j/Code/lllm/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- `/Users/j/Code/lllm/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`

## Architecture Guidance

The current codebase is organized around GGUF + format abstraction:

- `include/orome.h`
- `inference/src/main.m`
- `inference/src/engine.m`
- `inference/src/metal.m`
- `inference/src/format.m`
- `inference/src/gguf.m`
- `inference/src/shaders.metal`

Important rules for this dense campaign:

- Keep logic generic and driven by `ModelConfig` + GGUF metadata.
- Do not reintroduce hardcoded 27B-shaped tensor assumptions into shared paths.
- Dense-path wins must preserve 9B dense correctness and 35B MoE performance.
- For this phase, prefer hypotheses about output behavior and runtime semantics over broader quant-kernel enablement.

## What You CAN Modify

Primary hot-path files:

- `inference/src/engine.m`
- `inference/src/shaders.metal`
- `inference/src/metal.m`
- `inference/src/format.m`
- `include/orome.h`
- `Makefile`

Secondary files if needed for a specific optimization:

- `inference/src/main.m`
- `inference/src/gguf.m`

## What You Should Not Modify During Experiment Runs

- `tools/benchmark.py`
- `experiments/qwen35-27B/program.md`
- Model weights on disk

## Experiment Protocol

Each experiment must be atomic.

1. Re-benchmark current HEAD if the baseline is unclear.
2. State one concrete hypothesis.
   The hypothesis should directly target one of:
   - raw `<think>` leakage
   - punctuation collapse / repeated-character spam
   - quality variance between repeated runs
3. Modify only the files required for that idea.
4. Build with:

```bash
make clean && make 2>&1
```

5. Benchmark with:

```bash
python3 tools/benchmark.py \
  --model /Users/j/Code/lllm/models/Qwen3.5-27B-Q4_K_M.gguf \
  --prompt Hello \
  --tokens 100 \
  --k 0 \
  --trials 1 \
  --warmup-runs 1 \
  --cooldown-sec 0 \
  --quality-config experiments/qwen35-27B/cross_check.json \
  --json \
  2> experiments/qwen35-27B/bench_err.txt
```

6. If the benchmark command fails its quality gate, treat the experiment as a discard even if tok/s appears higher in partial output.
7. If you keep a result locally, expect the runner to also run the Orome-vs-llama throughput compare and the repeated Orome-vs-llama quality compare before the session is accepted.
8. Append a row to `results.tsv`.
9. Decide:
   - Better or equal quality at roughly the same throughput: keep it, `git add -A`, commit it.
   - Better tok/s with the same or better repeated quality: keep it, `git add -A`, commit it.
   - Quality regression: discard it and revert source changes cleanly, even if tok/s improves.
   - Material throughput regression: discard it and revert source changes cleanly, even if one quality case improves.
   - Crash or build failure after reasonable fix attempts: log `crash`, revert, move on.
10. Update `status.md` with:
   - current best result
   - what you tried
   - what worked / failed
   - the next 2-4 best ideas
11. Continue until interrupted.

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
