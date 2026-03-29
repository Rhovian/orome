# Research

Orome uses autonomous research sessions to optimize inference throughput and quality per model family.

## Experiment Targets

- **[experiments/qwen35-9B/program.md](../experiments/qwen35-9B/program.md)** — 9B dense GGUF optimization
- **[experiments/qwen35-27B/program.md](../experiments/qwen35-27B/program.md)** — 27B dense GGUF optimization
- **[experiments/qwen35-35B/program.md](../experiments/qwen35-35B/program.md)** — 35B MoE GGUF optimization

## Running Experiments

```bash
./scripts/run_experiments.sh qwen35-9B --agent codex --sessions 3
./scripts/run_experiments.sh qwen35-27B --agent codex --sessions 3
./scripts/run_experiments.sh qwen35-35B --sessions 3
./scripts/run_experiments.sh qwen35-35B --agent codex --sessions 3
```

Each session reads `status.md` for handoff, follows `program.md` for protocol, and appends to `results.tsv`. The runner reverts session commits if throughput or output quality regresses. Each target defines a `cross_check.json` used for self-check and cross-model regression gates.
