# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **48.61 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **2108.9 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `e2f98ff`
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf`

## Model Format Reality
- This GGUF is **not** a Q5_K mix in practice.
- Tensor counts from `--gguf-info`:
  - `Q8_0`: 311 tensors
  - `F32`: 301 tensors
  - `Q4_K`: 120 tensors
  - `Q6_K`: 1 tensor (LM head)
- Live MoE tensor formats:
  - routed expert `gate/up/down`: **Q4_K**
  - shared expert `gate/up/down`: **Q8_0**
  - `routing_gate`: **F32**
  - `shared_expert_gate`: **F32**
- Consequence: old Q5_K-focused hypotheses should be deprioritized for this model.

## What Worked This Session
1. Re-bench current HEAD at `3626e09`: **47.77 tok/s**, TTFT `2103.2 ms`
2. `44c47ea` — 2-row GGUF `Q5_K` general matvec: **48.10 tok/s**
   - Kept per benchmark protocol, but later metadata inspection showed the model has no `Q5_K` tensors, so treat this as noise rather than a real live-path win
3. `456acce` — fused `F32 routing_gate + shared_expert_gate` into one dispatch: **48.44 tok/s**
   - This was a real live-path win; the prior `shared_expert_gate` dispatch was wasting almost a full 512-thread group on a single output row
4. `e2f98ff` — fused `Q8_0 shared gate + up + SwiGLU`: **48.61 tok/s**
   - This is the current best and confirms that the old `Q4_K`-only shared fusion path was dead for this GGUF

## What Failed
1. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**, TTFT `2105.1 ms`
   - Slightly worse than the `48.61` baseline
   - Conclusion: broad 2-row `Q8_0` is not an automatic win even though `Q8_0` is a major live format

## Interpretation
- The best gains this session came from **removing live dispatches in the MoE side-path**, not from changing the benchmark’s projection micro-metric.
- `proj_avg_ms` stayed pinned at `1.1011`, so the `48.44 -> 48.61` improvement is coming from non-projection work.
- The old campaign’s historical signals remain useful, but only after checking the **actual tensor formats in the current GGUF**. Two dead assumptions surfaced immediately:
  - there are no `Q5_K` tensors in this model
  - shared expert `gate/up` are `Q8_0`, so the old `Q4_K` shared fusion path was never firing

## Next Best Ideas
1. Attack the live `Q8_0 shared_down` path next; it still runs as a separate matvec every MoE layer and is a stronger target than generic `Q8_0` 2-row.
2. Profile or inspect other format-specific dead paths the same way; only chase optimizations that match the actual GGUF tensor map.
3. If another `Q8_0` experiment is tried, prefer **specific fusion or dispatch elimination** over blanket 2-row kernel changes.
4. Use Metal System Trace if available to confirm whether `shared_down` or broader `Q8_0` attention projections now dominate after the two dispatch fusions.
