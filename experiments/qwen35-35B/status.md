# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **48.61 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **2108.9 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `e2f98ff`
- Current HEAD: `b765df3` (log-only; source state still matches `e2f98ff`)
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf`

## Current Campaign Reality
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
- Consequence: old Q5_K-focused hypotheses are historical context only, not primary live-path targets for this model.

## Important GGUF-Era Wins
1. `a386dbf` — specialize Q4_K scale unpack in the hot GGUF kernels: **48.22 tok/s**
2. `456acce` — fuse `F32 routing_gate + shared_expert_gate` into one dispatch: **48.44 tok/s**
3. `e2f98ff` — fuse `Q8_0 shared gate + up + SwiGLU` for the live shared-expert path: **48.61 tok/s**
4. `44c47ea` is still useful context, but later tensor inspection showed its `Q5_K` win is not a real live-path effect for this GGUF.

## Latest Session
1. Re-bench current HEAD at `b765df3`: **48.55 tok/s**, TTFT `2111.3 ms`
   - Confirms the current code stays near the established `48.6 tok/s` GGUF baseline
2. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**, TTFT `2098.5 ms`
   - Slightly above the re-baseline, but still below the standing `48.61` best
3. No new source change beat `e2f98ff`
   - Current best remains the `Q8_0 shared gate + up + SwiGLU` fusion

## What Failed
1. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**, TTFT `2105.1 ms`
2. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**, TTFT `2098.5 ms`
3. Conclusion: both broad and targeted standalone `Q8_0` 2-row variants are below the current best and should be treated as exhausted unless a trace shows a very specific shape/path worth revisiting.

## Interpretation
- The branch is stable around **48.5-48.6 tok/s**, and `proj_avg_ms` is still pinned at `1.1011`.
- The latest failed test matters because it was the strongest remaining simple `Q8_0` hypothesis from the prior handoff. Both broad and targeted `Q8_0` 2-row variants now look exhausted.
- The real wins in the GGUF era are still **dispatch elimination/fusion on live MoE side paths**, not projection micro-metrics by themselves.

## Next Best Ideas
1. Revisit `shared_down` only as a **fusion** problem, not a standalone matvec problem: test folding the shared contribution into `moe_combine_copy_sq` or otherwise removing the extra dispatch/writeback.
2. Use Metal System Trace if available to identify the new post-fusion top kernels before touching more shader code.
3. Audit the live GGUF tensor map for other one-row or low-occupancy dispatches similar to the earlier `shared_expert_gate` waste.
4. Prefer experiments that remove barriers or intermediate buffers only when the producer/consumer dependency is explicit.
