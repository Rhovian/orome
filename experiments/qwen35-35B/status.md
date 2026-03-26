# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **48.73 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **2096.2 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `e2f98ff`
- Current HEAD: `169939a` (log-only; source state still matches `e2f98ff`)
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
1. Re-bench current HEAD at `169939a`: **48.73 tok/s**, TTFT `2096.2 ms`
   - Confirms the current code still sits in the established `48.6-48.7 tok/s` GGUF regime
2. `n/a` — schedule fused `shared_down` earlier to overlap with routing and routed expert work: **48.54 tok/s**, TTFT `2121.5 ms`
   - Regression versus the fresh baseline; discarded and source reverted
3. No new source change beat the current source-state baseline
   - The best live code path is still the `e2f98ff` shared `Q8_0 gate + up + SwiGLU` fusion, with `169939a` only carrying log updates

## What Failed
1. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**, TTFT `2105.1 ms`
2. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**, TTFT `2098.5 ms`
3. `n/a` — early `shared_down` overlap scheduling: **48.54 tok/s**, TTFT `2121.5 ms`
4. Conclusion: both standalone `Q8_0` micro-kernel variants and the simple early-scheduling `shared_down` overlap idea are below the current best and should be treated as exhausted unless a trace reveals a more specific dependency issue.

## Interpretation
- The branch is stable around **48.5-48.7 tok/s**, and `proj_avg_ms` is still pinned at `1.1011`.
- The new failed test matters because it weakens the simplest remaining `shared_down` scheduling hypothesis: launching it earlier did not expose useful overlap and slightly hurt TTFT.
- The real wins in the GGUF era are still **dispatch elimination/fusion on live MoE side paths**, not projection micro-metrics or naive reordering by themselves.

## Next Best Ideas
1. Revisit `shared_down` only as a **true fusion/writeback elimination** problem, not a scheduling problem: test folding the shared contribution into `moe_combine_copy_sq` or otherwise removing the extra `buf_shared_out` write.
2. Specialize `moe_combine_copy_sq` for the live `K=8` benchmark path and hoist the scalar shared-gate work out of the per-element hot loop.
3. Use Metal System Trace if available to identify the top post-fusion kernels before touching more shader code.
4. Audit the live GGUF tensor map for other low-occupancy or scalar-like dispatches similar to the earlier `shared_expert_gate` waste.
