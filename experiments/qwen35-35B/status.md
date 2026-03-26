# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **58.45 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **1947.4 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `bd669d4`
- Source state: matches `bd669d4`
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf`

## Current Campaign Reality
- This GGUF is still the same live-format mix:
  - `Q8_0`: 311 tensors
  - `F32`: 301 tensors
  - `Q4_K`: 120 tensors
  - `Q6_K`: 1 tensor (LM head)
- Live MoE tensor formats:
  - routed expert `gate/up/down`: **Q4_K**
  - shared expert `gate/up/down`: **Q8_0**
  - `routing_gate`: **F32**
  - `shared_expert_gate`: **F32**
- Consequence: the best remaining wins are still in **live GGUF MoE dispatches and combine work**, not old packed-format ideas or Q5-focused hypotheses.

## Important GGUF-Era Wins
1. `bd669d4` — hoist the live `K=8` combine shared-gate sigmoid once per simdgroup without extra sync: **58.45 tok/s**
2. `44eeb52` — defer shared expert gate/up past routing so softmax waits only on routing logits: **57.63 tok/s**
3. `88553c3` — select top-K directly from routing logits and normalize only the chosen experts: **57.50 tok/s**
4. `48ce29e` — specialize `moe_combine_copy_sq` for the live `K=8` path without extra synchronization: **48.87 tok/s**
5. `e2f98ff` — fuse `Q8_0 shared gate + up + SwiGLU` for the live shared expert path: **48.61 tok/s**

## Latest Session
1. Re-bench current branch head `b23b089`: **57.36 tok/s**, TTFT `1924.4 ms`
   - Fresh same-session baseline on the retained routing path before new MoE-tail follow-ups
2. `44eeb52` — defer shared expert gate/up past routing so the first MoE barrier waits only on routing logits: **57.63 tok/s**
   - Moving the live shared expert projection out of the initial routing barrier bought a small but real gain while keeping the same dependency structure
3. `bd669d4` — hoist the live `K=8` combine shared-gate sigmoid once per simdgroup via `simd_broadcast_first`: **58.45 tok/s**
   - This revisits an old negative shared-gate-hoist idea, but without the threadgroup synchronization that previously erased the benefit

## Interpretation
- Routing was not the end of the GGUF MoE story. After `88553c3`, there was still measurable time in overlap-sensitive shared-expert and combine-side work even though `proj_avg_ms` stayed pinned at `1.1011`.
- `44eeb52` is positive evidence that the initial MoE routing barrier was still too conservative: letting shared expert gate/up overlap with routed expert work is better than making routing softmax wait on it.
- `bd669d4` is positive evidence that combine-side scalar hoists can help, but only when they preserve the current overlap structure. The old threadgroup-broadcast shared-gate experiment was negative; the barrier-free simdgroup-broadcast version is positive.
- A short Metal System Trace on `44eeb52` completed successfully, but the current command buffers and encoders are effectively unlabeled for this workload, so the trace was not yet sufficient to cleanly identify which individual `orome` dispatch replaced routing as the next dominant cost.

## What Failed
1. `n/a` — shrink `softmax_topk_route` launch from 256 threads to a single 32-thread simdgroup: **57.44 tok/s**
2. `n/a` — fuse live `Q8_0 shared_down` into the `K=8` combine path to eliminate `shared_out` writeback: **46.91 tok/s**
3. `n/a` — read combine params from constant address space and scalarize the fixed `K=8` weights: **48.39 tok/s**
4. `n/a` — specialize live `Q8_0 shared_down` for fixed `S=512` with smaller threadgroup scratch and full SIMD participation: **48.81 tok/s**
5. `n/a` — specialize `moe_combine_copy_sq` for live `K=8` and hoist `shared_gate` via threadgroup broadcast: **48.30 tok/s**
6. `n/a` — precompute shared gate sigmoid in `softmax_topk_route` and consume it directly in combine: **48.48 tok/s**
7. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**
8. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**
9. `n/a` — early `shared_down` overlap scheduling: **48.54 tok/s**

## Next Best Ideas
1. Add temporary Metal labels around `orome` compute command buffers and encoders before another Metal System Trace pass, so the trace can attribute the remaining MoE-tail cost to a specific dispatch instead of anonymous `Compute Command 0` work.
2. Try a shape-specific `softmax_topk_route` specialization for `K=8`, `n_experts=256` that preserves the current 256-thread launch but trims the remaining top-k overhead without changing overlap structure.
3. Revisit early `shared_down` overlap from the new `44eeb52` scheduling baseline; the old negative evidence predates deferring shared gate/up past routing.
4. Look for other combine-side scalar or reduction hoists that can use simdgroup broadcast without threadgroup barriers, following the positive `bd669d4` result.

## Current Log
- `bd669d4` is the retained source commit and current best result.
- `44eeb52` remains retained underneath it as the first scheduling win of this session.
- A short Metal System Trace was captured on `44eeb52`, but unlabeled `orome` compute work limited per-dispatch attribution.
- `results.tsv` has been updated with:
  - the fresh `57.36 tok/s` baseline on `b23b089`,
  - the retained `57.63 tok/s` shared-gate-up scheduling change on `44eeb52`,
  - the retained `58.45 tok/s` barrier-free combine shared-gate hoist on `bd669d4`.
