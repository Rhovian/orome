# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **48.87 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **2101.6 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `48ce29e`
- Source state: matches `48ce29e`
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
1. `a386dbf` — specialize Q4_K scale unpack in the hot GGUF kernels: **48.22 tok/s**
2. `456acce` — fuse `F32 routing_gate + shared_expert_gate` into one dispatch: **48.44 tok/s**
3. `e2f98ff` — fuse `Q8_0 shared gate + up + SwiGLU` for the live shared expert path: **48.61 tok/s**
4. `48ce29e` — specialize `moe_combine_copy_sq` for the live `K=8` path without extra synchronization: **48.87 tok/s**

## Latest Session
1. Re-bench current pre-session source state: **48.51 tok/s**, TTFT `2206.2 ms`
   - Confirms the branch was still in the established `48.5-48.7 tok/s` GGUF regime before new edits
2. `n/a` — specialize `moe_combine_copy_sq` for live `K=8` and hoist `shared_gate` via threadgroup broadcast: **48.30 tok/s**
   - Regression; the added intra-kernel synchronization cost more than it saved
3. `48ce29e` — specialize `moe_combine_copy_sq` for live `K=8` without extra synchronization: **48.87 tok/s**
   - New best result; fixed-`K` unrolling in combine is a real live-path win
4. `n/a` — precompute shared gate sigmoid in `softmax_topk_route` and consume it directly in combine: **48.48 tok/s**
   - Regression; moving the scalar sigmoid upstream did not translate into end-to-end throughput

## Interpretation
- The first real improvement past the prior `48.73 tok/s` plateau came from the **MoE combine kernel**, not projection kernels.
- The useful part of the combine hypothesis was **fixed-`K=8` specialization**; the harmful part was adding synchronization or redistributing the shared-gate scalar work.
- `proj_avg_ms` stayed pinned at `1.1011`, so this gain sits outside the benchmark's projection timing and reinforces that the MoE tail is still a live optimization target.

## What Failed
1. `n/a` — specialize `moe_combine_copy_sq` for live `K=8` and hoist `shared_gate` via threadgroup broadcast: **48.30 tok/s**
2. `n/a` — precompute shared gate sigmoid in `softmax_topk_route` and consume it directly in combine: **48.48 tok/s**
3. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**
4. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**
5. `n/a` — early `shared_down` overlap scheduling: **48.54 tok/s**

## Next Best Ideas
1. Revisit `shared_down` only as a **true writeback-elimination / fusion** problem: fold the shared contribution into combine or otherwise remove the extra `buf_shared_out` traffic.
2. If staying on the combine path, prefer **instruction-level** changes that add no new threadgroup barriers: fixed-slot parameter handling, constant-address-space experiments, or other no-sync simplifications.
3. Use Metal System Trace to confirm whether `moe_combine_copy_sq_k8`, `softmax_topk_route`, or `shared_down` is the top post-`48ce29e` target before making a larger shader change.
4. Audit the live GGUF tensor map again for other small scalar-like dispatches analogous to the earlier `shared_expert_gate` fusion win.

## Current Log
- `48ce29e` is the retained source commit.
- `results.tsv` has been updated with:
  - the fresh `48.51 tok/s` session baseline,
  - the discarded broadcast-hoist combine variant,
  - the kept `48.87 tok/s` `K=8` combine specialization,
  - the discarded route-side shared-gate hoist.
