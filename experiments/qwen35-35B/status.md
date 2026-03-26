# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **57.50 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **1944.5 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `88553c3`
- Source state: matches `88553c3`
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
1. `88553c3` — select top-K directly from routing logits and normalize only the chosen experts: **57.50 tok/s**
2. `a386dbf` — specialize Q4_K scale unpack in the hot GGUF kernels: **48.22 tok/s**
3. `456acce` — fuse `F32 routing_gate + shared_expert_gate` into one dispatch: **48.44 tok/s**
4. `e2f98ff` — fuse `Q8_0 shared gate + up + SwiGLU` for the live shared expert path: **48.61 tok/s**
5. `48ce29e` — specialize `moe_combine_copy_sq` for the live `K=8` path without extra synchronization: **48.87 tok/s**

## Latest Session
1. Re-bench current branch head `4954529`: **47.68 tok/s**, TTFT `1986.2 ms`
   - Fresh session baseline before touching routing; confirms the post-`48ce29e` code was still in the same `47-48 tok/s` GGUF regime
2. `88553c3` — select top-K directly from raw routing logits and softmax only the chosen experts: **57.50 tok/s**
   - This keeps the MoE math equivalent for the live path because the old kernel renormalized the selected experts after a full softmax; the global denominator canceled out
   - Sanity run reached `[tok 100]`, so the gain was not an early-EOS benchmark artifact
3. `n/a` — shrink `softmax_topk_route` launch from 256 threads to one 32-thread simdgroup: **57.44 tok/s**
   - Slight regression versus `88553c3`; once the redundant full-softmax work was removed, reducing the launch width alone did not buy additional end-to-end throughput

## Interpretation
- The prior GGUF-era focus on `shared_down` and combine was too narrow. The live routing kernel was still paying for a full 256-way softmax even though the selected expert weights were renormalized immediately afterward.
- For this inference path, selecting top-K on raw logits and normalizing only the chosen experts is mathematically equivalent to the old full-softmax-plus-renorm behavior, and it removed a large amount of redundant work from every layer.
- The jump from `47.68` to `57.50 tok/s` with `proj_avg_ms` still pinned at `1.1011` is strong evidence that a major remaining bottleneck was outside the benchmark's projection timer and inside the MoE routing tail.
- The discarded 32-thread follow-up suggests the main win was the removed math and reduction work, not simply the original 256-thread launch shape.
- The old `shared_down+combine` fusion regression is still relevant: overlap-sensitive MoE work remains real, but routing now has to be treated as a first-class hot path alongside combine and shared expert tail work.

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
1. Use Metal System Trace on `88553c3` to see what replaced routing as the dominant MoE-tail cost: `moe_combine_copy_sq_k8`, `shared_down`, expert down, or some other dispatch outside `proj_avg_ms`.
2. Re-audit the MoE path for other places where the live inference math cancels or renormalizes away work the kernels still perform, analogous to the removed full-softmax denominator in routing.
3. If routing still shows up in trace, try a shape-specific `K=8`, `n_experts=256` specialization that preserves the current 256-thread launch but trims remaining per-threadgroup overhead without changing overlap structure.
4. Revisit combine-side simplifications only if they preserve the existing standalone `shared_down` overlap; the earlier fused consumer-side versions are still negative evidence against serializing that path.

## Current Log
- `88553c3` is the retained source commit and current best result.
- Source code was restored after the discarded 32-thread `softmax_topk_route` launch experiment; the working tree build matches `88553c3` again.
- `results.tsv` has been updated with:
  - the fresh `47.68 tok/s` branch-head baseline on `4954529`,
  - the retained `57.50 tok/s` routing-kernel simplification on `88553c3`,
  - the discarded `57.44 tok/s` 32-thread routing launch follow-up.
