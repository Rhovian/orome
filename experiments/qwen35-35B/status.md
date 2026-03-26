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
1. Re-bench current pre-session source state: **48.63 tok/s**, TTFT `2119.0 ms`
   - Confirms the retained `48ce29e` source still reruns in the established `48.6-48.8 tok/s` GGUF regime on this machine
2. `n/a` — read combine params from constant address space and scalarize the fixed `K=8` weights: **48.39 tok/s**
   - Regression; removing the small device-buffer reads in combine did not help the live path
3. `n/a` — specialize live `Q8_0 shared_down` for fixed `S=512` with smaller threadgroup scratch and full SIMD participation: **48.81 tok/s**
   - Better than the session rerun, but still below the retained `48.87 tok/s` best; not enough gain to justify extra code

## Interpretation
- The retained winner is still the **no-sync `K=8` combine specialization** in `48ce29e`.
- The new constant-address-space combine attempt reinforces that the remaining combine cost is not a simple uniform-parameter fetch problem.
- The `shared_down` shape-specific kernel likely improved that local dispatch, but not enough end-to-end to beat the retained best. That points to overlap and surrounding MoE scheduling still mattering more than a standalone `shared_down` micro-kernel.
- `proj_avg_ms` stayed pinned at `1.1011` across the session, so the remaining headroom is still outside the benchmark's projection timer and inside the MoE tail / dispatch structure.

## What Failed
1. `n/a` — read combine params from constant address space and scalarize the fixed `K=8` weights: **48.39 tok/s**
2. `n/a` — specialize live `Q8_0 shared_down` for fixed `S=512` with smaller threadgroup scratch and full SIMD participation: **48.81 tok/s**
3. `n/a` — specialize `moe_combine_copy_sq` for live `K=8` and hoist `shared_gate` via threadgroup broadcast: **48.30 tok/s**
4. `n/a` — precompute shared gate sigmoid in `softmax_topk_route` and consume it directly in combine: **48.48 tok/s**
5. `n/a` — 2-row GGUF `Q8_0` general matvec: **48.60 tok/s**
6. `n/a` — dedicated 2-row `Q8_0 shared_down` kernel: **48.57 tok/s**
7. `n/a` — early `shared_down` overlap scheduling: **48.54 tok/s**

## Next Best Ideas
1. Revisit `shared_down` only as a **consumer-side fusion / true writeback-elimination** problem, and only if overlap with routed expert down work is preserved or replaced with a bigger win.
2. Use Metal System Trace to determine whether the local `shared_down` improvement is being hidden by existing overlap, or whether `moe_combine_copy_sq_k8` / routing dispatches remain the real top target post-`48ce29e`.
3. Audit the live GGUF tensor map again for other small scalar-like or low-width dispatches analogous to the earlier `shared_expert_gate` fusion win.
4. If staying on the combine path, try another no-sync simplification that does not move the shared-gate sigmoid upstream or just relocate uniform params.

## Current Log
- `48ce29e` remains the retained source commit.
- `results.tsv` has been updated with:
  - the fresh `48.63 tok/s` session baseline,
  - the discarded constant-address-space `K=8` combine variant,
  - the discarded live `Q8_0 shared_down` `S=512` specialization.
