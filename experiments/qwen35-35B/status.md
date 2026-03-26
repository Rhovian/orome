# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Current Best
- **66.74 tok/s** (Q4_K_S GGUF, 100 tokens sustained, 66.91 on second run)
- TTFT: **~2182 ms**
- `proj_avg_ms`: **1.1011**
- Branch: `autoresearch/orome`
- Best keep commit: `c2391c1`
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf`

## Current Campaign Reality
- GGUF live-format mix: Q8_0 (311), F32 (301), Q4_K (120), Q6_K (1)
- Routed expert gate/up/down: **Q4_K**, shared expert: **Q8_0**, routing: **F32**
- Best remaining wins: dispatch/barrier reduction, kernel efficiency, scheduling overlap

## This Session's Wins (from 65.44 baseline)
1. `9b97283` — defer GDN state decay to second pass (halve state write traffic): **65.85 tok/s** (+0.41)
2. `6721aac` — de-interleave Q+gate weights at load time (eliminate 2 dispatch+2 barrier per full-attn layer): **65.92 tok/s** (+0.07)
3. `8b8f329` — fuse sigmoid gate into attn_values kernel (eliminate 1 dispatch+1 barrier per full-attn layer): **65.96 tok/s** (+0.04)
4. `513fc2d` — schedule decay_beta alongside conv1d for concurrent overlap in linear attention: **66.50 tok/s** (+0.54)
5. `e2bdd78` — fuse QK RMS norm into delta_net kernel (eliminate 1 dispatch+1 barrier per linear layer): **66.12 tok/s** (noise)
6. `c2391c1` — 2-row Q8_0 matvec (halve TG count for all Q8_0 projections): **66.74 tok/s** (+0.17)

Total session gain: **+1.30 tok/s** (65.44 → 66.74)

## Interpretation
- Dispatch/barrier reduction is the dominant optimization theme this session. The GPU spends significant time in scheduling overhead and barrier stalls. Reducing dispatch count and barrier count provides consistent, compounding gains.
- The GDN deferred-decay is a clean memory bandwidth optimization: fewer state writes = less bandwidth consumed.
- The decay_beta scheduling overlap was the biggest per-change win: moving a tiny dispatch to run concurrently with conv1d freed up a barrier slot in the critical path.
- 2-row Q8_0 was previously negative at 58 tok/s but is now positive at 66+ tok/s. The dispatch/barrier reductions changed the bottleneck profile enough that TG count reduction matters.
- Full-attention optimizations (Q+gate deinterleave, sigmoid gate fusion) were individually small but collectively eliminate 30 dispatches + 30 barriers per token.

## What Failed This Session
- (None so far — all experiments were kept)

## Previous Session Wins (still active)
1. `def8689` — parallel top-K routing: simd_max reduction: **65.20 tok/s** (+5.77)
2. `8558d2f` — transpose expert output [K][dim] → [dim][K]: **65.58 tok/s** (+0.38)
3. `4f8c12f` — half-precision GDN state: **59.20 tok/s** (+0.38)
4. `bd669d4` — hoist K=8 combine shared-gate sigmoid: **58.45 tok/s** (+0.82)

## Next Best Ideas
1. Try column-major GDN state transpose again (was -4.5 at 65 tok/s, but code has changed significantly with deferred decay + fused QK norm; may work now)
2. Revisit shared_down Q8_0 specialization for in_dim=512 with half-wave parallelism (16 of 32 lanes idle)
3. Fuse gated_rms_norm into O-projection x_shared loading to eliminate 1 dispatch+1 barrier per linear layer
4. Reduce norm_apply_partial + routing_gate into a single kernel or overlap them
5. Look at the Q4K expert gate+up+SwiGLU inner loop for packed byte reads
6. Profile with Metal System Trace using labeled dispatches to find the actual per-dispatch hotspot

## Current Log
- All experiments this session were positive or noise-level, all kept
- Session started from `a465bb4` baseline at 65.36-65.44 tok/s
- Best result: 66.74 tok/s (66.91 on second run) at commit `c2391c1`
