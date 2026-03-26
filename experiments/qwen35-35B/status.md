# Orome Optimization Status — GGUF Q4_K Era

## Current Best
- **48.22 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: 2099 ms
- Branch: `autoresearch/orome`
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf` (19.3 GB)

## Historical Context
- `results.tsv` spans multiple 35B eras. The old `62.53 tok/s` peak came from the previous packed-format path, not the current GGUF-only codebase.
- Old wins are still useful as hypotheses, especially around 2-row kernels, expert fusion, GPU routing, concurrent dispatch, and barrier discipline.
- Old residency / `mlock` experiments are much less actionable for the current source tree.

## Architecture (post-cleanup)
- Legacy format removed entirely — GGUF-only, format-agnostic via TensorRef/LayerTensorCache
- No fallback paths — all expert projections use batched Q4K/Q5K dynamic kernels
- Single command buffer with concurrent dispatch for all 40 layers
- GPU-resident hidden state from embedding to argmax (only 4 bytes cross GPU→CPU)
- `moe_combine_copy_sq` fuses expert combine + residual copy + partial sum_sq

## What's Been Done (this session)
1. Fixed Q4_K nibble ordering in all 7 kernels (GGML uses 4 groups of 32 bytes, NOT per-sub-block or global split)
2. Fixed Q5_K qh bit extraction (qh[l] bit g*2 for low nibble, g*2+1 for high nibble)
3. Intra-superblock SIMD distribution: 128 bytes/superblock → 32 lanes × 4 bytes = 100% utilization (was 6-25%)
4. Shared memory input reads (x_shared half precision) in all dequant kernels
5. Q6K LM head optimization (same intra-superblock pattern) — biggest single win
6. Single CB with proper inter-layer barriers
7. Removed all fallback paths (no per-expert dispatch, no cpu_argmax)
8. Fixed MoE profiling instrumentation
9. 2-row-per-simdgroup GGUF matvec for Q4_K + Q6_K projection path
10. 2-row-per-simdgroup expert down kernels for GGUF Q4_K + Q5_K
11. Fused routed expert gate+up+SwiGLU for GGUF Q4_K experts
12. Fused shared expert gate+up+SwiGLU for GGUF Q4_K weights
13. Specialized Q4_K packed scale/min unpack to decode only the two pairs each SIMD group actually consumes

## Performance Breakdown
- Per additional layer: ~0.47ms (0.085ms theoretical at 400 GB/s → 5.5x overhead)
- LM head (Q6K, 398MB): ~2.5ms
- Benchmark `proj_avg_ms` remains ~1.101, so the latest gain is coming from non-projection GGUF Q4_K work rather than the main projection micro-metric

## Latest Experiment
- Current kept baseline at `a386dbf`: **48.22 tok/s**, TTFT 2099 ms, proj avg 1.101 ms
- Discarded: threadgroup-cache `x` in `matvec_f32` for `in_dim <= 2048`
- Result: **47.81 tok/s**, TTFT 2101 ms, proj avg 1.101 ms
- Interpretation: the extra threadgroup load + barrier cost is not amortized by the small F32 projection workload; routing/shared-gate F32 matvecs are not the next bottleneck

## Key Bottleneck Analysis
The Q4_K format has inherent overhead vs the old legacy 4-bit format:
- 12 bytes of packed scales/mins per 256 weights → scale/min unpack remains a real cost, even after trimming it to the per-group pair actually consumed
- 144 bytes per 256 weights (0.5625 B/w) vs ~128 bytes (0.5 B/w) in legacy
- Q6K LM head is 398MB vs ~254MB for simple 4-bit

The legacy system reached 62 tok/s with simpler dequant. The remaining 48→62 gap is still largely format overhead.

## Next Experiments (priority order)
1. Profile with Metal System Trace to confirm whether MoE-side Q4_K kernels now dominate after the scale-unpack trim
2. Fuse `routing_gate` + `shared_expert_gate` if their tensor formats line up, since they are tiny same-input projections that still cost separate dispatches
3. Specialize the Q5_K expert-down high-bit path next, because the Q4_K-scale trim likely increases the relative weight of Q5_K overhead
4. Reduce barrier count only where producer/consumer relationships are proven safe under concurrent dispatch
