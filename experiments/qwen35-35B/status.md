# Orome Optimization Status — GGUF Q4_K / Q8_0 Era

## Final Best
- **68.91 tok/s** (Q4_K_S GGUF, 100 tokens sustained)
- TTFT: **2184.0 ms**
- `proj_avg_ms`: **0.5887**
- Branch: `autoresearch/orome`
- Best source commit: `041860b`
- Model: `/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf`

## Final Campaign Reality
- GGUF live-format mix: Q8_0 (311), F32 (301), Q4_K (120), Q6_K (1)
- Routed expert gate/up/down: **Q4_K**, shared expert: **Q8_0**, routing: **F32**
- The campaign surpassed the old packed-format peak (`62.39-62.53 tok/s`) and finished near **69 tok/s** on the current GGUF codebase.

## Final Winning Themes
1. Dispatch and barrier reduction mattered the most. The biggest gains consistently came from removing dispatches from the critical path or overlapping work more aggressively.
2. Routing and combine work still had real headroom after the earlier 58 tok/s phase. Parallel top-K routing, shared-expert scheduling, and combine-side scalar hoists all produced meaningful wins.
3. Full-attention cleanup was the final push beyond the old historical peak. `041860b` fused QK RMS norm and RoPE for full-attention layers and became the best retained source state.

## Important Final Wins
1. `041860b` — fuse QK RMS norm + RoPE into a single kernel for full-attention layers: **68.01 tok/s**, with later re-benchmarks on the same code reaching **68.91 tok/s**
2. `c2391c1` — 2-row GGUF Q8_0 matvec: **66.74 tok/s**
3. `513fc2d` — schedule `decay_beta` alongside `conv1d` for concurrent overlap: **66.50 tok/s**
4. `def8689` — parallel top-K routing via simdgroup max reduction: **65.20 tok/s**
5. `bd669d4` — hoist K=8 combine shared-gate sigmoid per simdgroup: **58.45 tok/s**

## Late Negative Signals
- Post-`041860b` experiments around KV-cache fusion, routing-kernel fusion, 2-row expert gate/up Q4_K, 4-row Q4_K, reduced `MATVEC_X_SHARED_SIZE`, and delta-net vectorization were all negative or noise-level.
- That pattern suggests the campaign is genuinely near the wall for this architecture/model pairing without a more fundamental shift in approach.

## Campaign Closeout
- Campaign complete at `041860b`.
- Live GGUF results are in `results.tsv`.
- Older packed-format context is preserved in `results.historical.tsv`.
