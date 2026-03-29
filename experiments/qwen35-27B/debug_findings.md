# 27B NaN Overflow — Open Investigation

## Problem
Orome intermittently produces NaN in the hidden state during inference on
Qwen3.5-27B-Q4_K_M.gguf. This causes repeated `!` character output (token 0).
llama.cpp does NOT overflow on the same model and weights.

## What we know
- NaN originates in **linear attention (delta-net) layers**, in the attention
  output, not FFN
- It's a **single-step catastrophic failure** — not gradual accumulation. One
  layer's output jumps from clean (max ~64) to all-NaN instantly
- The failing layer varies per run (observed: 0, 17, 20, 26, 33, 42, 48, 50,
  52, 54, 58). Always a linear attention layer
- Occurs between positions 30-544, typically within one conversation
- In-kernel detection shows the delta-net output and state go bad at the same
  head simultaneously (usually head 32, mapping to k-head 0)
- The bug is intermittent and input-dependent

## What we ruled out
GPU concurrency, `-ffast-math`, think prefill mode, llama vs standard Q4K
kernels, delta-net state clamping, SwiGLU/gated-RMS-norm output clamping,
softplus decay floor, bad GGUF weights, cross-conversation state leakage,
buffer sizes, tensor layouts, conv1d weight order, tensor types, Q scaling
placement, decay computation math.

## Current mitigation
Clamp at ±65504 in `residual_add_sum_sq` (the hidden state chokepoint after
every layer). Verified: 0 collapses across 500+ stress test conversations,
no throughput or quality impact.

## Next steps to find root cause
- Numerical dump comparison: feed identical tokens to orome and llama.cpp, dump
  per-layer hidden states, find first point of divergence
- The math appears identical on paper but values must diverge somewhere since
  llama.cpp handles the same model without overflow

## Key debugging notes
- `-ffast-math` breaks `isnan()` in Metal shaders. Use bit-level check:
  `(bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) != 0`
- `--dump-stats` mode serializes GPU execution per-layer which can mask the bug
- In-kernel NaN checks (bit-level, atomic flags) work without altering timing
