# Qwen3.5-27B Quality Bug Investigation — 2026-03-29

## Bug 1: NaN Collapse (repeated character spam)

### Symptom
Model intermittently produces repeated `!` characters (token ID 0) during chat.
The entire logit buffer becomes NaN, causing `cpu_sample_topk` to return token 0
by default (first index wins when all values are below the -1e30 sentinel).

### Root Cause (narrowed)
Numerical overflow in **linear attention (delta-net) layer intermediates**
propagates through the residual stream. The NaN originates in the
ATTN-RESIDUAL-ADD phase of a linear attention layer (observed across many
different layers: 0, 17, 20, 26, 33, 42, 48, 50, 52, 54, 58 — varies per run).
The overflow is input-dependent and occurs within a single conversation,
typically between positions 30-420.

**Critical finding: llama.cpp does NOT overflow on the same model/weights.**
265+ requests with zero NaN on the same Qwen3.5-27B-Q4_K_M.gguf. This confirms
the bug is in orome's computation, not the model or quantization. The exact
numerical divergence between the two engines has not yet been identified — the
math appears identical on paper.

### Mitigation (verified)
Clamp `buf_moe_hidden` at ±65504 in `residual_add_sum_sq` (shaders.metal):
```metal
val = clamp(val, -65504.0f, 65504.0f);
```
This is the single chokepoint where the hidden state is written after every
layer. Verified: 0 NaN across 500+ conversations. This is a band-aid — it
silently truncates values that overflow, which may subtly affect output quality.

### What's needed to find the true root cause
Numerical comparison: feed identical token sequences to both orome and llama.cpp,
dump intermediate values at each stage of a linear attention layer, and find the
first point where values diverge significantly. The divergence must exist since
llama.cpp handles it fine while orome overflows.

### Exhaustively ruled out as root cause
- GPU concurrency / memory barriers (MTLDispatchTypeSerial still crashes)
- `-ffast-math` shader flag (removed, still crashes)
- Think prefill mode (enabled open `<think>`, still crashes)
- llama-style Q4K kernel (OROME_HYBRID_Q4=none still crashes)
- Standard orome Q4K kernel (same result)
- Delta-net state accumulation / clamping (state clamp at 1e6 didn't fix)
- SwiGLU output overflow (clamped all SwiGLU, still crashes)
- Gated RMS norm output overflow (clamped, still crashes)
- softplus decay floor (ensured g < 1, still crashes)
- Bad GGUF weights (scanned layer 63 gate/up/down blocks, zero bad)
- Cross-conversation state leakage (fixed state reset, still crashes within one conversation)
- Buffer sizes (all verified against llama.cpp dimensions)
- Tensor layouts / conv1d weight order (verified matches llama.cpp)
- Tensor types (F32 vs BF16 verified correct for each tensor)
- Q scaling placement (verified: both L2-norm then post-scale output)
- Decay computation (exp(ssm_a * softplus(alpha + dt_bias)) — identical)

### Key debugging insights

1. **`-ffast-math` breaks `isnan()` in Metal shaders.** Our GPU NaN check kernel
   used `isnan()` which was silently optimized away. This led us to incorrectly
   blame layer 63 (full attention) for hours when the real source was earlier
   linear attention layers. Fix: bit-level NaN check:
   `(bits & 0x7F800000) == 0x7F800000 && (bits & 0x007FFFFF) != 0`

2. **`engine_reset` was not clearing linear attention state.** The delta-net
   recurrence state and conv1d history persisted across requests. Fixed by adding
   memset of `buf_linear_state` and `buf_conv_state` in `engine_reset`. This is a
   correctness fix independent of the overflow bug.

3. **The residual add is the single chokepoint.** Every layer writes its output
   through `residual_add_sum_sq` into `buf_moe_hidden`. Clamping here catches
   overflow from any source — delta-net state, gated RMS norm, SwiGLU, O-proj —
   without needing to identify which specific kernel overflows.

---

## Bug 2: Think-Token Loop (empty response)

### Symptom
Model produces empty visible response. Server generates `</think>\n\n` in an
infinite loop, never producing actual content. The SSE think-token filter strips
all of this, resulting in an empty stream to the client.

### Status
Not yet investigated. Observed during stress testing. Low frequency (~1 in 200).
Separate from the NaN bug — no NaN present in these occurrences.

---

## Bug 3: Quality Gate Blind Spots (original issue)

### Fix Applied
`benchmark.py` — `strip_think_blocks()` function removes `<think>...</think>`
and truncated `<think>...` blocks from raw completion output before quality
evaluation. The "contains raw think markers" penalty was removed. This fixes
both the benchmark quality gate and the comparison tool (which imports
`evaluate_quality_reply`).

---

## Changes Made

### Production fixes:
- `src/shaders.metal` — residual_add_sum_sq clamp at ±65504
- `src/shaders.metal` — nan_check_kernel with bit-level NaN detection
- `src/engine.m` — engine_reset clears linear attention state and conv1d state
- `src/server.m` — generation prefix opens `<think>` tag instead of closed block
- `include/orome.h` — buf_nan_flag buffer and nan_check pipeline on MetalCtx
- `src/metal.m` — buf_nan_flag allocation and nan_check pipeline creation
- `tools/benchmark.py` — strip_think_blocks, fixed evaluate_quality_reply
- `tools/stress_chat.py` — new stress test tool for chat quality

### Debug instrumentation (to clean up before commit):
- `src/engine.m` — DISPATCH_NAN_CHECK macro, per-phase NaN checks (markers
  1000+ O-proj, 2000+ SwiGLU, 3000+ down-proj, 5000+ FFN-input, 7000+
  attn-residual), weight scan on first step, nan-probe/emb-nan logging
- `src/server.m` — sample_next_debug, debug_log_top5, debug_sampling flag,
  per-token debug logging, NaN-after-model_step check
