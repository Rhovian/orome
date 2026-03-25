# Qwen3.5-35B-A3B: GPU Dispatch Campaign

**1.04 → 62.53 tok/s (60x) on Mac Studio M2 Max**

## The Model

Qwen3.5-35B-A3B is a 35B-parameter Mixture-of-Experts model with 3B active parameters per token. It's small enough to fit entirely in 96 GB of unified memory — the optimization story here is pure GPU compute, not I/O.

| Spec | Value |
|------|-------|
| Layers | 40 (15 full attention + 25 linear/GatedDeltaNet) |
| Hidden dim | 2048 |
| Experts | 256 total, 8 routed + 1 shared per token |
| Expert FFN | 512 intermediate |
| Expert weights (4-bit) | ~18 GB |
| Non-expert weights | 1.4 GB |
| Total footprint | ~19.4 GB — fits in RAM |

Because everything fits, every expert layer is mlock'd into physical memory and wrapped as a Metal shared buffer. The GPU reads expert weights directly from unified memory with zero I/O overhead. The entire optimization surface is GPU dispatch efficiency.

## The Machine

Mac Studio (2023): M2 Max, 12 CPU cores, 38 GPU cores, 96 GB unified memory at ~400 GB/s bandwidth. Actively cooled — no thermal throttling.

## Starting Point: 1.04 tok/s

The initial implementation was naive:
- Single GPU command buffer per compute kernel (~600 dispatches per token)
- CPU-side expert forward pass using `cpu_dequant_matvec()`
- Calloc/free in the attention hot path
- Every matvec was its own GPU round-trip

The model produced correct output. It was just slow — death by a thousand command buffer commits.

## The Campaign

### Phase 1: Get Expert Compute on GPU (1.04 → 13.12 tok/s)

The single biggest win. Moving expert gate/up/down projections from CPU `dequant_matvec()` to GPU Metal kernels gave a **12.6x** speedup. The CPU was doing serial dequantization and multiplication; the GPU parallelizes across 38 cores.

### Phase 2: Fuse the GPU Pipeline (13 → 37 tok/s)

With expert compute on GPU, the bottleneck shifted to CPU-GPU synchronization. Each layer did ~34 individual GPU dispatches with command buffer commits between them. The fix: fuse related operations into single command buffers.

- Q/K/V projections + RoPE + attention + O-proj + routing → one command buffer per layer
- Eliminated 40 CPU-GPU round-trips per token
- GPU-resident hidden state — no readback between layers

### Phase 3: Batch and Overlap (37 → 52 tok/s)

With fused pipelines, the remaining overhead was dispatch count within each layer. Expert forward still did one dispatch per expert (8 dispatches for gate, 8 for up, 8 for down = 24 dispatches per layer).

- Batched expert kernels: gate+up+SwiGLU fused into one kernel, dispatches dropped from 34/layer to 6/layer
- Concurrent GPU dispatch (`MTLDispatchTypeConcurrent`): independent matvecs overlap on the GPU
- Per-layer command buffers: GPU pipelines layer N while CPU prepares layer N+1

### Phase 4: Squeeze the Kernels (52 → 62.53 tok/s)

The final 20% came from kernel-level optimization:

- **2-row matvec layout**: Each SIMD group processes 2 output rows instead of 1. Halved threadgroup count while keeping 512 threads per group. This was the single best kernel change (~17% improvement).
- **GPU argmax**: Moved softmax+topk routing from CPU to GPU, eliminating 993 KB logits readback per token.
- **Half-precision shared memory**: 8 KB instead of 16 KB for intermediate buffers.
- **Precompiled metallib**: Eliminated runtime shader compilation.

### What Didn't Work

- **Moving O-proj to CPU**: 82% regression (2.72 tok/s). CPU dequant_matvec is 5.7x slower than GPU.
- **Register-cached DeltaNet state**: Float[128] arrays caused register spilling. Worse than shared memory.
- **Fused residual+norm kernels**: Single threadgroup reduced parallelism. Within noise.
- **Per-head attention scores kernel**: Only 16 threadgroups underutilized 38 GPU cores.
- **ROWS_PER_TG tuning**: Tested 4, 8, 16, 32, 64. 16 was consistently optimal for 38 GPU cores.

## Final Architecture

At 62.53 tok/s, the forward pass is a tight GPU pipeline:

```
For each layer:
  [GPU] Norm → Q/K/V projections → RoPE → Attention → O-proj → Residual add
  [GPU] Norm → Routing gate → Softmax+TopK
  [GPU] Expert gate+up+SwiGLU (batched, concurrent) → Expert down (batched)
  [GPU] Expert combine + shared expert + residual add
Final:
  [GPU] Norm → LM head → Argmax
  [CPU] Read 1 token ID
```

Total GPU round-trips per token: ~40 (one per layer) + 1 (final readback). Down from ~600 at baseline.

## Key Insight

The 35B story is simple: **move everything to GPU, then reduce the number of times you talk to it.** Every win came from either (a) moving compute from CPU to GPU, or (b) batching GPU dispatches to reduce synchronization. The model fits in memory, so there's no I/O story. It's pure dispatch engineering.

This is the foundation that the 397B campaign builds on — same GPU kernels, same dispatch patterns, but with a fundamentally different bottleneck: the experts don't fit in RAM.
