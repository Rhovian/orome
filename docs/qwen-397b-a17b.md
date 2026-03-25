# Qwen3.5-397B-A17B: I/O Streaming Campaign

**4.08 → 7.87 tok/s (+93%) on Mac Studio M2 Max**

## The Problem

Qwen3.5-397B-A17B is a 397B-parameter MoE model with 17B active parameters per token. Its expert weights are 217 GB at 4-bit — more than double the machine's 96 GB of RAM. The experts *cannot* be GPU-resident. Every token requires streaming expert data from SSD through the page cache.

This is a fundamentally different optimization problem from the 35B model. The 35B campaign was about GPU dispatch efficiency. The 397B campaign is about I/O.

| Spec | Value |
|------|-------|
| Layers | 60 (15 full attention + 45 linear/GatedDeltaNet) |
| Hidden dim | 4096 |
| Experts | 512 total, 10 routed + 1 shared per token |
| Expert size (4-bit) | 6.75 MB |
| Expert size (2-bit) | 3.75 MB |
| Expert weights (4-bit) | 217.4 GB |
| Expert weights (2-bit) | 120.8 GB |
| Non-expert weights | 5.52 GB (mmap'd, mlock'd) |

Per token at 4-bit: 10 experts x 60 layers x 6.75 MB = **4,050 MB of I/O**. At 2-bit with caching: ~1,300 MB. The model reads over a gigabyte from disk every 127 milliseconds.

## The Machine

Same Mac Studio: M2 Max, 96 GB unified memory, NVMe SSD. The key constraint: after loading the non-expert weights (5.52 GB) and allocating the expert cache (2.7 GB), about **83 GB** remains for the OS page cache. The 2-bit expert data is 120.8 GB. The 38 GB gap is the source of all SSD reads.

## The Serial Dependency Chain

Every layer executes this sequence, and each step blocks the next:

```
GPU attention (~0.58ms) → routing readback → pread I/O (~0.93ms) → GPU expert forward (~0.53ms) → CPU combine
```

You can't prefetch experts during attention because you don't know which experts to load until attention + routing completes. You can't start the next layer until expert combine finishes. This chain is locked by the model's architecture.

## Starting Point: 4.08 tok/s

Baseline: pure pread, K=10, 4-bit experts. Each layer reads 10 experts x 6.75 MB = 67.5 MB from the expert files on SSD, processes them on GPU, combines results on CPU. No caching — every expert is re-read every token.

## The Campaign: 49 Experiments

### What Worked

**2-bit Expert Quantization (4.08 → 6.63 tok/s, +62%)**

The biggest single win. Repacking expert weights from 4-bit to 2-bit halved the I/O per expert from 6.75 MB to 3.75 MB. Total per-token I/O dropped from 4,050 MB to 2,250 MB. The quality impact was accepted — this is a throughput-first campaign.

**Concurrent GPU Dispatch (6.63 → 6.81 tok/s, +7%)**

Same optimization from the 35B campaign. `MTLDispatchTypeConcurrent` lets independent expert matvecs overlap on the GPU. Small win because GPU compute is only 25% of total time, but free.

**Per-Layer Expert Cache (6.81 → 7.66 tok/s, +12.5%)**

Consecutive tokens often route to the same experts. We allocate 12 Metal shared buffers per layer (K+2 = 10 active + 2 extra) and track which expert is in each slot. Cache hit → skip the pread entirely. Hit rate: ~40%. Per-token I/O dropped from 2,250 MB to ~1,300 MB.

Getting the cache right took several attempts. The first implementation used 30 shared GPU buffer slots across all layers — but layer N's data was overwritten by layers N+1 through 59. The fix: per-layer buffers so each layer's cache persists across tokens.

**K+2 Cache Sweet Spot (7.66 → 7.87 tok/s, +2.7%)**

We tested K+1 through K+4 extra cache slots. K+2 (12 slots, 2.7 GB) was optimal. K+3 was within noise. K+4 (3.15 GB) *regressed* because the extra 450 MB of pinned Metal buffers starved the OS page cache. This is the central tradeoff: every byte of application cache is a byte less of page cache.

### What Didn't Work (39 Discarded Experiments)

**Memory approaches — all regressed:**

- **Partial mlock (19 layers, 69 GB)**: 3.61 tok/s. Pinning 69 GB for 19 layers starved the page cache for the other 41 layers. The cure was worse than the disease.
- **mmap (with or without MADV_RANDOM)**: 0.95 tok/s. Page fault handling on macOS is catastrophically slow compared to explicit pread. The worst result in the entire campaign.
- **CPU-side expert cache (malloc)**: 5.49 tok/s. 2.36 GB of malloc'd buffers competed with the page cache differently than Metal shared buffers.
- **K+4 cache slots (3.15 GB)**: 6.89 tok/s. Even 450 MB extra pinned memory measurably increased SSD miss rate.

**I/O scheduling — all regressed or within noise:**

- **F_RDADVISE prefetch**: 1.81 tok/s. Read-ahead hints caused I/O contention.
- **Speculative expert prefetch (previous token's experts)**: 2.91 tok/s. Wrong predictions wasted SSD bandwidth — prediction accuracy was too low.
- **Pipelined pread+GPU (split K=10 into two groups)**: 3.99 tok/s. The overhead of two command buffers per layer exceeded any overlap benefit.
- **MTLSharedEvent pipelining**: 6.98 tok/s. Event signaling overhead (>0.5ms/layer) negated overlap savings.
- **Sorted expert indices, static GCD queues, fcntl(F_RDAHEAD)**: All within noise.

**GPU optimizations — all within noise:**

- **Batched 2-bit expert kernels (4 dispatches vs 31)**: 7.96 tok/s. Within noise. GPU dispatch overhead is not the bottleneck when I/O dominates.
- **GPU combine for pread path**: 7.64 tok/s. Extra barrier + kernel cost equaled the readback savings.
- **GPU norm + LM head + argmax**: 7.94 tok/s. Eliminated 993 KB readback, saved 0.5ms out of 127ms.
- **-O3 compiler optimization**: 7.89 tok/s. Within noise of -O2.

**Quantization:**

- **Tiered quantization (hot=4-bit, cold=2-bit)**: 6.83 tok/s. Cache slots had to be sized for 4-bit (6.75 MB each), doubling the cache from 2.7 GB to 4.86 GB. The page cache couldn't absorb the loss.
- **Fused 2-bit gate+up+SwiGLU kernel**: Crash. All-NaN output at layer 0.

## Where the Time Goes

At 7.87 tok/s (~127 ms/tok), measured with GPU profiling instrumentation:

| Component | Time | % |
|-----------|------|---|
| MoE I/O (pread) | 56 ms | 44% |
| Attention (GPU) | 35 ms | 28% |
| MoE GPU (expert forward) | 32 ms | 25% |
| LM head + other | 4 ms | 3% |

The page cache serves 90% of preads in <1ms. The remaining 10% hit SSD at 1-5ms each. Those SSD misses dominate the I/O time.

## The Memory Budget

Every byte matters. Here's how 96 GB is allocated:

| Category | Size | Notes |
|----------|------|-------|
| Non-expert weights | 5.52 GB | mmap'd + mlock'd |
| Expert cache (Metal) | 2.70 GB | K+2 per-layer buffers |
| OS + Metal + runtime | ~6 GB | Kernel, GPU pipelines, process overhead |
| **Page cache** | **~83 GB** | OS-managed, serves 69% of 120.8 GB expert data |

The 120.8 GB of 2-bit expert data minus 83 GB of page cache leaves a **38 GB gap**. This gap is why 10% of preads hit SSD. There is no free memory to close it — expanding the expert cache by even 450 MB (K+4) measurably degrades page cache performance.

## Why 7.87 tok/s is the Ceiling

Three hard walls:

1. **The serial dependency chain.** Each layer must complete attention before routing, routing before I/O, I/O before expert compute. These 127ms cannot overlap within a layer. Breaking this requires predicting expert routing before attention completes — externally demonstrated at 93-97% accuracy with a trained predictor model, but that's a research project.

2. **The page cache gap.** 83 GB of cache for 120.8 GB of data means 31% of expert pages are cold at any time. With temporal locality from the K+2 cache, this translates to a 10% pread miss rate. Closing the gap requires either more RAM or smaller expert data.

3. **GPU compute at noise floor.** Six experiments targeting GPU dispatch (batched kernels, GPU combine, fused kernels, -O3) were all within noise. GPU compute is 25% of time but already well-optimized from the 35B campaign.

## What Would Move the Needle

| Approach | Expected | Effort |
|----------|----------|--------|
| Learned expert predictor (overlap I/O with attention) | 10-14 tok/s | Research project — weeks |
| Token-level batching (amortize I/O over 2 tokens) | 10-12 tok/s | Inference rewrite — weeks |
| Better hardware (more RAM, faster NVMe) | 10-15 tok/s | Money |

These are architectural changes, not experiments. The experiment surface is flat — the last 10 experiments were all within noise or regression. 49 experiments proved that incremental optimization has reached diminishing returns.

## Key Insight

The 397B story is the inverse of the 35B story. On 35B, everything fits in RAM, so the bottleneck is GPU dispatch — reduce round-trips, batch dispatches, fuse kernels. On 397B, GPU dispatch is already optimized, so the bottleneck is I/O — reduce bytes read, cache what you can, and accept that the serial dependency chain sets a hard floor.

The most important lesson: **the OS page cache is the best I/O strategy.** Every attempt to outsmart it (mlock, mmap, prefetch, pipelining) made things worse. The page cache naturally adapts to access patterns without profiling, prediction, or manual tuning. Pread + page cache + a modest application-level cache (K+2) is the winning combination for models that exceed physical memory.
