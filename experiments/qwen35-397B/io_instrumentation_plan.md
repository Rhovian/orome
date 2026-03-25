# I/O Instrumentation & Memory Research Plan (Qwen3.5-397B)

## Problem Statement

With 96 GB unified memory and ~120.8 GB of 2-bit expert data (or ~217 GB at 4-bit), expert weights must stream from SSD. The OS page cache is the primary lever — ~83 GB available after model weights and expert cache buffers. But every memory optimization attempted blindly has regressed performance (partial mlock, speculative prefetch, mmap, F_RDADVISE). We need data before making decisions.

Current best: **7.87 tok/s** (2-bit, K+2 per-layer cache, commit 1942f27).

## Phase 1: Instrumentation (implemented)

Per-layer I/O stats are collected in `MoeLayerStats` structs attached to `ExpertFiles`. Each pread is timed with `mach_absolute_time` and bucketed by latency:

| Bucket | Range | Meaning |
|--------|-------|---------|
| cache_hits | < 200 us | Hot page cache |
| fast_ssd | 200 us - 1 ms | Warm cache or fast SSD |
| slow_ssd | 1 - 5 ms | Cold SSD read |
| very_slow | > 5 ms | Contention or thrashing |

Additional tracked metrics:
- `total_bytes_read` per layer — actual I/O volume
- Expert routing frequency per layer — which experts get called and how often
- `vm_statistics64` memory pressure snapshot every 10 tokens
- Per-token aggregate: total I/O time, effective bandwidth, cache hit rate

Report format (stderr, every 10 tokens):
```
[moe-io] avg/tok: io=XXms compute=XXms bw=XX.XGB/s cache=XX.XGB
[moe-io] pread latency: <200us=XX% 200us-1ms=XX% 1-5ms=XX% >5ms=XX%
[moe-io] mem: free=X.XGB inactive=X.XGB purgeable=X.XGB speculative=X.XGB
```

### Files Modified (already done)

| File | Changes |
|------|---------|
| `include/orome.h` | Added `MoeLayerStats`, `MemorySnapshot` structs; `layer_stats` field in `ExpertFiles`; `moe_print_layer_stats()` and `moe_sample_memory()` declarations |
| `src/moe.m` | Per-expert `mach_absolute_time` pread timing, GPU compute timing, routing frequency tracking, `vm_statistics64` memory sampling, periodic summary report |
| `src/engine.m` | Calls `moe_print_layer_stats()` every 10 tokens |

## Phase 2: Data Collection & Analysis

Run 3 benchmark sessions with instrumentation enabled. Key questions:

1. **Are we SSD-bandwidth-bound?** If effective bandwidth is near ~7 GB/s (M2 Max SSD ceiling), reducing bytes is the only path. If well below, we have scheduling headroom.

2. **What's the page cache hit rate?** Current data shows 74-78% of preads at 200us-1ms (warm cache), 21-25% at 1-5ms (SSD). How stable is this across tokens?

3. **Is there layer variance?** Do early/late layers have different I/O profiles? If so, per-layer strategy is viable.

4. **Expert concentration?** If a small set of experts handles most routing, selective caching (madvise WILLNEED for hot experts only) could help without the memory pressure of mlock.

5. **Memory pressure trajectory?** Does free+inactive RAM shrink during generation? If so, our Metal cache buffers may be crowding out page cache.

### Interpreting Results

**Scenario A: SSD-bandwidth-saturated**
- Signature: bandwidth near 7-8 GB/s, most preads in >5ms bucket
- Implication: no I/O scheduling trick helps. Must reduce bytes read.
- Next step: tiered quantization (2-bit cold experts, 4-bit hot), or reduce K

**Scenario B: Page cache underperforming**
- Signature: bandwidth well below SSD max, mixed latency distribution, cache_bytes low
- Implication: RAM is available but the OS isn't caching the right experts
- Next step: madvise(MADV_WILLNEED) for high-frequency experts, informed by routing data

**Scenario C: Compute-bound (unlikely)**
- Signature: compute_ms >> io_ms, most preads fast
- Implication: GPU is the bottleneck, not I/O
- Next step: GPU kernel optimization (35B playbook)

**Scenario D: High layer variance**
- Signature: fastest_layer << slowest_layer (>3x difference)
- Implication: some layers are well-cached, others aren't
- Next step: per-layer I/O scheduling, selective prefetch for cold layers only

## Phase 3: Strategy Selection (data-dependent)

Based on Phase 2 results, pick ONE of:

### A. Work with the page cache (if cache hit rate < 80% but memory is available)
- `madvise(MADV_WILLNEED)` for frequently-routed experts only (not whole layers)
- Hint next-layer experts during GPU compute
- No mlock, no mmap — let the OS manage pages but give it better hints

### B. Reduce I/O volume (if SSD-bandwidth-bound)
- Tiered quantization: 4-bit for hot experts, 2-bit for cold (infrastructure exists)
- Expert pruning: skip experts with very low routing weight
- Larger expert cache: only if memory pressure data shows headroom

### C. Overlap I/O with compute (if GPU is idle during pread)
- Async pread for next layer's experts during current layer's GPU dispatch
- Requires careful double-buffering to avoid GPU stalls
- Previous attempts failed from contention — retry only with instrumentation proving idle GPU time

### D. Reduce expert count (if routing is concentrated)
- Dynamic K reduction when most weight is in top-K/2 experts
- Thermal-aware K already exists but isn't triggered on actively-cooled machine

## Guard Rails

- **No partial mlock.** Proven harmful (3.61 tok/s). The all-or-nothing guard in `expert_files_open` must stay.
- **No mmap for expert I/O.** Proven catastrophic (0.95 tok/s). pread is the only safe path.
- **Expert cache stays at K+2 slots.** K+4 regressed from page cache pressure.
- **Total Metal buffer allocation must leave >80 GB for page cache.** Current: ~2.7 GB cache + ~5.5 GB weights = ~8.2 GB pinned. Page cache gets ~83 GB.
- **35B must not regress.** All changes must be backward-compatible with the mlock fast-path.

## Hardware Context

- Machine: M2 Max Mac Studio, 96 GB unified memory (~103.1 GB reported)
- SSD: ~7-8 GB/s sequential read
- GPU: 38 cores, ~400 GB/s memory bandwidth
- Expert data (2-bit): 60 layers x 512 experts x 3.75 MB = ~120.8 GB
- Expert data (4-bit): 60 layers x 512 experts x 6.75 MB = ~217 GB
- Per-token I/O (2-bit, no cache): 10 experts x 3.75 MB x 60 layers = ~2,250 MB
- Per-token I/O (2-bit, ~42% cache hit): ~1,300 MB
- Non-expert weights: 5.52 GB mmap'd
- Expert cache buffers: ~2.7 GB Metal shared memory (K+2 per layer)
