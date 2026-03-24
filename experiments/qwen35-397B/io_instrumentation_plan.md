# Phase 1: Pread I/O Instrumentation Research Plan

## What Was Built

The instrumentation is already implemented and builds clean. No code changes needed — just run and collect data.

### New output (every 10 tokens on stderr alongside existing `[profile]` line):

```
[moe-io] avg/tok: io=XX.Xms compute=XX.Xms combine=X.Xms  bandwidth=X.X GB/s  (X.X MB/tok)
[moe-io] pread latency: XX% <200us (cached)  XX% 200us-1ms  XX% 1-5ms  XX% >5ms (SSD)
[moe-io] layer variance: fastest=X.Xms (LNN) slowest=X.Xms (LNN)
[moe-io] memory: free=X.XGB active=X.XGB cache=X.XGB compressed=X.XGB / X.XGB total
[moe-io] L00 top experts: eN=X% eN=X% ...
```

### What each metric means:

| Metric | What it tells us |
|--------|-----------------|
| `io` ms/tok | Wall time spent in pread dispatch (all 60 layers summed) |
| `compute` ms/tok | Wall time GPU waits (`waitUntilCompleted`) for expert matvecs |
| `combine` ms/tok | CPU time for weighted expert sum + residual |
| `bandwidth` GB/s | Effective I/O bandwidth = bytes_read / io_time |
| `pread latency %` | **Key metric**: fraction of individual expert reads that are page-cache hits (<200us) vs SSD reads (>5ms) |
| `layer variance` | Whether some layers are consistently cached vs cold |
| `memory` snapshot | Live `vm_statistics64`: how much RAM is free, active, in page cache, compressed |
| `top experts` | Routing concentration — are some experts hit much more than others? |

## Research Protocol

### Experiment 1: Baseline Instrumentation Run

**Goal**: Collect I/O profile data at the current 4.08 tok/s baseline without any code changes.

1. Build: `make clean && make`
2. Run: `./orome --model /Users/j/models/Qwen3.5-397B-A17B-4bit --prompt "Explain quantum computing in detail" --tokens 100 --k 10 2>bench_err.txt`
3. Parse the `[moe-io]` lines from stderr (`bench_err.txt`)
4. Record in results.tsv with description: `instrumentation baseline: io profile at 4.08 tok/s`
5. **Do NOT discard** — this is a data collection run, not an optimization experiment
6. Verify tok/s is still ~4.08 (instrumentation overhead should be <1%)

### What to Record in status.md

After running, extract and record these key numbers:

```
## I/O Profile (from instrumentation run)
- io_ms_per_tok: ___
- compute_ms_per_tok: ___
- combine_ms_per_tok: ___
- effective_bandwidth_gbs: ___
- pread_cache_hit_pct: ___ (<200us bucket)
- pread_ssd_pct: ___ (>5ms bucket)
- fastest_layer: L__ (___ms)
- slowest_layer: L__ (___ms)
- memory_free_gb: ___
- memory_cache_gb: ___
- memory_compressed_gb: ___
```

### Interpreting the Results

The data reveals which optimization scenario we're in:

**Scenario A: SSD-bandwidth-saturated**
- Signature: bandwidth near 7-8 GB/s, most preads in >5ms bucket
- Implication: no I/O scheduling trick helps. Must reduce bytes read.
- Next step: tiered quantization (2-bit cold experts), or reduce K

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

**Prior estimate from status.md**: effective bandwidth ~16.5 GB/s at 4.08 tok/s suggests page cache is serving ~50%+ of reads. The instrumentation will confirm this and reveal the distribution.

### Experiment 2: Sustained Generation Profile (if time permits)

**Goal**: See how the I/O profile changes over longer generation (cache warming effects).

1. Run with `--tokens 200` instead of 100
2. Compare early reports (tokens 10-20) vs late reports (tokens 90-100, 190-200)
3. If cache hit rate improves over time, the working set is compressing

### Experiment 3: Cold Start Profile (if time permits)

**Goal**: Measure I/O profile with a cold page cache.

1. Run: `sudo purge` (clears file cache)
2. Immediately benchmark
3. Compare cache hit % vs warm run
4. This tells us the floor performance and how much the page cache is helping

## What NOT To Do Yet

- Do NOT attempt any memory optimization (mlock, mmap, madvise) in this session
- Do NOT change the pread path or I/O scheduling
- Do NOT modify expert loading or residency policy
- The ONLY goal is data collection — understand the problem before changing anything

## Files Modified (already done, just documenting)

| File | Changes |
|------|---------|
| `include/orome.h` | Added `MoeLayerStats`, `MemorySnapshot` structs; `layer_stats` field in `ExpertFiles`; `moe_print_layer_stats()` and `moe_sample_memory()` declarations |
| `src/moe.m` | Per-expert `mach_absolute_time` pread timing, GPU compute timing, routing frequency tracking, `vm_statistics64` memory sampling, periodic summary report |
| `src/engine.m` | Calls `moe_print_layer_stats()` every 10 tokens |

## Hardware Context (from program.md)

- Machine: M2 Max Mac Studio, 96 GB unified memory (~103.1 GB reported)
- SSD: ~7-8 GB/s sequential read
- GPU: 38 cores, ~400 GB/s memory bandwidth
- Expert data: 60 layers x 512 experts x 6.75 MB = ~207 GB (4-bit)
- Per-token I/O: 10 experts x 6.75 MB x 60 layers = ~4.05 GB
- Non-expert weights: 5.52 GB mmap'd
