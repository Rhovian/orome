/*
 * moe.m — Mixture of Experts: routing, expert I/O, expert forward pass.
 *
 * Expert loading is hardware-aware. Layers are only treated as GPU-resident
 * when they fit inside a conservative resident-memory budget on the current
 * machine; the remainder stay on the streaming pread path.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>

#include "orome.h"

#define ROWS_PER_TG 16  // must match metal.m and shaders.metal

// --- Timing helpers ---
static double ns_per_tick = 0;
static void ensure_timebase(void) {
    if (ns_per_tick > 0) return;
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ns_per_tick = (double)info.numer / (double)info.denom;
}

static bool g_profile_experts = false;

void moe_set_profile_experts(bool enabled) {
    g_profile_experts = enabled;
}

bool moe_get_profile_experts(void) {
    return g_profile_experts;
}

typedef struct {
    int fd;
    off_t offset;
    size_t size;
    void *dst;
    ssize_t result;
    uint64_t elapsed_ns;   // mach_absolute_time delta, converted to ns
} ExpertReadTask;

static void expert_pread_task(void *ctx) {
    ExpertReadTask *task = (ExpertReadTask *)ctx;
    if (g_profile_experts) {
        uint64_t t0 = mach_absolute_time();
        task->result = pread(task->fd, task->dst, task->size, task->offset);
        uint64_t t1 = mach_absolute_time();
        task->elapsed_ns = (uint64_t)((t1 - t0) * ns_per_tick);
    } else {
        task->result = pread(task->fd, task->dst, task->size, task->offset);
    }
}

// ============================================================================
// Expert file management — resident layers mmap + streaming fallback
// ============================================================================

static int load_hot_mask_json(ExpertFiles *ef, const char *path,
                              int num_layers, int num_experts) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
        if (!data) return -1;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (!root) return -1;

        int words_per_layer = (num_experts + 31) / 32;
        ef->hot_mask = calloc(num_layers * words_per_layer, sizeof(uint32_t));
        int total_hot = 0;
        for (int layer = 0; layer < num_layers; layer++) {
            NSString *key = [NSString stringWithFormat:@"%d", layer];
            NSArray *experts = root[key];
            if (!experts) continue;
            for (NSNumber *eid in experts) {
                int e = [eid intValue];
                if (e >= 0 && e < num_experts) {
                    ef->hot_mask[layer * words_per_layer + e / 32] |= (1U << (e % 32));
                    total_hot++;
                }
            }
        }
        double avg_pct = (num_layers > 0 && num_experts > 0)
            ? 100.0 * total_hot / (double)(num_layers * num_experts) : 0.0;
        printf("[moe] Loaded %d hot experts from %s (%.0f%% per layer avg)\n",
               total_hot, path, avg_pct);
        return 0;
    }
}

static size_t bytes_gib(size_t gib) {
    return gib * (size_t)1024 * 1024 * 1024;
}

static size_t clamp_subtract(size_t total, size_t used) {
    return total > used ? total - used : 0;
}

static size_t model_shared_weight_bytes(const char *model_dir) {
    if (!model_dir) return 0;

    char path[512];
    snprintf(path, sizeof(path), "%s/model_weights.bin", model_dir);

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) return 0;
    return (size_t)st.st_size;
}

static bool expert_layer_is_resident(const ExpertFiles *ef, int layer_idx) {
    return ef && ef->layer_resident && ef->layer_data
        && layer_idx >= 0 && layer_idx < ef->num_layers
        && ef->layer_resident[layer_idx]
        && ef->layer_data[layer_idx] != NULL;
}

static bool expert_layer_uses_pread(const ExpertFiles *ef, int layer_idx) {
    return ef && ef->layer_data && ef->layer_fds
        && layer_idx >= 0 && layer_idx < ef->num_layers
        && !expert_layer_is_resident(ef, layer_idx)
        && ef->layer_fds[layer_idx] >= 0;
}

static void log_expert_route(int layer_idx, const int *expert_indices, int K,
                             const ExpertFiles *ef) {
    if (!g_profile_experts) return;
    // Accumulate frequency stats only when profiling
    if (ef && ef->layer_stats && ef->layer_stats[layer_idx]) {
        MoeLayerStats *st = ef->layer_stats[layer_idx];
        for (int k = 0; k < K; k++) {
            if (expert_indices[k] < ef->num_experts)
                st->expert_freq[expert_indices[k]]++;
        }
        st->token_count++;
    }
    fprintf(stderr, "EXPERT_ROUTE layer=%d experts=", layer_idx);
    for (int k = 0; k < K; k++) fprintf(stderr, "%s%d", k ? "," : "", expert_indices[k]);
    fprintf(stderr, "\n");
}

static void cpu_dequant_matvec_quant(QuantType quant, const uint32_t *W,
                                     const uint16_t *scales, const uint16_t *biases,
                                     const float *x, float *out,
                                     int out_dim, int in_dim, int group_size) {
    if (quant == QUANT_2BIT) {
        cpu_dequant_matvec_2bit(W, scales, biases, x, out, out_dim, in_dim, group_size);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
    }
}

static void sync_hidden_buffer(MetalCtx *ctx, const float *hidden, int H) {
    if (!ctx || !ctx->buf_moe_hidden) return;
    memcpy([ctx->buf_moe_hidden contents], hidden, H * sizeof(float));
}

static void select_expert_source(const ExpertFiles *ef, const ModelConfig *cfg,
                                 int layer_idx, int expert_id, QuantType quant,
                                 int *fd_out, const ExpertLayout **layout_out,
                                 bool *is_2bit_out) {
    bool is_2bit = false;
    if (ef->tiered_quant && ef->layer_fds_2bit && ef->layer_fds_2bit[layer_idx] >= 0) {
        is_2bit = !expert_is_hot(ef, layer_idx, expert_id);
    } else if (quant == QUANT_2BIT && ef->layer_fds_2bit
               && ef->layer_fds_2bit[layer_idx] >= 0) {
        is_2bit = true;
    }

    *is_2bit_out = is_2bit;
    *layout_out = is_2bit ? &cfg->expert_2bit : &cfg->expert_4bit;
    *fd_out = is_2bit ? ef->layer_fds_2bit[layer_idx] : ef->layer_fds[layer_idx];
}

static void expert_forward_direct(const ModelConfig *cfg, const ExpertLayout *layout,
                                  const void *expert_base, const float *input,
                                  float *output, float *gate_buf, float *up_buf,
                                  float *act_buf, QuantType quant);

static int pread_experts_into_gpu_buffers(MetalCtx *ctx, const ModelConfig *cfg,
                                          const ExpertFiles *ef, int layer_idx,
                                          const int *expert_indices, int K,
                                          QuantType quant, bool *expert_is_2bit) {
    if (!ctx || !ctx->queue) return -1;

    ExpertReadTask tasks[OROME_MAX_ACTIVE];
    memset(tasks, 0, sizeof(tasks));
    ExpertReadTask *task_ptr = tasks;  // block captures pointer, not VLA

    for (int k = 0; k < K; k++) {
        const ExpertLayout *elayout = NULL;
        int fd = -1;
        select_expert_source(ef, cfg, layer_idx, expert_indices[k], quant,
                             &fd, &elayout, &expert_is_2bit[k]);
        if (fd < 0 || !elayout || !ctx->buf_multi_expert_data[k]) {
            fprintf(stderr, "ERROR: Missing expert source for layer %d expert %d\n",
                    layer_idx, expert_indices[k]);
            return -1;
        }

        tasks[k].fd = fd;
        tasks[k].offset = (off_t)expert_indices[k] * elayout->expert_size;
        tasks[k].size = elayout->expert_size;
        tasks[k].dst = [ctx->buf_multi_expert_data[k] contents];
        tasks[k].result = -1;
    }

    // dispatch_apply: batched parallel pread — less overhead than dispatch_group+async_f
    double t_io_start = g_profile_experts ? now_ms() : 0;
    dispatch_apply((size_t)K, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t k) { expert_pread_task(&task_ptr[k]); });
    double t_io_end = g_profile_experts ? now_ms() : 0;

    for (int k = 0; k < K; k++) {
        if (tasks[k].result != (ssize_t)tasks[k].size) {
            fprintf(stderr,
                    "ERROR: pread failed for layer %d expert %d (%zd/%zu bytes)\n",
                    layer_idx, expert_indices[k], tasks[k].result, tasks[k].size);
            return -1;
        }
    }

    // Accumulate per-layer I/O stats (only when profiling)
    if (g_profile_experts && ef->layer_stats && ef->layer_stats[layer_idx]) {
        MoeLayerStats *st = ef->layer_stats[layer_idx];
        st->io_ms += t_io_end - t_io_start;
        for (int k = 0; k < K; k++) {
            st->io_bytes += tasks[k].size;
            uint64_t us = tasks[k].elapsed_ns / 1000;
            if      (us < 200)  st->pread_us_buckets[0]++;
            else if (us < 1000) st->pread_us_buckets[1]++;
            else if (us < 5000) st->pread_us_buckets[2]++;
            else                st->pread_us_buckets[3]++;
        }
    }

    return 0;
}

static int pread_experts_cpu(const ModelConfig *cfg, const ExpertFiles *ef,
                             int layer_idx, const int *expert_indices, int K,
                             const float *h_post, QuantType quant,
                             float **expert_out, float **expert_gate,
                             float **expert_up, float **expert_act) {
    size_t max_expert_size = cfg->expert_4bit.expert_size;
    if (cfg->expert_2bit.expert_size > max_expert_size) {
        max_expert_size = cfg->expert_2bit.expert_size;
    }
    uint8_t *scratch = malloc(max_expert_size);
    if (!scratch) return -1;

    for (int k = 0; k < K; k++) {
        const ExpertLayout *elayout = NULL;
        int fd = -1;
        bool is_2bit = false;
        select_expert_source(ef, cfg, layer_idx, expert_indices[k], quant,
                             &fd, &elayout, &is_2bit);
        if (fd < 0 || !elayout) {
            fprintf(stderr, "ERROR: Missing expert source for layer %d expert %d\n",
                    layer_idx, expert_indices[k]);
            free(scratch);
            return -1;
        }

        ssize_t got = pread(fd, scratch, elayout->expert_size,
                            (off_t)expert_indices[k] * elayout->expert_size);
        if (got != (ssize_t)elayout->expert_size) {
            fprintf(stderr, "ERROR: pread failed for layer %d expert %d (%zd/%zu bytes)\n",
                    layer_idx, expert_indices[k], got, elayout->expert_size);
            free(scratch);
            return -1;
        }

        expert_forward_direct(cfg, elayout, scratch, h_post,
                              expert_out[k], expert_gate[k],
                              expert_up[k], expert_act[k],
                              is_2bit ? QUANT_2BIT : QUANT_4BIT);
    }

    free(scratch);
    return 0;
}

static bool gpu_forward_pread_experts(MetalCtx *ctx, const ModelConfig *cfg,
                                      uint32_t *sd_w, uint16_t *sd_s, uint16_t *sd_b,
                                      int K, const bool *expert_is_2bit,
                                      float **expert_out, float *shared_out) {
    if (!ctx || !ctx->queue || !ctx->buf_weights || !sd_w || !sd_s || !sd_b) return false;

    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int S = cfg->shared_intermediate;
    uint8_t *wbase = (uint8_t *)[ctx->buf_weights contents];

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return false;

    for (int k = 0; k < K; k++) {
        const ExpertLayout *elayout = expert_is_2bit[k] ? &cfg->expert_2bit : &cfg->expert_4bit;
        id<MTLComputePipelineState> mv_pipe = expert_is_2bit[k] ? ctx->matvec_2bit : ctx->matvec_4bit;
        if (!mv_pipe) return false;

        [enc setComputePipelineState:mv_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->gate_w_off atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->gate_s_off atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->gate_b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:4];
        {
            uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
            [enc setBytes:&od length:sizeof(uint) atIndex:5];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
            [enc setBytes:&gs length:sizeof(uint) atIndex:7];
        }
        NSUInteger tg_size = ROWS_PER_TG * 32;
        NSUInteger num_tgs = ((uint)M + ROWS_PER_TG - 1) / ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        [enc setComputePipelineState:mv_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->up_w_off atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->up_s_off atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->up_b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:4];
        {
            uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
            [enc setBytes:&od length:sizeof(uint) atIndex:5];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
            [enc setBytes:&gs length:sizeof(uint) atIndex:7];
        }
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }

    for (int k = 0; k < K; k++) {
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:2];
        uint dim_val = (uint)M;
        [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
        NSUInteger swi_tgs = ((uint)M + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
    {
        uint dim_val = (uint)S;
        [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
    }
    [enc dispatchThreadgroups:MTLSizeMake(((uint)S + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    for (int k = 0; k < K; k++) {
        const ExpertLayout *elayout = expert_is_2bit[k] ? &cfg->expert_2bit : &cfg->expert_4bit;
        id<MTLComputePipelineState> mv_pipe = expert_is_2bit[k] ? ctx->matvec_2bit : ctx->matvec_4bit;
        if (!mv_pipe) return false;

        [enc setComputePipelineState:mv_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->down_w_off atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->down_s_off atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:elayout->down_b_off atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
        {
            uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size;
            [enc setBytes:&od length:sizeof(uint) atIndex:5];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
            [enc setBytes:&gs length:sizeof(uint) atIndex:7];
        }
        NSUInteger tg_size = ROWS_PER_TG * 32;
        NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }

    [enc setComputePipelineState:ctx->matvec_4bit];
    [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_w - wbase atIndex:0];
    [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_s - wbase atIndex:1];
    [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_b - wbase atIndex:2];
    [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
    [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
    {
        uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
        [enc setBytes:&od length:sizeof(uint) atIndex:5];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
        [enc setBytes:&gs length:sizeof(uint) atIndex:7];
    }
    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(ROWS_PER_TG * 32, 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    for (int k = 0; k < K; k++) {
        memcpy(expert_out[k], [ctx->buf_multi_expert_out[k] contents], H * sizeof(float));
    }
    memcpy(shared_out, [ctx->buf_shared_out contents], H * sizeof(float));

    return true;
}

ExpertFiles *expert_files_open(const ModelConfig *cfg, const char *model_dir,
                               const char *hot_mask_path) {
    ensure_timebase();
    ExpertFiles *ef = calloc(1, sizeof(ExpertFiles));
    ef->layer_data = calloc(cfg->num_layers, sizeof(void *));
    ef->layer_size = calloc(cfg->num_layers, sizeof(size_t));
    ef->layer_fds  = calloc(cfg->num_layers, sizeof(int));
    ef->layer_resident = calloc(cfg->num_layers, sizeof(bool));
    ef->num_experts = cfg->num_experts;
    ef->num_layers = cfg->num_layers;

    // Per-layer I/O instrumentation (always allocated, ~1 KB/layer)
    ef->layer_stats = calloc(cfg->num_layers, sizeof(MoeLayerStats *));
    for (int i = 0; i < cfg->num_layers; i++) {
        size_t sz = sizeof(MoeLayerStats) + cfg->num_experts * sizeof(uint16_t);
        ef->layer_stats[i] = calloc(1, sz);
    }

    uint64_t total_ram = [NSProcessInfo processInfo].physicalMemory;
    size_t process_budget = (size_t)(total_ram * 0.8);
    size_t shared_weight_bytes = model_shared_weight_bytes(model_dir);
    // Leave extra headroom for Metal allocations, WindowServer, and swap breathing room
    // during long 397B runs on unified memory machines.
    size_t runtime_reserve = total_ram / 16;
    if (runtime_reserve < bytes_gib(6)) runtime_reserve = bytes_gib(6);

    size_t total_expert = (size_t)cfg->num_layers * cfg->num_experts
        * cfg->expert_4bit.expert_size;
    size_t resident_budget = clamp_subtract(process_budget,
                                            shared_weight_bytes + runtime_reserve);
    ef->resident_budget_bytes = resident_budget;
    ef->shared_weight_bytes = shared_weight_bytes;
    ef->runtime_reserve_bytes = runtime_reserve;

    bool fits_fully_resident = (total_expert <= resident_budget);
    printf("[moe] Hardware budget: RAM %.1f GB, process cap %.1f GB, shared %.2f GB, runtime reserve %.1f GB → resident experts %.1f GB\n",
           total_ram / 1e9, process_budget / 1e9, shared_weight_bytes / 1e9,
           runtime_reserve / 1e9, resident_budget / 1e9);
    printf("[moe] Expert footprint: %.1f GB → %s\n",
           total_expert / 1e9, fits_fully_resident ? "fully resident" : "hybrid/streaming");

    size_t resident_bytes = 0;
    int resident_layers = 0;
    int streamed_layers = 0;

    for (int i = 0; i < cfg->num_layers; i++) {
        ef->layer_fds[i] = -1;

        char path[512];
        snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin",
                 model_dir ? model_dir : ".", i);

        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;

        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            continue;
        }
        size_t size = st.st_size;
        ef->layer_size[i] = size;

        // Only mlock layers when the FULL expert set fits in the resident budget.
        // Partial mlock starves the OS page cache and is slower than pure pread.
        bool allow_resident_layer = fits_fully_resident
            && (resident_bytes + size <= resident_budget);
        if (allow_resident_layer) {
            void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                fprintf(stderr, "WARNING: mmap failed for expert layer %d, falling back to pread streaming\n",
                        i);
            } else {
                madvise(data, size, MADV_WILLNEED);
                if (mlock(data, size) == 0) {
                    ef->layer_data[i] = data;
                    ef->layer_fds[i] = fd;
                    ef->layer_resident[i] = true;
                    resident_bytes += size;
                    resident_layers++;
                    continue;
                }

                fprintf(stderr, "WARNING: mlock failed for expert layer %d, falling back to pread streaming\n",
                        i);
                munmap(data, size);
            }
        }

        ef->layer_fds[i] = fd;
        ef->layer_resident[i] = false;
        streamed_layers++;
    }

    ef->resident_bytes = resident_bytes;
    ef->all_resident = (resident_layers == cfg->num_layers);
    ef->pread_mode = !ef->all_resident;
    ef->gpu_resident_safe = ef->all_resident;

    if (ef->all_resident) {
        printf("[moe] resident policy: all %d/%d layers mlock'd (%.1f GB resident)\n",
               resident_layers, cfg->num_layers, resident_bytes / 1e9);
    } else {
        printf("[moe] resident policy: %d/%d layers mlock'd (%.1f GB), %d streamed via pread\n",
               resident_layers, cfg->num_layers, resident_bytes / 1e9, streamed_layers);
        printf("[moe] safety guard: non-resident layers stay on streaming path to avoid virtual-fit OOMs\n");
    }

    if (hot_mask_path) {
        if (load_hot_mask_json(ef, hot_mask_path, cfg->num_layers, cfg->num_experts) == 0) {
            ef->tiered_quant = true;
        } else {
            fprintf(stderr, "WARNING: Failed to load hot mask from %s\n", hot_mask_path);
        }
    }

    // Open 2-bit expert files if they exist (for --2bit or tiered quant modes)
    if (!ef->layer_fds_2bit) {
        ef->layer_fds_2bit = calloc(cfg->num_layers, sizeof(int));
        int opened_2bit = 0;
        for (int i = 0; i < cfg->num_layers; i++) {
            ef->layer_fds_2bit[i] = -1;
            char path2[512];
            snprintf(path2, sizeof(path2), "%s/packed_experts_2bit/layer_%02d.bin",
                     model_dir ? model_dir : ".", i);
            ef->layer_fds_2bit[i] = open(path2, O_RDONLY);
            if (ef->layer_fds_2bit[i] >= 0) opened_2bit++;
        }
        if (opened_2bit > 0) {
            printf("[moe] 2-bit expert files: %d/%d layers available\n",
                   opened_2bit, cfg->num_layers);
        }
        if (ef->tiered_quant && opened_2bit == 0) {
            fprintf(stderr, "WARNING: No 2-bit expert files found, disabling tiered quant\n");
            ef->tiered_quant = false;
        }
    }

    return ef;
}

void expert_files_close(ExpertFiles *ef, const ModelConfig *cfg) {
    if (!ef) return;
    int num_layers = ef->num_layers > 0 ? ef->num_layers : (cfg ? cfg->num_layers : 0);
    for (int i = 0; i < num_layers; i++) {
        if (ef->layer_data[i]) munmap(ef->layer_data[i], ef->layer_size[i]);
        if (ef->layer_fds[i] >= 0) close(ef->layer_fds[i]);
        if (ef->layer_fds_2bit && ef->layer_fds_2bit[i] >= 0) close(ef->layer_fds_2bit[i]);
    }
    free(ef->layer_data);
    free(ef->layer_size);
    free(ef->layer_fds);
    free(ef->layer_resident);
    free(ef->layer_fds_2bit);
    free(ef->hot_mask);
    if (ef->layer_stats) {
        for (int i = 0; i < num_layers; i++) free(ef->layer_stats[i]);
        free(ef->layer_stats);
    }
    free(ef);
}

// ============================================================================
// I/O instrumentation: periodic summary report
// ============================================================================

void moe_print_layer_stats(ExpertFiles *ef, bool reset) {
    if (!ef || !ef->layer_stats) return;
    int num_layers = ef->num_layers;
    if (num_layers <= 0) return;

    // Aggregate across all layers
    double total_io = 0, total_compute = 0, total_combine = 0;
    uint64_t total_bytes = 0;
    int total_tokens = 0;
    uint32_t total_buckets[4] = {0};
    double min_io = 1e30, max_io = 0;
    int min_layer = 0, max_layer = 0;

    for (int i = 0; i < num_layers; i++) {
        MoeLayerStats *st = ef->layer_stats[i];
        if (!st || st->token_count == 0) continue;
        double avg_io = st->io_ms / st->token_count;
        total_io += st->io_ms;
        total_compute += st->compute_ms;
        total_combine += st->combine_ms;
        total_bytes += st->io_bytes;
        if (i == 0 || total_tokens == 0) total_tokens = st->token_count;
        for (int b = 0; b < 4; b++) total_buckets[b] += st->pread_us_buckets[b];
        if (avg_io < min_io) { min_io = avg_io; min_layer = i; }
        if (avg_io > max_io) { max_io = avg_io; max_layer = i; }
    }

    if (total_tokens == 0) return;
    double inv = 1.0 / total_tokens;
    double bw = (total_bytes > 0 && total_io > 0)
        ? (total_bytes / 1e9) / (total_io / 1e3) : 0;
    uint32_t total_reads = total_buckets[0] + total_buckets[1]
                         + total_buckets[2] + total_buckets[3];
    double pct0 = total_reads ? 100.0 * total_buckets[0] / total_reads : 0;
    double pct1 = total_reads ? 100.0 * total_buckets[1] / total_reads : 0;
    double pct2 = total_reads ? 100.0 * total_buckets[2] / total_reads : 0;
    double pct3 = total_reads ? 100.0 * total_buckets[3] / total_reads : 0;

    fprintf(stderr,
        "[moe-io] avg/tok: io=%.1fms compute=%.1fms combine=%.1fms  "
        "bandwidth=%.2f GB/s  (%.1f MB/tok)\n",
        total_io * inv, total_compute * inv, total_combine * inv,
        bw, total_bytes * inv / 1e6);
    fprintf(stderr,
        "[moe-io] pread latency: %.0f%% <200us (cached)  %.0f%% 200us-1ms  "
        "%.0f%% 1-5ms  %.0f%% >5ms (SSD)\n",
        pct0, pct1, pct2, pct3);
    fprintf(stderr,
        "[moe-io] layer variance: fastest=%.1fms (L%02d) slowest=%.1fms (L%02d)\n",
        min_io, min_layer, max_io, max_layer);

    // Print top experts for sampled layers
    int sample_layers[] = {0, num_layers/4, num_layers/2, 3*num_layers/4, num_layers-1};
    for (int s = 0; s < 5; s++) {
        int li = sample_layers[s];
        if (li >= num_layers) continue;
        MoeLayerStats *st = ef->layer_stats[li];
        if (!st || st->token_count == 0) continue;
        // Find top 5 experts by frequency
        int top[5] = {-1,-1,-1,-1,-1};
        for (int e = 0; e < ef->num_experts; e++) {
            for (int t = 0; t < 5; t++) {
                if (top[t] < 0 || st->expert_freq[e] > st->expert_freq[top[t]]) {
                    for (int u = 4; u > t; u--) top[u] = top[u-1];
                    top[t] = e;
                    break;
                }
            }
        }
        fprintf(stderr, "[moe-io] L%02d top experts:", li);
        int total_freq = st->token_count * 10; // K experts per token (approximate)
        for (int t = 0; t < 5 && top[t] >= 0; t++) {
            int pct = total_freq > 0 ? (int)(100.0 * st->expert_freq[top[t]] / total_freq + 0.5) : 0;
            fprintf(stderr, " e%d=%d%%", top[t], pct);
        }
        fprintf(stderr, "\n");
    }

    if (reset) {
        for (int i = 0; i < num_layers; i++) {
            MoeLayerStats *st = ef->layer_stats[i];
            if (!st) continue;
            st->io_ms = st->compute_ms = st->combine_ms = 0;
            st->io_bytes = 0;
            st->token_count = 0;
            memset(st->pread_us_buckets, 0, sizeof(st->pread_us_buckets));
            memset(st->expert_freq, 0, ef->num_experts * sizeof(uint16_t));
        }
    }
}

bool expert_is_hot(const ExpertFiles *ef, int layer, int expert_id) {
    if (!ef->hot_mask) return true;
    int words_per_layer = (ef->num_experts + 31) / 32;
    int word = expert_id / 32;
    int bit = expert_id % 32;
    return (ef->hot_mask[layer * words_per_layer + word] >> bit) & 1;
}

// ============================================================================
// Expert forward — operates directly on mmap'd data, no copies
// ============================================================================

static void expert_forward_direct(const ModelConfig *cfg, const ExpertLayout *layout,
                                  const void *expert_base, const float *input,
                                  float *output, float *gate_buf, float *up_buf,
                                  float *act_buf, QuantType quant) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int G = cfg->group_size;

    const uint8_t *data = expert_base;
    cpu_dequant_matvec_quant(quant,
                             (const uint32_t *)(data + layout->gate_w_off),
                             (const uint16_t *)(data + layout->gate_s_off),
                             (const uint16_t *)(data + layout->gate_b_off),
                             input, gate_buf, M, H, G);
    cpu_dequant_matvec_quant(quant,
                             (const uint32_t *)(data + layout->up_w_off),
                             (const uint16_t *)(data + layout->up_s_off),
                             (const uint16_t *)(data + layout->up_b_off),
                             input, up_buf, M, H, G);
    cpu_swiglu(gate_buf, up_buf, act_buf, M);
    cpu_dequant_matvec_quant(quant,
                             (const uint32_t *)(data + layout->down_w_off),
                             (const uint16_t *)(data + layout->down_s_off),
                             (const uint16_t *)(data + layout->down_b_off),
                             act_buf, output, H, M, G);
}

// ============================================================================
// MoE forward — pre-allocated scratch, zero-copy expert access
// ============================================================================

// Static scratch buffers (allocated once on first call, reused)
static float *s_gate_scores = NULL;
static float *s_shared_gate = NULL, *s_shared_up = NULL, *s_shared_act = NULL;
static float *s_shared_out = NULL;
static float s_shared_gate_score = 0.0f;
static float *s_expert_out[OROME_MAX_ACTIVE];
static float *s_expert_gate[OROME_MAX_ACTIVE];
static float *s_expert_up[OROME_MAX_ACTIVE];
static float *s_expert_act[OROME_MAX_ACTIVE];
static int s_moe_alloc_H = 0;
static int s_moe_alloc_M = 0;
static int s_moe_alloc_S = 0;
static int s_moe_alloc_E = 0;

static void ensure_scratch(const ModelConfig *cfg) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int S = cfg->shared_intermediate;
    int E = cfg->num_experts;
    if (s_moe_alloc_H >= H && s_moe_alloc_M >= M &&
        s_moe_alloc_S >= S && s_moe_alloc_E >= E) return;

    free(s_gate_scores);
    free(s_shared_gate); free(s_shared_up); free(s_shared_act); free(s_shared_out);
    s_gate_scores = calloc(E, sizeof(float));
    s_shared_gate = calloc(S, sizeof(float));
    s_shared_up   = calloc(S, sizeof(float));
    s_shared_act  = calloc(S, sizeof(float));
    s_shared_out  = calloc(H, sizeof(float));

    for (int k = 0; k < OROME_MAX_ACTIVE; k++) {
        free(s_expert_out[k]); free(s_expert_gate[k]);
        free(s_expert_up[k]); free(s_expert_act[k]);
        s_expert_out[k]  = calloc(H, sizeof(float));
        s_expert_gate[k] = calloc(M, sizeof(float));
        s_expert_up[k]   = calloc(M, sizeof(float));
        s_expert_act[k]  = calloc(M, sizeof(float));
    }

    s_moe_alloc_H = H;
    s_moe_alloc_M = M;
    s_moe_alloc_S = S;
    s_moe_alloc_E = E;
}

void moe_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                 int layer_idx, float *hidden, float *h_post,
                 ExpertFiles *ef, int K, QuantType quant) {
    int H = cfg->hidden_dim;
    int n_experts = cfg->num_experts;
    void *layer_data = expert_layer_is_resident(ef, layer_idx)
        ? ef->layer_data[layer_idx] : NULL;
    bool use_pread = expert_layer_uses_pread(ef, layer_idx);
    if ((!layer_data && !use_pread) || K <= 0) return;

    const ExpertLayout *layout = (quant == QUANT_2BIT) ? &cfg->expert_2bit : &cfg->expert_4bit;
    ensure_scratch(cfg);

    // 1. Routing + shared expert projections — batch all h_post-input matvecs
    int expert_indices[OROME_MAX_ACTIVE];
    float expert_weights[OROME_MAX_ACTIVE];

    uint32_t *gate_w = weights_layer_ptr(wf, layer_idx, "mlp.gate.weight");
    uint16_t *gate_s = weights_layer_ptr(wf, layer_idx, "mlp.gate.scales");
    uint16_t *gate_b = weights_layer_ptr(wf, layer_idx, "mlp.gate.biases");
    if (!gate_w || !gate_s || !gate_b) return;

    uint32_t *sg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.weight");
    uint16_t *sg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.scales");
    uint16_t *sg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.biases");
    uint32_t *su_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.weight");
    uint16_t *su_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.scales");
    uint16_t *su_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.biases");
    uint32_t *sd_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.weight");
    uint16_t *sd_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.scales");
    uint16_t *sd_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.biases");
    uint32_t *sgg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.weight");
    uint16_t *sgg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.scales");
    uint16_t *sgg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.biases");

    int S = cfg->shared_intermediate;
    memset(s_shared_out, 0, H * sizeof(float));
    bool shared_gpu_ready = false;

    // Batch: routing gate + shared gate_proj + shared up_proj + shared_expert_gate
    // All use h_post as input. 4 dispatches → 1 GPU command buffer.
    // Shared gate/up go to dedicated GPU buffers for later GPU SwiGLU+down.
    if (ctx && ctx->buf_weights && sg_w && su_w && sgg_w) {
        memcpy([ctx->buf_input contents], h_post, H * sizeof(float));
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        size_t sgg_off = n_experts * sizeof(float);
        GpuMatvecJob jobs[4] = {
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)gate_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)gate_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)gate_b - base,
              .out_buf = ctx->buf_output, .out_off = 0,
              .out_ptr = s_gate_scores, .out_dim = n_experts, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)sg_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)sg_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)sg_b - base,
              .out_buf = ctx->buf_shared_gate, .out_off = 0,
              .out_ptr = NULL, .out_dim = S, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)su_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)su_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)su_b - base,
              .out_buf = ctx->buf_shared_up, .out_off = 0,
              .out_ptr = NULL, .out_dim = S, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)sgg_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)sgg_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)sgg_b - base,
              .out_buf = ctx->buf_output, .out_off = sgg_off,
              .out_ptr = &s_shared_gate_score, .out_dim = 1, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
        };
        gpu_run_matvec_batch(ctx, jobs, 4);
        shared_gpu_ready = true;
    } else {
        fast_dequant_matvec(ctx, cfg, gate_w, gate_s, gate_b, h_post, s_gate_scores,
                            n_experts, H, QUANT_4BIT);
        if (sg_w) fast_dequant_matvec(ctx, cfg, sg_w, sg_s, sg_b, h_post, s_shared_gate,
                                       S, H, QUANT_4BIT);
        if (su_w) fast_dequant_matvec(ctx, cfg, su_w, su_s, su_b, h_post, s_shared_up,
                                       S, H, QUANT_4BIT);
        if (sgg_w) fast_dequant_matvec(ctx, cfg, sgg_w, sgg_s, sgg_b, h_post,
                                        &s_shared_gate_score, 1, H, QUANT_4BIT);
    }

    cpu_softmax(s_gate_scores, n_experts);
    cpu_topk(s_gate_scores, n_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    log_expert_route(layer_idx, expert_indices, K, ef);

    // 3. Routed experts — GPU if a resident expert Metal buffer is available,
    // otherwise stay on the streaming/pread path.
    id<MTLBuffer> expert_layer_buf = (ctx && ctx->buf_expert_layers)
                                      ? ctx->buf_expert_layers[layer_idx] : nil;

    if (!expert_layer_buf && use_pread) {
        bool ran_gpu = false;
        bool expert_is_2bit[OROME_MAX_ACTIVE] = {false};

        if (ctx && ctx->buf_input && ctx->buf_weights && ctx->buf_shared_gate
            && ctx->buf_shared_up && ctx->buf_shared_act && ctx->buf_shared_out && sd_w) {
            memcpy([ctx->buf_input contents], h_post, H * sizeof(float));
            if (!shared_gpu_ready) {
                memcpy([ctx->buf_shared_gate contents], s_shared_gate, S * sizeof(float));
                memcpy([ctx->buf_shared_up contents], s_shared_up, S * sizeof(float));
            }
            if (pread_experts_into_gpu_buffers(ctx, cfg, ef, layer_idx,
                                               expert_indices, K, quant,
                                               expert_is_2bit) == 0) {
                ran_gpu = gpu_forward_pread_experts(ctx, cfg, sd_w, sd_s, sd_b,
                                                    K, expert_is_2bit,
                                                    s_expert_out, s_shared_out);
            }
            if (ran_gpu) {
                float sw = cpu_sigmoid(s_shared_gate_score);
                for (int i = 0; i < H; i++) s_shared_out[i] *= sw;
            }
        }

        if (!ran_gpu) {
            if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post,
                                         s_shared_gate, S, H, cfg->group_size);
            if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post,
                                         s_shared_up, S, H, cfg->group_size);
            cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
            if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act,
                                         s_shared_out, H, S, cfg->group_size);
            float sw = cpu_sigmoid(s_shared_gate_score);
            for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

            if (pread_experts_cpu(cfg, ef, layer_idx, expert_indices, K, h_post, quant,
                                  s_expert_out, s_expert_gate,
                                  s_expert_up, s_expert_act) != 0) {
                return;
            }
        }
    } else if (expert_layer_buf && ctx->buf_weights && sd_w) {
        // GPU expert+shared forward: all in ONE command buffer
        // Shared gate/up already in GPU buffers from routing batch above
        memcpy([ctx->buf_input contents], h_post, H * sizeof(float));

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        int M = cfg->moe_intermediate;
        id<MTLComputePipelineState> mv_pipe = (quant == QUANT_2BIT)
                                               ? ctx->matvec_2bit : ctx->matvec_4bit;
        uint8_t *wbase = (uint8_t *)[ctx->buf_weights contents];

        // Phase 1: gate + up projections for all K routed experts
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)M + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        // Phase 2: SwiGLU for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:2];
            uint dim_val = (uint)M;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)M + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        // Shared expert SwiGLU
        {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
            uint dim_val = (uint)S;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)S + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Phase 3: down projections for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_b_off atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        // Shared expert down
        {
            [enc setComputePipelineState:ctx->matvec_4bit];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_w - wbase atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_s - wbase atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_b - wbase atIndex:2];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Read back expert results
        for (int k = 0; k < K; k++) {
            memcpy(s_expert_out[k], [ctx->buf_multi_expert_out[k] contents],
                   H * sizeof(float));
        }
        memcpy(s_shared_out, [ctx->buf_shared_out contents], H * sizeof(float));

        // Apply shared expert gate
        float sw = cpu_sigmoid(s_shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;
    } else {
        // CPU fallback: shared expert
        if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post, s_shared_gate, S, H, cfg->group_size);
        if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post, s_shared_up, S, H, cfg->group_size);
        cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
        if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act, s_shared_out, H, S, cfg->group_size);
        float sw = cpu_sigmoid(s_shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

        // CPU fallback: routed experts
        for (int k = 0; k < K; k++) {
            const void *expert_base = (const uint8_t *)layer_data +
                (size_t)expert_indices[k] * layout->expert_size;
            expert_forward_direct(cfg, layout, expert_base, h_post,
                                  s_expert_out[k], s_expert_gate[k],
                                  s_expert_up[k], s_expert_act[k], quant);
        }
    }

    // 4. Combine: hidden += sum(weight[k] * expert_out[k]) + shared_out
    for (int k = 0; k < K; k++) {
        cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
    }
    cpu_vec_add(hidden, s_shared_out, H);
    if (use_pread) sync_hidden_buffer(ctx, hidden, H);

    // Debug: NaN check after MoE combine
    if (g_profile_experts) {
        int nans = 0;
        for (int i = 0; i < H; i++) {
            if (hidden[i] != hidden[i]) nans++;  // NaN != NaN
        }
        if (nans > 0) {
            fprintf(stderr, "NAN_CHECK moe_forward layer=%d nans=%d/%d\n", layer_idx, nans, H);
        }
    }
}

// ============================================================================
// MoE forward with pre-computed routing — skips routing GPU batch
// Assumes: gate_scores computed, shared expert gate/up already in GPU buffers,
//          h_post already in ctx->buf_input.
// ============================================================================

void moe_forward_routed(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                        int layer_idx, float *hidden, float *h_post,
                        float *gate_scores, float shared_gate_score,
                        ExpertFiles *ef, int K, QuantType quant,
                        bool gpu_combine) {
    int H = cfg->hidden_dim;
    int n_experts = cfg->num_experts;
    void *layer_data = expert_layer_is_resident(ef, layer_idx)
        ? ef->layer_data[layer_idx] : NULL;
    bool use_pread = expert_layer_uses_pread(ef, layer_idx);
    if ((!layer_data && !use_pread) || K <= 0) return;

    const ExpertLayout *layout = (quant == QUANT_2BIT) ? &cfg->expert_2bit : &cfg->expert_4bit;
    ensure_scratch(cfg);

    // 1. Softmax + topk on pre-computed gate scores
    int expert_indices[OROME_MAX_ACTIVE];
    float expert_weights[OROME_MAX_ACTIVE];

    cpu_softmax(gate_scores, n_experts);
    cpu_topk(gate_scores, n_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    log_expert_route(layer_idx, expert_indices, K, ef);

    // 2. Shared expert down weights
    uint32_t *sd_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.weight");
    uint16_t *sd_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.scales");
    uint16_t *sd_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.biases");
    int S = cfg->shared_intermediate;
    memset(s_shared_out, 0, H * sizeof(float));

    // 3. Routed experts + shared expert — GPU path
    // h_post is already in ctx->buf_input from fused command buffer
    // Shared gate/up already in GPU buffers from fused command buffer
    id<MTLBuffer> expert_layer_buf = (ctx && ctx->buf_expert_layers)
                                      ? ctx->buf_expert_layers[layer_idx] : nil;

    // Prefer direct Metal buffer access for resident layers over pread staging.
    if (!expert_layer_buf && use_pread) gpu_combine = false;

    if (!expert_layer_buf && use_pread) {
        bool ran_gpu = false;
        bool expert_is_2bit[OROME_MAX_ACTIVE] = {false};

        // h_post (eng->h_post) is stale in the GPU fused path — buf_input has the
        // correct post-norm data from the fused GPU command buffer. Read it back
        // so the CPU fallback path also uses correct data.
        if (ctx && ctx->buf_input) {
            memcpy(h_post, [ctx->buf_input contents], H * sizeof(float));
        }

        if (ctx && ctx->buf_input && ctx->buf_weights && ctx->buf_shared_gate
            && ctx->buf_shared_up && ctx->buf_shared_act && ctx->buf_shared_out && sd_w) {
            // buf_input already has correct h_post from fused GPU command buffer — don't overwrite
            if (pread_experts_into_gpu_buffers(ctx, cfg, ef, layer_idx,
                                               expert_indices, K, quant,
                                               expert_is_2bit) == 0) {
                ran_gpu = gpu_forward_pread_experts(ctx, cfg, sd_w, sd_s, sd_b,
                                                    K, expert_is_2bit,
                                                    s_expert_out, s_shared_out);
            }
            if (ran_gpu) {
                float sw = cpu_sigmoid(shared_gate_score);
                for (int i = 0; i < H; i++) s_shared_out[i] *= sw;
            }
        }

        if (!ran_gpu) {
            uint32_t *sg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.weight");
            uint16_t *sg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.scales");
            uint16_t *sg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.biases");
            uint32_t *su_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.weight");
            uint16_t *su_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.scales");
            uint16_t *su_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.biases");
            if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post,
                                         s_shared_gate, S, H, cfg->group_size);
            if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post,
                                         s_shared_up, S, H, cfg->group_size);
            cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
            if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act,
                                         s_shared_out, H, S, cfg->group_size);
            float sw = cpu_sigmoid(shared_gate_score);
            for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

            if (pread_experts_cpu(cfg, ef, layer_idx, expert_indices, K, h_post, quant,
                                  s_expert_out, s_expert_gate,
                                  s_expert_up, s_expert_act) != 0) {
                return;
            }
        }

        for (int k = 0; k < K; k++) {
            cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
        }
        cpu_vec_add(hidden, s_shared_out, H);
        sync_hidden_buffer(ctx, hidden, H);
    } else if (expert_layer_buf && ctx->buf_weights && sd_w &&
        ctx->batch_expert_mv && ctx->batch_expert_down) {
        // buf_input already has h_post — no memcpy needed
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        int M = cfg->moe_intermediate;
        uint8_t *wbase = (uint8_t *)[ctx->buf_weights contents];

        // Build per-expert offsets for gate, up, down projections
        typedef struct { uint32_t w_off, s_off, b_off; } GpuExpertOff;
        GpuExpertOff gate_offs[OROME_MAX_ACTIVE];
        GpuExpertOff up_offs[OROME_MAX_ACTIVE];
        GpuExpertOff down_offs[OROME_MAX_ACTIVE];
        for (int k = 0; k < K; k++) {
            size_t eo = (size_t)expert_indices[k] * layout->expert_size;
            gate_offs[k] = (GpuExpertOff){
                (uint32_t)(eo + layout->gate_w_off),
                (uint32_t)(eo + layout->gate_s_off),
                (uint32_t)(eo + layout->gate_b_off) };
            up_offs[k] = (GpuExpertOff){
                (uint32_t)(eo + layout->up_w_off),
                (uint32_t)(eo + layout->up_s_off),
                (uint32_t)(eo + layout->up_b_off) };
            down_offs[k] = (GpuExpertOff){
                (uint32_t)(eo + layout->down_w_off),
                (uint32_t)(eo + layout->down_s_off),
                (uint32_t)(eo + layout->down_b_off) };
        }

        NSUInteger tg_size = ROWS_PER_TG * 32;
        uint num_row_tgs_M = ((uint)M + ROWS_PER_TG - 1) / ROWS_PER_TG;
        uint num_row_tgs_H = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;

        // Phase 1: Batched gate projections (1 dispatch for all K experts)
        [enc setComputePipelineState:ctx->batch_expert_mv];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:2];
        [enc setBytes:gate_offs length:K * sizeof(GpuExpertOff) atIndex:3];
        { uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size, nrt = num_row_tgs_M;
          [enc setBytes:&od length:sizeof(uint) atIndex:4];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:5];
          [enc setBytes:&gs length:sizeof(uint) atIndex:6];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:7]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        // Phase 1b: Batched up projections (1 dispatch for all K experts)
        [enc setComputePipelineState:ctx->batch_expert_mv];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:2];
        [enc setBytes:up_offs length:K * sizeof(GpuExpertOff) atIndex:3];
        { uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size, nrt = num_row_tgs_M;
          [enc setBytes:&od length:sizeof(uint) atIndex:4];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:5];
          [enc setBytes:&gs length:sizeof(uint) atIndex:6];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:7]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Phase 2: Batched SwiGLU for K experts (1 dispatch)
        [enc setComputePipelineState:ctx->batch_swiglu];
        [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:2];
        { uint td = (uint)(K * M); [enc setBytes:&td length:sizeof(uint) atIndex:3]; }
        [enc dispatchThreadgroups:MTLSizeMake(((uint)(K * M) + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // Shared expert SwiGLU
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
        { uint dim_val = (uint)S; [enc setBytes:&dim_val length:sizeof(uint) atIndex:3]; }
        [enc dispatchThreadgroups:MTLSizeMake(((uint)S + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Phase 3: Batched down projections (1 dispatch, per-expert input from packed act)
        [enc setComputePipelineState:ctx->batch_expert_down];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_out offset:0 atIndex:2];
        [enc setBytes:down_offs length:K * sizeof(GpuExpertOff) atIndex:3];
        { uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size, nrt = num_row_tgs_H;
          [enc setBytes:&od length:sizeof(uint) atIndex:4];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:5];
          [enc setBytes:&gs length:sizeof(uint) atIndex:6];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:7]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_H * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        // Shared expert down
        {
            [enc setComputePipelineState:ctx->matvec_4bit];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_w - wbase atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_s - wbase atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_b - wbase atIndex:2];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
            { uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
              [enc setBytes:&od length:sizeof(uint) atIndex:5];
              [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
              [enc setBytes:&gs length:sizeof(uint) atIndex:7]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_H, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        if (gpu_combine && ctx->moe_combine_packed) {
            // Add barrier before combine
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Fill combine params: weights[0..K-1] + shared_gate_score
            float *params = (float *)[ctx->buf_combine_params contents];
            for (int k = 0; k < K; k++) params[k] = expert_weights[k];
            params[8] = shared_gate_score;

            // Dispatch packed combine: reads from buf_batch_expert_out [K * H]
            [enc setComputePipelineState:ctx->moe_combine_packed];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_batch_expert_out offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_combine_params offset:0 atIndex:4];
            { uint d = (uint)H, kk = (uint)K;
              [enc setBytes:&d length:sizeof(uint) atIndex:5];
              [enc setBytes:&kk length:sizeof(uint) atIndex:6]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
            [cmd commit];
            // DON'T wait — caller handles synchronization via queue ordering
        } else {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Read back expert results
            for (int k = 0; k < K; k++) {
                memcpy(s_expert_out[k],
                       (uint8_t *)[ctx->buf_batch_expert_out contents] + (size_t)k * H * sizeof(float),
                       H * sizeof(float));
            }
            memcpy(s_shared_out, [ctx->buf_shared_out contents], H * sizeof(float));

            // Apply shared expert gate
            float sw = cpu_sigmoid(shared_gate_score);
            for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

            // CPU combine
            for (int k = 0; k < K; k++) {
                cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
            }
            cpu_vec_add(hidden, s_shared_out, H);
            sync_hidden_buffer(ctx, hidden, H);
        }
    } else {
        // CPU fallback
        uint32_t *sg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.weight");
        uint16_t *sg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.scales");
        uint16_t *sg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.biases");
        uint32_t *su_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.weight");
        uint16_t *su_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.scales");
        uint16_t *su_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.biases");
        if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post, s_shared_gate, S, H, cfg->group_size);
        if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post, s_shared_up, S, H, cfg->group_size);
        cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
        if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act, s_shared_out, H, S, cfg->group_size);
        float sw = cpu_sigmoid(shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

        for (int k = 0; k < K; k++) {
            const void *expert_base = (const uint8_t *)layer_data +
                (size_t)expert_indices[k] * layout->expert_size;
            expert_forward_direct(cfg, layout, expert_base, h_post,
                                  s_expert_out[k], s_expert_gate[k],
                                  s_expert_up[k], s_expert_act[k], quant);
        }

        // CPU combine
        for (int k = 0; k < K; k++) {
            cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
        }
        cpu_vec_add(hidden, s_shared_out, H);
        sync_hidden_buffer(ctx, hidden, H);
    }

    // Debug: NaN check after MoE combine
    if (g_profile_experts) {
        int nans = 0;
        for (int i = 0; i < H; i++) {
            if (hidden[i] != hidden[i]) nans++;
        }
        if (nans > 0) {
            fprintf(stderr, "NAN_CHECK moe_forward_routed layer=%d nans=%d/%d\n", layer_idx, nans, H);
        }
    }
}
