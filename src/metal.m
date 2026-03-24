/*
 * metal.m — Metal GPU context, pipeline setup, buffer management, GPU dispatch.
 *
 * All buffer sizes are derived from ModelConfig — no hardcoded model dimensions.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orome.h"

// ============================================================================
// Pipeline creation
// ============================================================================

static id<MTLComputePipelineState> make_pipeline(MetalCtx *ctx, NSString *name) {
    id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "WARNING: shader '%s' not found\n", [name UTF8String]);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> ps =
        [ctx->device newComputePipelineStateWithFunction:fn error:&error];
    if (!ps) {
        fprintf(stderr, "ERROR: pipeline '%s': %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return ps;
}

// ============================================================================
// Setup
// ============================================================================

MetalCtx *metal_setup(const ModelConfig *cfg) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx);
        return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx);
        return NULL;
    }

    // Load precompiled metallib (fast) or fall back to source compilation
    NSError *error = nil;
    double t0 = now_ms();

    NSString *metallib_path = @"src/shaders.metallib";
    if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path]) {
        NSURL *url = [NSURL fileURLWithPath:metallib_path];
        ctx->library = [ctx->device newLibraryWithURL:url error:&error];
        if (ctx->library) {
            printf("[metal] Loaded precompiled metallib: %.0f ms\n", now_ms() - t0);
        } else {
            fprintf(stderr, "WARNING: Failed to load metallib: %s, falling back to source\n",
                    [[error localizedDescription] UTF8String]);
        }
    }

    if (!ctx->library) {
        NSString *src = [NSString stringWithContentsOfFile:@"src/shaders.metal"
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
        if (!src) {
            fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
            free(ctx);
            return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_1;

        ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
        if (!ctx->library) {
            fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }
        printf("[metal] Shader compile from source: %.0f ms\n", now_ms() - t0);
    }

    // Create pipelines
    ctx->matvec_4bit      = make_pipeline(ctx, @"dequant_matvec_4bit_v3");
    ctx->matvec_2bit      = make_pipeline(ctx, @"dequant_matvec_2bit");
    ctx->norm_sum_sq      = make_pipeline(ctx, @"rms_norm_sum_sq");
    ctx->norm_apply       = make_pipeline(ctx, @"rms_norm_apply_bf16");
    ctx->residual_add     = make_pipeline(ctx, @"residual_add");
    ctx->attn_scores      = make_pipeline(ctx, @"attn_scores_batched");
    ctx->attn_softmax     = make_pipeline(ctx, @"attn_softmax_batched");
    ctx->attn_values      = make_pipeline(ctx, @"attn_values_batched");
    ctx->sigmoid_gate     = make_pipeline(ctx, @"sigmoid_gate");
    ctx->swiglu           = make_pipeline(ctx, @"swiglu_fused");
    ctx->moe_combine      = make_pipeline(ctx, @"moe_combine_residual");
    ctx->delta_net        = make_pipeline(ctx, @"gated_delta_net_step");
    ctx->conv1d           = make_pipeline(ctx, @"conv1d_step");
    ctx->rms_norm_qk      = make_pipeline(ctx, @"rms_norm_qk");
    ctx->decay_beta       = make_pipeline(ctx, @"compute_decay_beta");
    ctx->gated_rms_norm   = make_pipeline(ctx, @"gated_rms_norm");
    ctx->batch_expert_mv  = make_pipeline(ctx, @"batch_expert_matvec_4bit");
    ctx->batch_swiglu     = make_pipeline(ctx, @"batch_swiglu");
    ctx->batch_expert_down = make_pipeline(ctx, @"batch_expert_down_4bit");
    ctx->moe_combine_packed = make_pipeline(ctx, @"moe_combine_residual_packed");
    ctx->rms_norm_qk_w    = make_pipeline(ctx, @"rms_norm_qk_weighted");
    ctx->rope_apply       = make_pipeline(ctx, @"rope_apply");
    ctx->kv_cache_write   = make_pipeline(ctx, @"kv_cache_write");
    ctx->softmax_topk     = make_pipeline(ctx, @"softmax_topk_route");
    ctx->batch_expert_mv_dyn  = make_pipeline(ctx, @"batch_expert_mv_dyn");
    ctx->batch_expert_down_dyn = make_pipeline(ctx, @"batch_expert_down_dyn");
    ctx->expert_gate_up_swiglu = make_pipeline(ctx, @"expert_gate_up_swiglu_dyn");
    ctx->copy_buffer     = make_pipeline(ctx, @"copy_buffer");
    ctx->residual_add_sq = make_pipeline(ctx, @"residual_add_sum_sq");
    ctx->norm_apply_partial = make_pipeline(ctx, @"rms_norm_apply_partial");
    ctx->moe_combine_copy_sq = make_pipeline(ctx, @"moe_combine_copy_sq");
    ctx->matvec_4bit_2row = make_pipeline(ctx, @"dequant_matvec_4bit_2row");
    ctx->batch_expert_down_dyn_2row = make_pipeline(ctx, @"batch_expert_down_dyn_2row");
    ctx->argmax = make_pipeline(ctx, @"argmax_kernel");
    ctx->deinterleave_qgate = make_pipeline(ctx, @"deinterleave_qgate");
    ctx->copy_tmp_to_buf = make_pipeline(ctx, @"copy_tmp_to_buf");

    if (!ctx->matvec_4bit || !ctx->norm_sum_sq || !ctx->norm_apply) {
        fprintf(stderr, "ERROR: Required Metal pipelines missing\n");
        metal_free(ctx);
        return NULL;
    }

    // ---- Allocate buffers based on model config ----

    int H = cfg->hidden_dim;
    int V = cfg->vocab_size;
    int max_dim = (V > H) ? V : H;  // output buffer must fit vocab logits

    ctx->buf_input = [ctx->device newBufferWithLength:H * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_dim * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    ctx->buf_argmax_result = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq = [ctx->device newBufferWithLength:32 * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    ctx->buf_residual = [ctx->device newBufferWithLength:H * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid = [ctx->device newBufferWithLength:H * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    ctx->buf_moe_hidden = [ctx->device newBufferWithLength:H * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:(OROME_MAX_ACTIVE + 2) * sizeof(float)
                                                       options:MTLResourceStorageModeShared];

    // KV cache GPU buffers (full attention layers only)
    int n_full = cfg->num_full_attn_layers;
    ctx->buf_kv_k = (__strong id<MTLBuffer> *)calloc(n_full, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v = (__strong id<MTLBuffer> *)calloc(n_full, sizeof(id<MTLBuffer>));
    size_t kv_size = (size_t)OROME_GPU_KV_SEQ * cfg->kv_dim * sizeof(float);
    for (int i = 0; i < n_full; i++) {
        ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_size
                                                    options:MTLResourceStorageModeShared];
        ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_size
                                                    options:MTLResourceStorageModeShared];
        if (ctx->buf_kv_k[i]) memset([ctx->buf_kv_k[i] contents], 0, kv_size);
        if (ctx->buf_kv_v[i]) memset([ctx->buf_kv_v[i] contents], 0, kv_size);
    }

    ctx->buf_attn_scores = [ctx->device newBufferWithLength:
        (size_t)cfg->num_attn_heads * OROME_GPU_KV_SEQ * sizeof(float)
        options:MTLResourceStorageModeShared];
    // 2× for attn_output_gate: Q projection stores [Q, gate] concatenated
    ctx->buf_attn_output = [ctx->device newBufferWithLength:
        (size_t)cfg->num_attn_heads * cfg->head_dim * 2 * sizeof(float)
        options:MTLResourceStorageModeShared];

    // Linear attention GPU state
    int n_lin = cfg->num_linear_layers;
    ctx->buf_linear_state = (__strong id<MTLBuffer> *)calloc(n_lin, sizeof(id<MTLBuffer>));
    ctx->buf_conv_state = (__strong id<MTLBuffer> *)calloc(n_lin, sizeof(id<MTLBuffer>));
    size_t delta_size = (size_t)cfg->linear_num_v_heads * cfg->linear_value_dim
                        * cfg->linear_key_dim * sizeof(float);  // float32 state
    size_t conv_size = (size_t)(cfg->conv_kernel_size - 1) * cfg->linear_conv_dim * sizeof(float);
    for (int i = 0; i < n_lin; i++) {
        ctx->buf_linear_state[i] = [ctx->device newBufferWithLength:delta_size
                                                            options:MTLResourceStorageModeShared];
        ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:conv_size
                                                          options:MTLResourceStorageModeShared];
        if (ctx->buf_linear_state[i]) memset([ctx->buf_linear_state[i] contents], 0, delta_size);
        if (ctx->buf_conv_state[i]) memset([ctx->buf_conv_state[i] contents], 0, conv_size);
    }

    // buf_linear_q is reused for gated_rms_norm output (linear_total_value elements)
    // which is larger than linear_total_key, so allocate the max of both
    { size_t q_size = cfg->linear_total_key > cfg->linear_total_value
                        ? cfg->linear_total_key : cfg->linear_total_value;
    ctx->buf_linear_q = [ctx->device newBufferWithLength:q_size * sizeof(float)
                                                 options:MTLResourceStorageModeShared]; }
    ctx->buf_linear_k = [ctx->device newBufferWithLength:cfg->linear_total_key * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_linear_v = [ctx->device newBufferWithLength:cfg->linear_total_value * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_linear_decay = [ctx->device newBufferWithLength:cfg->linear_num_v_heads * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_linear_beta = [ctx->device newBufferWithLength:cfg->linear_num_v_heads * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_linear_output = [ctx->device newBufferWithLength:cfg->linear_total_value * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    ctx->buf_conv_input = [ctx->device newBufferWithLength:cfg->linear_conv_dim * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    ctx->buf_conv_output = [ctx->device newBufferWithLength:cfg->linear_conv_dim * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Expert buffers
    size_t expert_alloc = (cfg->expert_4bit.expert_size + 16383) & ~((size_t)16383);
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:H * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    // Allocate extra data buffer slots for expert caching across tokens.
    // Data buffers hold raw expert weights; gate/up/act/out are only needed for K active experts.
    int cache_slots = cfg->num_experts_per_tok * 3;  // 3× K for caching headroom
    if (cache_slots > OROME_EXPERT_CACHE_SLOTS) cache_slots = OROME_EXPERT_CACHE_SLOTS;
    for (int k = 0; k < cache_slots; k++) {
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithLength:expert_alloc
                                                                 options:MTLResourceStorageModeShared];
    }
    ctx->num_expert_data_slots = cache_slots;
    printf("[metal] Expert data buffer slots: %d (%d active + %d cache)\n",
           cache_slots, cfg->num_experts_per_tok, cache_slots - cfg->num_experts_per_tok);
    for (int k = 0; k < cfg->num_experts_per_tok && k < OROME_MAX_ACTIVE; k++) {
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:cfg->moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k] = [ctx->device newBufferWithLength:cfg->moe_intermediate * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k] = [ctx->device newBufferWithLength:cfg->moe_intermediate * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k] = [ctx->device newBufferWithLength:H * sizeof(float)
                                                                options:MTLResourceStorageModeShared];
    }

    // Packed batch expert buffers (K × intermediate_dim)
    int K_max = cfg->num_experts_per_tok < OROME_MAX_ACTIVE ? cfg->num_experts_per_tok : OROME_MAX_ACTIVE;
    size_t batch_gate_size = (size_t)K_max * cfg->moe_intermediate * sizeof(float);
    size_t batch_out_size = (size_t)K_max * H * sizeof(float);
    ctx->buf_batch_expert_gate = [ctx->device newBufferWithLength:batch_gate_size
                                                          options:MTLResourceStorageModeShared];
    ctx->buf_batch_expert_up = [ctx->device newBufferWithLength:batch_gate_size
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_batch_expert_act = [ctx->device newBufferWithLength:batch_gate_size
                                                         options:MTLResourceStorageModeShared];
    ctx->buf_batch_expert_out = [ctx->device newBufferWithLength:batch_out_size
                                                         options:MTLResourceStorageModeShared];
    ctx->buf_expert_offsets = [ctx->device newBufferWithLength:K_max * 3 * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_topk_indices = [ctx->device newBufferWithLength:K_max * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];

    ctx->buf_shared_gate = [ctx->device newBufferWithLength:cfg->shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up = [ctx->device newBufferWithLength:cfg->shared_intermediate * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    ctx->buf_shared_act = [ctx->device newBufferWithLength:cfg->shared_intermediate * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    ctx->buf_shared_out = [ctx->device newBufferWithLength:H * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    printf("[metal] Buffers allocated for %s (%d layers, %d hidden)\n",
           cfg->name, cfg->num_layers, cfg->hidden_dim);
    return ctx;
}

void metal_set_weights(MetalCtx *ctx, WeightFile *wf) {
    size_t page_size = 16384;
    size_t aligned = (wf->size + page_size - 1) & ~(page_size - 1);

    ctx->buf_weights = [ctx->device newBufferWithBytesNoCopy:wf->data
                                                      length:aligned
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!ctx->buf_weights) {
        fprintf(stderr, "WARNING: Cannot wrap weights as Metal buffer (%.2f GB) — GPU will use copies\n",
                wf->size / 1e9);
    } else {
        printf("[metal] Weights wrapped as Metal buffer (%.2f GB)\n", aligned / 1e9);
    }
}

void metal_set_expert_weights(MetalCtx *ctx, ExpertFiles *ef, const ModelConfig *cfg) {
    if (!ctx || !ef) return;
    int n = cfg->num_layers;
    ctx->buf_expert_layers = (__strong id<MTLBuffer> *)calloc(n, sizeof(id<MTLBuffer>));
    ctx->num_expert_layers = n;
    int wrapped = 0;
    for (int i = 0; i < n; i++) {
        // Only wrap layers that are explicitly inside the resident budget.
        if (!ef->layer_resident || !ef->layer_resident[i]
            || !ef->layer_data[i] || ef->layer_size[i] == 0) continue;
        size_t page_size = 16384;
        size_t aligned = (ef->layer_size[i] + page_size - 1) & ~(page_size - 1);
        ctx->buf_expert_layers[i] = [ctx->device newBufferWithBytesNoCopy:ef->layer_data[i]
                                                                    length:aligned
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
        if (ctx->buf_expert_layers[i]) wrapped++;
    }
    printf("[metal] Resident expert layers wrapped as Metal buffers: %d/%d\n", wrapped, n);

    // Per-layer expert cache buffers for cross-token caching on pread layers.
    // Each non-resident layer gets a buffer holding K expert slots so data
    // persists across tokens without inter-layer contamination.
    int pread_layers = n - wrapped;
    if (pread_layers > 0) {
        bool has_2bit = false;
        if (ef->layer_fds_2bit) {
            for (int i = 0; i < n; i++) {
                if (ef->layer_fds_2bit[i] >= 0) { has_2bit = true; break; }
            }
        }
        size_t slot_bytes;
        if (has_2bit && !ef->tiered_quant) {
            slot_bytes = (cfg->expert_2bit.expert_size + 16383) & ~((size_t)16383);
        } else {
            size_t max_es = cfg->expert_4bit.expert_size;
            if (cfg->expert_2bit.expert_size > max_es) max_es = cfg->expert_2bit.expert_size;
            slot_bytes = (max_es + 16383) & ~((size_t)16383);
        }
        ctx->expert_cache_slot_bytes = slot_bytes;
        int K = cfg->num_experts_per_tok;
        size_t per_layer = (size_t)K * slot_bytes;
        ctx->buf_expert_layer_cache = (__strong id<MTLBuffer> *)calloc(n, sizeof(id<MTLBuffer>));
        size_t total_alloc = 0;
        int alloc_count = 0;
        for (int i = 0; i < n; i++) {
            if (ef->layer_resident && ef->layer_resident[i]) continue;
            ctx->buf_expert_layer_cache[i] = [ctx->device newBufferWithLength:per_layer
                                                                       options:MTLResourceStorageModeShared];
            if (ctx->buf_expert_layer_cache[i]) { total_alloc += per_layer; alloc_count++; }
        }
        printf("[metal] Per-layer expert cache: %d layers × %d slots × %.2f MB = %.0f MB\n",
               alloc_count, K, (float)slot_bytes / 1048576.0f, (float)total_alloc / 1048576.0f);
    }
}

void metal_free(MetalCtx *ctx) {
    if (!ctx) return;
    free(ctx->buf_kv_k);
    free(ctx->buf_kv_v);
    free(ctx->buf_linear_state);
    free(ctx->buf_conv_state);
    free(ctx->buf_expert_layers);
    free(ctx->buf_expert_layer_cache);
    free(ctx);
}

// ============================================================================
// GPU matvec dispatch
// ============================================================================

#define ROWS_PER_TG 16  // tuning parameter for autoresearch

void gpu_encode_matvec_job(id<MTLComputeCommandEncoder> enc,
                           MetalCtx *ctx,
                           GpuMatvecJob *job) {
    // Use 2-row kernel for 4-bit matvecs (halves TG count)
    bool use_2row = !job->is_2bit && ctx->matvec_4bit_2row;
    id<MTLComputePipelineState> pipe;
    NSUInteger rows_per_tg;
    if (job->is_2bit) {
        pipe = ctx->matvec_2bit;
        rows_per_tg = ROWS_PER_TG;
    } else if (use_2row) {
        pipe = ctx->matvec_4bit_2row;
        rows_per_tg = ROWS_PER_TG * 2;  // 32 effective rows per TG (2 rows/simdgroup)
    } else {
        pipe = ctx->matvec_4bit;
        rows_per_tg = ROWS_PER_TG;
    }
    if (!pipe) return;

    [enc setComputePipelineState:pipe];
    [enc setBuffer:job->w_buf offset:job->w_off atIndex:0];
    [enc setBuffer:job->s_buf offset:job->s_off atIndex:1];
    [enc setBuffer:job->b_buf offset:job->b_off atIndex:2];
    id<MTLBuffer> in_buf = job->in_buf ? job->in_buf : ctx->buf_input;
    [enc setBuffer:in_buf offset:job->in_off atIndex:3];
    [enc setBuffer:job->out_buf offset:job->out_off atIndex:4];

    uint out_dim = (uint)job->out_dim;
    uint in_dim = (uint)job->in_dim;
    uint gs = (uint)job->group_size;
    [enc setBytes:&out_dim length:sizeof(uint) atIndex:5];
    [enc setBytes:&in_dim  length:sizeof(uint) atIndex:6];
    [enc setBytes:&gs      length:sizeof(uint) atIndex:7];

    NSUInteger tg_size = ROWS_PER_TG * 32;  // thread count stays at 512
    NSUInteger num_tgs = ((uint)job->out_dim + rows_per_tg - 1) / rows_per_tg;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
}

void gpu_run_matvec_batch(MetalCtx *ctx, GpuMatvecJob *jobs, int count) {
    if (!ctx || count <= 0) return;

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    for (int i = 0; i < count; i++) {
        gpu_encode_matvec_job(enc, ctx, &jobs[i]);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Copy results to CPU pointers if requested
    for (int i = 0; i < count; i++) {
        if (jobs[i].out_ptr) {
            memcpy(jobs[i].out_ptr,
                   (uint8_t *)[jobs[i].out_buf contents] + jobs[i].out_off,
                   jobs[i].out_dim * sizeof(float));
        }
    }
}

void gpu_dequant_matvec(MetalCtx *ctx, const ModelConfig *cfg,
                        uint32_t *W, uint16_t *scales, uint16_t *biases,
                        float *x, float *out, int out_dim, int in_dim,
                        QuantType quant) {
    if (!ctx || !ctx->buf_weights) return;

    // Copy input to GPU buffer
    memcpy([ctx->buf_input contents], x, in_dim * sizeof(float));

    // Compute offsets into mmap'd weight buffer
    uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
    size_t w_off = (uint8_t *)W - base;
    size_t s_off = (uint8_t *)scales - base;
    size_t b_off = (uint8_t *)biases - base;

    GpuMatvecJob job = {
        .w_buf = ctx->buf_weights, .w_off = w_off,
        .s_buf = ctx->buf_weights, .s_off = s_off,
        .b_buf = ctx->buf_weights, .b_off = b_off,
        .out_buf = ctx->buf_output, .out_off = 0,
        .out_ptr = out,
        .out_dim = out_dim, .in_dim = in_dim,
        .group_size = cfg->group_size,
        .is_2bit = (quant == QUANT_2BIT),
    };

    gpu_run_matvec_batch(ctx, &job, 1);
}

void fast_dequant_matvec(MetalCtx *ctx, const ModelConfig *cfg,
                         uint32_t *W, uint16_t *scales, uint16_t *biases,
                         float *x, float *out, int out_dim, int in_dim,
                         QuantType quant) {
    if (ctx && ctx->buf_weights) {
        gpu_dequant_matvec(ctx, cfg, W, scales, biases, x, out,
                           out_dim, in_dim, quant);
    } else if (quant == QUANT_2BIT) {
        cpu_dequant_matvec_2bit(W, scales, biases, x, out,
                                out_dim, in_dim, cfg->group_size);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out,
                           out_dim, in_dim, cfg->group_size);
    }
}
