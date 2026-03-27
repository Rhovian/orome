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
    ctx->norm_sum_sq      = make_pipeline(ctx, @"rms_norm_sum_sq");
    ctx->norm_apply       = make_pipeline(ctx, @"rms_norm_apply_bf16");
    ctx->attn_scores      = make_pipeline(ctx, @"attn_scores_batched");
    ctx->attn_softmax     = make_pipeline(ctx, @"attn_softmax_batched");
    ctx->attn_values      = make_pipeline(ctx, @"attn_values_batched");
    ctx->sigmoid_gate     = make_pipeline(ctx, @"sigmoid_gate");
    ctx->swiglu           = make_pipeline(ctx, @"swiglu_fused");
    ctx->delta_net        = make_pipeline(ctx, @"gated_delta_net_step");
    ctx->rms_norm_qk      = make_pipeline(ctx, @"rms_norm_qk");
    ctx->gated_rms_norm   = make_pipeline(ctx, @"gated_rms_norm");
    ctx->batch_swiglu     = make_pipeline(ctx, @"batch_swiglu");
    ctx->rms_norm_qk_w    = make_pipeline(ctx, @"rms_norm_qk_weighted");
    ctx->rope_apply       = make_pipeline(ctx, @"rope_apply");
    ctx->rms_norm_qk_rope = make_pipeline(ctx, @"rms_norm_qk_rope");
    ctx->kv_cache_write   = make_pipeline(ctx, @"kv_cache_write");
    ctx->softmax_topk     = make_pipeline(ctx, @"softmax_topk_route");
    ctx->copy_buffer      = make_pipeline(ctx, @"copy_buffer");
    ctx->residual_add_sq  = make_pipeline(ctx, @"residual_add_sum_sq");
    ctx->norm_apply_partial = make_pipeline(ctx, @"rms_norm_apply_partial");
    ctx->moe_combine_copy_sq = make_pipeline(ctx, @"moe_combine_copy_sq");
    ctx->moe_combine_copy_sq_k8 = make_pipeline(ctx, @"moe_combine_copy_sq_k8");
    ctx->argmax           = make_pipeline(ctx, @"argmax_kernel");
    ctx->deinterleave_qgate = make_pipeline(ctx, @"deinterleave_qgate");
    ctx->copy_tmp_to_buf  = make_pipeline(ctx, @"copy_tmp_to_buf");
    ctx->matvec_q4k       = make_pipeline(ctx, @"dequant_matvec_q4k");
    ctx->matvec_q8_0      = make_pipeline(ctx, @"dequant_matvec_q8_0");
    ctx->batch_expert_mv_q4k_dyn = make_pipeline(ctx, @"batch_expert_mv_q4k_dyn");
    ctx->batch_expert_gate_up_swiglu_q4k_dyn = make_pipeline(ctx, @"batch_expert_gate_up_swiglu_q4k_dyn");
    ctx->shared_gate_up_swiglu_q4k = make_pipeline(ctx, @"shared_gate_up_swiglu_q4k");
    ctx->shared_gate_up_swiglu_q8_0 = make_pipeline(ctx, @"shared_gate_up_swiglu_q8_0");
    ctx->batch_expert_down_q4k_dyn = make_pipeline(ctx, @"batch_expert_down_q4k_dyn");
    ctx->batch_expert_down_q5k_dyn = make_pipeline(ctx, @"batch_expert_down_q5k_dyn");
    ctx->conv1d_f32 = make_pipeline(ctx, @"conv1d_step_f32");
    ctx->decay_beta_f32 = make_pipeline(ctx, @"compute_decay_beta_f32");
    ctx->matvec_f32 = make_pipeline(ctx, @"matvec_f32");
    ctx->matvec_f32_pair = make_pipeline(ctx, @"matvec_f32_pair");
    ctx->matvec_q5k = make_pipeline(ctx, @"dequant_matvec_q5k");
    ctx->matvec_q6k = make_pipeline(ctx, @"dequant_matvec_q6k");

    if (!ctx->matvec_q4k || !ctx->norm_sum_sq || !ctx->norm_apply) {
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
                        * cfg->linear_key_dim * sizeof(uint16_t);  // half state
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

    // Expert buffers (packed batch) (K × intermediate_dim)
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

void metal_free(MetalCtx *ctx) {
    if (!ctx) return;
    free(ctx->buf_kv_k);
    free(ctx->buf_kv_v);
    free(ctx->buf_linear_state);
    free(ctx->buf_conv_state);
    free(ctx);
}

// ============================================================================
