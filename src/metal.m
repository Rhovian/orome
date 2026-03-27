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
#include <limits.h>

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

    ctx->kv_cache_seq = 0;

    // KV cache GPU buffers (full attention layers only)
    int n_full = cfg->num_full_attn_layers;
    ctx->buf_kv_k = (__strong id<MTLBuffer> *)calloc(n_full, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v = (__strong id<MTLBuffer> *)calloc(n_full, sizeof(id<MTLBuffer>));
    ctx->buf_attn_scores = nil;
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
    int K_max = cfg->num_experts_per_tok > 0 ? cfg->num_experts_per_tok : 1;
    if (K_max > OROME_MAX_ACTIVE) K_max = OROME_MAX_ACTIVE;
    int expert_intermediate = cfg->moe_intermediate > 0 ? cfg->moe_intermediate : 1;
    int shared_scratch = cfg->shared_intermediate > cfg->moe_intermediate
        ? cfg->shared_intermediate : cfg->moe_intermediate;
    if (shared_scratch <= 0) shared_scratch = 1;
    size_t batch_gate_size = (size_t)K_max * expert_intermediate * sizeof(float);
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

    ctx->buf_shared_gate = [ctx->device newBufferWithLength:(size_t)shared_scratch * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up = [ctx->device newBufferWithLength:(size_t)shared_scratch * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    ctx->buf_shared_act = [ctx->device newBufferWithLength:(size_t)shared_scratch * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    ctx->buf_shared_out = [ctx->device newBufferWithLength:H * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    printf("[metal] Buffers allocated for %s (%d layers, %d hidden)\n",
           cfg->name, cfg->num_layers, cfg->hidden_dim);
    return ctx;
}

bool metal_ensure_kv_capacity(MetalCtx *ctx, const ModelConfig *cfg, int seq_len) {
    if (!ctx || !cfg || seq_len <= 0) return seq_len <= 0;
    if (cfg->num_full_attn_layers == 0) return true;
    if (cfg->context_length > 0 && seq_len > cfg->context_length) {
        fprintf(stderr, "ERROR: requested sequence length %d exceeds model context %d\n",
                seq_len, cfg->context_length);
        return false;
    }
    if (seq_len <= ctx->kv_cache_seq) return true;

    int new_seq = ctx->kv_cache_seq > 0 ? ctx->kv_cache_seq : 256;
    while (new_seq < seq_len) {
        if (new_seq > INT_MAX / 2) {
            new_seq = seq_len;
            break;
        }
        new_seq *= 2;
    }
    if (cfg->context_length > 0 && new_seq > cfg->context_length) {
        new_seq = cfg->context_length;
    }
    if (new_seq < seq_len) {
        fprintf(stderr, "ERROR: could not grow KV cache to %d tokens\n", seq_len);
        return false;
    }

    size_t old_kv_size = (size_t)ctx->kv_cache_seq * cfg->kv_dim * sizeof(float);
    size_t new_kv_size = (size_t)new_seq * cfg->kv_dim * sizeof(float);
    for (int i = 0; i < cfg->num_full_attn_layers; i++) {
        id<MTLBuffer> new_k = [ctx->device newBufferWithLength:new_kv_size
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> new_v = [ctx->device newBufferWithLength:new_kv_size
                                                       options:MTLResourceStorageModeShared];
        if (!new_k || !new_v) {
            fprintf(stderr, "ERROR: failed to allocate KV cache buffers for %d tokens\n", new_seq);
            return false;
        }
        memset([new_k contents], 0, new_kv_size);
        memset([new_v contents], 0, new_kv_size);
        if (ctx->buf_kv_k[i] && old_kv_size > 0) {
            memcpy([new_k contents], [ctx->buf_kv_k[i] contents], old_kv_size);
        }
        if (ctx->buf_kv_v[i] && old_kv_size > 0) {
            memcpy([new_v contents], [ctx->buf_kv_v[i] contents], old_kv_size);
        }
        ctx->buf_kv_k[i] = new_k;
        ctx->buf_kv_v[i] = new_v;
    }

    size_t score_size = (size_t)cfg->num_attn_heads * new_seq * sizeof(float);
    id<MTLBuffer> new_scores = [ctx->device newBufferWithLength:score_size
                                                        options:MTLResourceStorageModeShared];
    if (!new_scores) {
        fprintf(stderr, "ERROR: failed to allocate attention score buffer for %d tokens\n", new_seq);
        return false;
    }
    memset([new_scores contents], 0, score_size);
    ctx->buf_attn_scores = new_scores;
    ctx->kv_cache_seq = new_seq;
    fprintf(stderr, "[metal] KV cache capacity: %d tokens\n", ctx->kv_cache_seq);
    return true;
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
