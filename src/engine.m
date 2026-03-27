/*
 * engine.m — Forward pass orchestration.
 *
 * Engine owns the full inference pipeline:
 *   embedding → layer loop (attention + MoE) → final norm → lm_head → sample
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orome.h"

static void thermal_k_reset(ThermalKState *t) {
    t->proj_ema_ms = 0.0;
    t->generated = 0;
    t->engaged = false;
    t->have_proj = false;
}

static int thermal_k_effective(ThermalKState *t, int requested_K) {
    if (!t->enabled || !t->engaged || t->hot_k <= 0) return requested_K;
    return requested_K > t->hot_k ? t->hot_k : requested_K;
}

static void thermal_k_record(ThermalKState *t, double proj_ms) {
    if (!t->enabled || proj_ms <= 0.0) return;
    if (!t->have_proj) {
        t->proj_ema_ms = proj_ms;
        t->have_proj = true;
    } else {
        t->proj_ema_ms = 0.75 * t->proj_ema_ms + 0.25 * proj_ms;
    }
    t->generated++;
    if (t->generated < t->min_gen) return;
    if (!t->engaged && t->hot_k > 0 && t->proj_ema_ms >= t->proj_threshold_ms) {
        t->engaged = true;
        fprintf(stderr, "[thermal-k] engaged: K→%d after %d tokens (ema=%.1fms thresh=%.1fms)\n",
                t->hot_k, t->generated, t->proj_ema_ms, t->proj_threshold_ms);
    }
}

// ============================================================================
// Engine lifecycle
// ============================================================================

Engine *engine_create(ModelConfig *cfg, MetalCtx *ctx, int active_experts) {
    Engine *eng = calloc(1, sizeof(Engine));
    eng->cfg = cfg;
    eng->ctx = ctx;
    eng->active_experts = (cfg->ffn_type == FFN_MOE)
        ? (active_experts > 0 ? active_experts : cfg->num_experts_per_tok)
        : 0;
    eng->thermal.enabled = false;
    eng->thermal.hot_k = 0;
    eng->thermal.min_gen = 16;
    eng->thermal.proj_threshold_ms = 85.0;
    thermal_k_reset(&eng->thermal);

    eng->hidden = calloc(cfg->hidden_dim, sizeof(float));
    eng->pos = 0;
    return eng;
}

void engine_free(Engine *eng) {
    if (!eng) return;
    free(eng->expert_layer_cache);
    free(eng->hidden);
    free(eng);
}

// Profiling accumulators (reset in engine_reset)
static double t_attn_total = 0, t_ffn_total = 0, t_lmhead_total = 0;
static int profile_count = 0;

void engine_reset(Engine *eng) {
    eng->pos = 0;
    thermal_k_reset(&eng->thermal);
    // Reset profile accumulators
    t_attn_total = 0; t_ffn_total = 0; t_lmhead_total = 0;
    profile_count = 0;
}

// ============================================================================
// Embedding lookup (GGUF format)
// ============================================================================

// Embedding lookup for GGUF Q8_0 format
// Q8_0 block: 32 int8 weights + fp16 scale, so one row = H/32 blocks
static void embed_lookup_gguf(GGUFFile *gf, const ModelConfig *cfg,
                               int token_id, float *out) {
    GGUFTensorInfo *ti = gguf_find_tensor(gf, "token_embd.weight");
    if (!ti) { fprintf(stderr, "ERROR: token_embd.weight not found in GGUF\n"); return; }

    int H = cfg->hidden_dim;
    uint8_t *data = (uint8_t *)gf->mmap_base + gf->data_offset + ti->offset;

    if (ti->type == 8) { // Q8_0
        int blocks_per_row = H / 32;
        int bytes_per_row = blocks_per_row * 34; // 34 bytes per Q8_0 block
        uint8_t *row = data + (size_t)token_id * bytes_per_row;

        for (int blk = 0; blk < blocks_per_row; blk++) {
            uint8_t *block = row + blk * 34;
            uint16_t d_raw = block[0] | ((uint16_t)block[1] << 8);
            // fp16 to float
            uint32_t sign = (d_raw >> 15) & 1;
            uint32_t exp = (d_raw >> 10) & 0x1F;
            uint32_t mant = d_raw & 0x3FF;
            float d;
            if (exp == 0) d = 0.0f;
            else {
                uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&d, &f32, 4);
            }

            int8_t *qs = (int8_t *)(block + 2);
            int base = blk * 32;
            for (int j = 0; j < 32; j++) {
                out[base + j] = d * (float)qs[j];
            }
        }
    } else if (ti->type == 0) { // F32
        float *row = (float *)(data + (size_t)token_id * H * sizeof(float));
        memcpy(out, row, H * sizeof(float));
    } else if (ti->type == 1) { // F16
        uint16_t *row = (uint16_t *)(data + (size_t)token_id * H * sizeof(uint16_t));
        for (int j = 0; j < H; j++) {
            uint16_t h = row[j];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            if (exp == 0) out[j] = 0.0f;
            else {
                uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&out[j], &f32, 4);
            }
        }
    } else {
        fprintf(stderr, "ERROR: unsupported embedding type %d\n", ti->type);
    }
}

// ============================================================================
// Fused expert encoding — GPU softmax+topk + dynamic expert matvecs
// ============================================================================
// Encodes softmax_topk + expert gate/up/swiglu/down + shared expert + combine
// into an already-open compute command encoder. Eliminates CPU readback.

#define ENGINE_ROWS_PER_TG 16  // must match ROWS_PER_TG in metal.m and shaders

// ============================================================================
// GPU-resident expert forward for GGUF (Q4K/Q5K format)
// ============================================================================
// All work stays on the GPU encoder — no CPU readback, no memcpy.
// Routing logits and shared expert gate score must already be in buf_output
// (from the routing dispatch in the current layer).

static void encode_experts_gguf(id<MTLComputeCommandEncoder> enc,
                                 MetalCtx *ctx, const ModelConfig *cfg,
                                 const ExpertLayerRef *elr,
                                 const LayerTensorCache *lt,
                                 int K) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int S = cfg->shared_intermediate;
    int n_experts = cfg->num_experts;

    NSUInteger tg_size = ENGINE_ROWS_PER_TG * 32;
    uint num_row_tgs_M = ((uint)M + ENGINE_ROWS_PER_TG - 1) / ENGINE_ROWS_PER_TG;
    uint num_row_tgs_H_2row = ((uint)H + (ENGINE_ROWS_PER_TG * 2) - 1)
        / (ENGINE_ROWS_PER_TG * 2);

    // --- 1. GPU softmax + topK routing ---
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:ctx->softmax_topk];
    [enc setBuffer:ctx->buf_output offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_combine_params offset:0 atIndex:2];
    { uint ne = (uint)n_experts, kk = (uint)K, sgg = (uint)n_experts;
      [enc setBytes:&ne length:sizeof(uint) atIndex:3];
      [enc setBytes:&kk length:sizeof(uint) atIndex:4];
      [enc setBytes:&sgg length:sizeof(uint) atIndex:5]; }
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    bool can_fuse_expert_gate_up =
        ctx->batch_expert_gate_up_swiglu_q4k_dyn
        && elr->gate.format == QFMT_GGUF_Q4_K
        && elr->up.format == QFMT_GGUF_Q4_K;
    id<MTLComputePipelineState> shared_gate_up_pipe = nil;
    if (lt->shared_gate.format == QFMT_GGUF_Q4_K
            && lt->shared_up.format == QFMT_GGUF_Q4_K) {
        shared_gate_up_pipe = ctx->shared_gate_up_swiglu_q4k;
    } else if (lt->shared_gate.format == QFMT_GGUF_Q8_0
                   && lt->shared_up.format == QFMT_GGUF_Q8_0) {
        shared_gate_up_pipe = ctx->shared_gate_up_swiglu_q8_0;
    }
    bool fuse_shared_gate_up = shared_gate_up_pipe != nil;

    if (can_fuse_expert_gate_up) {
        uint gate_es = (uint)elr->gate.expert_stride;
        uint up_es = (uint)elr->up.expert_stride;
        uint od = (uint)M, id_ = (uint)H, nrt = num_row_tgs_M;
        [enc setComputePipelineState:ctx->batch_expert_gate_up_swiglu_q4k_dyn];
        [enc setBuffer:elr->buffer offset:elr->gate.offset atIndex:0];
        [enc setBuffer:elr->buffer offset:elr->up.offset atIndex:1];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:3];
        [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:4];
        [enc setBytes:&gate_es length:sizeof(uint) atIndex:5];
        [enc setBytes:&up_es length:sizeof(uint) atIndex:6];
        [enc setBytes:&od length:sizeof(uint) atIndex:7];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:8];
        [enc setBytes:&nrt length:sizeof(uint) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    } else {
        // --- 2. Batched gate projection ---
        { uint es = (uint)elr->gate.expert_stride;
          uint od = (uint)M, id_ = (uint)H, nrt = num_row_tgs_M;
          [enc setComputePipelineState:ctx->batch_expert_mv_q4k_dyn];
          [enc setBuffer:elr->buffer offset:elr->gate.offset atIndex:0];
          [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
          [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:2];
          [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
          [enc setBytes:&es length:sizeof(uint) atIndex:4];
          [enc setBytes:&od length:sizeof(uint) atIndex:5];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:7];
          [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)]; }

        // --- 3. Batched up projection ---
        { uint es = (uint)elr->up.expert_stride;
          uint od = (uint)M, id_ = (uint)H, nrt = num_row_tgs_M;
          [enc setComputePipelineState:ctx->batch_expert_mv_q4k_dyn];
          [enc setBuffer:elr->buffer offset:elr->up.offset atIndex:0];
          [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
          [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:2];
          [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
          [enc setBytes:&es length:sizeof(uint) atIndex:4];
          [enc setBytes:&od length:sizeof(uint) atIndex:5];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:7];
          [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)]; }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // --- 4. Batched SwiGLU ---
        [enc setComputePipelineState:ctx->batch_swiglu];
        [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:2];
        { uint td = (uint)(K * M); [enc setBytes:&td length:sizeof(uint) atIndex:3]; }
        [enc dispatchThreadgroups:MTLSizeMake(((uint)(K * M) + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // Shared expert gate/up is independent of routing softmax and expert top-k.
    // Schedule it here so the initial routing barrier only waits on routing.
    if (fuse_shared_gate_up) {
        [enc setComputePipelineState:shared_gate_up_pipe];
        [enc setBuffer:lt->shared_gate.buffer offset:lt->shared_gate.offset atIndex:0];
        [enc setBuffer:lt->shared_up.buffer offset:lt->shared_up.offset atIndex:1];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
        { uint od = (uint)lt->shared_gate.out_dim, id_ = (uint)lt->shared_gate.in_dim;
          [enc setBytes:&od length:sizeof(uint) atIndex:4];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:5]; }
        NSUInteger num_tgs = (lt->shared_gate.out_dim + ENGINE_ROWS_PER_TG - 1)
            / ENGINE_ROWS_PER_TG;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    } else {
        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->shared_gate,
            ctx->buf_input, 0, ctx->buf_shared_gate, 0);
        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->shared_up,
            ctx->buf_input, 0, ctx->buf_shared_up, 0);
    }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // --- 5. Shared expert SwiGLU ---
    if (!fuse_shared_gate_up) {
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
        { uint dim_val = (uint)S; [enc setBytes:&dim_val length:sizeof(uint) atIndex:3]; }
        [enc dispatchThreadgroups:MTLSizeMake(((uint)S + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // --- 6. Expert down projection (Q4K or Q5K) ---
    { id<MTLComputePipelineState> down_pipe = (elr->down.format == QFMT_GGUF_Q5_K)
          ? ctx->batch_expert_down_q5k_dyn : ctx->batch_expert_down_q4k_dyn;
      uint es = (uint)elr->down.expert_stride;
      uint od = (uint)H, id_ = (uint)M, nrt = num_row_tgs_H_2row;
      [enc setComputePipelineState:down_pipe];
      [enc setBuffer:elr->buffer offset:elr->down.offset atIndex:0];
      [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:1];
      [enc setBuffer:ctx->buf_batch_expert_out offset:0 atIndex:2];
      [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
      [enc setBytes:&es length:sizeof(uint) atIndex:4];
      [enc setBytes:&od length:sizeof(uint) atIndex:5];
      [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
      [enc setBytes:&nrt length:sizeof(uint) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_H_2row * K, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)]; }

    // --- 7. Shared expert down projection ---
    format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->shared_down,
        ctx->buf_shared_act, 0, ctx->buf_shared_out, 0);

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // --- 8. Combine: hidden += experts + shared, copy residual, compute partial sum_sq ---
    id<MTLComputePipelineState> combine_pipe =
        (K == 8 && ctx->moe_combine_copy_sq_k8)
        ? ctx->moe_combine_copy_sq_k8
        : ctx->moe_combine_copy_sq;
    [enc setComputePipelineState:combine_pipe];
    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
    [enc setBuffer:ctx->buf_batch_expert_out offset:0 atIndex:3];
    [enc setBuffer:ctx->buf_combine_params offset:0 atIndex:4];
    { uint d = (uint)H, kk = (uint)K;
      [enc setBytes:&d length:sizeof(uint) atIndex:5];
      [enc setBytes:&kk length:sizeof(uint) atIndex:6]; }
    [enc setBuffer:ctx->buf_residual offset:0 atIndex:7];
    [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void encode_dense_ffn(id<MTLComputeCommandEncoder> enc,
                             MetalCtx *ctx, const ModelConfig *cfg,
                             const LayerTensorCache *lt) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;

    format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->dense_gate,
        ctx->buf_input, 0, ctx->buf_shared_gate, 0);
    format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->dense_up,
        ctx->buf_input, 0, ctx->buf_shared_up, 0);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
    { uint dim_val = (uint)M; [enc setBytes:&dim_val length:sizeof(uint) atIndex:3]; }
    [enc dispatchThreadgroups:MTLSizeMake(((uint)M + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->dense_down,
        ctx->buf_shared_act, 0, ctx->buf_h_mid, 0);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ctx->residual_add_sq];
    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_h_mid offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
    [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:3];
    { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:4]; }
    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// ============================================================================
// Forward pass: one token
// ============================================================================

// (profiling accumulators declared earlier, near engine_reset)

int engine_step(Engine *eng, int token_id) {
    ModelConfig *cfg = eng->cfg;
    int H = cfg->hidden_dim;
    int pos = eng->pos;
    double step_start = now_ms();
    MetalCtx *ctx = eng->ctx;
    LayerTensorCache *tcache = eng->tensor_cache;

    if (cfg->num_full_attn_layers > 0 &&
        !metal_ensure_kv_capacity(ctx, cfg, pos + 1)) {
        fprintf(stderr, "ERROR: KV cache unavailable for token position %d\n", pos);
        return cfg->eos_tokens[0] >= 0 ? cfg->eos_tokens[0] : 0;
    }

    // 1. Embedding
    memset(eng->hidden, 0, H * sizeof(float));
    embed_lookup_gguf(eng->gf, cfg, token_id, eng->hidden);

    // 2. Layer loop — fused O-proj + post-norm + MoE routing in one GPU commit
    int full_idx = 0, linear_idx = 0;
    double t0, t1;
    // Upload hidden to GPU once before the layer loop.
    // Hidden state stays on GPU (in buf_moe_hidden) throughout all layers.
    memcpy([ctx->buf_moe_hidden contents], eng->hidden, H * sizeof(float));

    // Single command buffer for all layers — eliminates per-layer CB overhead
    id<MTLCommandBuffer> fwd_cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> fwd_enc = [fwd_cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    for (int layer = 0; layer < cfg->num_layers; layer++) {
        int n_experts = cfg->num_experts;
        int effective_k = thermal_k_effective(&eng->thermal, eng->active_experts);

        // --- LINEAR ATTENTION: fully fused GPU path ---
        // Projections + conv1d + QK norm + decay/beta + delta_net + gated_rms_norm
        // + O-proj + residual + post-norm + routing — all in ONE GPU commit.
        if (cfg->layer_types[layer] == ATTN_LINEAR) {
            t0 = now_ms();

            int total_key = cfg->linear_total_key;
            int conv_dim = cfg->linear_conv_dim;
            int n_v_heads = cfg->linear_num_v_heads;
            int key_dim = cfg->linear_key_dim;
            int value_dim = cfg->linear_value_dim;

            bool skip_copy_norm =
                (cfg->ffn_type == FFN_MOE && layer > 0 && ctx->moe_combine_copy_sq);
            id<MTLComputeCommandEncoder> enc = fwd_enc;
            if (!skip_copy_norm) {
                [enc setComputePipelineState:ctx->copy_buffer];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // --- Phase A: Input norm → buf_input ---
            if (skip_copy_norm) {
                // Use partial sums from previous layer's moe_combine_copy_sq
                uint num_tgs = ((uint)H + 255) / 256;
                [enc setComputePipelineState:ctx->norm_apply_partial];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:tcache[layer].input_norm.buffer
                   offset:tcache[layer].input_norm.offset atIndex:1];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                { uint d = (uint)H; float e = cfg->rms_norm_eps;
                  uint np = num_tgs;
                  [enc setBytes:&d length:sizeof(uint) atIndex:4];
                  [enc setBytes:&e length:sizeof(float) atIndex:5];
                  [enc setBytes:&np length:sizeof(uint) atIndex:6]; }
                [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc setComputePipelineState:ctx->norm_sum_sq];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
                { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:2]; }
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:ctx->norm_apply];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:tcache[layer].input_norm.buffer
                   offset:tcache[layer].input_norm.offset atIndex:1];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                { uint d = (uint)H; float e = cfg->rms_norm_eps;
                  [enc setBytes:&d length:sizeof(uint) atIndex:4];
                  [enc setBytes:&e length:sizeof(float) atIndex:5]; }
                [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // --- Phase B: 4 projections from buf_input ---
            // QKV → buf_conv_input, Z → buf_linear_output, alpha → buf_linear_decay, beta → buf_linear_beta
            { const LayerTensorCache *lt = &tcache[layer];
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->lin.qkv,
                    ctx->buf_input, 0, ctx->buf_conv_input, 0);
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->lin.z,
                    ctx->buf_input, 0, ctx->buf_linear_output, 0);
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->lin.a,
                    ctx->buf_input, 0, ctx->buf_linear_decay, 0);
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->lin.b,
                    ctx->buf_input, 0, ctx->buf_linear_beta, 0);
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase C: Conv1d step + decay/beta (concurrent) ---
            // Conv1d depends on buf_conv_input (from QKV proj).
            // Decay/beta depends on buf_linear_decay/beta (from A/B projs).
            // Both are ready after the projection barrier. Schedule together
            // so decay_beta overlaps with conv1d in the concurrent encoder.
            [enc setComputePipelineState:ctx->conv1d_f32];
            [enc setBuffer:ctx->buf_conv_state[linear_idx] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_conv_input offset:0 atIndex:1];
            [enc setBuffer:tcache[layer].lin.conv.buffer
                   offset:tcache[layer].lin.conv.offset atIndex:2];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:3];
            { uint cd = (uint)conv_dim; [enc setBytes:&cd length:sizeof(uint) atIndex:4]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)conv_dim + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Decay/beta: reads A/B projection outputs, writes decay/beta gates
            [enc setComputePipelineState:ctx->decay_beta_f32];
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:1];
            [enc setBuffer:tcache[layer].lin.A_log.buffer
                   offset:tcache[layer].lin.A_log.offset atIndex:2];
            [enc setBuffer:tcache[layer].lin.dt_bias.buffer
                   offset:tcache[layer].lin.dt_bias.offset atIndex:3];
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:4];
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(((uint)n_v_heads + 63) / 64, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase D+F: GatedDeltaNet with fused QK RMS norm ---
            // QK norm is now computed inline within delta_net's shared memory loading.
            // Q/K from buf_conv_output (raw, pre-norm), V at offset 2*total_key
            { uint k_per_v = (uint)cfg->linear_v_heads_per_k;
              float inv_s = 1.0f / sqrtf((float)key_dim);
            [enc setComputePipelineState:ctx->delta_net];
            [enc setBuffer:ctx->buf_linear_state[linear_idx] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];  // q (raw)
            [enc setBuffer:ctx->buf_conv_output offset:total_key * sizeof(float) atIndex:2]; // k (raw)
            [enc setBuffer:ctx->buf_conv_output offset:2 * total_key * sizeof(float) atIndex:3]; // v
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:4];
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:5];
            [enc setBuffer:ctx->buf_linear_v offset:0 atIndex:6];  // output
            [enc setBytes:&k_per_v length:sizeof(uint) atIndex:7];
            [enc setBytes:&inv_s length:sizeof(float) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(n_v_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(value_dim, 1, 1)]; }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase G: Gated RMS norm ---
            // values=buf_linear_v, z=buf_linear_output (Z proj stored there), output→buf_linear_q (reuse)
            [enc setComputePipelineState:ctx->gated_rms_norm];
            [enc setBuffer:ctx->buf_linear_v offset:0 atIndex:0];       // values
            [enc setBuffer:ctx->buf_linear_output offset:0 atIndex:1];  // z
            [enc setBuffer:tcache[layer].lin.o_norm.buffer
                   offset:tcache[layer].lin.o_norm.offset
                  atIndex:2];
            [enc setBuffer:ctx->buf_linear_q offset:0 atIndex:3];       // output (reuse buf_linear_q)
            { uint vd = (uint)value_dim; float e = cfg->rms_norm_eps;
              [enc setBytes:&vd length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_v_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(value_dim, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase H: O projection (attn_out in buf_linear_q → buf_h_mid) ---
            format_dispatch_matvec(enc, ctx, (TensorRef *)&tcache[layer].lin.o,
                ctx->buf_linear_q, 0, ctx->buf_h_mid, 0);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // --- Phase I+J: Fused residual add + sum_sq → buf_moe_hidden ---
            { uint num_tgs = ((uint)H + 255) / 256;
            [enc setComputePipelineState:ctx->residual_add_sq];
            [enc setBuffer:ctx->buf_residual offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_h_mid offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:3];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:4]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Post-attention RMS norm with partial sums → buf_input
            [enc setComputePipelineState:ctx->norm_apply_partial];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:tcache[layer].post_norm.buffer
                   offset:tcache[layer].post_norm.offset atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              uint np = num_tgs;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5];
              [enc setBytes:&np length:sizeof(uint) atIndex:6]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }


            if (cfg->ffn_type == FFN_MOE) {
                // --- Phase K: MoE routing + shared expert projections ---
                { size_t sgg_off = n_experts * sizeof(float);
                    const LayerTensorCache *lt = &tcache[layer];
                    bool fuse_routing_shared_gate =
                        ctx->matvec_f32_pair
                        && lt->routing_gate.format == QFMT_F32
                        && lt->shared_expert_gate.format == QFMT_F32
                        && lt->routing_gate.out_dim == (uint32_t)n_experts
                        && lt->shared_expert_gate.out_dim == 1
                        && lt->routing_gate.in_dim == lt->shared_expert_gate.in_dim;
                    if (fuse_routing_shared_gate) {
                        uint od = (uint)lt->routing_gate.out_dim;
                        uint id_ = (uint)lt->routing_gate.in_dim;
                        NSUInteger num_tgs = (od + 1 + ENGINE_ROWS_PER_TG - 1) / ENGINE_ROWS_PER_TG;
                        [enc setComputePipelineState:ctx->matvec_f32_pair];
                        [enc setBuffer:lt->routing_gate.buffer offset:lt->routing_gate.offset atIndex:0];
                        [enc setBuffer:lt->shared_expert_gate.buffer offset:lt->shared_expert_gate.offset atIndex:1];
                        [enc setBuffer:ctx->buf_input offset:0 atIndex:2];
                        [enc setBuffer:ctx->buf_output offset:0 atIndex:3];
                        [enc setBytes:&od length:sizeof(uint) atIndex:4];
                        [enc setBytes:&id_ length:sizeof(uint) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(ENGINE_ROWS_PER_TG * 32, 1, 1)];
                    } else {
                        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->routing_gate,
                            ctx->buf_input, 0, ctx->buf_output, 0);
                        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->shared_expert_gate,
                            ctx->buf_input, 0, ctx->buf_output, sgg_off);
                    }
                    // GPU-resident expert forward
                    double ffn_t0 = now_ms();
                    encode_experts_gguf(enc, ctx, cfg,
                                        &eng->expert_layer_cache[layer],
                                        &tcache[layer], effective_k);
                    t1 = now_ms();
                    t_ffn_total += t1 - ffn_t0;
                    t_attn_total += t1 - t0;
                }
            } else {
                double ffn_t0 = now_ms();
                encode_dense_ffn(enc, ctx, cfg, &tcache[layer]);
                t1 = now_ms();
                t_ffn_total += t1 - ffn_t0;
                t_attn_total += t1 - t0;
            }

            linear_idx++;
            continue;
        }

        // --- FULL ATTENTION: fully fused GPU path ---
        // Input norm + QKV projections + QK weighted RMS norm + RoPE + KV cache write
        // + attention (scores/softmax/values) + O-proj + residual + post-norm + routing
        // — all in ONE GPU commit.
        if (cfg->layer_types[layer] == ATTN_FULL) {
            t0 = now_ms();

            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;
            int kv_dim = cfg->kv_dim;
            int seq_len = pos + 1;

            bool skip_copy_norm =
                (cfg->ffn_type == FFN_MOE && layer > 0 && ctx->moe_combine_copy_sq);
            id<MTLComputeCommandEncoder> enc = fwd_enc;
            if (!skip_copy_norm) {
                [enc setComputePipelineState:ctx->copy_buffer];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // --- Phase A: Input norm → buf_input ---
            if (skip_copy_norm) {
                // Use partial sums from previous layer's moe_combine_copy_sq
                uint num_tgs = ((uint)H + 255) / 256;
                [enc setComputePipelineState:ctx->norm_apply_partial];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:tcache[layer].input_norm.buffer
                   offset:tcache[layer].input_norm.offset atIndex:1];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                { uint d = (uint)H; float e = cfg->rms_norm_eps;
                  uint np = num_tgs;
                  [enc setBytes:&d length:sizeof(uint) atIndex:4];
                  [enc setBytes:&e length:sizeof(float) atIndex:5];
                  [enc setBytes:&np length:sizeof(uint) atIndex:6]; }
                [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                [enc setComputePipelineState:ctx->norm_sum_sq];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
                { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:2]; }
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:ctx->norm_apply];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:tcache[layer].input_norm.buffer
                   offset:tcache[layer].input_norm.offset atIndex:1];
                [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                { uint d = (uint)H; float e = cfg->rms_norm_eps;
                  [enc setBytes:&d length:sizeof(uint) atIndex:4];
                  [enc setBytes:&e length:sizeof(float) atIndex:5]; }
                [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            // --- Phase B: Q/K/V projections ---
            { const LayerTensorCache *lt = &tcache[layer];
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->full.q,
                    ctx->buf_input, 0, ctx->buf_attn_output, 0);
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->full.k,
                    ctx->buf_input, 0, ctx->buf_conv_output, 0);
                format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->full.v,
                    ctx->buf_input, 0, ctx->buf_conv_input, 0);
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Q+gate weights are de-interleaved at load time, so the matvec
            // output is already [Q_h0..h15, gate_h0..h15]. No runtime permutation needed.

            // --- Phase C+D: Fused QK RMS norm + RoPE ---
            // Q in buf_attn_output (16 heads × 256), K in buf_conv_output (2 heads × 256)
            [enc setComputePipelineState:ctx->rms_norm_qk_rope];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];  // Q
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];  // K
            [enc setBuffer:tcache[layer].full.q_norm.buffer
                   offset:tcache[layer].full.q_norm.offset atIndex:2];
            [enc setBuffer:tcache[layer].full.k_norm.buffer
                   offset:tcache[layer].full.k_norm.offset atIndex:3];
            { uint hd_val = (uint)hd, nq = (uint)n_heads, nkv = (uint)n_kv;
              float inv_s = 1.0f / sqrtf((float)hd);
              uint rd = (uint)cfg->rotary_dim;
              uint p = (uint)pos; float th = cfg->rope_theta;
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:4];
              [enc setBytes:&nq length:sizeof(uint) atIndex:5];
              [enc setBytes:&nkv length:sizeof(uint) atIndex:6];
              [enc setBytes:&inv_s length:sizeof(float) atIndex:7];
              [enc setBytes:&rd length:sizeof(uint) atIndex:8];
              [enc setBytes:&p length:sizeof(uint) atIndex:9];
              [enc setBytes:&th length:sizeof(float) atIndex:10]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase E: KV cache write ---
            [enc setComputePipelineState:ctx->kv_cache_write];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:0];  // K
            [enc setBuffer:ctx->buf_conv_input offset:0 atIndex:1];   // V
            [enc setBuffer:ctx->buf_kv_k[full_idx] offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_kv_v[full_idx] offset:0 atIndex:3];
            { uint kvd = (uint)kv_dim, p = (uint)pos;
              [enc setBytes:&kvd length:sizeof(uint) atIndex:4];
              [enc setBytes:&p length:sizeof(uint) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)kv_dim + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase F: Attention scores (Q @ K^T) ---
            [enc setComputePipelineState:ctx->attn_scores];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];   // Q
            [enc setBuffer:ctx->buf_kv_k[full_idx] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:2];
            { uint hd_val = (uint)hd, kvd = (uint)kv_dim;
              uint sl = (uint)seq_len, ss = (uint)ctx->kv_cache_seq;
              float scale = 1.0f;  // already scaled in QK norm
              uint hpk = (uint)cfg->q_heads_per_kv;
              uint nst = (uint)seq_len;
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:3];
              [enc setBytes:&kvd length:sizeof(uint) atIndex:4];
              [enc setBytes:&sl length:sizeof(uint) atIndex:5];
              [enc setBytes:&ss length:sizeof(uint) atIndex:6];
              [enc setBytes:&scale length:sizeof(float) atIndex:7];
              [enc setBytes:&hpk length:sizeof(uint) atIndex:8];
              [enc setBytes:&nst length:sizeof(uint) atIndex:9]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_heads * seq_len, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase G: Softmax ---
            [enc setComputePipelineState:ctx->attn_softmax];
            [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:0];
            { uint sl = (uint)seq_len, ss = (uint)ctx->kv_cache_seq;
              [enc setBytes:&sl length:sizeof(uint) atIndex:1];
              [enc setBytes:&ss length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase H: Attention values (scores @ V) + fused sigmoid gate ---
            // Output → buf_attn_output (reuse Q buffer, now overwritten with gated attn output)
            [enc setComputePipelineState:ctx->attn_values];
            [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_kv_v[full_idx] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:2];
            { uint hd_val = (uint)hd, kvd = (uint)kv_dim;
              uint sl = (uint)seq_len, ss = (uint)ctx->kv_cache_seq;
              uint hpk = (uint)cfg->q_heads_per_kv;
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:3];
              [enc setBytes:&kvd length:sizeof(uint) atIndex:4];
              [enc setBytes:&sl length:sizeof(uint) atIndex:5];
              [enc setBytes:&ss length:sizeof(uint) atIndex:6];
              [enc setBytes:&hpk length:sizeof(uint) atIndex:7]; }
            [enc setBuffer:ctx->buf_attn_output offset:n_heads * hd * sizeof(float) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(((uint)(n_heads * hd) + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase I: O-proj (buf_attn_output → buf_h_mid) ---
            format_dispatch_matvec(enc, ctx, (TensorRef *)&tcache[layer].full.o,
                ctx->buf_attn_output, 0, ctx->buf_h_mid, 0);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase J+K: Fused residual add + sum_sq → buf_moe_hidden ---
            { uint num_tgs = ((uint)H + 255) / 256;
            [enc setComputePipelineState:ctx->residual_add_sq];
            [enc setBuffer:ctx->buf_residual offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_h_mid offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:3];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:4]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Post-attention RMS norm with partial sums → buf_input
            [enc setComputePipelineState:ctx->norm_apply_partial];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:tcache[layer].post_norm.buffer
                   offset:tcache[layer].post_norm.offset atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              uint np = num_tgs;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5];
              [enc setBytes:&np length:sizeof(uint) atIndex:6]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }


            if (cfg->ffn_type == FFN_MOE) {
                // --- Phase L: MoE routing + shared expert projections ---
                { size_t sgg_off = n_experts * sizeof(float);
                    const LayerTensorCache *lt = &tcache[layer];
                    bool fuse_routing_shared_gate =
                        ctx->matvec_f32_pair
                        && lt->routing_gate.format == QFMT_F32
                        && lt->shared_expert_gate.format == QFMT_F32
                        && lt->routing_gate.out_dim == (uint32_t)n_experts
                        && lt->shared_expert_gate.out_dim == 1
                        && lt->routing_gate.in_dim == lt->shared_expert_gate.in_dim;
                    if (fuse_routing_shared_gate) {
                        uint od = (uint)lt->routing_gate.out_dim;
                        uint id_ = (uint)lt->routing_gate.in_dim;
                        NSUInteger num_tgs = (od + 1 + ENGINE_ROWS_PER_TG - 1) / ENGINE_ROWS_PER_TG;
                        [enc setComputePipelineState:ctx->matvec_f32_pair];
                        [enc setBuffer:lt->routing_gate.buffer offset:lt->routing_gate.offset atIndex:0];
                        [enc setBuffer:lt->shared_expert_gate.buffer offset:lt->shared_expert_gate.offset atIndex:1];
                        [enc setBuffer:ctx->buf_input offset:0 atIndex:2];
                        [enc setBuffer:ctx->buf_output offset:0 atIndex:3];
                        [enc setBytes:&od length:sizeof(uint) atIndex:4];
                        [enc setBytes:&id_ length:sizeof(uint) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(ENGINE_ROWS_PER_TG * 32, 1, 1)];
                    } else {
                        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->routing_gate,
                            ctx->buf_input, 0, ctx->buf_output, 0);
                        format_dispatch_matvec(enc, ctx, (TensorRef *)&lt->shared_expert_gate,
                            ctx->buf_input, 0, ctx->buf_output, sgg_off);
                    }
                    // GPU-resident expert forward
                    double ffn_t0 = now_ms();
                    encode_experts_gguf(enc, ctx, cfg,
                                        &eng->expert_layer_cache[layer],
                                        &tcache[layer], effective_k);
                    t1 = now_ms();
                    t_ffn_total += t1 - ffn_t0;
                    t_attn_total += t1 - t0;
                }
            } else {
                double ffn_t0 = now_ms();
                encode_dense_ffn(enc, ctx, cfg, &tcache[layer]);
                t1 = now_ms();
                t_ffn_total += t1 - ffn_t0;
                t_attn_total += t1 - t0;
            }

            full_idx++;
            continue;
        }

    }

    // Sync GPU + final norm + LM head
    t0 = now_ms();
    {
        id<MTLComputeCommandEncoder> enc = fwd_enc;

        // RMS norm: buf_moe_hidden → buf_input
        if (ctx->moe_combine_copy_sq) {
            uint num_tgs = ((uint)H + 255) / 256;
            [enc setComputePipelineState:ctx->norm_apply_partial];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:eng->globals.final_norm.buffer
                   offset:eng->globals.final_norm.offset
                  atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              uint np = num_tgs;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5];
              [enc setBytes:&np length:sizeof(uint) atIndex:6]; }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            [enc setComputePipelineState:ctx->norm_sum_sq];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->norm_apply];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:eng->globals.final_norm.buffer
                   offset:eng->globals.final_norm.offset
                  atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // LM head matvec: buf_input → buf_output
        format_dispatch_matvec(enc, ctx, (TensorRef *)&eng->globals.lm_head,
            ctx->buf_input, 0, ctx->buf_output, 0);

        // GPU argmax
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {   [enc setComputePipelineState:ctx->argmax];
            [enc setBuffer:ctx->buf_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_argmax_result offset:0 atIndex:1];
            { uint v = (uint)cfg->vocab_size;
              [enc setBytes:&v length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        [fwd_cmd commit];
        [fwd_cmd waitUntilCompleted];

    }
    t1 = now_ms();
    t_lmhead_total += t1 - t0;

    // 5. Sample (GPU argmax — read 4-byte result)
    int next_token = (int)(*(uint32_t *)[ctx->buf_argmax_result contents]);

    eng->pos++;
    profile_count++;

    // Print profile every 10 tokens
    if (profile_count % 10 == 0) {
        double inv = 1.0 / profile_count;
        double attn_only = t_attn_total - t_ffn_total;
        fprintf(stderr, "[profile] avg/tok: attn=%.1fms ffn=%.1fms lmhead=%.1fms\n",
                attn_only * inv, t_ffn_total * inv, t_lmhead_total * inv);
    }

    double step_ms = now_ms() - step_start;
    thermal_k_record(&eng->thermal, step_ms);

    return next_token;
}
