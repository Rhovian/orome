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

// ============================================================================
// Engine lifecycle
// ============================================================================

Engine *engine_create(ModelConfig *cfg, WeightFile *wf, MetalCtx *ctx,
                      ExpertFiles *ef, QuantType quant, int active_experts) {
    Engine *eng = calloc(1, sizeof(Engine));
    eng->cfg = cfg;
    eng->wf = wf;
    eng->ctx = ctx;
    eng->ef = ef;
    eng->quant = quant;
    eng->active_experts = active_experts > 0 ? active_experts : cfg->num_experts_per_tok;

    int H = cfg->hidden_dim;
    eng->hidden   = calloc(H, sizeof(float));
    eng->residual = calloc(H, sizeof(float));
    eng->h_post   = calloc(H, sizeof(float));
    eng->logits   = calloc(cfg->vocab_size, sizeof(float));

    // KV caches (full attention layers)
    eng->kv_caches = calloc(cfg->num_full_attn_layers, sizeof(KVCache *));
    for (int i = 0; i < cfg->num_full_attn_layers; i++) {
        eng->kv_caches[i] = kv_cache_new(cfg);
    }

    // Linear attention states
    eng->linear_states = calloc(cfg->num_linear_layers, sizeof(LinearAttnState *));
    for (int i = 0; i < cfg->num_linear_layers; i++) {
        eng->linear_states[i] = linear_state_new(cfg);
    }

    eng->pos = 0;
    return eng;
}

void engine_free(Engine *eng) {
    if (!eng) return;
    free(eng->hidden);
    free(eng->residual);
    free(eng->h_post);
    free(eng->logits);

    if (eng->kv_caches) {
        for (int i = 0; i < eng->cfg->num_full_attn_layers; i++) {
            kv_cache_free(eng->kv_caches[i]);
        }
        free(eng->kv_caches);
    }
    if (eng->linear_states) {
        for (int i = 0; i < eng->cfg->num_linear_layers; i++) {
            linear_state_free(eng->linear_states[i]);
        }
        free(eng->linear_states);
    }
    free(eng);
}

void engine_reset(Engine *eng) {
    eng->pos = 0;
    for (int i = 0; i < eng->cfg->num_full_attn_layers; i++) {
        eng->kv_caches[i]->len = 0;
        int kv_size = OROME_GPU_KV_SEQ * eng->cfg->kv_dim;
        memset(eng->kv_caches[i]->k_cache, 0, kv_size * sizeof(float));
        memset(eng->kv_caches[i]->v_cache, 0, kv_size * sizeof(float));
    }
    for (int i = 0; i < eng->cfg->num_linear_layers; i++) {
        int conv_size = (eng->cfg->conv_kernel_size - 1) * eng->cfg->linear_conv_dim;
        int ssm_size = eng->cfg->linear_num_v_heads * eng->cfg->linear_key_dim
                       * eng->cfg->linear_value_dim;
        memset(eng->linear_states[i]->conv_state, 0, conv_size * sizeof(float));
        memset(eng->linear_states[i]->ssm_state, 0, ssm_size * sizeof(float));
    }
}

// ============================================================================
// Embedding lookup (4-bit dequantized)
// ============================================================================

static void embed_lookup(WeightFile *wf, const ModelConfig *cfg,
                         int token_id, float *out) {
    TensorInfo *w_info = weights_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = weights_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = weights_tensor_info(wf, "model.embed_tokens.biases");
    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embed_tokens tensors not found\n");
        return;
    }

    int H = cfg->hidden_dim;
    int G = cfg->group_size;
    int packed_cols = H / 8;
    int num_groups = H / G;

    uint32_t *W = (uint32_t *)((uint8_t *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((uint8_t *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((uint8_t *)wf->data + b_info->offset);

    uint32_t *row_w = W + token_id * packed_cols;
    uint16_t *row_s = S + token_id * num_groups;
    uint16_t *row_b = B + token_id * num_groups;

    for (int col = 0; col < packed_cols; col++) {
        int g = col / (G / 8);
        float scale = bf16_to_f32(row_s[g]);
        float bias  = bf16_to_f32(row_b[g]);
        uint32_t packed = row_w[col];
        int x_base = col * 8;
        for (int b = 0; b < 8; b++) {
            float nibble = (float)((packed >> (b * 4)) & 0xF);
            out[x_base + b] = nibble * scale + bias;
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                             float *hidden, float *logits) {
    uint32_t *w = weights_tensor_ptr(wf, "lm_head.weight");
    uint16_t *s = weights_tensor_ptr(wf, "lm_head.scales");
    uint16_t *b = weights_tensor_ptr(wf, "lm_head.biases");
    if (!w || !s || !b) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }
    fast_dequant_matvec(ctx, cfg, w, s, b, hidden, logits,
                        cfg->vocab_size, cfg->hidden_dim, QUANT_4BIT);
}

// ============================================================================
// Forward pass: one token
// ============================================================================

// Profiling accumulators (reset externally if needed)
static double t_attn_total = 0, t_moe_total = 0, t_norm_total = 0, t_lmhead_total = 0;
static int profile_count = 0;

int engine_step(Engine *eng, int token_id) {
    ModelConfig *cfg = eng->cfg;
    int H = cfg->hidden_dim;
    int pos = eng->pos;

    // 1. Embedding
    memset(eng->hidden, 0, H * sizeof(float));
    embed_lookup(eng->wf, cfg, token_id, eng->hidden);

    // 2. Layer loop — fused O-proj + post-norm + MoE routing in one GPU commit
    int full_idx = 0, linear_idx = 0;
    double t0, t1;
    MetalCtx *ctx = eng->ctx;

    // Static scratch for gate scores readback
    static float *s_fused_gate_scores = NULL;
    static int s_fused_gate_alloc = 0;
    if (s_fused_gate_alloc < cfg->num_experts) {
        free(s_fused_gate_scores);
        s_fused_gate_scores = calloc(cfg->num_experts, sizeof(float));
        s_fused_gate_alloc = cfg->num_experts;
    }

    for (int layer = 0; layer < cfg->num_layers; layer++) {
        // Save pre-attention hidden state for residual
        memcpy(eng->residual, eng->hidden, H * sizeof(float));

        int n_experts = cfg->num_experts;
        int S = cfg->shared_intermediate;

        // --- LINEAR ATTENTION: fully fused GPU path ---
        // Projections + conv1d + QK norm + decay/beta + delta_net + gated_rms_norm
        // + O-proj + residual + post-norm + routing — all in ONE GPU commit.
        if (cfg->layer_types[layer] == ATTN_LINEAR && ctx && ctx->buf_weights) {
            t0 = now_ms();

            uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
            int total_key = cfg->linear_total_key;
            int total_value = cfg->linear_total_value;
            int conv_dim = cfg->linear_conv_dim;
            int n_v_heads = cfg->linear_num_v_heads;
            int n_k_heads = cfg->linear_num_k_heads;
            int key_dim = cfg->linear_key_dim;
            int value_dim = cfg->linear_value_dim;

            // Look up all weight pointers
            uint16_t *input_norm_w = weights_layer_ptr(eng->wf, layer, "input_layernorm.weight");
            uint32_t *qkv_w = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_qkv.weight");
            uint16_t *qkv_s = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_qkv.scales");
            uint16_t *qkv_b = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_qkv.biases");
            uint32_t *z_w = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_z.weight");
            uint16_t *z_s = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_z.scales");
            uint16_t *z_b = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_z.biases");
            uint32_t *a_w = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_a.weight");
            uint16_t *a_s = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_a.scales");
            uint16_t *a_b = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_a.biases");
            uint32_t *b_w = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_b.weight");
            uint16_t *b_s = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_b.scales");
            uint16_t *b_b = weights_layer_ptr(eng->wf, layer, "linear_attn.in_proj_b.biases");
            uint16_t *conv_w = weights_layer_ptr(eng->wf, layer, "linear_attn.conv1d.weight");
            float *A_log = weights_layer_ptr(eng->wf, layer, "linear_attn.A_log");
            uint16_t *dt_bias = weights_layer_ptr(eng->wf, layer, "linear_attn.dt_bias");
            uint16_t *o_norm_w = weights_layer_ptr(eng->wf, layer, "linear_attn.norm.weight");
            uint32_t *o_w = weights_layer_ptr(eng->wf, layer, "linear_attn.out_proj.weight");
            uint16_t *o_s = weights_layer_ptr(eng->wf, layer, "linear_attn.out_proj.scales");
            uint16_t *o_b = weights_layer_ptr(eng->wf, layer, "linear_attn.out_proj.biases");
            uint16_t *post_norm_w = weights_layer_ptr(eng->wf, layer, "post_attention_layernorm.weight");
            uint32_t *gate_w = weights_layer_ptr(eng->wf, layer, "mlp.gate.weight");
            uint16_t *gate_s = weights_layer_ptr(eng->wf, layer, "mlp.gate.scales");
            uint16_t *gate_b = weights_layer_ptr(eng->wf, layer, "mlp.gate.biases");
            uint32_t *sg_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.weight");
            uint16_t *sg_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.scales");
            uint16_t *sg_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.biases");
            uint32_t *su_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.weight");
            uint16_t *su_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.scales");
            uint16_t *su_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.biases");
            uint32_t *sgg_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.weight");
            uint16_t *sgg_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.scales");
            uint16_t *sgg_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.biases");

            // Copy hidden to GPU
            memcpy([ctx->buf_moe_hidden contents], eng->hidden, H * sizeof(float));
            memcpy([ctx->buf_residual contents], eng->residual, H * sizeof(float));

            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // --- Phase A: Input norm → buf_input ---
            [enc setComputePipelineState:ctx->norm_sum_sq];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->norm_apply];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)input_norm_w - base atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase B: 4 projections from buf_input ---
            // QKV → buf_conv_input, Z → buf_linear_output, alpha → buf_linear_decay, beta → buf_linear_beta
            // (reuse buf_linear_output for Z since delta_net hasn't run yet)
            GpuMatvecJob proj_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)qkv_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)qkv_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)qkv_b - base,
                  .out_buf = ctx->buf_conv_input, .out_off = 0,
                  .out_ptr = NULL, .out_dim = conv_dim, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)z_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)z_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)z_b - base,
                  .out_buf = ctx->buf_linear_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = total_value, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)a_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)a_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)a_b - base,
                  .out_buf = ctx->buf_linear_decay, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_v_heads, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)b_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)b_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)b_b - base,
                  .out_buf = ctx->buf_linear_beta, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_v_heads, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 4; j++)
                gpu_encode_matvec_job(enc, ctx, &proj_jobs[j]);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase C: Conv1d step ---
            // buf_conv_input → buf_conv_output, updates buf_conv_state[linear_idx]
            [enc setComputePipelineState:ctx->conv1d];
            [enc setBuffer:ctx->buf_conv_state[linear_idx] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_conv_input offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)conv_w - base atIndex:2];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:3];
            { uint cd = (uint)conv_dim; [enc setBytes:&cd length:sizeof(uint) atIndex:4]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)conv_dim + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase D: QK RMS norm (in-place on buf_conv_output) ---
            // Q at offset 0, K at offset total_key * sizeof(float)
            [enc setComputePipelineState:ctx->rms_norm_qk];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_conv_output offset:total_key * sizeof(float) atIndex:1];
            { uint kd = (uint)key_dim;
              float inv_s = 1.0f / sqrtf((float)key_dim);
              [enc setBytes:&kd length:sizeof(uint) atIndex:2];
              [enc setBytes:&inv_s length:sizeof(float) atIndex:3]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_k_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(key_dim, 1, 1)];

            // --- Phase E: Compute decay + beta gate ---
            // alpha from buf_linear_decay (reused), beta from buf_linear_beta
            // A_log and dt_bias from weights
            [enc setComputePipelineState:ctx->decay_beta];
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)A_log - base atIndex:2];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)dt_bias - base atIndex:3];
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:4];  // output overwrites
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:5];   // output overwrites
            [enc dispatchThreadgroups:MTLSizeMake(((uint)n_v_heads + 63) / 64, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase F: GatedDeltaNet recurrence ---
            // Q/K from buf_conv_output, V at offset 2*total_key
            // state: buf_linear_state[linear_idx], output → buf_linear_v (reuse)
            { uint k_per_v = (uint)(n_v_heads / n_k_heads);
            [enc setComputePipelineState:ctx->delta_net];
            [enc setBuffer:ctx->buf_linear_state[linear_idx] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];  // q
            [enc setBuffer:ctx->buf_conv_output offset:total_key * sizeof(float) atIndex:2]; // k
            [enc setBuffer:ctx->buf_conv_output offset:2 * total_key * sizeof(float) atIndex:3]; // v
            [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:4];
            [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:5];
            [enc setBuffer:ctx->buf_linear_v offset:0 atIndex:6];  // output
            [enc setBytes:&k_per_v length:sizeof(uint) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_v_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(value_dim, 1, 1)]; }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase G: Gated RMS norm ---
            // values=buf_linear_v, z=buf_linear_output (Z proj stored there), output→buf_linear_q (reuse)
            [enc setComputePipelineState:ctx->gated_rms_norm];
            [enc setBuffer:ctx->buf_linear_v offset:0 atIndex:0];       // values
            [enc setBuffer:ctx->buf_linear_output offset:0 atIndex:1];  // z
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)o_norm_w - base atIndex:2];
            [enc setBuffer:ctx->buf_linear_q offset:0 atIndex:3];       // output (reuse buf_linear_q)
            { uint vd = (uint)value_dim; float e = cfg->rms_norm_eps;
              [enc setBytes:&vd length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_v_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(value_dim, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase H: O projection (attn_out in buf_linear_q → buf_h_mid) ---
            GpuMatvecJob o_job = {
                .w_buf = ctx->buf_weights, .w_off = (uint8_t *)o_w - base,
                .s_buf = ctx->buf_weights, .s_off = (uint8_t *)o_s - base,
                .b_buf = ctx->buf_weights, .b_off = (uint8_t *)o_b - base,
                .in_buf = ctx->buf_linear_q, .in_off = 0,
                .out_buf = ctx->buf_h_mid, .out_off = 0,
                .out_ptr = NULL, .out_dim = H, .in_dim = total_value,
                .group_size = cfg->group_size, .is_2bit = false
            };
            gpu_encode_matvec_job(enc, ctx, &o_job);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase I: Residual add → buf_moe_hidden ---
            [enc setComputePipelineState:ctx->residual_add];
            [enc setBuffer:ctx->buf_residual offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_h_mid offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:3]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase J: Post-attention RMS norm → buf_input ---
            [enc setComputePipelineState:ctx->norm_sum_sq];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
            { uint d = (uint)H; [enc setBytes:&d length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->norm_apply];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)post_norm_w - base atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            { uint d = (uint)H; float e = cfg->rms_norm_eps;
              [enc setBytes:&d length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase K: MoE routing + shared expert projections ---
            size_t sgg_off = n_experts * sizeof(float);
            GpuMatvecJob routing_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)gate_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)gate_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)gate_b - base,
                  .out_buf = ctx->buf_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_experts, .in_dim = H,
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
                  .out_ptr = NULL, .out_dim = 1, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 4; j++)
                gpu_encode_matvec_job(enc, ctx, &routing_jobs[j]);

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Read back routing scores and hidden
            memcpy(s_fused_gate_scores, [ctx->buf_output contents],
                   n_experts * sizeof(float));
            float shared_gate_score;
            memcpy(&shared_gate_score,
                   (uint8_t *)[ctx->buf_output contents] + sgg_off,
                   sizeof(float));
            memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));
            memcpy(eng->residual, eng->hidden, H * sizeof(float));

            t1 = now_ms();
            t_attn_total += t1 - t0;

            // MoE expert forward (separate commit)
            t0 = now_ms();
            moe_forward_routed(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                               s_fused_gate_scores, shared_gate_score,
                               eng->ef, eng->active_experts, eng->quant);
            t1 = now_ms();
            t_moe_total += t1 - t0;

            linear_idx++;
            continue;
        }

        // --- FULL ATTENTION path (unchanged) ---
        float *attn_out = NULL;
        int attn_out_dim = 0;

        t0 = now_ms();
        if (cfg->layer_types[layer] == ATTN_FULL) {
            full_attention_forward(eng->wf, ctx, cfg, layer, pos,
                                   eng->hidden, eng->residual, eng->h_post,
                                   eng->kv_caches[full_idx],
                                   &attn_out, &attn_out_dim);
            full_idx++;
        } else {
            // CPU fallback for linear attention (no Metal)
            linear_attention_forward(eng->wf, ctx, cfg, layer, pos,
                                     eng->hidden, eng->residual, eng->h_post,
                                     eng->linear_states[linear_idx],
                                     &attn_out, &attn_out_dim);
            linear_idx++;
        }
        t1 = now_ms();
        t_attn_total += t1 - t0;

        // --- Fused O-proj + residual + post-norm + MoE routing ---
        t0 = now_ms();

        // Look up O projection weights
        const char *o_proj_name = "self_attn.o_proj";
        char wname[128], sname[128], bname[128];
        snprintf(wname, sizeof(wname), "%s.weight", o_proj_name);
        snprintf(sname, sizeof(sname), "%s.scales", o_proj_name);
        snprintf(bname, sizeof(bname), "%s.biases", o_proj_name);
        uint32_t *o_w = weights_layer_ptr(eng->wf, layer, wname);
        uint16_t *o_s = weights_layer_ptr(eng->wf, layer, sname);
        uint16_t *o_b = weights_layer_ptr(eng->wf, layer, bname);

        // Post-attention norm weight
        uint16_t *post_norm_w = weights_layer_ptr(eng->wf, layer,
                                                   "post_attention_layernorm.weight");

        // MoE routing + shared expert weights
        uint32_t *gate_w = weights_layer_ptr(eng->wf, layer, "mlp.gate.weight");
        uint16_t *gate_s = weights_layer_ptr(eng->wf, layer, "mlp.gate.scales");
        uint16_t *gate_b = weights_layer_ptr(eng->wf, layer, "mlp.gate.biases");
        uint32_t *sg_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.weight");
        uint16_t *sg_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.scales");
        uint16_t *sg_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.gate_proj.biases");
        uint32_t *su_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.weight");
        uint16_t *su_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.scales");
        uint16_t *su_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert.up_proj.biases");
        uint32_t *sgg_w = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.weight");
        uint16_t *sgg_s = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.scales");
        uint16_t *sgg_b = weights_layer_ptr(eng->wf, layer, "mlp.shared_expert_gate.biases");

        // Use fused GPU path if available
        if (ctx && ctx->buf_weights && gate_w && sg_w && su_w && sgg_w && o_w) {
            uint8_t *base = (uint8_t *)[ctx->buf_weights contents];

            // Copy attn output to buf_output (large enough for 8192+ floats)
            memcpy([ctx->buf_output contents], attn_out, attn_out_dim * sizeof(float));
            // Copy residual (pre-attention hidden) to buf_residual
            memcpy([ctx->buf_residual contents], eng->residual, H * sizeof(float));

            // Build fused command buffer
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // 1. O projection: buf_output(attn_out) → buf_h_mid
            GpuMatvecJob o_job = {
                .w_buf = ctx->buf_weights, .w_off = (uint8_t *)o_w - base,
                .s_buf = ctx->buf_weights, .s_off = (uint8_t *)o_s - base,
                .b_buf = ctx->buf_weights, .b_off = (uint8_t *)o_b - base,
                .in_buf = ctx->buf_output, .in_off = 0,
                .out_buf = ctx->buf_h_mid, .out_off = 0,
                .out_ptr = NULL, .out_dim = H, .in_dim = attn_out_dim,
                .group_size = cfg->group_size, .is_2bit = false
            };
            gpu_encode_matvec_job(enc, ctx, &o_job);

            // Barrier: O proj writes → residual_add reads
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 2. Residual add: buf_residual + buf_h_mid → buf_moe_hidden
            [enc setComputePipelineState:ctx->residual_add];
            [enc setBuffer:ctx->buf_residual offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_h_mid offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];
            { uint dim_val = (uint)H; [enc setBytes:&dim_val length:sizeof(uint) atIndex:3]; }
            NSUInteger res_tgs = ((uint)H + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(res_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Barrier: residual_add writes → rms_norm reads
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 3. RMS norm sum_sq: buf_moe_hidden → buf_sum_sq
            [enc setComputePipelineState:ctx->norm_sum_sq];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:1];
            { uint dim_val = (uint)H; [enc setBytes:&dim_val length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Barrier: sum_sq write → norm_apply reads
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 4. RMS norm apply: buf_moe_hidden + norm_w + sum_sq → buf_input
            [enc setComputePipelineState:ctx->norm_apply];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)post_norm_w - base atIndex:1];
            [enc setBuffer:ctx->buf_sum_sq offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            {
                uint dim_val = (uint)H;
                float eps_val = cfg->rms_norm_eps;
                [enc setBytes:&dim_val length:sizeof(uint) atIndex:4];
                [enc setBytes:&eps_val length:sizeof(float) atIndex:5];
            }
            NSUInteger norm_tgs = ((uint)H + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(norm_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Barrier: norm output → routing reads
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 5. MoE routing + shared expert projections (all read from buf_input)
            size_t sgg_off = n_experts * sizeof(float);
            GpuMatvecJob routing_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)gate_w - base,
                  .s_buf = ctx->buf_weights, .s_off = (uint8_t *)gate_s - base,
                  .b_buf = ctx->buf_weights, .b_off = (uint8_t *)gate_b - base,
                  .out_buf = ctx->buf_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_experts, .in_dim = H,
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
                  .out_ptr = NULL, .out_dim = 1, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 4; j++) {
                gpu_encode_matvec_job(enc, ctx, &routing_jobs[j]);
            }

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Read back: gate_scores, shared_gate_score, hidden
            memcpy(s_fused_gate_scores, [ctx->buf_output contents],
                   n_experts * sizeof(float));
            float shared_gate_score;
            memcpy(&shared_gate_score,
                   (uint8_t *)[ctx->buf_output contents] + sgg_off,
                   sizeof(float));
            // Read back hidden (updated with O proj + residual) from buf_moe_hidden
            memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));

            // Save pre-MoE state for residual (hidden is now updated)
            memcpy(eng->residual, eng->hidden, H * sizeof(float));

            // MoE expert forward with pre-computed routing
            // h_post is in buf_input on GPU, shared gate/up in their GPU buffers
            moe_forward_routed(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                               s_fused_gate_scores, shared_gate_score,
                               eng->ef, eng->active_experts, eng->quant);
        } else {
            // CPU fallback: original flow
            const char *o_proj_name_fallback = (cfg->layer_types[layer] == ATTN_FULL)
                ? "self_attn.o_proj" : "linear_attn.out_proj";
            char wn2[128], sn2[128], bn2[128];
            snprintf(wn2, sizeof(wn2), "%s.weight", o_proj_name_fallback);
            snprintf(sn2, sizeof(sn2), "%s.scales", o_proj_name_fallback);
            snprintf(bn2, sizeof(bn2), "%s.biases", o_proj_name_fallback);
            uint32_t *o_w2 = weights_layer_ptr(eng->wf, layer, wn2);
            uint16_t *o_s2 = weights_layer_ptr(eng->wf, layer, sn2);
            uint16_t *o_b2 = weights_layer_ptr(eng->wf, layer, bn2);
            fast_dequant_matvec(ctx, cfg, o_w2, o_s2, o_b2, attn_out, eng->h_post,
                                H, attn_out_dim, QUANT_4BIT);
            for (int i = 0; i < H; i++) eng->hidden[i] += eng->h_post[i];

            cpu_rms_norm(eng->hidden, post_norm_w, eng->h_post, H, cfg->rms_norm_eps);
            memcpy(eng->residual, eng->hidden, H * sizeof(float));
            moe_forward(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                         eng->ef, eng->active_experts, eng->quant);
        }

        t1 = now_ms();
        t_moe_total += t1 - t0;
    }

    // 3. Final norm
    uint16_t *final_norm_w = weights_tensor_ptr(eng->wf, "model.norm.weight");
    float normed[H];
    cpu_rms_norm(eng->hidden, final_norm_w, normed, H, cfg->rms_norm_eps);

    // 4. LM head
    t0 = now_ms();
    lm_head_forward(eng->wf, eng->ctx, cfg, normed, eng->logits);
    t1 = now_ms();
    t_lmhead_total += t1 - t0;

    // 5. Sample (greedy argmax)
    int next_token = cpu_argmax(eng->logits, cfg->vocab_size);

    eng->pos++;
    profile_count++;

    // Print profile every 10 tokens
    if (profile_count % 10 == 0) {
        double inv = 1.0 / profile_count;
        fprintf(stderr, "[profile] avg/tok: attn=%.1fms moe=%.1fms norm=%.2fms lmhead=%.1fms\n",
                t_attn_total * inv, t_moe_total * inv, t_norm_total * inv, t_lmhead_total * inv);
    }

    return next_token;
}
