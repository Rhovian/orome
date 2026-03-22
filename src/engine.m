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

int engine_step(Engine *eng, int token_id) {
    ModelConfig *cfg = eng->cfg;
    int H = cfg->hidden_dim;
    int pos = eng->pos;

    // 1. Embedding
    memset(eng->hidden, 0, H * sizeof(float));
    embed_lookup(eng->wf, cfg, token_id, eng->hidden);

    // 2. Layer loop
    int full_idx = 0, linear_idx = 0;
    for (int layer = 0; layer < cfg->num_layers; layer++) {
        // Save pre-attention hidden state for residual
        memcpy(eng->residual, eng->hidden, H * sizeof(float));

        if (cfg->layer_types[layer] == ATTN_FULL) {
            full_attention_forward(eng->wf, eng->ctx, cfg, layer, pos,
                                   eng->hidden, eng->residual, eng->h_post,
                                   eng->kv_caches[full_idx]);
            full_idx++;
        } else {
            linear_attention_forward(eng->wf, eng->ctx, cfg, layer, pos,
                                     eng->hidden, eng->residual, eng->h_post,
                                     eng->linear_states[linear_idx]);
            linear_idx++;
        }

        // Post-attention norm → h_post (input to MoE)
        uint16_t *post_norm_w = weights_layer_ptr(eng->wf, layer,
                                                   "post_attention_layernorm.weight");
        cpu_rms_norm(eng->hidden, post_norm_w, eng->h_post, H, cfg->rms_norm_eps);

        // Save pre-MoE state for residual
        memcpy(eng->residual, eng->hidden, H * sizeof(float));

        // MoE forward
        moe_forward(eng->wf, eng->ctx, cfg, layer, eng->hidden, eng->h_post,
                     eng->ef, eng->active_experts, eng->quant);
    }

    // 3. Final norm
    uint16_t *final_norm_w = weights_tensor_ptr(eng->wf, "model.norm.weight");
    float normed[H];
    cpu_rms_norm(eng->hidden, final_norm_w, normed, H, cfg->rms_norm_eps);

    // 4. LM head
    lm_head_forward(eng->wf, eng->ctx, cfg, normed, eng->logits);

    // 5. Sample (greedy argmax)
    int next_token = cpu_argmax(eng->logits, cfg->vocab_size);

    eng->pos++;
    return next_token;
}
