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
// Precomputed weight byte offsets — eliminates ~1200 hash lookups per token
// ============================================================================

// All offsets are relative to wf->data base pointer.
// Linear attention layers use .lin, full attention layers use .full.
typedef struct {
    size_t input_norm_w;
    // MoE routing + shared expert (common to both attn types)
    size_t gate_w, gate_s, gate_b;
    size_t sg_w, sg_s, sg_b;   // shared expert gate_proj
    size_t su_w, su_s, su_b;   // shared expert up_proj
    size_t sgg_w, sgg_s, sgg_b; // shared_expert_gate
    size_t sd_w, sd_s, sd_b;   // shared expert down_proj
    size_t post_norm_w;
    union {
        struct {
            size_t qkv_w, qkv_s, qkv_b;
            size_t z_w, z_s, z_b;
            size_t a_w, a_s, a_b;
            size_t b_w, b_s, b_b;
            size_t conv_w, A_log, dt_bias, o_norm_w;
            size_t o_w, o_s, o_b;
        } lin;
        struct {
            size_t q_w, q_s, q_b;
            size_t k_w, k_s, k_b;
            size_t v_w, v_s, v_b;
            size_t qnorm_w, knorm_w;
            size_t o_w, o_s, o_b;
        } full;
    };
} LayerWeightCache;

// GGUF weight cache builder — maps GGUF tensor names to offsets within the mmap'd buffer
static LayerWeightCache *build_weight_cache_gguf_impl(GGUFFile *gf, const ModelConfig *cfg,
                                                   size_t *out_embed_off, size_t *out_lmhead_off,
                                                   size_t *out_norm_off) {
    LayerWeightCache *cache = calloc(cfg->num_layers, sizeof(LayerWeightCache));

    // Helper: resolve a GGUF tensor to its absolute file offset
    #define GGUF_OFF(name) ({ \
        GGUFTensorInfo *_ti = gguf_find_tensor(gf, (name)); \
        _ti ? (gf->data_offset + _ti->offset) : 0; \
    })

    // Global tensors
    if (out_embed_off)  *out_embed_off  = GGUF_OFF("token_embd.weight");
    if (out_lmhead_off) *out_lmhead_off = GGUF_OFF("output.weight");
    if (out_norm_off)   *out_norm_off   = GGUF_OFF("output_norm.weight");

    char name[128];
    for (int i = 0; i < cfg->num_layers; i++) {
        LayerWeightCache *c = &cache[i];

        // Norms (F32, used directly — not matvec'd)
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        c->input_norm_w = GGUF_OFF(name);
        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
        c->post_norm_w = GGUF_OFF(name);

        // MoE routing gate (F32)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", i);
        c->gate_w = GGUF_OFF(name);
        c->gate_s = 0; c->gate_b = 0; // no separate scales for GGUF

        // Shared expert gate (scalar gate, F32)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", i);
        c->sgg_w = GGUF_OFF(name);
        c->sgg_s = 0; c->sgg_b = 0;

        // Shared expert projections
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", i);
        c->sg_w = GGUF_OFF(name);
        c->sg_s = 0; c->sg_b = 0;
        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", i);
        c->su_w = GGUF_OFF(name);
        c->su_s = 0; c->su_b = 0;
        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", i);
        c->sd_w = GGUF_OFF(name);
        c->sd_s = 0; c->sd_b = 0;

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            // Linear attention (GatedDeltaNet / SSM-style)
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
            c->lin.qkv_w = GGUF_OFF(name);
            c->lin.qkv_s = 0; c->lin.qkv_b = 0;

            // Z gate (attn_gate in GGUF = in_proj_z in our code)
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
            c->lin.z_w = GGUF_OFF(name);
            c->lin.z_s = 0; c->lin.z_b = 0;

            // SSM parameters
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", i);
            c->lin.a_w = GGUF_OFF(name);
            c->lin.a_s = 0; c->lin.a_b = 0;
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", i);
            c->lin.b_w = GGUF_OFF(name);
            c->lin.b_s = 0; c->lin.b_b = 0;

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            c->lin.conv_w = GGUF_OFF(name);
            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            c->lin.A_log = GGUF_OFF(name);
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            c->lin.dt_bias = GGUF_OFF(name);

            // Output norm (if present)
            // Note: GGUF may not have a separate o_norm for linear attention
            c->lin.o_norm_w = 0;

            // Output projection — GGUF may not have a separate one for GatedDeltaNet
            c->lin.o_w = 0; c->lin.o_s = 0; c->lin.o_b = 0;
        } else {
            // Full attention (GQA)
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            c->full.q_w = GGUF_OFF(name);
            c->full.q_s = 0; c->full.q_b = 0;
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            c->full.k_w = GGUF_OFF(name);
            c->full.k_s = 0; c->full.k_b = 0;
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            c->full.v_w = GGUF_OFF(name);
            c->full.v_s = 0; c->full.v_b = 0;
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
            c->full.qnorm_w = GGUF_OFF(name);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
            c->full.knorm_w = GGUF_OFF(name);
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            c->full.o_w = GGUF_OFF(name);
            c->full.o_s = 0; c->full.o_b = 0;
        }
    }
    #undef GGUF_OFF
    return cache;
}

// External entry point for main.m
void *build_weight_cache_gguf_ext(GGUFFile *gf, const ModelConfig *cfg) {
    size_t e, l, n;
    return build_weight_cache_gguf_impl(gf, cfg, &e, &l, &n);
}

static LayerWeightCache *build_weight_cache(WeightFile *wf, const ModelConfig *cfg) {
    LayerWeightCache *cache = calloc(cfg->num_layers, sizeof(LayerWeightCache));
    uint8_t *base = (uint8_t *)wf->data;
    for (int i = 0; i < cfg->num_layers; i++) {
        LayerWeightCache *c = &cache[i];
        c->input_norm_w = (uint8_t *)weights_layer_ptr(wf, i, "input_layernorm.weight") - base;
        c->post_norm_w = (uint8_t *)weights_layer_ptr(wf, i, "post_attention_layernorm.weight") - base;
        c->gate_w = (uint8_t *)weights_layer_ptr(wf, i, "mlp.gate.weight") - base;
        c->gate_s = (uint8_t *)weights_layer_ptr(wf, i, "mlp.gate.scales") - base;
        c->gate_b = (uint8_t *)weights_layer_ptr(wf, i, "mlp.gate.biases") - base;
        c->sg_w = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.gate_proj.weight") - base;
        c->sg_s = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.gate_proj.scales") - base;
        c->sg_b = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.gate_proj.biases") - base;
        c->su_w = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.up_proj.weight") - base;
        c->su_s = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.up_proj.scales") - base;
        c->su_b = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.up_proj.biases") - base;
        c->sgg_w = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert_gate.weight") - base;
        c->sgg_s = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert_gate.scales") - base;
        c->sgg_b = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert_gate.biases") - base;
        c->sd_w = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.down_proj.weight") - base;
        c->sd_s = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.down_proj.scales") - base;
        c->sd_b = (uint8_t *)weights_layer_ptr(wf, i, "mlp.shared_expert.down_proj.biases") - base;

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            c->lin.qkv_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_qkv.weight") - base;
            c->lin.qkv_s = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_qkv.scales") - base;
            c->lin.qkv_b = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_qkv.biases") - base;
            c->lin.z_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_z.weight") - base;
            c->lin.z_s = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_z.scales") - base;
            c->lin.z_b = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_z.biases") - base;
            c->lin.a_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_a.weight") - base;
            c->lin.a_s = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_a.scales") - base;
            c->lin.a_b = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_a.biases") - base;
            c->lin.b_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_b.weight") - base;
            c->lin.b_s = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_b.scales") - base;
            c->lin.b_b = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.in_proj_b.biases") - base;
            c->lin.conv_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.conv1d.weight") - base;
            c->lin.A_log = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.A_log") - base;
            c->lin.dt_bias = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.dt_bias") - base;
            c->lin.o_norm_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.norm.weight") - base;
            c->lin.o_w = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.out_proj.weight") - base;
            c->lin.o_s = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.out_proj.scales") - base;
            c->lin.o_b = (uint8_t *)weights_layer_ptr(wf, i, "linear_attn.out_proj.biases") - base;
        } else {
            c->full.q_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.q_proj.weight") - base;
            c->full.q_s = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.q_proj.scales") - base;
            c->full.q_b = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.q_proj.biases") - base;
            c->full.k_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.k_proj.weight") - base;
            c->full.k_s = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.k_proj.scales") - base;
            c->full.k_b = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.k_proj.biases") - base;
            c->full.v_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.v_proj.weight") - base;
            c->full.v_s = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.v_proj.scales") - base;
            c->full.v_b = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.v_proj.biases") - base;
            c->full.qnorm_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.q_norm.weight") - base;
            c->full.knorm_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.k_norm.weight") - base;
            c->full.o_w = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.o_proj.weight") - base;
            c->full.o_s = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.o_proj.scales") - base;
            c->full.o_b = (uint8_t *)weights_layer_ptr(wf, i, "self_attn.o_proj.biases") - base;
        }
    }
    return cache;
}

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

Engine *engine_create(ModelConfig *cfg, WeightFile *wf, MetalCtx *ctx,
                      ExpertFiles *ef, QuantType quant, int active_experts) {
    Engine *eng = calloc(1, sizeof(Engine));
    eng->cfg = cfg;
    eng->wf = wf;
    eng->ctx = ctx;
    eng->ef = ef;
    eng->quant = quant;
    eng->active_experts = active_experts > 0 ? active_experts : cfg->num_experts_per_tok;
    eng->thermal.enabled = false;
    eng->thermal.hot_k = 0;
    eng->thermal.min_gen = 16;
    eng->thermal.proj_threshold_ms = 85.0;
    thermal_k_reset(&eng->thermal);

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
    // Weight cache is built in engine_create for legacy format.
    // For GGUF, main.m sets eng->gf and we rebuild the cache below.
    if (wf->manifest) {
        eng->weight_cache = build_weight_cache(wf, cfg);
    }
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
    free(eng->weight_cache);
    free(eng);
}

// Profiling accumulators (reset in engine_reset)
static double t_attn_total = 0, t_moe_total = 0, t_norm_total = 0, t_lmhead_total = 0;
static int profile_count = 0;

void engine_reset(Engine *eng) {
    eng->pos = 0;
    thermal_k_reset(&eng->thermal);
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
    // Reset profile accumulators so they don't bleed across requests
    t_attn_total = 0; t_moe_total = 0; t_norm_total = 0; t_lmhead_total = 0;
    profile_count = 0;
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
// Fused expert encoding — GPU softmax+topk + dynamic expert matvecs
// ============================================================================
// Encodes softmax_topk + expert gate/up/swiglu/down + shared expert + combine
// into an already-open compute command encoder. Eliminates CPU readback.

#define ENGINE_ROWS_PER_TG 16  // must match ROWS_PER_TG in metal.m and shaders

static void encode_fused_experts(id<MTLComputeCommandEncoder> enc,
                                  MetalCtx *ctx, const ModelConfig *cfg,
                                  int layer, int K,
                                  QuantType quant,
                                  const LayerWeightCache *lw) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int S = cfg->shared_intermediate;
    int n_experts = cfg->num_experts;

    const ExpertLayout *layout = (quant == QUANT_2BIT) ? &cfg->expert_2bit : &cfg->expert_4bit;
    id<MTLBuffer> expert_layer_buf = ctx->buf_expert_layers[layer];

    NSUInteger tg_size = ENGINE_ROWS_PER_TG * 32;
    uint num_row_tgs_M = ((uint)M + ENGINE_ROWS_PER_TG - 1) / ENGINE_ROWS_PER_TG;
    uint num_row_tgs_H = ((uint)H + ENGINE_ROWS_PER_TG - 1) / ENGINE_ROWS_PER_TG;

    // --- Softmax + TopK on GPU ---
    // Routing logits are in buf_output[0..n_experts-1],
    // shared gate score at buf_output[n_experts]
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

    // --- Fused expert gate+up+SwiGLU (1 dispatch instead of 3) ---
    uint expert_sz = (uint)layout->expert_size;

    if (ctx->expert_gate_up_swiglu) {
        [enc setComputePipelineState:ctx->expert_gate_up_swiglu];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
        { uint es = expert_sz;
          uint gw = (uint)layout->gate_w_off, gs_ = (uint)layout->gate_s_off, gb = (uint)layout->gate_b_off;
          uint uw = (uint)layout->up_w_off, us = (uint)layout->up_s_off, ub = (uint)layout->up_b_off;
          uint od = (uint)M, id_ = (uint)H, gsize = (uint)cfg->group_size, nrt = num_row_tgs_M;
          [enc setBytes:&es length:sizeof(uint) atIndex:4];
          [enc setBytes:&gw length:sizeof(uint) atIndex:5];
          [enc setBytes:&gs_ length:sizeof(uint) atIndex:6];
          [enc setBytes:&gb length:sizeof(uint) atIndex:7];
          [enc setBytes:&uw length:sizeof(uint) atIndex:8];
          [enc setBytes:&us length:sizeof(uint) atIndex:9];
          [enc setBytes:&ub length:sizeof(uint) atIndex:10];
          [enc setBytes:&od length:sizeof(uint) atIndex:11];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:12];
          [enc setBytes:&gsize length:sizeof(uint) atIndex:13];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:14]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    } else {
        // Fallback: separate gate + up + swiglu
        [enc setComputePipelineState:ctx->batch_expert_mv_dyn];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
        { uint es = expert_sz;
          uint pw = (uint)layout->gate_w_off, ps = (uint)layout->gate_s_off, pb = (uint)layout->gate_b_off;
          uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size, nrt = num_row_tgs_M;
          [enc setBytes:&es length:sizeof(uint) atIndex:4];
          [enc setBytes:&pw length:sizeof(uint) atIndex:5];
          [enc setBytes:&ps length:sizeof(uint) atIndex:6];
          [enc setBytes:&pb length:sizeof(uint) atIndex:7];
          [enc setBytes:&od length:sizeof(uint) atIndex:8];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:9];
          [enc setBytes:&gs length:sizeof(uint) atIndex:10];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:11]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        [enc setComputePipelineState:ctx->batch_expert_mv_dyn];
        [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
        { uint es = expert_sz;
          uint pw = (uint)layout->up_w_off, ps = (uint)layout->up_s_off, pb = (uint)layout->up_b_off;
          uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size, nrt = num_row_tgs_M;
          [enc setBytes:&es length:sizeof(uint) atIndex:4];
          [enc setBytes:&pw length:sizeof(uint) atIndex:5];
          [enc setBytes:&ps length:sizeof(uint) atIndex:6];
          [enc setBytes:&pb length:sizeof(uint) atIndex:7];
          [enc setBytes:&od length:sizeof(uint) atIndex:8];
          [enc setBytes:&id_ length:sizeof(uint) atIndex:9];
          [enc setBytes:&gs length:sizeof(uint) atIndex:10];
          [enc setBytes:&nrt length:sizeof(uint) atIndex:11]; }
        [enc dispatchThreadgroups:MTLSizeMake(num_row_tgs_M * K, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:ctx->batch_swiglu];
        [enc setBuffer:ctx->buf_batch_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_batch_expert_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:2];
        { uint td = (uint)(K * M); [enc setBytes:&td length:sizeof(uint) atIndex:3]; }
        [enc dispatchThreadgroups:MTLSizeMake(((uint)(K * M) + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // --- Shared expert SwiGLU ---
    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
    { uint dim_val = (uint)S; [enc setBytes:&dim_val length:sizeof(uint) atIndex:3]; }
    [enc dispatchThreadgroups:MTLSizeMake(((uint)S + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // --- Expert down projections (dynamic, per-expert packed input) ---
    // Use 2-row kernel if available (halves TG count)
    { bool use_2row_down = (ctx->batch_expert_down_dyn_2row != nil);
      id<MTLComputePipelineState> down_pipe = use_2row_down ? ctx->batch_expert_down_dyn_2row : ctx->batch_expert_down_dyn;
      uint down_rows_per_tg = use_2row_down ? ENGINE_ROWS_PER_TG * 2 : ENGINE_ROWS_PER_TG;
      uint down_num_row_tgs = ((uint)H + down_rows_per_tg - 1) / down_rows_per_tg;
      [enc setComputePipelineState:down_pipe];
      [enc setBuffer:expert_layer_buf offset:0 atIndex:0];
      [enc setBuffer:ctx->buf_batch_expert_act offset:0 atIndex:1];
      [enc setBuffer:ctx->buf_batch_expert_out offset:0 atIndex:2];
      [enc setBuffer:ctx->buf_topk_indices offset:0 atIndex:3];
      { uint es = expert_sz;
        uint pw = (uint)layout->down_w_off, ps = (uint)layout->down_s_off, pb = (uint)layout->down_b_off;
        uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size, nrt = down_num_row_tgs;
        [enc setBytes:&es length:sizeof(uint) atIndex:4];
        [enc setBytes:&pw length:sizeof(uint) atIndex:5];
        [enc setBytes:&ps length:sizeof(uint) atIndex:6];
        [enc setBytes:&pb length:sizeof(uint) atIndex:7];
        [enc setBytes:&od length:sizeof(uint) atIndex:8];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:9];
        [enc setBytes:&gs length:sizeof(uint) atIndex:10];
        [enc setBytes:&nrt length:sizeof(uint) atIndex:11]; }
      [enc dispatchThreadgroups:MTLSizeMake(down_num_row_tgs * K, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)]; }

    // --- Shared expert down (use 2-row kernel if available) ---
    { id<MTLComputePipelineState> sd_pipe = ctx->matvec_4bit_2row ? ctx->matvec_4bit_2row : ctx->matvec_4bit;
      uint sd_rows_per_tg = ctx->matvec_4bit_2row ? ENGINE_ROWS_PER_TG * 2 : ENGINE_ROWS_PER_TG;
      uint sd_num_tgs = ((uint)H + sd_rows_per_tg - 1) / sd_rows_per_tg;
      [enc setComputePipelineState:sd_pipe];
      [enc setBuffer:ctx->buf_weights offset:lw->sd_w atIndex:0];
      [enc setBuffer:ctx->buf_weights offset:lw->sd_s atIndex:1];
      [enc setBuffer:ctx->buf_weights offset:lw->sd_b atIndex:2];
      [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
      [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
      { uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
        [enc setBytes:&od length:sizeof(uint) atIndex:5];
        [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
        [enc setBytes:&gs length:sizeof(uint) atIndex:7]; }
      [enc dispatchThreadgroups:MTLSizeMake(sd_num_tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)]; }

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // --- Combine: hidden += experts + shared + copy residual + partial sum_sq ---
    if (ctx->moe_combine_copy_sq) {
        [enc setComputePipelineState:ctx->moe_combine_copy_sq];
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
    } else {
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
    }
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

    // 1. Embedding
    memset(eng->hidden, 0, H * sizeof(float));
    embed_lookup(eng->wf, cfg, token_id, eng->hidden);

    // Debug: dump embedding
    if (moe_get_profile_experts() && pos < 3) {
        float hmin = 1e30f, hmax = -1e30f, hsum = 0;
        for (int i = 0; i < H; i++) {
            if (eng->hidden[i] < hmin) hmin = eng->hidden[i];
            if (eng->hidden[i] > hmax) hmax = eng->hidden[i];
            hsum += eng->hidden[i];
        }
        fprintf(stderr, "EMBED pos=%d token=%d h[0..3]=[%.4f,%.4f,%.4f,%.4f] min=%.4f max=%.4f mean=%.6f\n",
                pos, token_id, eng->hidden[0], eng->hidden[1], eng->hidden[2], eng->hidden[3],
                hmin, hmax, hsum / H);
    }

    // 2. Layer loop — fused O-proj + post-norm + MoE routing in one GPU commit
    int full_idx = 0, linear_idx = 0;
    double t0, t1;
    MetalCtx *ctx = eng->ctx;
    LayerWeightCache *wcache = (LayerWeightCache *)eng->weight_cache;

    // Static scratch for gate scores readback
    static float *s_fused_gate_scores = NULL;
    static int s_fused_gate_alloc = 0;
    if (s_fused_gate_alloc < cfg->num_experts) {
        free(s_fused_gate_scores);
        s_fused_gate_scores = calloc(cfg->num_experts, sizeof(float));
        s_fused_gate_alloc = cfg->num_experts;
    }

    // Upload hidden to GPU once before the layer loop.
    // Hidden state stays on GPU (in buf_moe_hidden) throughout all layers.
    bool gpu_resident = (ctx && ctx->buf_weights && ctx->moe_combine
                         && eng->ef->gpu_resident_safe);
    if (gpu_resident) {
        memcpy([ctx->buf_moe_hidden contents], eng->hidden, H * sizeof(float));
    }

    // Per-layer command buffers with compute copy (no blit encoder transition).
    id<MTLCommandBuffer> fwd_cmd = nil;
    id<MTLComputeCommandEncoder> fwd_enc = nil;
    // fwd_enc = nil means per-layer CBs will be used

    for (int layer = 0; layer < cfg->num_layers; layer++) {
        int n_experts = cfg->num_experts;
        int S = cfg->shared_intermediate;
        int effective_k = thermal_k_effective(&eng->thermal, eng->active_experts);
        // For non-GPU-resident path, save residual on CPU
        if (!gpu_resident) {
            memcpy(eng->residual, eng->hidden, H * sizeof(float));
        }

        // --- Debug: check hidden state entering each layer ---
        if (moe_get_profile_experts() && !gpu_resident && layer >= 50) {
            int nans = 0; float maxv = 0;
            for (int i = 0; i < H; i++) {
                if (eng->hidden[i] != eng->hidden[i]) nans++;
                float av = fabsf(eng->hidden[i]);
                if (av > maxv) maxv = av;
            }
            fprintf(stderr, "DIAG_HIDDEN_IN layer=%d nans=%d max=%.3e\n", layer, nans, maxv);
        }

        // --- LINEAR ATTENTION: fully fused GPU path ---
        // Projections + conv1d + QK norm + decay/beta + delta_net + gated_rms_norm
        // + O-proj + residual + post-norm + routing — all in ONE GPU commit.
        if (cfg->layer_types[layer] == ATTN_LINEAR && ctx && ctx->buf_weights) {
            t0 = now_ms();

            int total_key = cfg->linear_total_key;
            int total_value = cfg->linear_total_value;
            int conv_dim = cfg->linear_conv_dim;
            int n_v_heads = cfg->linear_num_v_heads;
            int n_k_heads = cfg->linear_num_k_heads;
            int key_dim = cfg->linear_key_dim;
            int value_dim = cfg->linear_value_dim;

            const LayerWeightCache *lw = &wcache[layer];

            // GPU-resident: copy buf_moe_hidden → buf_residual
            if (!gpu_resident) {
                memcpy([ctx->buf_moe_hidden contents], eng->hidden, H * sizeof(float));
                memcpy([ctx->buf_residual contents], eng->residual, H * sizeof(float));
            }

            // For layers > 0 with moe_combine_copy_sq: residual + sum_sq already computed
            bool skip_copy_norm = (layer > 0 && gpu_resident && ctx->moe_combine_copy_sq);

            id<MTLCommandBuffer> cmd = nil;
            id<MTLComputeCommandEncoder> enc = nil;
            if (gpu_resident && fwd_enc) {
                enc = fwd_enc;
                if (!skip_copy_norm) {
                    [enc setComputePipelineState:ctx->copy_buffer];
                    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                    { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                cmd = [ctx->queue commandBuffer];
                enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                if (gpu_resident && !skip_copy_norm) {
                    [enc setComputePipelineState:ctx->copy_buffer];
                    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                    { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // --- Phase A: Input norm → buf_input ---
            if (skip_copy_norm) {
                // Use partial sums from previous layer's moe_combine_copy_sq
                uint num_tgs = ((uint)H + 255) / 256;
                [enc setComputePipelineState:ctx->norm_apply_partial];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_weights offset:lw->input_norm_w atIndex:1];
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
                [enc setBuffer:ctx->buf_weights offset:lw->input_norm_w atIndex:1];
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
            // (reuse buf_linear_output for Z since delta_net hasn't run yet)
            GpuMatvecJob proj_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = lw->lin.qkv_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->lin.qkv_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->lin.qkv_b,
                  .out_buf = ctx->buf_conv_input, .out_off = 0,
                  .out_ptr = NULL, .out_dim = conv_dim, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->lin.z_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->lin.z_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->lin.z_b,
                  .out_buf = ctx->buf_linear_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = total_value, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->lin.a_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->lin.a_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->lin.a_b,
                  .out_buf = ctx->buf_linear_decay, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_v_heads, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->lin.b_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->lin.b_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->lin.b_b,
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
            [enc setBuffer:ctx->buf_weights offset:lw->lin.conv_w atIndex:2];
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
            [enc setBuffer:ctx->buf_weights offset:lw->lin.A_log atIndex:2];
            [enc setBuffer:ctx->buf_weights offset:lw->lin.dt_bias atIndex:3];
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
            [enc setBuffer:ctx->buf_weights offset:lw->lin.o_norm_w atIndex:2];
            [enc setBuffer:ctx->buf_linear_q offset:0 atIndex:3];       // output (reuse buf_linear_q)
            { uint vd = (uint)value_dim; float e = cfg->rms_norm_eps;
              [enc setBytes:&vd length:sizeof(uint) atIndex:4];
              [enc setBytes:&e length:sizeof(float) atIndex:5]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_v_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(value_dim, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase H: O projection (attn_out in buf_linear_q → buf_h_mid) ---
            { GpuMatvecJob o_job = {
                .w_buf = ctx->buf_weights, .w_off = lw->lin.o_w,
                .s_buf = ctx->buf_weights, .s_off = lw->lin.o_s,
                .b_buf = ctx->buf_weights, .b_off = lw->lin.o_b,
                .in_buf = ctx->buf_linear_q, .in_off = 0,
                .out_buf = ctx->buf_h_mid, .out_off = 0,
                .out_ptr = NULL, .out_dim = H, .in_dim = total_value,
                .group_size = cfg->group_size, .is_2bit = false
            };
            gpu_encode_matvec_job(enc, ctx, &o_job); }
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
            [enc setBuffer:ctx->buf_weights offset:lw->post_norm_w atIndex:1];
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


            // --- Phase K: MoE routing + shared expert projections ---
            size_t sgg_off = n_experts * sizeof(float);
            GpuMatvecJob routing_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = lw->gate_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->gate_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->gate_b,
                  .out_buf = ctx->buf_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_experts, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->sg_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->sg_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->sg_b,
                  .out_buf = ctx->buf_shared_gate, .out_off = 0,
                  .out_ptr = NULL, .out_dim = S, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->su_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->su_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->su_b,
                  .out_buf = ctx->buf_shared_up, .out_off = 0,
                  .out_ptr = NULL, .out_dim = S, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->sgg_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->sgg_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->sgg_b,
                  .out_buf = ctx->buf_output, .out_off = sgg_off,
                  .out_ptr = NULL, .out_dim = 1, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 4; j++)
                gpu_encode_matvec_job(enc, ctx, &routing_jobs[j]);

            // Per-layer fused path: use fused experts when layer has Metal buffer
            bool layer_fused = ctx->softmax_topk && ctx->batch_expert_mv_dyn
                && ctx->buf_expert_layers && ctx->buf_expert_layers[layer]
                && !moe_get_profile_experts();

            if (gpu_resident && layer_fused) {
                // Fully GPU-resident: no readback between layers
                encode_fused_experts(enc, ctx, cfg, layer,
                                     effective_k, eng->quant, lw);
                if (!fwd_enc) {
                    [enc endEncoding];
                    [cmd commit];
                }
            } else if (layer_fused) {
                // Hybrid fused: this resident layer has a direct Metal buffer
                // GPU does softmax+topk+experts — no CPU routing readback
                encode_fused_experts(enc, ctx, cfg, layer,
                                     effective_k, eng->quant, lw);
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                // Sync hidden back to CPU for next layer
                memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));
            } else {
                // Fallback: CPU routing readback + moe_forward_routed
                [enc endEncoding];
                if (cmd) [cmd commit];
                if (cmd) [cmd waitUntilCompleted];

                memcpy(s_fused_gate_scores, [ctx->buf_output contents],
                       n_experts * sizeof(float));
                float shared_gate_score;
                memcpy(&shared_gate_score,
                       (uint8_t *)[ctx->buf_output contents] + sgg_off,
                       sizeof(float));

                if (!gpu_resident) {
                    memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));
                    memcpy(eng->residual, eng->hidden, H * sizeof(float));
                }

                double t0_moe = now_ms();
                moe_forward_routed(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                                   s_fused_gate_scores, shared_gate_score,
                                   eng->ef, effective_k, eng->quant,
                                   gpu_resident);
                double moe_ms = now_ms() - t0_moe;
                t_moe_total += moe_ms;
            }

            t1 = now_ms();
            t_attn_total += t1 - t0;
            // Note: t_attn_total includes moe time; true attn = t_attn_total - t_moe_total

            linear_idx++;
            continue;
        }

        // --- FULL ATTENTION: fully fused GPU path ---
        // Input norm + QKV projections + QK weighted RMS norm + RoPE + KV cache write
        // + attention (scores/softmax/values) + O-proj + residual + post-norm + routing
        // — all in ONE GPU commit.
        if (cfg->layer_types[layer] == ATTN_FULL && ctx && ctx->buf_weights
            && ctx->rms_norm_qk_w && ctx->rope_apply && ctx->kv_cache_write) {
            t0 = now_ms();

            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;
            int kv_dim = cfg->kv_dim;
            int seq_len = pos + 1;

            const LayerWeightCache *lw = &wcache[layer];

            // GPU-resident: copy buf_moe_hidden → buf_residual
            if (!gpu_resident) {
                memcpy([ctx->buf_moe_hidden contents], eng->hidden, H * sizeof(float));
                memcpy([ctx->buf_residual contents], eng->residual, H * sizeof(float));
            }

            // For layers > 0 with moe_combine_copy_sq: residual + sum_sq already computed
            bool skip_copy_norm = (layer > 0 && gpu_resident && ctx->moe_combine_copy_sq);

            id<MTLCommandBuffer> cmd = nil;
            id<MTLComputeCommandEncoder> enc = nil;
            if (gpu_resident && fwd_enc) {
                enc = fwd_enc;
                if (!skip_copy_norm) {
                    [enc setComputePipelineState:ctx->copy_buffer];
                    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                    { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            } else {
                cmd = [ctx->queue commandBuffer];
                enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                if (gpu_resident && !skip_copy_norm) {
                    [enc setComputePipelineState:ctx->copy_buffer];
                    [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_residual offset:0 atIndex:1];
                    { uint c = (uint)H; [enc setBytes:&c length:sizeof(uint) atIndex:2]; }
                    [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }

            // --- Phase A: Input norm → buf_input ---
            if (skip_copy_norm) {
                // Use partial sums from previous layer's moe_combine_copy_sq
                uint num_tgs = ((uint)H + 255) / 256;
                [enc setComputePipelineState:ctx->norm_apply_partial];
                [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_weights offset:lw->input_norm_w atIndex:1];
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
                [enc setBuffer:ctx->buf_weights offset:lw->input_norm_w atIndex:1];
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
            // Q → buf_attn_output (n_heads*hd floats)
            // K → buf_conv_output (reuse, kv_dim = 512 floats)
            // V → buf_conv_input (reuse, kv_dim = 512 floats)
            GpuMatvecJob qkv_jobs[3] = {
                { .w_buf = ctx->buf_weights, .w_off = lw->full.q_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->full.q_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->full.q_b,
                  .out_buf = ctx->buf_attn_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_heads * hd * 2, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->full.k_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->full.k_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->full.k_b,
                  .out_buf = ctx->buf_conv_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_kv * hd, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->full.v_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->full.v_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->full.v_b,
                  .out_buf = ctx->buf_conv_input, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_kv * hd, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 3; j++)
                gpu_encode_matvec_job(enc, ctx, &qkv_jobs[j]);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase B2: De-interleave Q+gate ---
            // Q projection output layout: [Q_h0(hd), gate_h0(hd), Q_h1(hd), gate_h1(hd), ...]
            // Rearrange to: [Q_h0, Q_h1, ..., Q_h15, gate_h0, gate_h1, ..., gate_h15]
            // Use buf_output as scratch (large enough: vocab_size floats)
            [enc setComputePipelineState:ctx->deinterleave_qgate];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_output offset:0 atIndex:1];
            { uint hd_val = (uint)hd, nh = (uint)n_heads;
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:2];
              [enc setBytes:&nh length:sizeof(uint) atIndex:3]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)(n_heads * hd) + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Copy de-interleaved data back from scratch to buf_attn_output
            [enc setComputePipelineState:ctx->copy_tmp_to_buf];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_output offset:0 atIndex:1];
            { uint cnt = (uint)(n_heads * hd * 2);
              [enc setBytes:&cnt length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)(n_heads * hd * 2) + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase C: Weighted QK RMS norm ---
            // Q in buf_attn_output (16 heads × 256), K in buf_conv_output (2 heads × 256)
            [enc setComputePipelineState:ctx->rms_norm_qk_w];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];  // Q
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];  // K
            [enc setBuffer:ctx->buf_weights offset:lw->full.qnorm_w atIndex:2];
            [enc setBuffer:ctx->buf_weights offset:lw->full.knorm_w atIndex:3];
            { uint hd_val = (uint)hd, nq = (uint)n_heads, nkv = (uint)n_kv;
              float inv_s = 1.0f / sqrtf((float)hd);
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:4];
              [enc setBytes:&nq length:sizeof(uint) atIndex:5];
              [enc setBytes:&nkv length:sizeof(uint) atIndex:6];
              [enc setBytes:&inv_s length:sizeof(float) atIndex:7]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase D: RoPE ---
            [enc setComputePipelineState:ctx->rope_apply];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];  // Q
            [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];  // K
            { uint hd_val = (uint)hd, rd = (uint)cfg->rotary_dim;
              uint nq = (uint)n_heads, nkv = (uint)n_kv;
              uint p = (uint)pos; float th = cfg->rope_theta;
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:2];
              [enc setBytes:&rd length:sizeof(uint) atIndex:3];
              [enc setBytes:&nq length:sizeof(uint) atIndex:4];
              [enc setBytes:&nkv length:sizeof(uint) atIndex:5];
              [enc setBytes:&p length:sizeof(uint) atIndex:6];
              [enc setBytes:&th length:sizeof(float) atIndex:7]; }
            { uint half_rot = (uint)cfg->rotary_dim / 2;
              uint max_heads = n_heads > n_kv ? n_heads : n_kv;
              [enc dispatchThreadgroups:MTLSizeMake(max_heads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(half_rot, 1, 1)]; }
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
              uint sl = (uint)seq_len, ss = (uint)OROME_GPU_KV_SEQ;
              float scale = 1.0f;  // already scaled in QK norm
              uint hpk = (uint)(n_heads / n_kv);
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
            { uint sl = (uint)seq_len, ss = (uint)OROME_GPU_KV_SEQ;
              [enc setBytes:&sl length:sizeof(uint) atIndex:1];
              [enc setBytes:&ss length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase H: Attention values (scores @ V) ---
            // Output → buf_attn_output (reuse Q buffer, now overwritten with attn output)
            [enc setComputePipelineState:ctx->attn_values];
            [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_kv_v[full_idx] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:2];
            { uint hd_val = (uint)hd, kvd = (uint)kv_dim;
              uint sl = (uint)seq_len, ss = (uint)OROME_GPU_KV_SEQ;
              uint hpk = (uint)(n_heads / n_kv);
              [enc setBytes:&hd_val length:sizeof(uint) atIndex:3];
              [enc setBytes:&kvd length:sizeof(uint) atIndex:4];
              [enc setBytes:&sl length:sizeof(uint) atIndex:5];
              [enc setBytes:&ss length:sizeof(uint) atIndex:6];
              [enc setBytes:&hpk length:sizeof(uint) atIndex:7]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)(n_heads * hd) + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase I-pre: Attention output gate (sigmoid) ---
            [enc setComputePipelineState:ctx->sigmoid_gate];
            [enc setBuffer:ctx->buf_attn_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_attn_output offset:n_heads * hd * sizeof(float) atIndex:1];
            { uint gd = (uint)(n_heads * hd);
              [enc setBytes:&gd length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)(n_heads * hd) + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Phase I: O-proj (buf_attn_output → buf_h_mid) ---
            { GpuMatvecJob o_job = {
                .w_buf = ctx->buf_weights, .w_off = lw->full.o_w,
                .s_buf = ctx->buf_weights, .s_off = lw->full.o_s,
                .b_buf = ctx->buf_weights, .b_off = lw->full.o_b,
                .in_buf = ctx->buf_attn_output, .in_off = 0,
                .out_buf = ctx->buf_h_mid, .out_off = 0,
                .out_ptr = NULL, .out_dim = H, .in_dim = n_heads * hd,
                .group_size = cfg->group_size, .is_2bit = false
            };
            gpu_encode_matvec_job(enc, ctx, &o_job); }
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
            [enc setBuffer:ctx->buf_weights offset:lw->post_norm_w atIndex:1];
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


            // --- Phase L: MoE routing + shared expert projections ---
            size_t sgg_off = n_experts * sizeof(float);
            GpuMatvecJob routing_jobs[4] = {
                { .w_buf = ctx->buf_weights, .w_off = lw->gate_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->gate_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->gate_b,
                  .out_buf = ctx->buf_output, .out_off = 0,
                  .out_ptr = NULL, .out_dim = n_experts, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->sg_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->sg_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->sg_b,
                  .out_buf = ctx->buf_shared_gate, .out_off = 0,
                  .out_ptr = NULL, .out_dim = S, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->su_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->su_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->su_b,
                  .out_buf = ctx->buf_shared_up, .out_off = 0,
                  .out_ptr = NULL, .out_dim = S, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
                { .w_buf = ctx->buf_weights, .w_off = lw->sgg_w,
                  .s_buf = ctx->buf_weights, .s_off = lw->sgg_s,
                  .b_buf = ctx->buf_weights, .b_off = lw->sgg_b,
                  .out_buf = ctx->buf_output, .out_off = sgg_off,
                  .out_ptr = NULL, .out_dim = 1, .in_dim = H,
                  .group_size = cfg->group_size, .is_2bit = false },
            };
            for (int j = 0; j < 4; j++)
                gpu_encode_matvec_job(enc, ctx, &routing_jobs[j]);

            // Per-layer fused path: use fused experts when layer has Metal buffer
            bool layer_fused_fa = ctx->softmax_topk && ctx->batch_expert_mv_dyn
                && ctx->buf_expert_layers && ctx->buf_expert_layers[layer]
                && !moe_get_profile_experts();

            if (gpu_resident && layer_fused_fa) {
                encode_fused_experts(enc, ctx, cfg, layer,
                                     effective_k, eng->quant, lw);
                if (!fwd_enc) {
                    [enc endEncoding];
                    [cmd commit];
                }
            } else if (layer_fused_fa) {
                // Hybrid fused: this resident layer has a direct Metal buffer
                encode_fused_experts(enc, ctx, cfg, layer,
                                     effective_k, eng->quant, lw);
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));
            } else {
                [enc endEncoding];
                if (cmd) [cmd commit];
                if (cmd) [cmd waitUntilCompleted];

                memcpy(s_fused_gate_scores, [ctx->buf_output contents],
                       n_experts * sizeof(float));
                float shared_gate_score;
                memcpy(&shared_gate_score,
                       (uint8_t *)[ctx->buf_output contents] + sgg_off,
                       sizeof(float));

                if (!gpu_resident) {
                    memcpy(eng->hidden, [ctx->buf_moe_hidden contents], H * sizeof(float));
                    memcpy(eng->residual, eng->hidden, H * sizeof(float));
                }

                double t0_moe = now_ms();
                moe_forward_routed(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                                   s_fused_gate_scores, shared_gate_score,
                                   eng->ef, effective_k, eng->quant,
                                   gpu_resident);
                double moe_ms = now_ms() - t0_moe;
                t_moe_total += moe_ms;
            }

            t1 = now_ms();
            t_attn_total += t1 - t0;

            full_idx++;
            continue;
        }

        // --- CPU FALLBACK path (no Metal or missing pipelines) ---
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
            linear_attention_forward(eng->wf, ctx, cfg, layer, pos,
                                     eng->hidden, eng->residual, eng->h_post,
                                     eng->linear_states[linear_idx],
                                     &attn_out, &attn_out_dim);
            linear_idx++;
        }
        t1 = now_ms();
        t_attn_total += t1 - t0;

        // CPU fallback O-proj + norm + MoE
        t0 = now_ms();
        {
            const char *o_proj_name = (cfg->layer_types[layer] == ATTN_FULL)
                ? "self_attn.o_proj" : "linear_attn.out_proj";
            char wname[128], sname[128], bname[128];
            snprintf(wname, sizeof(wname), "%s.weight", o_proj_name);
            snprintf(sname, sizeof(sname), "%s.scales", o_proj_name);
            snprintf(bname, sizeof(bname), "%s.biases", o_proj_name);
            uint32_t *o_w = weights_layer_ptr(eng->wf, layer, wname);
            uint16_t *o_s = weights_layer_ptr(eng->wf, layer, sname);
            uint16_t *o_b = weights_layer_ptr(eng->wf, layer, bname);
            uint16_t *post_norm_w = weights_layer_ptr(eng->wf, layer,
                                                       "post_attention_layernorm.weight");
            fast_dequant_matvec(ctx, cfg, o_w, o_s, o_b, attn_out, eng->h_post,
                                H, attn_out_dim, QUANT_4BIT);
            for (int i = 0; i < H; i++) eng->hidden[i] += eng->h_post[i];

            cpu_rms_norm(eng->hidden, post_norm_w, eng->h_post, H, cfg->rms_norm_eps);
            memcpy(eng->residual, eng->hidden, H * sizeof(float));
            moe_forward(eng->wf, ctx, cfg, layer, eng->hidden, eng->h_post,
                        eng->ef, effective_k, eng->quant);
        }

        t1 = now_ms();
        t_moe_total += t1 - t0;
    }

    // Sync GPU + final norm + LM head
    t0 = now_ms();
    if (gpu_resident) {
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        uint16_t *final_norm_w = weights_tensor_ptr(eng->wf, "model.norm.weight");
        uint32_t *lm_w = weights_tensor_ptr(eng->wf, "lm_head.weight");
        uint16_t *lm_s = weights_tensor_ptr(eng->wf, "lm_head.scales");
        uint16_t *lm_b = weights_tensor_ptr(eng->wf, "lm_head.biases");

        // Use shared forward-pass encoder if available
        id<MTLCommandBuffer> cmd = nil;
        id<MTLComputeCommandEncoder> enc = nil;
        if (fwd_enc) {
            enc = fwd_enc;
        } else {
            cmd = [ctx->queue commandBuffer];
            enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        }

        // RMS norm: buf_moe_hidden → buf_input
        // Use partial sums from last layer's moe_combine_copy_sq if available
        if (ctx->moe_combine_copy_sq) {
            uint num_tgs = ((uint)H + 255) / 256;
            [enc setComputePipelineState:ctx->norm_apply_partial];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)final_norm_w - base atIndex:1];
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
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)final_norm_w - base atIndex:1];
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
        GpuMatvecJob lm_job = {
            .w_buf = ctx->buf_weights, .w_off = (uint8_t *)lm_w - base,
            .s_buf = ctx->buf_weights, .s_off = (uint8_t *)lm_s - base,
            .b_buf = ctx->buf_weights, .b_off = (uint8_t *)lm_b - base,
            .in_buf = ctx->buf_input, .in_off = 0,
            .out_buf = ctx->buf_output, .out_off = 0,
            .out_ptr = NULL, .out_dim = cfg->vocab_size, .in_dim = H,
            .group_size = cfg->group_size, .is_2bit = false
        };
        gpu_encode_matvec_job(enc, ctx, &lm_job);

        // GPU argmax: find max index without copying 993KB logits to CPU
        if (ctx->argmax && ctx->buf_argmax_result) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->argmax];
            [enc setBuffer:ctx->buf_output offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_argmax_result offset:0 atIndex:1];
            { uint v = (uint)cfg->vocab_size;
              [enc setBytes:&v length:sizeof(uint) atIndex:2]; }
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        [enc endEncoding];
        id<MTLCommandBuffer> wait_cmd = fwd_cmd ? fwd_cmd : cmd;
        [wait_cmd commit];
        [wait_cmd waitUntilCompleted];
    } else {
        uint16_t *final_norm_w = weights_tensor_ptr(eng->wf, "model.norm.weight");
        float normed[H];
        cpu_rms_norm(eng->hidden, final_norm_w, normed, H, cfg->rms_norm_eps);
        lm_head_forward(eng->wf, eng->ctx, cfg, normed, eng->logits);
    }
    t1 = now_ms();
    t_lmhead_total += t1 - t0;

    // 5. Sample (greedy argmax)
    int next_token;
    if (gpu_resident && ctx->argmax && ctx->buf_argmax_result) {
        // GPU argmax already computed — just read the 4-byte result
        next_token = (int)(*(uint32_t *)[ctx->buf_argmax_result contents]);

    } else {
        next_token = cpu_argmax(eng->logits, cfg->vocab_size);
    }

    // Debug: print top logits and hidden state summary
    if (moe_get_profile_experts()) {
        float maxl = -1e30f, minl = 1e30f;
        int maxi = 0;
        for (int i = 0; i < cfg->vocab_size; i++) {
            if (eng->logits[i] > maxl) { maxl = eng->logits[i]; maxi = i; }
            if (eng->logits[i] < minl) minl = eng->logits[i];
        }
        float hmax = 0;
        for (int i = 0; i < H; i++) {
            float av = fabsf(eng->hidden[i]);
            if (av > hmax) hmax = av;
        }
        fprintf(stderr, "LOGITS pos=%d token=%d maxlogit=%.3f@%d minlogit=%.3f hmax=%.3e\n",
                pos, next_token, maxl, maxi, minl, hmax);
    }

    eng->pos++;
    profile_count++;

    // Print profile every 10 tokens
    if (profile_count % 10 == 0) {
        double inv = 1.0 / profile_count;
        double attn_only = t_attn_total - t_moe_total;
        fprintf(stderr, "[profile] avg/tok: attn=%.1fms moe=%.1fms norm=%.2fms lmhead=%.1fms\n",
                attn_only * inv, t_moe_total * inv, t_norm_total * inv, t_lmhead_total * inv);
        if (eng->ef && moe_get_profile_experts()) moe_print_layer_stats(eng->ef, true);
    }

    double step_ms = now_ms() - step_start;
    thermal_k_record(&eng->thermal, step_ms);

    return next_token;
}
