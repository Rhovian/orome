/*
 * attention.m — Full (GQA) and linear (GatedDeltaNet) attention layers.
 *
 * Tensor name mapping (from MLX-community Qwen3.5-35B-A3B-4bit):
 *   Full attention layers:  self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm
 *   Linear attention layers: linear_attn.in_proj_{qkv,z,a,b}, linear_attn.out_proj,
 *                            linear_attn.conv1d, linear_attn.{A_log,dt_bias,norm}
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#include "orome.h"

// ============================================================================
// Full attention (grouped-query, softmax-based)
// Tensor names: self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales,biases}
//               self_attn.{q_norm,k_norm}.weight
// ============================================================================

void full_attention_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                            int layer_idx, int pos, float *hidden, float *residual,
                            float *h_post, KVCache *kv,
                            float **attn_out, int *attn_out_dim) {
    (void)residual; (void)h_post;
    int H = cfg->hidden_dim;
    int n_heads = cfg->num_attn_heads;
    int n_kv = cfg->num_kv_heads;
    int hd = cfg->head_dim;
    int kv_dim = cfg->kv_dim;
    float eps = cfg->rms_norm_eps;

    // Scratch buffers (allocated once, reused)
    static float *s_normed = NULL, *s_q = NULL, *s_k = NULL, *s_v = NULL;
    static float *s_attn_out = NULL, *s_o_proj = NULL;
    static int s_alloc_H = 0;
    if (s_alloc_H < H) {
        free(s_normed); free(s_q); free(s_k); free(s_v);
        free(s_attn_out); free(s_o_proj);
        s_normed   = calloc(H, sizeof(float));
        s_q        = calloc(n_heads * hd, sizeof(float));
        s_k        = calloc(n_kv * hd, sizeof(float));
        s_v        = calloc(n_kv * hd, sizeof(float));
        s_attn_out = calloc(n_heads * hd, sizeof(float));
        s_o_proj   = calloc(H, sizeof(float));
        s_alloc_H = H;
    }

    // 1. Input norm
    uint16_t *norm_w = weights_layer_ptr(wf, layer_idx, "input_layernorm.weight");
    cpu_rms_norm(hidden, norm_w, s_normed, H, eps);

    // 2. Q/K/V projections — batched in one GPU command buffer
    uint32_t *q_w = weights_layer_ptr(wf, layer_idx, "self_attn.q_proj.weight");
    uint16_t *q_s = weights_layer_ptr(wf, layer_idx, "self_attn.q_proj.scales");
    uint16_t *q_b = weights_layer_ptr(wf, layer_idx, "self_attn.q_proj.biases");
    uint32_t *k_w = weights_layer_ptr(wf, layer_idx, "self_attn.k_proj.weight");
    uint16_t *k_s = weights_layer_ptr(wf, layer_idx, "self_attn.k_proj.scales");
    uint16_t *k_b = weights_layer_ptr(wf, layer_idx, "self_attn.k_proj.biases");
    uint32_t *v_w = weights_layer_ptr(wf, layer_idx, "self_attn.v_proj.weight");
    uint16_t *v_s = weights_layer_ptr(wf, layer_idx, "self_attn.v_proj.scales");
    uint16_t *v_b = weights_layer_ptr(wf, layer_idx, "self_attn.v_proj.biases");

    if (ctx && ctx->buf_weights) {
        memcpy([ctx->buf_input contents], s_normed, H * sizeof(float));
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        GpuMatvecJob jobs[3] = {
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)q_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)q_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)q_b - base,
              .out_buf = ctx->buf_output, .out_off = 0,
              .out_ptr = s_q, .out_dim = n_heads * hd, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)k_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)k_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)k_b - base,
              .out_buf = ctx->buf_h_mid, .out_off = 0,
              .out_ptr = s_k, .out_dim = n_kv * hd, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)v_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)v_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)v_b - base,
              .out_buf = ctx->buf_moe_hidden, .out_off = 0,
              .out_ptr = s_v, .out_dim = n_kv * hd, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
        };
        gpu_run_matvec_batch(ctx, jobs, 3);
    } else {
        cpu_dequant_matvec((void *)q_w, q_s, q_b, s_normed, s_q, n_heads * hd, H, cfg->group_size);
        cpu_dequant_matvec((void *)k_w, k_s, k_b, s_normed, s_k, n_kv * hd, H, cfg->group_size);
        cpu_dequant_matvec((void *)v_w, v_s, v_b, s_normed, s_v, n_kv * hd, H, cfg->group_size);
    }

    // 3. Per-head Q/K RMS norm + scale
    uint16_t *qnorm_w = weights_layer_ptr(wf, layer_idx, "self_attn.q_norm.weight");
    uint16_t *knorm_w = weights_layer_ptr(wf, layer_idx, "self_attn.k_norm.weight");
    float inv_scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_heads; h++) {
        cpu_rms_norm(s_q + h * hd, qnorm_w, s_q + h * hd, hd, 1e-6f);
        for (int d = 0; d < hd; d++) s_q[h * hd + d] *= inv_scale;
    }
    for (int h = 0; h < n_kv; h++) {
        cpu_rms_norm(s_k + h * hd, knorm_w, s_k + h * hd, hd, 1e-6f);
    }

    // 4. RoPE
    apply_rotary_emb(s_q, s_k, pos, n_kv, hd, cfg->rotary_dim, cfg->rope_theta);

    // 5. Store K/V in cache
    memcpy(kv->k_cache + pos * kv_dim, s_k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + pos * kv_dim, s_v, kv_dim * sizeof(float));
    int seq_len = pos + 1;

    // 6. Scaled dot-product attention (Accelerate-optimized, per head)
    // Static scratch for scores — avoids calloc/free per head
    static float *s_scores = NULL;
    static int s_scores_cap = 0;
    if (s_scores_cap < seq_len) {
        free(s_scores);
        s_scores_cap = seq_len < 256 ? 256 : seq_len;
        s_scores = malloc(s_scores_cap * sizeof(float));
    }

    int heads_per_kv = n_heads / n_kv;
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = s_q + h * hd;

        // scores = K_cache @ q  (seq_len × hd matrix times hd vector)
        // K_cache layout: [seq_len × kv_dim], stride kv_dim, starting at kv_h * hd
        cblas_sgemv(CblasRowMajor, CblasNoTrans, seq_len, hd, 1.0f,
                    kv->k_cache + kv_h * hd, kv_dim, qh, 1, 0.0f, s_scores, 1);
        cpu_softmax(s_scores, seq_len);

        // out = V_cache^T @ scores  (hd × seq_len matrix times seq_len vector)
        float *out_h = s_attn_out + h * hd;
        cblas_sgemv(CblasRowMajor, CblasTrans, seq_len, hd, 1.0f,
                    kv->v_cache + kv_h * hd, kv_dim, s_scores, 1, 0.0f, out_h, 1);
    }

    // 7. Export pre-O-proj output (caller handles O proj + residual)
    *attn_out = s_attn_out;
    *attn_out_dim = n_heads * hd;
}

// ============================================================================
// Linear attention (GatedDeltaNet)
// Tensor names: linear_attn.in_proj_{qkv,z,a,b}.{weight,scales,biases}
//               linear_attn.out_proj.{weight,scales,biases}
//               linear_attn.conv1d.weight (BF16, not quantized)
//               linear_attn.A_log (F32), linear_attn.dt_bias (BF16)
//               linear_attn.norm.weight (BF16)
// ============================================================================

void linear_attention_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                              int layer_idx, int pos, float *hidden, float *residual,
                              float *h_post, LinearAttnState *state,
                              float **attn_out, int *attn_out_dim) {
    (void)pos; (void)residual; (void)h_post;
    int H = cfg->hidden_dim;
    int n_v_heads = cfg->linear_num_v_heads;
    int n_k_heads = cfg->linear_num_k_heads;
    int key_dim = cfg->linear_key_dim;
    int val_dim = cfg->linear_value_dim;
    int total_key = cfg->linear_total_key;
    int total_value = cfg->linear_total_value;
    int conv_dim = cfg->linear_conv_dim;
    float eps = cfg->rms_norm_eps;

    // Scratch
    static float *s_normed = NULL, *s_qkv = NULL, *s_z = NULL;
    static float *s_alpha = NULL, *s_beta = NULL, *s_conv_out = NULL;
    static float *s_values_out = NULL, *s_gated_out = NULL, *s_o_proj = NULL;
    static int s_alloc = 0;
    if (s_alloc < H) {
        free(s_normed); free(s_qkv); free(s_z);
        free(s_alpha); free(s_beta); free(s_conv_out);
        free(s_values_out); free(s_gated_out); free(s_o_proj);
        s_normed     = calloc(H, sizeof(float));
        s_qkv        = calloc(conv_dim, sizeof(float));
        s_z          = calloc(total_value, sizeof(float));
        s_alpha      = calloc(n_v_heads, sizeof(float));
        s_beta       = calloc(n_v_heads, sizeof(float));
        s_conv_out   = calloc(conv_dim, sizeof(float));
        s_values_out = calloc(total_value, sizeof(float));
        s_gated_out  = calloc(total_value, sizeof(float));
        s_o_proj     = calloc(H, sizeof(float));
        s_alloc = H;
    }

    // 1. Input norm
    uint16_t *norm_w = weights_layer_ptr(wf, layer_idx, "input_layernorm.weight");
    cpu_rms_norm(hidden, norm_w, s_normed, H, eps);

    // 2. Projections (linear_attn.in_proj_*) — batched in one GPU command buffer
    uint32_t *qkv_w = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_qkv.weight");
    uint16_t *qkv_s = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_qkv.scales");
    uint16_t *qkv_b = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_qkv.biases");
    uint32_t *z_w = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_z.weight");
    uint16_t *z_s = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_z.scales");
    uint16_t *z_b = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_z.biases");
    uint32_t *a_w = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_a.weight");
    uint16_t *a_s = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_a.scales");
    uint16_t *a_b = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_a.biases");
    uint32_t *b_w = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_b.weight");
    uint16_t *b_s = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_b.scales");
    uint16_t *b_b = weights_layer_ptr(wf, layer_idx, "linear_attn.in_proj_b.biases");

    if (ctx && ctx->buf_weights) {
        memcpy([ctx->buf_input contents], s_normed, H * sizeof(float));
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        // Pack all 4 outputs into buf_output at different offsets
        size_t z_off = conv_dim * sizeof(float);
        size_t a_off = z_off + total_value * sizeof(float);
        size_t b_off = a_off + n_v_heads * sizeof(float);
        GpuMatvecJob jobs[4] = {
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)qkv_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)qkv_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)qkv_b - base,
              .out_buf = ctx->buf_output, .out_off = 0,
              .out_ptr = s_qkv, .out_dim = conv_dim, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)z_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)z_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)z_b - base,
              .out_buf = ctx->buf_output, .out_off = z_off,
              .out_ptr = s_z, .out_dim = total_value, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)a_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)a_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)a_b - base,
              .out_buf = ctx->buf_output, .out_off = a_off,
              .out_ptr = s_alpha, .out_dim = n_v_heads, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)b_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)b_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)b_b - base,
              .out_buf = ctx->buf_output, .out_off = b_off,
              .out_ptr = s_beta, .out_dim = n_v_heads, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
        };
        gpu_run_matvec_batch(ctx, jobs, 4);
    } else {
        cpu_dequant_matvec((void *)qkv_w, qkv_s, qkv_b, s_normed, s_qkv, conv_dim, H, cfg->group_size);
        cpu_dequant_matvec((void *)z_w, z_s, z_b, s_normed, s_z, total_value, H, cfg->group_size);
        cpu_dequant_matvec((void *)a_w, a_s, a_b, s_normed, s_alpha, n_v_heads, H, cfg->group_size);
        cpu_dequant_matvec((void *)b_w, b_s, b_b, s_normed, s_beta, n_v_heads, H, cfg->group_size);
    }

    // 3. Conv1d step (linear_attn.conv1d.weight is BF16, shape [conv_dim, kernel, 1])
    uint16_t *conv_w = weights_layer_ptr(wf, layer_idx, "linear_attn.conv1d.weight");
    cpu_conv1d_step(state->conv_state, s_qkv, conv_w, s_conv_out,
                    conv_dim, cfg->conv_kernel_size);

    // 4. Split conv output into Q, K, V
    float *q = s_conv_out;                          // [total_key]
    float *k = s_conv_out + total_key;              // [total_key]
    float *v = s_conv_out + total_key * 2;          // [total_value]

    // 5. Q/K per-head RMS norm + scaling
    for (int h = 0; h < n_k_heads; h++) {
        cpu_rms_norm_bare(q + h * key_dim, q + h * key_dim, key_dim, 1e-6f);
        cpu_rms_norm_bare(k + h * key_dim, k + h * key_dim, key_dim, 1e-6f);
    }
    float inv_scale = 1.0f / sqrtf((float)key_dim);
    for (int i = 0; i < total_key; i++) {
        q[i] *= inv_scale * inv_scale;  // Q gets double scale
        k[i] *= inv_scale;
    }

    // 6. GatedDeltaNet recurrence (per v-head)
    // A_log is F32, dt_bias is BF16
    float *A_log = weights_layer_ptr(wf, layer_idx, "linear_attn.A_log");
    uint16_t *dt_bias = weights_layer_ptr(wf, layer_idx, "linear_attn.dt_bias");

    int k_per_v = n_v_heads / n_k_heads;
    for (int vh = 0; vh < n_v_heads; vh++) {
        int kh = vh / k_per_v;
        float *S = state->ssm_state + vh * val_dim * key_dim;

        // Compute decay and beta gate
        float A_val = expf(A_log[vh]);  // A_log is F32
        float softplus = logf(1.0f + expf(s_alpha[vh] + bf16_to_f32(dt_bias[vh])));
        float decay = expf(-A_val * softplus);
        float beta_gate = cpu_sigmoid(s_beta[vh]);

        float *kh_ptr = k + kh * key_dim;
        float *qh_ptr = q + kh * key_dim;

        for (int vi = 0; vi < val_dim; vi++) {
            // Decay + memory read
            float kv_mem = 0.0f;
            for (int ki = 0; ki < key_dim; ki++) {
                S[vi * key_dim + ki] *= decay;
                kv_mem += S[vi * key_dim + ki] * kh_ptr[ki];
            }

            // Delta update
            float delta = (v[vh * val_dim + vi] - kv_mem) * beta_gate;
            for (int ki = 0; ki < key_dim; ki++) {
                S[vi * key_dim + ki] += kh_ptr[ki] * delta;
            }

            // Output
            float out_val = 0.0f;
            for (int ki = 0; ki < key_dim; ki++) {
                out_val += S[vi * key_dim + ki] * qh_ptr[ki];
            }
            s_values_out[vh * val_dim + vi] = out_val;
        }
    }

    // 7. Gated RMS norm (z-gated output, linear_attn.norm.weight)
    uint16_t *o_norm_w = weights_layer_ptr(wf, layer_idx, "linear_attn.norm.weight");
    cpu_rms_norm_gated(s_values_out, s_z, o_norm_w, s_gated_out,
                       n_v_heads, val_dim, eps);

    // 8. Export pre-O-proj output (caller handles O proj + residual)
    *attn_out = s_gated_out;
    *attn_out_dim = total_value;
}
