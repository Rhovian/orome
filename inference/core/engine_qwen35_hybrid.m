/*
 * engine_qwen35_hybrid.m — isolated 27B-style dense hybrid backend.
 *
 * This reuses the shared engine implementation, but overrides quantized matvec
 * dispatch so mixed linear/full dense models can use llama-style Q5_K/Q6_K
 * kernels without perturbing the shared path.
 */

#include <stdlib.h>
#include <string.h>

#include "orome.h"

typedef enum {
    HYBRID_Q4_NONE       = 0,
    HYBRID_Q4_LINEAR_Z   = 1u << 0,
    HYBRID_Q4_FULL_Q     = 1u << 1,
    HYBRID_Q4_FULL_O     = 1u << 2,
    HYBRID_Q4_LINEAR_O   = 1u << 3,
    HYBRID_Q4_DENSE_UP   = 1u << 4,
    HYBRID_Q4_OTHER      = 1u << 5,
    HYBRID_Q4_ALL        = 0x3Fu,
} HybridQ4Mask;

#define HYBRID_Q4_DEFAULT_MASK (HYBRID_Q4_OTHER | HYBRID_Q4_FULL_O | HYBRID_Q4_LINEAR_O | HYBRID_Q4_FULL_Q)

static uint32_t hybrid_q4_mask(void) {
    static bool initialized = false;
    static uint32_t mask = HYBRID_Q4_DEFAULT_MASK;
    if (initialized) return mask;
    initialized = true;

    const char *env = getenv("OROME_HYBRID_Q4");
    if (!env || !*env) return mask;
    if (strcmp(env, "all") == 0) { mask = HYBRID_Q4_ALL; return mask; }
    if (strcmp(env, "none") == 0) { mask = HYBRID_Q4_NONE; return mask; }

    mask = HYBRID_Q4_NONE;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", env);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (strcmp(tok, "linear_z") == 0) mask |= HYBRID_Q4_LINEAR_Z;
        else if (strcmp(tok, "full_q") == 0) mask |= HYBRID_Q4_FULL_Q;
        else if (strcmp(tok, "full_o") == 0) mask |= HYBRID_Q4_FULL_O;
        else if (strcmp(tok, "linear_o") == 0) mask |= HYBRID_Q4_LINEAR_O;
        else if (strcmp(tok, "dense_up") == 0) mask |= HYBRID_Q4_DENSE_UP;
        else if (strcmp(tok, "other") == 0) mask |= HYBRID_Q4_OTHER;
    }
    return mask;
}

static const char *hybrid_q4_family_name(uint32_t family) {
    switch (family) {
        case HYBRID_Q4_LINEAR_Z: return "linear_z";
        case HYBRID_Q4_FULL_Q:   return "full_q";
        case HYBRID_Q4_FULL_O:   return "full_o";
        case HYBRID_Q4_LINEAR_O: return "linear_o";
        case HYBRID_Q4_DENSE_UP: return "dense_up";
        case HYBRID_Q4_OTHER:    return "other";
        default:                 return "unknown";
    }
}

static uint32_t hybrid_q4_family(MetalCtx *ctx,
                                 id<MTLBuffer> in_buf,
                                 id<MTLBuffer> out_buf) {
    if (in_buf == ctx->buf_input && out_buf == ctx->buf_linear_output) {
        return HYBRID_Q4_LINEAR_Z;
    }
    if (in_buf == ctx->buf_input && out_buf == ctx->buf_attn_output) {
        return HYBRID_Q4_FULL_Q;
    }
    if (in_buf == ctx->buf_attn_output && out_buf == ctx->buf_h_mid) {
        return HYBRID_Q4_FULL_O;
    }
    if (in_buf == ctx->buf_linear_q && out_buf == ctx->buf_h_mid) {
        return HYBRID_Q4_LINEAR_O;
    }
    if (in_buf == ctx->buf_input &&
        (out_buf == ctx->buf_shared_gate || out_buf == ctx->buf_shared_up)) {
        return HYBRID_Q4_DENSE_UP;
    }
    return HYBRID_Q4_OTHER;
}

static void hybrid_dispatch_matvec(id<MTLComputeCommandEncoder> enc,
                                   MetalCtx *ctx,
                                   TensorRef *ref,
                                   id<MTLBuffer> in_buf, size_t in_off,
                                   id<MTLBuffer> out_buf, size_t out_off) {
    if (!ref || !ref->buffer) return;

    id<MTLComputePipelineState> pipe = nil;
    NSUInteger num_tgs = 0;
    MTLSize tg_size = MTLSizeMake(1, 1, 1);

    switch (ref->format) {
        case QFMT_GGUF_Q4_K:
        {
            uint32_t family = hybrid_q4_family(ctx, in_buf, out_buf);
            static uint32_t logged = 0;
            if ((logged & family) == 0) {
                logged |= family;
                fprintf(stderr, "[hybrid-q4] family=%s enabled=%s\n",
                        hybrid_q4_family_name(family),
                        (hybrid_q4_mask() & family) ? "yes" : "no");
            }
            if (ctx->matvec_q4k_llama && (hybrid_q4_mask() & family)) {
                pipe = ctx->matvec_q4k_llama;
                num_tgs = (ref->out_dim + 3) / 4;
                tg_size = MTLSizeMake(32, 2, 1);
            }
            break;
        }
        case QFMT_GGUF_Q5_K:
            if (ctx->matvec_q5k_llama) {
                pipe = ctx->matvec_q5k_llama;
                num_tgs = (ref->out_dim + 1) / 2;
                tg_size = MTLSizeMake(32, 2, 1);
            }
            break;
        case QFMT_GGUF_Q6_K:
            if (ctx->matvec_q6k_llama) {
                pipe = ctx->matvec_q6k_llama;
                num_tgs = (ref->out_dim + 3) / 4;
                tg_size = MTLSizeMake(32, 2, 1);
            }
            break;
        default:
            break;
    }

    if (!pipe) {
        format_dispatch_matvec(enc, ctx, ref, in_buf, in_off, out_buf, out_off);
        return;
    }

    [enc setComputePipelineState:pipe];
    [enc setBuffer:ref->buffer offset:ref->offset atIndex:0];
    [enc setBuffer:in_buf offset:in_off atIndex:1];
    [enc setBuffer:out_buf offset:out_off atIndex:2];
    uint od = ref->out_dim;
    uint id_ = ref->in_dim;
    [enc setBytes:&od length:sizeof(uint) atIndex:3];
    [enc setBytes:&id_ length:sizeof(uint) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:tg_size];
}

static void hybrid_linear_delta_net_dispatch(id<MTLComputeCommandEncoder> enc,
                                             MetalCtx *ctx,
                                             const ModelConfig *cfg,
                                             int linear_idx,
                                             int total_key,
                                             int n_v_heads,
                                             int num_k_heads,
                                             int key_dim,
                                             int value_dim) {
    (void)cfg;
    float inv_s = 1.0f / sqrtf((float)key_dim);
    uint kd = (uint)key_dim;
    [enc setComputePipelineState:ctx->rms_norm_qk];
    [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_conv_output offset:total_key * sizeof(float) atIndex:1];
    [enc setBytes:&kd length:sizeof(uint) atIndex:2];
    [enc setBytes:&inv_s length:sizeof(float) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_k_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake((NSUInteger)key_dim, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    uint nk = (uint)num_k_heads;
    [enc setComputePipelineState:ctx->delta_net];
    [enc setBuffer:ctx->buf_linear_state[linear_idx] offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_conv_output offset:total_key * sizeof(float) atIndex:2];
    [enc setBuffer:ctx->buf_conv_output offset:2 * total_key * sizeof(float) atIndex:3];
    [enc setBuffer:ctx->buf_linear_decay offset:0 atIndex:4];
    [enc setBuffer:ctx->buf_linear_beta offset:0 atIndex:5];
    [enc setBuffer:ctx->buf_linear_v offset:0 atIndex:6];
    [enc setBytes:&nk length:sizeof(uint) atIndex:7];
    [enc setBytes:&inv_s length:sizeof(float) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((value_dim + 3) / 4), (NSUInteger)n_v_heads, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

#define engine_create engine_create_qwen35_dense_hybrid_unused
#define engine_free engine_free_qwen35_dense_hybrid_unused
#define engine_reset engine_reset_qwen35_dense_hybrid_unused
#define engine_step engine_step_qwen35_dense_hybrid
#define format_dispatch_matvec hybrid_dispatch_matvec
#define ENGINE_LINEAR_DELTA_NET_DISPATCH(enc, ctx, cfg, linear_idx, total_key, n_v_heads, num_k_heads, key_dim, value_dim) \
    hybrid_linear_delta_net_dispatch((enc), (ctx), (cfg), (linear_idx), (total_key), (n_v_heads), (num_k_heads), (key_dim), (value_dim))
#include "engine.m"
#undef ENGINE_LINEAR_DELTA_NET_DISPATCH
#undef format_dispatch_matvec
#undef engine_step
#undef engine_reset
#undef engine_free
#undef engine_create
