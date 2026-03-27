// format.m — Weight format abstraction layer
//
// Decouples tensor storage format (GGUF Q4_K, custom 4-bit, etc.) from
// GPU dispatch. The engine asks for tensors by name, gets back opaque
// TensorRefs that carry everything needed for dispatch.
//
// Adding a new quant type: add kernel to shaders.metal, add case to
// format_dispatch_matvec(). Adding a new file format: implement
// format_provider_open_xxx(), no dispatch changes needed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <string.h>
#include "orome.h"

static inline float format_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return ldexpf((float)mant, -24) * (sign ? -1.0f : 1.0f);
    }
    if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float out;
    memcpy(&out, &f32, sizeof(out));
    return out;
}

static inline float format_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline void format_unpack_q4k_scale_min_pair(const uint8_t *sc_data,
                                                    uint32_t g,
                                                    float *sc_lo,
                                                    float *sc_hi,
                                                    float *mn_lo,
                                                    float *mn_hi) {
    if (g < 2) {
        uint32_t base = g * 2;
        *sc_lo = (float)(sc_data[base] & 63);
        *sc_hi = (float)(sc_data[base + 1] & 63);
        *mn_lo = (float)(sc_data[base + 4] & 63);
        *mn_hi = (float)(sc_data[base + 5] & 63);
        return;
    }

    uint32_t base = (g - 2) * 2;
    uint32_t upper = g - 2;
    uint8_t sc_pack = sc_data[8 + upper];
    uint8_t mn_pack = sc_data[10 + upper];
    *sc_lo = (float)((sc_pack & 0xF) | ((sc_data[base] >> 6) << 4));
    *sc_hi = (float)((sc_pack >> 4) | ((sc_data[base + 1] >> 6) << 4));
    *mn_lo = (float)((mn_pack & 0xF) | ((sc_data[base + 4] >> 6) << 4));
    *mn_hi = (float)((mn_pack >> 4) | ((sc_data[base + 5] >> 6) << 4));
}

static bool format_decode_q8_0_row(const uint8_t *row, uint32_t in_dim, float *out) {
    if (in_dim % 32 != 0) return false;
    uint32_t blocks_per_row = in_dim / 32;
    for (uint32_t blk = 0; blk < blocks_per_row; blk++) {
        const uint8_t *block = row + blk * 34;
        float d = format_fp16_to_f32((uint16_t)(block[0] | ((uint16_t)block[1] << 8)));
        const int8_t *qs = (const int8_t *)(block + 2);
        uint32_t base = blk * 32;
        for (uint32_t j = 0; j < 32; j++) {
            out[base + j] = d * (float)qs[j];
        }
    }
    return true;
}

static bool format_decode_q4_k_row(const uint8_t *row, uint32_t in_dim, float *out) {
    if (in_dim % 256 != 0) return false;
    uint32_t num_superblocks = in_dim / 256;
    for (uint32_t sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        const uint8_t *sb = row + sb_idx * 144;
        float d = format_fp16_to_f32((uint16_t)(sb[0] | ((uint16_t)sb[1] << 8)));
        float dmin = format_fp16_to_f32((uint16_t)(sb[2] | ((uint16_t)sb[3] << 8)));
        const uint8_t *scales = sb + 4;
        const uint8_t *qs = sb + 16;
        uint32_t base = sb_idx * 256;

        for (uint32_t g = 0; g < 4; g++) {
            float sc_lo, sc_hi, mn_lo, mn_hi;
            format_unpack_q4k_scale_min_pair(scales, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            const uint8_t *group_qs = qs + g * 32;
            float q_sc_lo = d * sc_lo;
            float q_sc_hi = d * sc_hi;
            float q_mn_lo = dmin * mn_lo;
            float q_mn_hi = dmin * mn_hi;
            for (uint32_t l = 0; l < 32; l++) {
                uint8_t byte = group_qs[l];
                out[base + g * 64 + l] = q_sc_lo * (float)(byte & 0xF) - q_mn_lo;
                out[base + g * 64 + 32 + l] = q_sc_hi * (float)(byte >> 4) - q_mn_hi;
            }
        }
    }
    return true;
}

static bool format_decode_q5_k_row(const uint8_t *row, uint32_t in_dim, float *out) {
    if (in_dim % 256 != 0) return false;
    uint32_t num_superblocks = in_dim / 256;
    for (uint32_t sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        const uint8_t *sb = row + sb_idx * 176;
        float d = format_fp16_to_f32((uint16_t)(sb[0] | ((uint16_t)sb[1] << 8)));
        float dmin = format_fp16_to_f32((uint16_t)(sb[2] | ((uint16_t)sb[3] << 8)));
        const uint8_t *scales = sb + 4;
        const uint8_t *qh = sb + 16;
        const uint8_t *ql = sb + 48;
        uint32_t base = sb_idx * 256;

        for (uint32_t g = 0; g < 4; g++) {
            float sc_lo, sc_hi, mn_lo, mn_hi;
            format_unpack_q4k_scale_min_pair(scales, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            const uint8_t *group_ql = ql + g * 32;
            uint8_t hm_lo = (uint8_t)(1u << (g * 2));
            uint8_t hm_hi = (uint8_t)(1u << (g * 2 + 1));
            float q_sc_lo = d * sc_lo;
            float q_sc_hi = d * sc_hi;
            float q_mn_lo = dmin * mn_lo;
            float q_mn_hi = dmin * mn_hi;
            for (uint32_t l = 0; l < 32; l++) {
                uint8_t byte = group_ql[l];
                uint8_t q5_lo = (uint8_t)((byte & 0xF) | ((qh[l] & hm_lo) ? 16 : 0));
                uint8_t q5_hi = (uint8_t)((byte >> 4) | ((qh[l] & hm_hi) ? 16 : 0));
                out[base + g * 64 + l] = q_sc_lo * (float)q5_lo - q_mn_lo;
                out[base + g * 64 + 32 + l] = q_sc_hi * (float)q5_hi - q_mn_hi;
            }
        }
    }
    return true;
}

static bool format_decode_q6_k_row(const uint8_t *row, uint32_t in_dim, float *out) {
    if (in_dim % 256 != 0) return false;
    uint32_t num_superblocks = in_dim / 256;
    for (uint32_t sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        const uint8_t *sb = row + sb_idx * 210;
        const uint8_t *ql = sb;
        const uint8_t *qh = sb + 128;
        const int8_t *sc = (const int8_t *)(sb + 192);
        float d = format_fp16_to_f32((uint16_t)(sb[208] | ((uint16_t)sb[209] << 8)));
        uint32_t base = sb_idx * 256;

        for (uint32_t blk = 0; blk < 2; blk++) {
            const uint8_t *ql_blk = ql + blk * 64;
            const uint8_t *qh_blk = qh + blk * 32;
            const int8_t *sc_blk = sc + blk * 8;
            uint32_t blk_base = base + blk * 128;
            for (uint32_t l = 0; l < 32; l++) {
                uint32_t is = l / 16;
                int q1 = (int)((ql_blk[l] & 0xF) | (((qh_blk[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_blk[l + 32] & 0xF) | (((qh_blk[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_blk[l] >> 4) | (((qh_blk[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_blk[l + 32] >> 4) | (((qh_blk[l] >> 6) & 3) << 4)) - 32;
                out[blk_base + l] = d * (float)sc_blk[is + 0] * (float)q1;
                out[blk_base + l + 32] = d * (float)sc_blk[is + 2] * (float)q2;
                out[blk_base + l + 64] = d * (float)sc_blk[is + 4] * (float)q3;
                out[blk_base + l + 96] = d * (float)sc_blk[is + 6] * (float)q4;
            }
        }
    }
    return true;
}

bool format_decode_row_f32(TensorRef *ref, uint32_t row_idx, float *out) {
    if (!ref || !ref->buffer || !out || row_idx >= ref->out_dim) return false;
    const uint8_t *base = (const uint8_t *)[ref->buffer contents] + ref->offset;

    switch (ref->format) {
        case QFMT_F32: {
            const float *row = (const float *)(base + (size_t)row_idx * ref->in_dim * sizeof(float));
            memcpy(out, row, (size_t)ref->in_dim * sizeof(float));
            return true;
        }
        case QFMT_F16: {
            const uint16_t *row = (const uint16_t *)(base + (size_t)row_idx * ref->in_dim * sizeof(uint16_t));
            for (uint32_t j = 0; j < ref->in_dim; j++) out[j] = format_fp16_to_f32(row[j]);
            return true;
        }
        case QFMT_BF16: {
            const uint16_t *row = (const uint16_t *)(base + (size_t)row_idx * ref->in_dim * sizeof(uint16_t));
            for (uint32_t j = 0; j < ref->in_dim; j++) out[j] = format_bf16_to_f32(row[j]);
            return true;
        }
        case QFMT_GGUF_Q8_0: {
            if (ref->in_dim % 32 != 0) return false;
            size_t row_bytes = (size_t)(ref->in_dim / 32) * 34;
            return format_decode_q8_0_row(base + (size_t)row_idx * row_bytes, ref->in_dim, out);
        }
        case QFMT_GGUF_Q4_K: {
            if (ref->in_dim % 256 != 0) return false;
            size_t row_bytes = (size_t)(ref->in_dim / 256) * 144;
            return format_decode_q4_k_row(base + (size_t)row_idx * row_bytes, ref->in_dim, out);
        }
        case QFMT_GGUF_Q5_K: {
            if (ref->in_dim % 256 != 0) return false;
            size_t row_bytes = (size_t)(ref->in_dim / 256) * 176;
            return format_decode_q5_k_row(base + (size_t)row_idx * row_bytes, ref->in_dim, out);
        }
        case QFMT_GGUF_Q6_K: {
            if (ref->in_dim % 256 != 0) return false;
            size_t row_bytes = (size_t)(ref->in_dim / 256) * 210;
            return format_decode_q6_k_row(base + (size_t)row_idx * row_bytes, ref->in_dim, out);
        }
    }
    return false;
}

// ============================================================================
// Dispatch: encode a matvec into a command encoder using a TensorRef
// ============================================================================

#define FORMAT_ROWS_PER_TG 16
#define FORMAT_TWO_ROW_MULTIPLIER 2

static inline uint format_effective_rows_per_tg(QuantFormat fmt) {
    switch (fmt) {
        case QFMT_GGUF_Q4_K:
        case QFMT_GGUF_Q5_K:
        case QFMT_GGUF_Q6_K:
        case QFMT_GGUF_Q8_0:
            return FORMAT_ROWS_PER_TG * FORMAT_TWO_ROW_MULTIPLIER;
        default:
            return FORMAT_ROWS_PER_TG;
    }
}

void format_dispatch_matvec(
    id<MTLComputeCommandEncoder> enc,
    MetalCtx *ctx,
    TensorRef *ref,
    id<MTLBuffer> in_buf, size_t in_off,
    id<MTLBuffer> out_buf, size_t out_off
) {
    if (!ref->buffer) {
        // Tensor not found — skip silently (may be optional like o_norm)
        return;
    }
    id<MTLComputePipelineState> pipe = ref->pipeline;
    if (!pipe) {
        // Try to resolve pipeline at dispatch time (in case it was NULL at cache build time)
        pipe = format_pipeline_for(ctx, ref->format);
        if (!pipe) {
            fprintf(stderr, "[format] no pipeline for format %d (od=%u id=%u)\n",
                    ref->format, ref->out_dim, ref->in_dim);
            return;
        }
    }

    [enc setComputePipelineState:pipe];

    switch (ref->format) {
        case QFMT_GGUF_Q4_K:
        case QFMT_GGUF_Q5_K:
        case QFMT_GGUF_Q8_0:
        case QFMT_GGUF_Q6_K: {
            // GGUF formats: single contiguous buffer, interleaved blocks
            [enc setBuffer:ref->buffer offset:ref->offset  atIndex:0]; // data
            [enc setBuffer:in_buf      offset:in_off       atIndex:1]; // input
            [enc setBuffer:out_buf     offset:out_off      atIndex:2]; // output
            uint od = ref->out_dim, id_ = ref->in_dim;
            [enc setBytes:&od  length:sizeof(uint) atIndex:3];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:4];
            break;
        }

        case QFMT_F16:
        case QFMT_BF16:
        case QFMT_F32: {
            // Unquantized: direct buffer, no dequant needed
            [enc setBuffer:ref->buffer offset:ref->offset  atIndex:0]; // data
            [enc setBuffer:in_buf      offset:in_off       atIndex:1]; // input
            [enc setBuffer:out_buf     offset:out_off      atIndex:2]; // output
            uint od = ref->out_dim, id_ = ref->in_dim;
            [enc setBytes:&od  length:sizeof(uint) atIndex:3];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:4];
            break;
        }
    }

    uint rows_per_tg = format_effective_rows_per_tg(ref->format);
    NSUInteger num_tgs = (ref->out_dim + rows_per_tg - 1) / rows_per_tg;
    NSUInteger tg_size = FORMAT_ROWS_PER_TG * 32; // 512 threads
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
}

// ============================================================================
// Pipeline selection: map QuantFormat → Metal pipeline
// ============================================================================

id<MTLComputePipelineState> format_pipeline_for(MetalCtx *ctx, QuantFormat fmt) {
    switch (fmt) {
        case QFMT_GGUF_Q4_K:  return ctx->matvec_q4k;
        case QFMT_GGUF_Q5_K:  return ctx->matvec_q5k;
        case QFMT_GGUF_Q8_0:  return ctx->matvec_q8_0;
        case QFMT_GGUF_Q6_K:  return ctx->matvec_q6k;
        case QFMT_F16:        return ctx->matvec_f32;  // F16 uses F32 kernel (upcast at load)
        case QFMT_BF16:       return ctx->matvec_f32;
        case QFMT_F32:        return ctx->matvec_f32;
    }
    return NULL;
}

// ============================================================================
// GGML type → QuantFormat mapping
// ============================================================================

QuantFormat format_from_ggml_type(uint32_t ggml_type) {
    switch (ggml_type) {
        case 8:  return QFMT_GGUF_Q8_0;   // GGML_TYPE_Q8_0
        case 12: return QFMT_GGUF_Q4_K;   // GGML_TYPE_Q4_K
        case 13: return QFMT_GGUF_Q5_K;   // GGML_TYPE_Q5_K
        case 14: return QFMT_GGUF_Q6_K;   // GGML_TYPE_Q6_K
        case 1:  return QFMT_F16;         // GGML_TYPE_F16
        case 30: return QFMT_BF16;        // GGML_TYPE_BF16
        case 0:  return QFMT_F32;         // GGML_TYPE_F32
        default: return QFMT_GGUF_Q4_K;   // fallback
    }
}

// ============================================================================
// FormatProvider: build TensorRefs from a loaded model
// ============================================================================

FormatProvider *format_provider_open_gguf(GGUFFile *gf, MetalCtx *ctx) {
    FormatProvider *fp = calloc(1, sizeof(FormatProvider));
    fp->gguf = gf;

    // Wrap the entire GGUF mmap as a single Metal buffer
    size_t aligned = (gf->file_size + 4095) & ~4095;
    fp->model_buf = [ctx->device newBufferWithBytesNoCopy:gf->mmap_base
                                                   length:aligned
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!fp->model_buf) {
        fprintf(stderr, "[format] failed to wrap GGUF as Metal buffer\n");
        free(fp);
        return NULL;
    }

    fprintf(stderr, "[format] GGUF wrapped as Metal buffer (%.1f GB)\n",
            gf->file_size / (1024.0 * 1024.0 * 1024.0));

    return fp;
}

void format_provider_close(FormatProvider *fp) {
    if (!fp) return;
    // Metal buffer is released by ARC
    free(fp);
}

// Resolve a tensor by name → TensorRef ready for dispatch

// Resolve expert layer tensors → ExpertLayerRef
ExpertLayerRef format_resolve_expert_layer(FormatProvider *fp, int layer_idx,
                                            uint32_t num_experts) {
    ExpertLayerRef elr = {0};
    elr.buffer = fp->model_buf;

    char name[128];

    // Gate: shape [hidden, intermediate, num_experts] → per-expert stride
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_idx);
    GGUFTensorInfo *gate_ti = gguf_find_tensor(fp->gguf, name);
    if (gate_ti) {
        elr.gate.offset = fp->gguf->data_offset + gate_ti->offset;
        elr.gate.format = format_from_ggml_type(gate_ti->type);
        elr.gate.expert_stride = gguf_tensor_size(gate_ti) / num_experts;
    }

    // Up: same layout as gate
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer_idx);
    GGUFTensorInfo *up_ti = gguf_find_tensor(fp->gguf, name);
    if (up_ti) {
        elr.up.offset = fp->gguf->data_offset + up_ti->offset;
        elr.up.format = format_from_ggml_type(up_ti->type);
        elr.up.expert_stride = gguf_tensor_size(up_ti) / num_experts;
    }

    // Down: shape [intermediate, hidden, num_experts]
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_idx);
    GGUFTensorInfo *down_ti = gguf_find_tensor(fp->gguf, name);
    if (down_ti) {
        elr.down.offset = fp->gguf->data_offset + down_ti->offset;
        elr.down.format = format_from_ggml_type(down_ti->type);
        elr.down.expert_stride = gguf_tensor_size(down_ti) / num_experts;
    }

    return elr;
}

// ============================================================================
// Tensor cache builders — produce format-agnostic LayerTensorCache
// ============================================================================

// Helper: create a GGUF TensorRef from a tensor name
static TensorRef gguf_ref(GGUFFile *gf, id<MTLBuffer> buf, MetalCtx *ctx,
                           const char *name, uint32_t out_dim, uint32_t in_dim) {
    GGUFTensorInfo *ti = gguf_find_tensor(gf, name);
    if (!ti) return (TensorRef){0};
    QuantFormat fmt = format_from_ggml_type(ti->type);
    return (TensorRef){
        .buffer = buf, .offset = gf->data_offset + ti->offset,
        .pipeline = format_pipeline_for(ctx, fmt),
        .format = fmt, .out_dim = out_dim, .in_dim = in_dim,
    };
}

// Helper: GGUF raw tensor.
// If convert_bf16=true: converts F32 to BF16 (for norm weights read by BF16 kernels).
// If convert_bf16=false: keeps as-is (for F32 data used as matvec or direct float reads).
static TensorRef gguf_raw(GGUFFile *gf, id<MTLBuffer> buf, id<MTLDevice> device,
                           const char *name, bool convert_bf16) {
    GGUFTensorInfo *ti = gguf_find_tensor(gf, name);
    if (!ti) return (TensorRef){0};

    if (ti->type == 0 && convert_bf16) { // F32 — convert to BF16 for kernel compatibility
        size_t num_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) num_elements *= ti->dims[d];
        size_t bf16_size = num_elements * sizeof(uint16_t);

        id<MTLBuffer> bf16_buf = [device newBufferWithLength:bf16_size
                                                     options:MTLResourceStorageModeShared];
        float *src = (float *)((uint8_t *)[buf contents] + gf->data_offset + ti->offset);
        uint16_t *dst = (uint16_t *)[bf16_buf contents];
        for (size_t i = 0; i < num_elements; i++) {
            uint32_t f32;
            memcpy(&f32, &src[i], 4);
            dst[i] = (uint16_t)(f32 >> 16); // F32 → BF16 truncation
        }
        return (TensorRef){ .buffer = bf16_buf, .offset = 0, .format = QFMT_BF16 };
    }

    return (TensorRef){ .buffer = buf, .offset = gf->data_offset + ti->offset, .format = QFMT_F32 };
}

// ---- GGUF format builder ----

LayerTensorCache *build_tensor_cache_gguf(GGUFFile *gf, id<MTLBuffer> model_buf,
                                           MetalCtx *ctx,
                                           const ModelConfig *cfg,
                                           GlobalTensorCache *globals) {
    LayerTensorCache *cache = calloc(cfg->num_layers, sizeof(LayerTensorCache));
    id<MTLBuffer> buf = model_buf;
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    char name[128];

    #define G_REF(tname, od, id) gguf_ref(gf, buf, ctx, (tname), (od), (id))
    #define G_RAW(tname) gguf_raw(gf, buf, ctx->device, (tname), true)
    #define G_RAW_F32(tname) gguf_raw(gf, buf, ctx->device, (tname), false)

    // Global tensors
    if (globals) {
        globals->token_embd = G_REF("token_embd.weight", cfg->vocab_size, H);
        globals->lm_head = G_REF("output.weight", cfg->vocab_size, H);
        if (!globals->lm_head.buffer) {
            // Qwen3.5 ties the LM head to token embeddings in GGUF exports.
            globals->lm_head = globals->token_embd;
        }
        globals->final_norm = G_RAW("output_norm.weight");
    }

    for (int i = 0; i < cfg->num_layers; i++) {
        LayerTensorCache *c = &cache[i];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        c->input_norm = G_RAW(name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        c->post_norm = G_RAW(name);
        if (!c->post_norm.buffer) { // fallback name
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
            c->post_norm = G_RAW(name);
        }

        if (cfg->ffn_type == FFN_MOE) {
            // MoE routing gate (F32)
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", i);
            c->routing_gate = G_REF(name, cfg->num_experts, H);

            // Shared expert (may have different intermediate dim than routed experts)
            int S = cfg->shared_intermediate > 0 ? cfg->shared_intermediate : M;
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", i);
            c->shared_gate = G_REF(name, S, H);
            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", i);
            c->shared_up = G_REF(name, S, H);
            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", i);
            c->shared_down = G_REF(name, H, S);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", i);
            c->shared_expert_gate = G_REF(name, 1, H); // scalar gate [1, H] matvec
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
            c->dense_gate = G_REF(name, M, H);
            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
            c->dense_up = G_REF(name, M, H);
            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
            c->dense_down = G_REF(name, H, M);
        }

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            int conv_dim = cfg->linear_conv_dim;
            int total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
            int n_v = cfg->linear_num_v_heads;
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
            c->lin.qkv = G_REF(name, conv_dim, H);
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
            c->lin.z = G_REF(name, total_value, H);
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", i);
            c->lin.a = G_REF(name, n_v, H);
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", i);
            c->lin.b = G_REF(name, n_v, H);
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            c->lin.o = G_REF(name, H, total_value);
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            c->lin.conv = G_RAW_F32(name);
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
            c->lin.o_norm = G_RAW(name);     // gated_rms_norm expects BF16
            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            c->lin.A_log = G_RAW_F32(name);
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            c->lin.dt_bias = G_RAW_F32(name);
        } else {
            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;

            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            { GGUFTensorInfo *qi = gguf_find_tensor(gf, name);
              // Gated attention stores Q and gate rows interleaved per head.
              // Reorder rows at load time using generic per-row tensor bytes so
              // all quant formats stay on the same fast runtime path.
              int q_out = (qi && qi->n_dims >= 2) ? (int)qi->dims[1] : n_heads * hd;
              if (qi && qi->n_dims >= 2 && q_out == n_heads * hd * 2) {
                  int rows = q_out;
                  size_t row_bytes = gguf_tensor_size(qi) / (size_t)rows;
                  size_t total_bytes = (size_t)rows * row_bytes;
                  id<MTLBuffer> db = [ctx->device newBufferWithLength:total_bytes
                                                              options:MTLResourceStorageModeShared];
                  uint8_t *s = (uint8_t *)[buf contents] + gf->data_offset + qi->offset;
                  uint8_t *d = (uint8_t *)[db contents];
                  for (int h = 0; h < n_heads; h++) {
                      memcpy(d + (size_t)(h * hd) * row_bytes,
                             s + (size_t)(h * 2 * hd) * row_bytes,
                             (size_t)hd * row_bytes);
                      memcpy(d + (size_t)(n_heads * hd + h * hd) * row_bytes,
                             s + (size_t)(h * 2 * hd + hd) * row_bytes,
                             (size_t)hd * row_bytes);
                  }
                  QuantFormat fmt = format_from_ggml_type(qi->type);
                  c->full.q = (TensorRef){ .buffer = db, .offset = 0,
                      .pipeline = format_pipeline_for(ctx, fmt),
                      .format = fmt, .out_dim = (uint32_t)q_out, .in_dim = (uint32_t)H };
              } else {
                  c->full.q = G_REF(name, q_out, H);
              } }
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            c->full.k = G_REF(name, n_kv * hd, H);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            c->full.v = G_REF(name, n_kv * hd, H);
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            c->full.o = G_REF(name, H, n_heads * hd);
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
            c->full.q_norm = G_RAW(name);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
            c->full.k_norm = G_RAW(name);
        }
    }
    #undef G_REF
    #undef G_RAW
    return cache;
}
