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
#include "orome.h"

// ============================================================================
// Dispatch: encode a matvec into a command encoder using a TensorRef
// ============================================================================

#define FORMAT_ROWS_PER_TG 16

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
        case QFMT_OROME_4BIT:
        case QFMT_OROME_2BIT: {
            // Legacy format: 3 separate buffer regions (weights, scales, biases)
            [enc setBuffer:ref->buffer offset:ref->offset            atIndex:0]; // W
            [enc setBuffer:ref->buffer offset:ref->scale_offset      atIndex:1]; // S
            [enc setBuffer:ref->buffer offset:ref->bias_offset       atIndex:2]; // B
            [enc setBuffer:in_buf      offset:in_off                 atIndex:3]; // input
            [enc setBuffer:out_buf     offset:out_off                atIndex:4]; // output
            uint od = ref->out_dim, id_ = ref->in_dim, gs = ref->group_size;
            [enc setBytes:&od  length:sizeof(uint) atIndex:5];
            [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
            [enc setBytes:&gs  length:sizeof(uint) atIndex:7];
            break;
        }

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

    uint rows_per_tg = FORMAT_ROWS_PER_TG;
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
        case QFMT_OROME_4BIT: return ctx->matvec_4bit;
        case QFMT_OROME_2BIT: return ctx->matvec_2bit;
        case QFMT_GGUF_Q4_K:  return ctx->matvec_q4k;
        case QFMT_GGUF_Q5_K:  return ctx->matvec_q5k;
        case QFMT_GGUF_Q8_0:  return ctx->matvec_q8_0;
        case QFMT_GGUF_Q6_K:  return ctx->matvec_q6k;
        case QFMT_F16:        return ctx->matvec_f16;
        case QFMT_BF16:       return ctx->matvec_f16;
        case QFMT_F32:        return ctx->matvec_f32;
    }
    return NULL;
}

// ============================================================================
// GGML type → QuantFormat mapping
// ============================================================================

QuantFormat format_from_ggml_type(uint32_t ggml_type) {
    switch (ggml_type) {
        case 2:  return QFMT_GGUF_Q4_0;   // GGML_TYPE_Q4_0
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
    fp->ctx = ctx;

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
TensorRef format_resolve_tensor(FormatProvider *fp, const char *name,
                                uint32_t out_dim, uint32_t in_dim) {
    TensorRef ref = {0};

    GGUFTensorInfo *ti = gguf_find_tensor(fp->gguf, name);
    if (!ti) {
        fprintf(stderr, "[format] tensor not found: %s\n", name);
        return ref;
    }

    ref.buffer = fp->model_buf;
    ref.offset = fp->gguf->data_offset + ti->offset;
    ref.format = format_from_ggml_type(ti->type);
    ref.pipeline = format_pipeline_for(fp->ctx, ref.format);
    ref.out_dim = out_dim;
    ref.in_dim = in_dim;
    ref.group_size = 0; // not used for GGUF formats
    ref.scale_offset = 0;
    ref.bias_offset = 0;

    return ref;
}

// Resolve expert layer tensors → ExpertLayerRef
ExpertLayerRef format_resolve_expert_layer(FormatProvider *fp, int layer_idx,
                                            uint32_t hidden_dim, uint32_t intermediate,
                                            uint32_t num_experts) {
    ExpertLayerRef elr = {0};
    elr.buffer = fp->model_buf;
    elr.num_experts = num_experts;

    // Build tensor names
    char name[128];

    // Gate: shape [hidden, intermediate, num_experts] → per-expert stride
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_idx);
    GGUFTensorInfo *gate_ti = gguf_find_tensor(fp->gguf, name);
    if (gate_ti) {
        elr.gate.offset = fp->gguf->data_offset + gate_ti->offset;
        elr.gate.format = format_from_ggml_type(gate_ti->type);
        elr.gate.pipeline = format_pipeline_for(fp->ctx, elr.gate.format);
        // Expert stride: total tensor size / num_experts
        elr.gate.expert_stride = gguf_tensor_size(gate_ti) / num_experts;
        elr.gate.out_dim = intermediate;
        elr.gate.in_dim = hidden_dim;
    }

    // Up: same layout as gate
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer_idx);
    GGUFTensorInfo *up_ti = gguf_find_tensor(fp->gguf, name);
    if (up_ti) {
        elr.up.offset = fp->gguf->data_offset + up_ti->offset;
        elr.up.format = format_from_ggml_type(up_ti->type);
        elr.up.pipeline = format_pipeline_for(fp->ctx, elr.up.format);
        elr.up.expert_stride = gguf_tensor_size(up_ti) / num_experts;
        elr.up.out_dim = intermediate;
        elr.up.in_dim = hidden_dim;
    }

    // Down: shape [intermediate, hidden, num_experts]
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_idx);
    GGUFTensorInfo *down_ti = gguf_find_tensor(fp->gguf, name);
    if (down_ti) {
        elr.down.offset = fp->gguf->data_offset + down_ti->offset;
        elr.down.format = format_from_ggml_type(down_ti->type);
        elr.down.pipeline = format_pipeline_for(fp->ctx, elr.down.format);
        elr.down.expert_stride = gguf_tensor_size(down_ti) / num_experts;
        elr.down.out_dim = hidden_dim;
        elr.down.in_dim = intermediate;
    }

    // Shared expert
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer_idx);
    GGUFTensorInfo *sgate_ti = gguf_find_tensor(fp->gguf, name);
    if (sgate_ti) {
        elr.shared_gate.offset = fp->gguf->data_offset + sgate_ti->offset;
        elr.shared_gate.format = format_from_ggml_type(sgate_ti->type);
        elr.shared_gate.pipeline = format_pipeline_for(fp->ctx, elr.shared_gate.format);
        elr.shared_gate.out_dim = intermediate;
        elr.shared_gate.in_dim = hidden_dim;
    }

    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer_idx);
    GGUFTensorInfo *sup_ti = gguf_find_tensor(fp->gguf, name);
    if (sup_ti) {
        elr.shared_up.offset = fp->gguf->data_offset + sup_ti->offset;
        elr.shared_up.format = format_from_ggml_type(sup_ti->type);
        elr.shared_up.pipeline = format_pipeline_for(fp->ctx, elr.shared_up.format);
        elr.shared_up.out_dim = intermediate;
        elr.shared_up.in_dim = hidden_dim;
    }

    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer_idx);
    GGUFTensorInfo *sdown_ti = gguf_find_tensor(fp->gguf, name);
    if (sdown_ti) {
        elr.shared_down.offset = fp->gguf->data_offset + sdown_ti->offset;
        elr.shared_down.format = format_from_ggml_type(sdown_ti->type);
        elr.shared_down.pipeline = format_pipeline_for(fp->ctx, elr.shared_down.format);
        elr.shared_down.out_dim = hidden_dim;
        elr.shared_down.in_dim = intermediate;
    }

    return elr;
}

// ============================================================================
// Tensor cache builders — produce format-agnostic LayerTensorCache
// ============================================================================

// Helper: create a legacy TensorRef (our packed format, 3 separate regions)
static TensorRef legacy_ref(id<MTLBuffer> buf, size_t w, size_t s, size_t b,
                             MetalCtx *ctx, uint32_t out_dim, uint32_t in_dim,
                             uint32_t group_size, bool is_2bit) {
    return (TensorRef){
        .buffer = buf, .offset = w, .scale_offset = s, .bias_offset = b,
        .pipeline = is_2bit ? ctx->matvec_2bit : ctx->matvec_4bit,
        .format = is_2bit ? QFMT_OROME_2BIT : QFMT_OROME_4BIT,
        .out_dim = out_dim, .in_dim = in_dim, .group_size = group_size,
    };
}

// Helper: create a TensorRef for a raw F32 tensor (norms, biases — not matvec'd)
static TensorRef raw_ref(id<MTLBuffer> buf, size_t offset) {
    return (TensorRef){ .buffer = buf, .offset = offset, .format = QFMT_F32 };
}

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

// ---- Legacy format builder ----

LayerTensorCache *build_tensor_cache_legacy(WeightFile *wf, MetalCtx *ctx,
                                             const ModelConfig *cfg,
                                             GlobalTensorCache *globals) {
    LayerTensorCache *cache = calloc(cfg->num_layers, sizeof(LayerTensorCache));
    id<MTLBuffer> buf = ctx->buf_weights;
    uint8_t *base = (uint8_t *)wf->data;
    uint32_t gs = cfg->group_size;
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;

    #define L_OFF(layer, suffix) ((size_t)((uint8_t *)weights_layer_ptr(wf, layer, suffix) - base))
    #define L_REF(layer, suffix, od, id) legacy_ref(buf, L_OFF(layer, suffix ".weight"), \
        L_OFF(layer, suffix ".scales"), L_OFF(layer, suffix ".biases"), ctx, od, id, gs, false)
    #define L_RAW(layer, suffix) raw_ref(buf, L_OFF(layer, suffix))

    // Global tensors
    if (globals) {
        globals->embedding = raw_ref(buf, L_OFF(0, "../../model.embed_tokens.weight"));
        globals->lm_head = legacy_ref(buf,
            (size_t)((uint8_t *)weights_tensor_ptr(wf, "lm_head.weight") - base),
            (size_t)((uint8_t *)weights_tensor_ptr(wf, "lm_head.scales") - base),
            (size_t)((uint8_t *)weights_tensor_ptr(wf, "lm_head.biases") - base),
            ctx, cfg->vocab_size, H, gs, false);
        globals->final_norm = raw_ref(buf,
            (size_t)((uint8_t *)weights_tensor_ptr(wf, "model.norm.weight") - base));
    }

    for (int i = 0; i < cfg->num_layers; i++) {
        LayerTensorCache *c = &cache[i];
        c->input_norm = L_RAW(i, "input_layernorm.weight");
        c->post_norm = L_RAW(i, "post_attention_layernorm.weight");
        c->routing_gate = L_REF(i, "mlp.gate", cfg->num_experts, H);
        c->shared_gate = L_REF(i, "mlp.shared_expert.gate_proj", M, H);
        c->shared_up = L_REF(i, "mlp.shared_expert.up_proj", M, H);
        c->shared_down = L_REF(i, "mlp.shared_expert.down_proj", H, M);
        c->shared_expert_gate = L_RAW(i, "mlp.shared_expert_gate.weight");

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            int conv_dim = cfg->linear_num_v_heads * (cfg->linear_key_dim + cfg->linear_value_dim)
                         + cfg->linear_num_k_heads * cfg->linear_key_dim;
            int total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
            int n_v = cfg->linear_num_v_heads;

            c->lin.qkv = L_REF(i, "linear_attn.in_proj_qkv", conv_dim, H);
            c->lin.z = L_REF(i, "linear_attn.in_proj_z", total_value, H);
            c->lin.a = L_REF(i, "linear_attn.in_proj_a", n_v, H);
            c->lin.b = L_REF(i, "linear_attn.in_proj_b", n_v, H);
            c->lin.o = L_REF(i, "linear_attn.out_proj", H, total_value);
            c->lin.conv = L_RAW(i, "linear_attn.conv1d.weight");
            c->lin.A_log = L_RAW(i, "linear_attn.A_log");
            c->lin.dt_bias = L_RAW(i, "linear_attn.dt_bias");
            c->lin.o_norm = L_RAW(i, "linear_attn.norm.weight");
        } else {
            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;
            c->full.q = L_REF(i, "self_attn.q_proj", n_heads * hd, H);
            c->full.k = L_REF(i, "self_attn.k_proj", n_kv * hd, H);
            c->full.v = L_REF(i, "self_attn.v_proj", n_kv * hd, H);
            c->full.o = L_REF(i, "self_attn.o_proj", H, n_heads * hd);
            c->full.q_norm = L_RAW(i, "self_attn.q_norm.weight");
            c->full.k_norm = L_RAW(i, "self_attn.k_norm.weight");
        }
    }
    #undef L_OFF
    #undef L_REF
    #undef L_RAW
    return cache;
}

// ---- GGUF format builder ----

LayerTensorCache *build_tensor_cache_gguf(GGUFFile *gf, MetalCtx *ctx,
                                           const ModelConfig *cfg,
                                           GlobalTensorCache *globals) {
    LayerTensorCache *cache = calloc(cfg->num_layers, sizeof(LayerTensorCache));
    id<MTLBuffer> buf = ctx->buf_weights; // GGUF mmap wrapped as Metal buffer
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    char name[128];

    #define G_REF(tname, od, id) gguf_ref(gf, buf, ctx, (tname), (od), (id))
    #define G_RAW(tname) gguf_raw(gf, buf, ctx->device, (tname), true)
    #define G_RAW_F32(tname) gguf_raw(gf, buf, ctx->device, (tname), false)

    // Global tensors
    if (globals) {
        globals->embedding = G_RAW("token_embd.weight");
        globals->lm_head = G_REF("output.weight", cfg->vocab_size, H);
        globals->final_norm = G_RAW("output_norm.weight");
    }

    for (int i = 0; i < cfg->num_layers; i++) {
        LayerTensorCache *c = &cache[i];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        c->input_norm = G_RAW(name);
        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
        c->post_norm = G_RAW(name);

        // MoE routing gate (F32)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", i);
        c->routing_gate = G_REF(name, cfg->num_experts, H);

        // Shared expert
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", i);
        c->shared_gate = G_REF(name, M, H);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", i);
        c->shared_up = G_REF(name, M, H);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", i);
        c->shared_down = G_REF(name, H, M);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", i);
        c->shared_expert_gate = G_RAW_F32(name); // scalar gate, dispatched as F32 matvec

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            int conv_dim = cfg->linear_num_v_heads * (cfg->linear_key_dim + cfg->linear_value_dim)
                         + cfg->linear_num_k_heads * cfg->linear_key_dim;
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
            // Output projection (ssm_out in GGUF)
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            c->lin.o = G_REF(name, H, total_value);
            // Conv1d weights: GGUF stores [kernel_size, conv_dim] F32
            // Our kernel expects [conv_dim * kernel_size] BF16 (channel-major)
            // Need transpose + F32→BF16 conversion
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            {
                GGUFTensorInfo *conv_ti = gguf_find_tensor(gf, name);
                if (conv_ti && conv_ti->type == 0 && conv_ti->n_dims == 2) {
                    int kernel_size = (int)conv_ti->dims[0]; // ne0 = 4 (contiguous)
                    int cdim = (int)conv_ti->dims[1];        // ne1 = 8192
                    size_t bf16_size = kernel_size * cdim * sizeof(uint16_t);
                    id<MTLBuffer> conv_buf = [ctx->device newBufferWithLength:bf16_size
                                                                     options:MTLResourceStorageModeShared];
                    float *src = (float *)((uint8_t *)[buf contents] + gf->data_offset + conv_ti->offset);
                    uint16_t *dst = (uint16_t *)[conv_buf contents];
                    // Transpose [kernel_size, conv_dim] → [conv_dim, kernel_size] and convert F32→BF16
                    for (int ch = 0; ch < cdim; ch++) {
                        for (int k = 0; k < kernel_size; k++) {
                            float val = src[k * cdim + ch]; // GGUF: row k, col ch
                            uint32_t f32; memcpy(&f32, &val, 4);
                            dst[ch * kernel_size + k] = (uint16_t)(f32 >> 16);
                        }
                    }
                    c->lin.conv = (TensorRef){ .buffer = conv_buf, .offset = 0, .format = QFMT_BF16 };
                } else {
                    c->lin.conv = G_RAW(name);
                }
            }
            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            c->lin.A_log = G_RAW_F32(name);  // decay_beta kernel expects F32
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            c->lin.dt_bias = G_RAW(name);     // decay_beta kernel expects BF16
            // Output norm (ssm_norm in GGUF)
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
            c->lin.o_norm = G_RAW(name);     // gated_rms_norm expects BF16
        } else {
            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;

            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            c->full.q = G_REF(name, n_heads * hd, H);
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
