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
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        c->post_norm = G_RAW(name);
        if (!c->post_norm.buffer) { // fallback name
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
            c->post_norm = G_RAW(name);
        }

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

        if (cfg->layer_types[i] == ATTN_LINEAR) {
            int conv_dim = cfg->linear_conv_dim;
            int total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
            int n_v = cfg->linear_num_v_heads;

            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
            c->lin.qkv = G_REF(name, conv_dim, H);
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
            c->lin.z = G_REF(name, total_value, H);
            // Alpha, beta, A_log, dt_bias: GGUF stores with head-interleaved rows.
            // GGUF order: [head0, head2, head4, ..., head30, head1, head3, ..., head31]
            // De-interleave by building new buffers/tensors with sequential head order.

            // Helper: de-interleave a Q8_0 weight tensor with n_v rows
            #define DEINTERLEAVE_Q8_ROWS(tname, out_ref) do { \
                snprintf(name, sizeof(name), "blk.%d." tname, i); \
                GGUFTensorInfo *_ti = gguf_find_tensor(gf, name); \
                if (_ti && _ti->n_dims >= 2 && _ti->dims[1] == (uint64_t)n_v) { \
                    int _rows = n_v, _half = _rows / 2; \
                    int _bpr = ((int)_ti->dims[0] / 32) * 34; \
                    size_t _total = (size_t)_rows * _bpr; \
                    id<MTLBuffer> _db = [ctx->device newBufferWithLength:_total options:MTLResourceStorageModeShared]; \
                    uint8_t *_s = (uint8_t *)[buf contents] + gf->data_offset + _ti->offset; \
                    uint8_t *_d = (uint8_t *)[_db contents]; \
                    for (int _r = 0; _r < _half; _r++) { \
                        memcpy(_d + (_r*2) * _bpr, _s + _r * _bpr, _bpr); \
                        memcpy(_d + (_r*2+1) * _bpr, _s + (_half+_r) * _bpr, _bpr); \
                    } \
                    QuantFormat _fmt = format_from_ggml_type(_ti->type); \
                    out_ref = (TensorRef){ .buffer = _db, .offset = 0, \
                        .pipeline = format_pipeline_for(ctx, _fmt), \
                        .format = _fmt, .out_dim = (uint32_t)n_v, .in_dim = (uint32_t)H }; \
                } else { out_ref = G_REF(name, n_v, H); } \
            } while(0)

            DEINTERLEAVE_Q8_ROWS("ssm_alpha.weight", c->lin.a);
            DEINTERLEAVE_Q8_ROWS("ssm_beta.weight", c->lin.b);
            #undef DEINTERLEAVE_Q8_ROWS
            // Output projection (ssm_out in GGUF)
            // The GGUF stores ssm_out with head-interleaved Q8_0 blocks along in_dim.
            // 32 v-heads × 4 blocks each (128 values), interleaved in 2 groups of 16 heads.
            // De-interleave at load time by creating a new buffer with correct block order.
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            {
                GGUFTensorInfo *ti = gguf_find_tensor(gf, name);
                if (ti && ti->n_dims >= 2) {
                    int in_dim = (int)ti->dims[0];  // 4096 (total_value)
                    int out_dim = (int)ti->dims[1];  // 2048 (H)
                    int n_heads = cfg->linear_num_v_heads;  // 32
                    int head_dim_vals = cfg->linear_value_dim;  // 128
                    int blocks_per_head = head_dim_vals / 32;  // 4 Q8_0 blocks per head
                    int total_blocks = in_dim / 32;  // 128 blocks per row
                    int bytes_per_block = 34;  // Q8_0: 32 int8 + fp16 scale
                    int bytes_per_row = total_blocks * bytes_per_block;
                    size_t total_bytes = (size_t)out_dim * bytes_per_row;

                    // Allocate de-interleaved buffer
                    id<MTLBuffer> deint_buf = [ctx->device newBufferWithLength:total_bytes
                                                                      options:MTLResourceStorageModeShared];
                    uint8_t *src = (uint8_t *)[buf contents] + gf->data_offset + ti->offset;
                    uint8_t *dst = (uint8_t *)[deint_buf contents];

                    // For each row, rearrange blocks from interleaved to sequential order
                    // GGUF block i maps to: ref_block = (i/blocks_per_head % (n_heads/2)) * (2*blocks_per_head)
                    //                                 + (i / (total_blocks/2)) * blocks_per_head
                    //                                 + (i % blocks_per_head)
                    // Simplified: half = i / (total_blocks/2), group = (i % (total_blocks/2)) / blocks_per_head
                    //             pos = i % blocks_per_head
                    //             ref = group * (2*blocks_per_head) + half * blocks_per_head + pos
                    int half_blocks = total_blocks / 2;
                    for (int row = 0; row < out_dim; row++) {
                        uint8_t *src_row = src + (size_t)row * bytes_per_row;
                        uint8_t *dst_row = dst + (size_t)row * bytes_per_row;
                        for (int gb = 0; gb < total_blocks; gb++) {
                            int half = gb / half_blocks;
                            int group = (gb % half_blocks) / blocks_per_head;
                            int pos = gb % blocks_per_head;
                            int ref_blk = group * (2 * blocks_per_head) + half * blocks_per_head + pos;
                            memcpy(dst_row + ref_blk * bytes_per_block,
                                   src_row + gb * bytes_per_block,
                                   bytes_per_block);
                        }
                    }

                    QuantFormat fmt = format_from_ggml_type(ti->type);
                    c->lin.o = (TensorRef){
                        .buffer = deint_buf, .offset = 0,
                        .pipeline = format_pipeline_for(ctx, fmt),
                        .format = fmt, .out_dim = (uint32_t)out_dim, .in_dim = (uint32_t)in_dim,
                    };
                    if (i == 0) fprintf(stderr, "[format] De-interleaved ssm_out (%d blocks/row, %d heads)\n",
                                        total_blocks, n_heads);
                } else {
                    c->lin.o = G_REF(name, H, total_value);
                }
            }
            // Conv1d weights: GGUF stores [ne0=kernel_size, ne1=conv_dim] F32
            // Already in [conv_dim, kernel_size] memory order (ne0 contiguous per channel)
            // Use F32 conv1d kernel directly — no conversion needed
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            c->lin.conv = G_RAW_F32(name);
            // A_log and dt_bias: F32, 32 elements, head-interleaved
            // De-interleave: deint[r*2] = gguf[r], deint[r*2+1] = gguf[half+r] for r<half
            #define DEINTERLEAVE_F32_RAW(tname, out_ref) do { \
                snprintf(name, sizeof(name), "blk.%d." tname, i); \
                GGUFTensorInfo *_ti = gguf_find_tensor(gf, name); \
                if (_ti && _ti->dims[0] == (uint64_t)n_v) { \
                    int _half = n_v / 2; \
                    size_t _bytes = n_v * sizeof(float); \
                    id<MTLBuffer> _db = [ctx->device newBufferWithLength:_bytes options:MTLResourceStorageModeShared]; \
                    float *_s = (float *)((uint8_t *)[buf contents] + gf->data_offset + _ti->offset); \
                    float *_d = (float *)[_db contents]; \
                    for (int _r = 0; _r < _half; _r++) { \
                        _d[_r*2] = _s[_r]; \
                        _d[_r*2+1] = _s[_half+_r]; \
                    } \
                    out_ref = (TensorRef){ .buffer = _db, .offset = 0, .format = QFMT_F32 }; \
                } else { out_ref = G_RAW_F32(name); } \
            } while(0)

            DEINTERLEAVE_F32_RAW("ssm_a", c->lin.A_log);
            DEINTERLEAVE_F32_RAW("ssm_dt.bias", c->lin.dt_bias);
            #undef DEINTERLEAVE_F32_RAW
            // Output norm (ssm_norm in GGUF)
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
            c->lin.o_norm = G_RAW(name);     // gated_rms_norm expects BF16
        } else {
            int n_heads = cfg->num_attn_heads;
            int n_kv = cfg->num_kv_heads;
            int hd = cfg->head_dim;

            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            { GGUFTensorInfo *qi = gguf_find_tensor(gf, name);
              // Gated attention: Q tensor has n_heads*head_dim*2 rows (Q + gate interleaved)
              int q_out = (qi && qi->n_dims >= 2) ? (int)qi->dims[1] : n_heads * hd;
              c->full.q = G_REF(name, q_out, H); }
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
