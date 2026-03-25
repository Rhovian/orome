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
    id<MTLComputePipelineState> pipe = ref->pipeline;
    if (!pipe) {
        fprintf(stderr, "[format] no pipeline for format %d\n", ref->format);
        return;
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
        case QFMT_BF16:       return ctx->matvec_f16; // same kernel, different decode
        case QFMT_F32:        return ctx->matvec_f16; // placeholder
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
