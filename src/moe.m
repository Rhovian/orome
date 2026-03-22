/*
 * moe.m — Mixture of Experts: routing, expert I/O, expert forward pass.
 *
 * On machines with enough RAM (e.g. 96GB M2 Max), all expert weights are
 * mmap'd at startup — zero per-token I/O overhead. Expert data is accessed
 * directly from the mmap'd region with no copies.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "orome.h"

#define ROWS_PER_TG 16  // must match metal.m and shaders.metal

// ============================================================================
// Expert file management — mmap all layers at startup
// ============================================================================

ExpertFiles *expert_files_open(const ModelConfig *cfg, const char *model_dir,
                               const char *hot_mask_path) {
    ExpertFiles *ef = calloc(1, sizeof(ExpertFiles));
    ef->layer_data = calloc(cfg->num_layers, sizeof(void *));
    ef->layer_size = calloc(cfg->num_layers, sizeof(size_t));
    ef->layer_fds  = calloc(cfg->num_layers, sizeof(int));

    size_t total_mmaped = 0;
    int opened = 0;

    for (int i = 0; i < cfg->num_layers; i++) {
        ef->layer_fds[i] = -1;

        char path[512];
        snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin",
                 model_dir ? model_dir : ".", i);

        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;

        struct stat st;
        fstat(fd, &st);
        size_t size = st.st_size;

        void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            continue;
        }

        // Advise the kernel we'll access this randomly (expert selection is unpredictable)
        madvise(data, size, MADV_RANDOM);

        ef->layer_data[i] = data;
        ef->layer_size[i] = size;
        ef->layer_fds[i] = fd;
        total_mmaped += size;
        opened++;
    }

    ef->all_mmaped = (opened == cfg->num_layers);
    printf("[moe] mmap'd %d/%d layers (%.1f GB total)\n",
           opened, cfg->num_layers, total_mmaped / 1e9);

    if (hot_mask_path) {
        int words_per_layer = (cfg->num_experts + 31) / 32;
        ef->hot_mask = calloc(cfg->num_layers * words_per_layer, sizeof(uint32_t));
        ef->tiered_quant = true;
    }

    return ef;
}

void expert_files_close(ExpertFiles *ef, const ModelConfig *cfg) {
    if (!ef) return;
    for (int i = 0; i < cfg->num_layers; i++) {
        if (ef->layer_data[i]) munmap(ef->layer_data[i], ef->layer_size[i]);
        if (ef->layer_fds[i] >= 0) close(ef->layer_fds[i]);
    }
    free(ef->layer_data);
    free(ef->layer_size);
    free(ef->layer_fds);
    free(ef->hot_mask);
    free(ef);
}

bool expert_is_hot(const ExpertFiles *ef, int layer, int expert_id) {
    if (!ef->hot_mask) return true;
    int words_per_layer = 8;
    int word = expert_id / 32;
    int bit = expert_id % 32;
    return (ef->hot_mask[layer * words_per_layer + word] >> bit) & 1;
}

// ============================================================================
// Expert forward — operates directly on mmap'd data, no copies
// ============================================================================

static void expert_forward_direct(const ModelConfig *cfg, const ExpertLayout *layout,
                                  const void *expert_base, const float *input,
                                  float *output, float *gate_buf, float *up_buf,
                                  float *act_buf) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int G = cfg->group_size;

    const uint8_t *data = expert_base;
    cpu_dequant_matvec((const uint32_t *)(data + layout->gate_w_off),
                       (const uint16_t *)(data + layout->gate_s_off),
                       (const uint16_t *)(data + layout->gate_b_off),
                       input, gate_buf, M, H, G);
    cpu_dequant_matvec((const uint32_t *)(data + layout->up_w_off),
                       (const uint16_t *)(data + layout->up_s_off),
                       (const uint16_t *)(data + layout->up_b_off),
                       input, up_buf, M, H, G);
    cpu_swiglu(gate_buf, up_buf, act_buf, M);
    cpu_dequant_matvec((const uint32_t *)(data + layout->down_w_off),
                       (const uint16_t *)(data + layout->down_s_off),
                       (const uint16_t *)(data + layout->down_b_off),
                       act_buf, output, H, M, G);
}

// ============================================================================
// MoE forward — pre-allocated scratch, zero-copy expert access
// ============================================================================

// Static scratch buffers (allocated once on first call, reused)
static float *s_gate_scores = NULL;
static float *s_shared_gate = NULL, *s_shared_up = NULL, *s_shared_act = NULL;
static float *s_shared_out = NULL;
static float s_shared_gate_score = 0.0f;
static float *s_expert_out[OROME_MAX_ACTIVE];
static float *s_expert_gate[OROME_MAX_ACTIVE];
static float *s_expert_up[OROME_MAX_ACTIVE];
static float *s_expert_act[OROME_MAX_ACTIVE];
static int s_moe_alloc_H = 0;
static int s_moe_alloc_M = 0;
static int s_moe_alloc_S = 0;
static int s_moe_alloc_E = 0;

static void ensure_scratch(const ModelConfig *cfg) {
    int H = cfg->hidden_dim;
    int M = cfg->moe_intermediate;
    int S = cfg->shared_intermediate;
    int E = cfg->num_experts;
    if (s_moe_alloc_H >= H && s_moe_alloc_M >= M &&
        s_moe_alloc_S >= S && s_moe_alloc_E >= E) return;

    free(s_gate_scores);
    free(s_shared_gate); free(s_shared_up); free(s_shared_act); free(s_shared_out);
    s_gate_scores = calloc(E, sizeof(float));
    s_shared_gate = calloc(S, sizeof(float));
    s_shared_up   = calloc(S, sizeof(float));
    s_shared_act  = calloc(S, sizeof(float));
    s_shared_out  = calloc(H, sizeof(float));

    for (int k = 0; k < OROME_MAX_ACTIVE; k++) {
        free(s_expert_out[k]); free(s_expert_gate[k]);
        free(s_expert_up[k]); free(s_expert_act[k]);
        s_expert_out[k]  = calloc(H, sizeof(float));
        s_expert_gate[k] = calloc(M, sizeof(float));
        s_expert_up[k]   = calloc(M, sizeof(float));
        s_expert_act[k]  = calloc(M, sizeof(float));
    }

    s_moe_alloc_H = H;
    s_moe_alloc_M = M;
    s_moe_alloc_S = S;
    s_moe_alloc_E = E;
}

void moe_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                 int layer_idx, float *hidden, float *h_post,
                 ExpertFiles *ef, int K, QuantType quant) {
    int H = cfg->hidden_dim;
    int n_experts = cfg->num_experts;
    void *layer_data = ef->layer_data[layer_idx];
    if (!layer_data || K <= 0) return;

    const ExpertLayout *layout = (quant == QUANT_2BIT) ? &cfg->expert_2bit : &cfg->expert_4bit;
    ensure_scratch(cfg);

    // 1. Routing + shared expert projections — batch all h_post-input matvecs
    int expert_indices[OROME_MAX_ACTIVE];
    float expert_weights[OROME_MAX_ACTIVE];

    uint32_t *gate_w = weights_layer_ptr(wf, layer_idx, "mlp.gate.weight");
    uint16_t *gate_s = weights_layer_ptr(wf, layer_idx, "mlp.gate.scales");
    uint16_t *gate_b = weights_layer_ptr(wf, layer_idx, "mlp.gate.biases");
    if (!gate_w || !gate_s || !gate_b) return;

    uint32_t *sg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.weight");
    uint16_t *sg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.scales");
    uint16_t *sg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.biases");
    uint32_t *su_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.weight");
    uint16_t *su_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.scales");
    uint16_t *su_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.biases");
    uint32_t *sd_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.weight");
    uint16_t *sd_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.scales");
    uint16_t *sd_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.biases");
    uint32_t *sgg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.weight");
    uint16_t *sgg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.scales");
    uint16_t *sgg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert_gate.biases");

    int S = cfg->shared_intermediate;
    memset(s_shared_out, 0, H * sizeof(float));

    // Batch: routing gate + shared gate_proj + shared up_proj + shared_expert_gate
    // All use h_post as input. 4 dispatches → 1 GPU command buffer.
    // Shared gate/up go to dedicated GPU buffers for later GPU SwiGLU+down.
    if (ctx && ctx->buf_weights && sg_w && su_w && sgg_w) {
        memcpy([ctx->buf_input contents], h_post, H * sizeof(float));
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        size_t sgg_off = n_experts * sizeof(float);
        GpuMatvecJob jobs[4] = {
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)gate_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)gate_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)gate_b - base,
              .out_buf = ctx->buf_output, .out_off = 0,
              .out_ptr = s_gate_scores, .out_dim = n_experts, .in_dim = H,
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
              .out_ptr = &s_shared_gate_score, .out_dim = 1, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
        };
        gpu_run_matvec_batch(ctx, jobs, 4);
    } else {
        fast_dequant_matvec(ctx, cfg, gate_w, gate_s, gate_b, h_post, s_gate_scores,
                            n_experts, H, QUANT_4BIT);
        if (sg_w) fast_dequant_matvec(ctx, cfg, sg_w, sg_s, sg_b, h_post, s_shared_gate,
                                       S, H, QUANT_4BIT);
        if (su_w) fast_dequant_matvec(ctx, cfg, su_w, su_s, su_b, h_post, s_shared_up,
                                       S, H, QUANT_4BIT);
        if (sgg_w) fast_dequant_matvec(ctx, cfg, sgg_w, sgg_s, sgg_b, h_post,
                                        &s_shared_gate_score, 1, H, QUANT_4BIT);
    }

    cpu_softmax(s_gate_scores, n_experts);
    cpu_topk(s_gate_scores, n_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    // 3. Routed experts — GPU if expert Metal buffers available, else CPU
    id<MTLBuffer> expert_layer_buf = (ctx && ctx->buf_expert_layers)
                                      ? ctx->buf_expert_layers[layer_idx] : nil;

    if (expert_layer_buf && ctx->buf_weights && sd_w) {
        // GPU expert+shared forward: all in ONE command buffer
        // Shared gate/up already in GPU buffers from routing batch above
        memcpy([ctx->buf_input contents], h_post, H * sizeof(float));

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        int M = cfg->moe_intermediate;
        id<MTLComputePipelineState> mv_pipe = (quant == QUANT_2BIT)
                                               ? ctx->matvec_2bit : ctx->matvec_4bit;
        uint8_t *wbase = (uint8_t *)[ctx->buf_weights contents];

        // Phase 1: gate + up projections for all K routed experts
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)M + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        // Phase 2: SwiGLU for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:2];
            uint dim_val = (uint)M;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)M + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        // Shared expert SwiGLU
        {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
            uint dim_val = (uint)S;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)S + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Phase 3: down projections for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_b_off atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        // Shared expert down
        {
            [enc setComputePipelineState:ctx->matvec_4bit];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_w - wbase atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_s - wbase atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_b - wbase atIndex:2];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Read back expert results
        for (int k = 0; k < K; k++) {
            memcpy(s_expert_out[k], [ctx->buf_multi_expert_out[k] contents],
                   H * sizeof(float));
        }
        memcpy(s_shared_out, [ctx->buf_shared_out contents], H * sizeof(float));

        // Apply shared expert gate
        float sw = cpu_sigmoid(s_shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;
    } else {
        // CPU fallback: shared expert
        if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post, s_shared_gate, S, H, cfg->group_size);
        if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post, s_shared_up, S, H, cfg->group_size);
        cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
        if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act, s_shared_out, H, S, cfg->group_size);
        float sw = cpu_sigmoid(s_shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

        // CPU fallback: routed experts
        for (int k = 0; k < K; k++) {
            const void *expert_base = (const uint8_t *)layer_data +
                (size_t)expert_indices[k] * layout->expert_size;
            expert_forward_direct(cfg, layout, expert_base, h_post,
                                  s_expert_out[k], s_expert_gate[k],
                                  s_expert_up[k], s_expert_act[k]);
        }
    }

    // 4. Combine: hidden += sum(weight[k] * expert_out[k]) + shared_out
    for (int k = 0; k < K; k++) {
        cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
    }
    cpu_vec_add(hidden, s_shared_out, H);
}

// ============================================================================
// MoE forward with pre-computed routing — skips routing GPU batch
// Assumes: gate_scores computed, shared expert gate/up already in GPU buffers,
//          h_post already in ctx->buf_input.
// ============================================================================

void moe_forward_routed(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                        int layer_idx, float *hidden, float *h_post,
                        float *gate_scores, float shared_gate_score,
                        ExpertFiles *ef, int K, QuantType quant,
                        bool gpu_combine) {
    int H = cfg->hidden_dim;
    int n_experts = cfg->num_experts;
    void *layer_data = ef->layer_data[layer_idx];
    if (!layer_data || K <= 0) return;

    const ExpertLayout *layout = (quant == QUANT_2BIT) ? &cfg->expert_2bit : &cfg->expert_4bit;
    ensure_scratch(cfg);

    // 1. Softmax + topk on pre-computed gate scores
    int expert_indices[OROME_MAX_ACTIVE];
    float expert_weights[OROME_MAX_ACTIVE];

    cpu_softmax(gate_scores, n_experts);
    cpu_topk(gate_scores, n_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    // 2. Shared expert down weights
    uint32_t *sd_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.weight");
    uint16_t *sd_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.scales");
    uint16_t *sd_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.down_proj.biases");
    int S = cfg->shared_intermediate;
    memset(s_shared_out, 0, H * sizeof(float));

    // 3. Routed experts + shared expert — GPU path
    // h_post is already in ctx->buf_input from fused command buffer
    // Shared gate/up already in GPU buffers from fused command buffer
    id<MTLBuffer> expert_layer_buf = (ctx && ctx->buf_expert_layers)
                                      ? ctx->buf_expert_layers[layer_idx] : nil;

    if (expert_layer_buf && ctx->buf_weights && sd_w) {
        // buf_input already has h_post — no memcpy needed
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        int M = cfg->moe_intermediate;
        id<MTLComputePipelineState> mv_pipe = (quant == QUANT_2BIT)
                                               ? ctx->matvec_2bit : ctx->matvec_4bit;
        uint8_t *wbase = (uint8_t *)[ctx->buf_weights contents];

        // Phase 1: gate + up projections for all K routed experts
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->gate_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)M + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->up_b_off atIndex:2];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:4];
            {
                uint od = (uint)M, id_ = (uint)H, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        // Phase 2: SwiGLU for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k] offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:2];
            uint dim_val = (uint)M;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)M + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        // Shared expert SwiGLU
        {
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_shared_up offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:2];
            uint dim_val = (uint)S;
            [enc setBytes:&dim_val length:sizeof(uint) atIndex:3];
            NSUInteger swi_tgs = ((uint)S + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swi_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Phase 3: down projections for all K experts + shared expert
        for (int k = 0; k < K; k++) {
            size_t expert_off = (size_t)expert_indices[k] * layout->expert_size;

            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_w_off atIndex:0];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_s_off atIndex:1];
            [enc setBuffer:expert_layer_buf offset:expert_off + layout->down_b_off atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)M, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        // Shared expert down
        {
            [enc setComputePipelineState:ctx->matvec_4bit];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_w - wbase atIndex:0];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_s - wbase atIndex:1];
            [enc setBuffer:ctx->buf_weights offset:(uint8_t *)sd_b - wbase atIndex:2];
            [enc setBuffer:ctx->buf_shared_act offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:4];
            {
                uint od = (uint)H, id_ = (uint)S, gs = (uint)cfg->group_size;
                [enc setBytes:&od length:sizeof(uint) atIndex:5];
                [enc setBytes:&id_ length:sizeof(uint) atIndex:6];
                [enc setBytes:&gs length:sizeof(uint) atIndex:7];
            }
            NSUInteger tg_size = ROWS_PER_TG * 32;
            NSUInteger num_tgs = ((uint)H + ROWS_PER_TG - 1) / ROWS_PER_TG;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }

        if (gpu_combine && ctx->moe_combine) {
            // Add barrier before combine
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Fill combine params: weights[0..K-1] + shared_gate_score
            float *params = (float *)[ctx->buf_combine_params contents];
            for (int k = 0; k < K; k++) params[k] = expert_weights[k];
            params[8] = shared_gate_score;

            // Dispatch moe_combine: buf_moe_hidden += experts + shared
            [enc setComputePipelineState:ctx->moe_combine];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:0];  // h_mid (pre-MoE residual)
            [enc setBuffer:ctx->buf_shared_out offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_moe_hidden offset:0 atIndex:2];  // hidden_out (in-place)
            for (int k = 0; k < 8 && k < K; k++) {
                [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:3 + k];
            }
            // Fill remaining expert slots with expert_out0 (won't be used, K check in kernel)
            for (int k = K; k < 8; k++) {
                [enc setBuffer:ctx->buf_multi_expert_out[0] offset:0 atIndex:3 + k];
            }
            [enc setBuffer:ctx->buf_combine_params offset:0 atIndex:11];
            { uint d = (uint)H, kk = (uint)K;
              [enc setBytes:&d length:sizeof(uint) atIndex:12];
              [enc setBytes:&kk length:sizeof(uint) atIndex:13]; }
            [enc dispatchThreadgroups:MTLSizeMake(((uint)H + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
            [cmd commit];
            // DON'T wait — caller handles synchronization via queue ordering
        } else {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Read back expert results
            for (int k = 0; k < K; k++) {
                memcpy(s_expert_out[k], [ctx->buf_multi_expert_out[k] contents],
                       H * sizeof(float));
            }
            memcpy(s_shared_out, [ctx->buf_shared_out contents], H * sizeof(float));

            // Apply shared expert gate
            float sw = cpu_sigmoid(shared_gate_score);
            for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

            // CPU combine
            for (int k = 0; k < K; k++) {
                cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
            }
            cpu_vec_add(hidden, s_shared_out, H);
        }
    } else {
        // CPU fallback
        uint32_t *sg_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.weight");
        uint16_t *sg_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.scales");
        uint16_t *sg_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.gate_proj.biases");
        uint32_t *su_w = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.weight");
        uint16_t *su_s = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.scales");
        uint16_t *su_b = weights_layer_ptr(wf, layer_idx, "mlp.shared_expert.up_proj.biases");
        if (sg_w) cpu_dequant_matvec((void *)sg_w, sg_s, sg_b, h_post, s_shared_gate, S, H, cfg->group_size);
        if (su_w) cpu_dequant_matvec((void *)su_w, su_s, su_b, h_post, s_shared_up, S, H, cfg->group_size);
        cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
        if (sd_w) cpu_dequant_matvec((void *)sd_w, sd_s, sd_b, s_shared_act, s_shared_out, H, S, cfg->group_size);
        float sw = cpu_sigmoid(shared_gate_score);
        for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

        for (int k = 0; k < K; k++) {
            const void *expert_base = (const uint8_t *)layer_data +
                (size_t)expert_indices[k] * layout->expert_size;
            expert_forward_direct(cfg, layout, expert_base, h_post,
                                  s_expert_out[k], s_expert_gate[k],
                                  s_expert_up[k], s_expert_act[k]);
        }

        // CPU combine
        for (int k = 0; k < K; k++) {
            cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
        }
        cpu_vec_add(hidden, s_shared_out, H);
    }
}
