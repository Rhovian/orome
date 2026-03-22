/*
 * moe.m — Mixture of Experts: routing, expert I/O, expert forward pass.
 *
 * On machines with enough RAM (e.g. 96GB M2 Max), all expert weights are
 * mmap'd at startup — zero per-token I/O overhead. Expert data is accessed
 * directly from the mmap'd region with no copies.
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "orome.h"

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
    if (ctx && ctx->buf_weights && sg_w && su_w && sgg_w) {
        memcpy([ctx->buf_input contents], h_post, H * sizeof(float));
        uint8_t *base = (uint8_t *)[ctx->buf_weights contents];
        // Pack outputs into buf_output at different offsets
        size_t sg_off = n_experts * sizeof(float);
        size_t su_off = sg_off + S * sizeof(float);
        size_t sgg_off = su_off + S * sizeof(float);
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
              .out_buf = ctx->buf_output, .out_off = sg_off,
              .out_ptr = s_shared_gate, .out_dim = S, .in_dim = H,
              .group_size = cfg->group_size, .is_2bit = false },
            { .w_buf = ctx->buf_weights, .w_off = (uint8_t *)su_w - base,
              .s_buf = ctx->buf_weights, .s_off = (uint8_t *)su_s - base,
              .b_buf = ctx->buf_weights, .b_off = (uint8_t *)su_b - base,
              .out_buf = ctx->buf_output, .out_off = su_off,
              .out_ptr = s_shared_up, .out_dim = S, .in_dim = H,
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

    // 2. Shared expert SwiGLU + down projection
    cpu_swiglu(s_shared_gate, s_shared_up, s_shared_act, S);
    if (sd_w) fast_dequant_matvec(ctx, cfg, sd_w, sd_s, sd_b, s_shared_act, s_shared_out,
                                   H, S, QUANT_4BIT);

    float sw = cpu_sigmoid(s_shared_gate_score);
    for (int i = 0; i < H; i++) s_shared_out[i] *= sw;

    // 3. Routed experts — zero-copy from mmap'd data
    for (int k = 0; k < K; k++) {
        const void *expert_base = (const uint8_t *)layer_data +
            (size_t)expert_indices[k] * layout->expert_size;
        expert_forward_direct(cfg, layout, expert_base, h_post,
                              s_expert_out[k], s_expert_gate[k],
                              s_expert_up[k], s_expert_act[k]);
    }

    // 4. Combine: hidden += sum(weight[k] * expert_out[k]) + shared_out
    for (int k = 0; k < K; k++) {
        cpu_vec_madd(hidden, s_expert_out[k], expert_weights[k], H);
    }
    cpu_vec_add(hidden, s_shared_out, H);
}
