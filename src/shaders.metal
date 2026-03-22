/*
 * shaders.metal — Phase 1 kernels for Qwen3.5-35B-A3B
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================

#define ROWS_PER_TG 16

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];

    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float x0 = float(x_shared[x_base + 0]);
        float x1 = float(x_shared[x_base + 1]);
        float x2 = float(x_shared[x_base + 2]);
        float x3 = float(x_shared[x_base + 3]);
        float x4 = float(x_shared[x_base + 4]);
        float x5 = float(x_shared[x_base + 5]);
        float x6 = float(x_shared[x_base + 6]);
        float x7 = float(x_shared[x_base + 7]);

        float sx0 = scale * x0;  float bx0 = bias * x0;
        float sx1 = scale * x1;  float bx1 = bias * x1;
        float sx2 = scale * x2;  float bx2 = bias * x2;
        float sx3 = scale * x3;  float bx3 = bias * x3;
        float sx4 = scale * x4;  float bx4 = bias * x4;
        float sx5 = scale * x5;  float bx5 = bias * x5;
        float sx6 = scale * x6;  float bx6 = bias * x6;
        float sx7 = scale * x7;  float bx7 = bias * x7;

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);

    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Batched multi-expert 4-bit matvec — processes K experts in one dispatch
// Grid: 2D (row_groups × K), threadgroup: ROWS_PER_TG * 32
// ============================================================================

struct ExpertOffsets {
    uint w_off;  // byte offset to weights within expert layer data
    uint s_off;  // byte offset to scales
    uint b_off;  // byte offset to biases
};

kernel void batch_expert_matvec_4bit(
    device const uint8_t*  layer_data  [[buffer(0)]],  // full layer expert data
    device const float*    x           [[buffer(1)]],  // input vector [in_dim]
    device float*          out         [[buffer(2)]],  // packed output [K * out_dim]
    constant ExpertOffsets* offsets     [[buffer(3)]],  // [K] per-expert offsets
    constant uint&         out_dim     [[buffer(4)]],
    constant uint&         in_dim      [[buffer(5)]],
    constant uint&         group_size  [[buffer(6)]],
    constant uint&         num_row_tgs [[buffer(7)]],   // = ceil(out_dim / ROWS_PER_TG)
    uint tgid    [[threadgroup_position_in_grid]],      // linearized: row_group + expert * num_row_tgs
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* W = (device const uint32_t*)(layer_data + offsets[expert].w_off);
    device const uint16_t* S = (device const uint16_t*)(layer_data + offsets[expert].s_off);
    device const uint16_t* B = (device const uint16_t*)(layer_data + offsets[expert].b_off);

    device const uint32_t* w_row = W + row * packed_cols;
    device const uint16_t* s_row = S + row * num_groups;
    device const uint16_t* b_row = B + row * num_groups;

    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert * out_dim + row] = sum;
    }
}

// Batched multi-expert 4-bit matvec with per-expert packed input
// Same as batch_expert_matvec_4bit but input is packed [K * in_dim]
kernel void batch_expert_down_4bit(
    device const uint8_t*  layer_data  [[buffer(0)]],
    device const float*    x           [[buffer(1)]],  // packed [K * in_dim]
    device float*          out         [[buffer(2)]],  // packed [K * out_dim]
    constant ExpertOffsets* offsets     [[buffer(3)]],
    constant uint&         out_dim     [[buffer(4)]],
    constant uint&         in_dim      [[buffer(5)]],
    constant uint&         group_size  [[buffer(6)]],
    constant uint&         num_row_tgs [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    device const float* x_expert = x + expert * in_dim;
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x_expert[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* W = (device const uint32_t*)(layer_data + offsets[expert].w_off);
    device const uint16_t* S = (device const uint16_t*)(layer_data + offsets[expert].s_off);
    device const uint16_t* B = (device const uint16_t*)(layer_data + offsets[expert].b_off);

    device const uint32_t* w_row = W + row * packed_cols;
    device const uint16_t* s_row = S + row * num_groups;
    device const uint16_t* b_row = B + row * num_groups;

    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert * out_dim + row] = sum;
    }
}

// Batched SwiGLU — processes K experts' activations packed sequentially
kernel void batch_swiglu(
    device const float* gate [[buffer(0)]],  // [K * dim] packed
    device const float* up   [[buffer(1)]],  // [K * dim] packed
    device float*       out  [[buffer(2)]],  // [K * dim] packed
    constant uint&      total_dim [[buffer(3)]],  // K * dim
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_dim) return;
    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 16;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];

    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        float sx0 = scale * x_shared[x_base + 0];   float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];   float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];   float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];   float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];   float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];   float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];   float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];   float bx7 = bias * x_shared[x_base + 7];
        float sx8 = scale * x_shared[x_base + 8];   float bx8 = bias * x_shared[x_base + 8];
        float sx9 = scale * x_shared[x_base + 9];   float bx9 = bias * x_shared[x_base + 9];
        float sx10 = scale * x_shared[x_base + 10]; float bx10 = bias * x_shared[x_base + 10];
        float sx11 = scale * x_shared[x_base + 11]; float bx11 = bias * x_shared[x_base + 11];
        float sx12 = scale * x_shared[x_base + 12]; float bx12 = bias * x_shared[x_base + 12];
        float sx13 = scale * x_shared[x_base + 13]; float bx13 = bias * x_shared[x_base + 13];
        float sx14 = scale * x_shared[x_base + 14]; float bx14 = bias * x_shared[x_base + 14];
        float sx15 = scale * x_shared[x_base + 15]; float bx15 = bias * x_shared[x_base + 15];

        acc += fma(float((packed >>  0) & 0x3), sx0, bx0);
        acc += fma(float((packed >>  2) & 0x3), sx1, bx1);
        acc += fma(float((packed >>  4) & 0x3), sx2, bx2);
        acc += fma(float((packed >>  6) & 0x3), sx3, bx3);
        acc += fma(float((packed >>  8) & 0x3), sx4, bx4);
        acc += fma(float((packed >> 10) & 0x3), sx5, bx5);
        acc += fma(float((packed >> 12) & 0x3), sx6, bx6);
        acc += fma(float((packed >> 14) & 0x3), sx7, bx7);
        acc += fma(float((packed >> 16) & 0x3), sx8, bx8);
        acc += fma(float((packed >> 18) & 0x3), sx9, bx9);
        acc += fma(float((packed >> 20) & 0x3), sx10, bx10);
        acc += fma(float((packed >> 22) & 0x3), sx11, bx11);
        acc += fma(float((packed >> 24) & 0x3), sx12, bx12);
        acc += fma(float((packed >> 26) & 0x3), sx13, bx13);
        acc += fma(float((packed >> 28) & 0x3), sx14, bx14);
        acc += fma(float((packed >> 30) & 0x3), sx15, bx15);
    }

    float sum = simd_sum(acc);

    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}

// ============================================================================
// Kernel 5: Residual add
// ============================================================================

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}

// Fused residual add + per-TG partial sum of squares (for subsequent RMS norm)
// Each threadgroup writes its partial sum_sq to sum_sq_parts[tgid]
kernel void residual_add_sum_sq(
    device const float* a          [[buffer(0)]],
    device const float* b          [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    device float*       sum_sq_parts [[buffer(3)]],
    constant uint&      dim        [[buffer(4)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float val = 0.0f;
    if (tid < dim) {
        val = a[tid] + b[tid];
        out[tid] = val;
    }

    // Compute partial sum of squares within this threadgroup
    float sq = val * val;
    float simd_val = simd_sum(sq);

    threadgroup float shared[32];
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) {
            sum_sq_parts[tgid] = v;
        }
    }
}

// RMS norm apply with partial sum_sq (reads N partial sums, computes total)
kernel void rms_norm_apply_partial(
    device const float*    x           [[buffer(0)]],
    device const uint16_t* weight      [[buffer(1)]],
    device const float*    sum_sq_parts [[buffer(2)]],
    device float*          out         [[buffer(3)]],
    constant uint&         dim         [[buffer(4)]],
    constant float&        eps         [[buffer(5)]],
    constant uint&         num_parts   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Sum partial sums (num_parts is small, e.g. 8)
    float total_sq = 0.0f;
    for (uint i = 0; i < num_parts; i++) {
        total_sq += sum_sq_parts[i];
    }

    float rms = rsqrt(total_sq / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}

// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device half *state,              // [64 * 128 * 128] persistent state (half precision)
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Load k and q into threadgroup memory (shared by all 128 threads)
    threadgroup float k_shared[128];
    threadgroup float q_shared[128];
    if (vi < 128) {
        k_shared[vi] = k[k_base + vi];
        q_shared[vi] = q[k_base + vi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    // Read half → float for compute, write back as half (halves bandwidth)
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = float(state[state_base + ki]) * g;
        state[state_base + ki] = half(s);
        kv_mem += s * k_shared[ki];
    }

    // Step 3+4+5: Delta update + output in one pass (saves 1 full state read)
    float delta = (v[v_base + vi] - kv_mem) * beta;
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = float(state[state_base + ki]) + k_shared[ki] * delta;
        state[state_base + ki] = half(s);
        out_val += s * q_shared[ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q — simd reduction instead of serial
    float qval = (tid < key_dim) ? q[base + tid] : 0;
    float q_sq = qval * qval;
    float q_simd = simd_sum(q_sq);
    threadgroup float q_shared[4];
    if (simd_lane == 0) q_shared[simd_group] = q_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = 0;
    if (tid < 4) q_total = q_shared[tid];
    q_total = simd_sum(q_total);
    float q_inv_rms = rsqrt(q_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;
    }

    // RMS norm for k — simd reduction
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    float k_sq = kval * kval;
    float k_simd = simd_sum(k_sq);
    threadgroup float k_shared[4];
    if (simd_lane == 0) k_shared[simd_group] = k_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_total = 0;
    if (tid < 4) k_total = k_shared[tid];
    k_total = simd_sum(k_total);
    float k_inv_rms = rsqrt(k_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}

// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction using simd_sum (128 threads = 4 simdgroups)
    float sq = val * val;
    float simd_val = simd_sum(sq);
    threadgroup float shared_sums[4];
    if (simd_lane == 0) shared_sums[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    if (tid < 4) total = shared_sums[tid];
    total = simd_sum(total);  // first simdgroup reduces
    float inv_rms = rsqrt(total / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}

// ============================================================================
// Kernel 17: Per-head weighted RMS norm for full attention Q/K
// ============================================================================
// Q: [num_q_heads * head_dim], K: [num_kv_heads * head_dim]
// Apply weighted RMS norm per head (separate weights for Q and K)
// Q heads also get scaled by inv_scale
// Dispatch: max(num_q_heads, num_kv_heads) threadgroups, head_dim threads each

kernel void rms_norm_qk_weighted(
    device float *q,                  // [num_q_heads * head_dim] in/out
    device float *k,                  // [num_kv_heads * head_dim] in/out
    device const uint16_t *q_weight,  // [head_dim] bf16 norm weight for Q
    device const uint16_t *k_weight,  // [head_dim] bf16 norm weight for K
    constant uint &head_dim,          // = 256
    constant uint &num_q_heads,       // = 16
    constant uint &num_kv_heads,      // = 2
    constant float &inv_scale,        // = 1/sqrt(head_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Process Q head if in range
    if (head < num_q_heads && tid < head_dim) {
        uint base = head * head_dim;
        float qval = q[base + tid];

        // simd reduction for sum of squares (256 threads = 8 simdgroups)
        float q_sq = qval * qval;
        float q_simd = simd_sum(q_sq);
        threadgroup float q_shared[8];
        if (simd_lane == 0) q_shared[simd_group] = q_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = (tid < 8) ? q_shared[tid] : 0;
        q_total = simd_sum(q_total);

        float inv_rms = rsqrt(q_total / float(head_dim) + 1e-6f);
        float w = bf16_to_f32(q_weight[tid]);
        q[base + tid] = qval * inv_rms * w * inv_scale;
    }

    // Process K head if in range
    if (head < num_kv_heads && tid < head_dim) {
        uint base = head * head_dim;
        float kval = k[base + tid];

        float k_sq = kval * kval;
        float k_simd = simd_sum(k_sq);
        threadgroup float k_shared[8];
        if (simd_lane == 0) k_shared[simd_group] = k_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float k_total = (tid < 8) ? k_shared[tid] : 0;
        k_total = simd_sum(k_total);

        float inv_rms = rsqrt(k_total / float(head_dim) + 1e-6f);
        float w = bf16_to_f32(k_weight[tid]);
        k[base + tid] = kval * inv_rms * w;
    }
}

// ============================================================================
// Kernel 18: RoPE (Rotary Position Embedding)
// ============================================================================
// Apply partial rotary embedding to Q and K.
// Q: [num_q_heads * head_dim], K: [num_kv_heads * head_dim]
// Only rotary_dim elements per head are rotated (first half_rot pairs).
// Dispatch: max(num_q_heads, num_kv_heads) threadgroups, half_rot threads each

kernel void rope_apply(
    device float *q,                  // [num_q_heads * head_dim] in/out
    device float *k,                  // [num_kv_heads * head_dim] in/out
    constant uint &head_dim,          // = 256
    constant uint &rotary_dim,        // = 64
    constant uint &num_q_heads,       // = 16
    constant uint &num_kv_heads,      // = 2
    constant uint &pos,               // current position
    constant float &theta,            // rope_theta = 10M
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint half_rot = rotary_dim / 2;
    if (tid >= half_rot) return;

    float freq = 1.0f / pow(theta, float(2 * tid) / float(rotary_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // Rotate Q head
    if (head < num_q_heads) {
        uint base = head * head_dim;
        float q0 = q[base + tid];
        float q1 = q[base + tid + half_rot];
        q[base + tid]            = q0 * cos_a - q1 * sin_a;
        q[base + tid + half_rot] = q0 * sin_a + q1 * cos_a;
    }

    // Rotate K head
    if (head < num_kv_heads) {
        uint base = head * head_dim;
        float k0 = k[base + tid];
        float k1 = k[base + tid + half_rot];
        k[base + tid]            = k0 * cos_a - k1 * sin_a;
        k[base + tid + half_rot] = k0 * sin_a + k1 * cos_a;
    }
}

// ============================================================================
// Kernel 19: KV cache write (scatter one token's K/V into cache)
// ============================================================================
// Write K[kv_dim] and V[kv_dim] at position pos into the KV cache.
// Cache layout: [max_seq * kv_dim], stored row-major.
// Dispatch: kv_dim threads

kernel void kv_cache_write(
    device const float *k_in,         // [kv_dim] new K vector
    device const float *v_in,         // [kv_dim] new V vector
    device float *k_cache,            // [max_seq * kv_dim]
    device float *v_cache,            // [max_seq * kv_dim]
    constant uint &kv_dim,
    constant uint &pos,               // write position
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= kv_dim) return;
    uint offset = pos * kv_dim + tid;
    k_cache[offset] = k_in[tid];
    v_cache[offset] = v_in[tid];
}

// ============================================================================
// Kernel 15: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// ============================================================================
// Kernel 16: MoE combine + residual + shared expert gate
// ============================================================================

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],
    device const float* shared_out  [[buffer(1)]],
    device float*       hidden_out  [[buffer(2)]],
    device const float* expert_out0 [[buffer(3)]],
    device const float* expert_out1 [[buffer(4)]],
    device const float* expert_out2 [[buffer(5)]],
    device const float* expert_out3 [[buffer(6)]],
    device const float* expert_out4 [[buffer(7)]],
    device const float* expert_out5 [[buffer(8)]],
    device const float* expert_out6 [[buffer(9)]],
    device const float* expert_out7 [[buffer(10)]],
    device const float* params      [[buffer(11)]],
    constant uint&      dim         [[buffer(12)]],
    constant uint&      K           [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float shared_gate = 1.0f / (1.0f + exp(-params[8]));

    float moe = 0.0f;
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

// MoE combine with packed expert output buffer [K * dim]
kernel void moe_combine_residual_packed(
    device const float* h_mid       [[buffer(0)]],
    device const float* shared_out  [[buffer(1)]],
    device float*       hidden_out  [[buffer(2)]],
    device const float* expert_out  [[buffer(3)]],  // packed [K * dim]
    device const float* params      [[buffer(4)]],  // [0..K-1]=weights, [8]=shared_gate_score
    constant uint&      dim         [[buffer(5)]],
    constant uint&      K           [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float shared_gate = 1.0f / (1.0f + exp(-params[8]));

    float moe = 0.0f;
    for (uint k = 0; k < K && k < 8; k++) {
        moe += params[k] * expert_out[k * dim + tid];
    }

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

// MoE combine + copy residual + partial sum_sq for next layer's norm
// Fuses moe_combine_packed + copy_buffer + norm_sum_sq into 1 dispatch
// Saves 2 dispatches + 2 barriers per layer
kernel void moe_combine_copy_sq(
    device const float* h_mid        [[buffer(0)]],
    device const float* shared_out   [[buffer(1)]],
    device float*       hidden_out   [[buffer(2)]],
    device const float* expert_out   [[buffer(3)]],
    device const float* params       [[buffer(4)]],
    constant uint&      dim          [[buffer(5)]],
    constant uint&      K            [[buffer(6)]],
    device float*       residual_out [[buffer(7)]],
    device float*       sum_sq_parts [[buffer(8)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float val = 0.0f;
    if (tid < dim) {
        float shared_gate = 1.0f / (1.0f + exp(-params[8]));
        float moe = 0.0f;
        for (uint k = 0; k < K && k < 8; k++) {
            moe += params[k] * expert_out[k * dim + tid];
        }
        val = h_mid[tid] + moe + shared_gate * shared_out[tid];
        hidden_out[tid] = val;
        residual_out[tid] = val;
    }

    // Partial sum of squares for next layer's RMS norm
    float sq = val * val;
    float simd_val = simd_sum(sq);
    threadgroup float shared[32];
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) {
            sum_sq_parts[tgid] = v;
        }
    }
}

// ============================================================================
// GPU softmax + top-K routing — eliminates CPU readback sync point
// ============================================================================

kernel void softmax_topk_route(
    device const float* logits       [[buffer(0)]],
    device uint32_t*    out_indices  [[buffer(1)]],
    device float*       out_params   [[buffer(2)]],
    constant uint&      n_experts    [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant uint&      sgg_off_f    [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float vals[256];
    if (lid < n_experts) vals[lid] = logits[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float reduce[32];
    float local_max = (lid < n_experts) ? vals[lid] : -1e30f;
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    if (simd_lane == 0) reduce[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < (tg_size + 31) / 32) {
        global_max = simd_max(reduce[simd_lane]);
    }
    threadgroup float bcast;
    if (lid == 0) bcast = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = bcast;

    float exp_val = (lid < n_experts) ? exp(vals[lid] - global_max) : 0.0f;
    if (lid < n_experts) vals[lid] = exp_val;
    float ss = simd_sum(exp_val);
    if (simd_lane == 0) reduce[simd_group] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < (tg_size + 31) / 32) {
        global_sum = simd_sum(reduce[simd_lane]);
    }
    if (lid == 0) bcast = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = bcast;

    if (lid < n_experts) vals[lid] = exp_val / global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        for (uint k = 0; k < K && k < 8; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint i = 0; i < n_experts; i++) {
                if (vals[i] > best_val) {
                    best_val = vals[i];
                    best_idx = i;
                }
            }
            out_indices[k] = best_idx;
            out_params[k] = best_val;
            vals[best_idx] = -1.0f;
        }
        float wsum = 0.0f;
        for (uint k = 0; k < K && k < 8; k++) wsum += out_params[k];
        float inv_wsum = 1.0f / wsum;
        for (uint k = 0; k < K && k < 8; k++) out_params[k] *= inv_wsum;
        out_params[8] = logits[sgg_off_f];
    }
}

// ============================================================================
// Fused expert gate+up+SwiGLU — compute both projections and activation in one dispatch
// ============================================================================
// Eliminates separate up projection and SwiGLU dispatches + 1 barrier.
// Each simdgroup computes both gate and up dot products for one output row,
// then applies SwiGLU inline. x_shared is loaded once instead of twice.

kernel void expert_gate_up_swiglu_dyn(
    device const uint8_t*  layer_data     [[buffer(0)]],
    device const float*    x              [[buffer(1)]],
    device float*          act_out        [[buffer(2)]],   // [K * out_dim]
    device const uint32_t* expert_indices [[buffer(3)]],
    constant uint&         expert_size    [[buffer(4)]],
    constant uint&         gate_w_off     [[buffer(5)]],
    constant uint&         gate_s_off     [[buffer(6)]],
    constant uint&         gate_b_off     [[buffer(7)]],
    constant uint&         up_w_off       [[buffer(8)]],
    constant uint&         up_s_off       [[buffer(9)]],
    constant uint&         up_b_off       [[buffer(10)]],
    constant uint&         out_dim        [[buffer(11)]],
    constant uint&         in_dim         [[buffer(12)]],
    constant uint&         group_size     [[buffer(13)]],
    constant uint&         num_row_tgs    [[buffer(14)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    uint expert_id = expert_indices[expert_k];
    uint ebase = expert_id * expert_size;

    device const uint32_t* gW = (device const uint32_t*)(layer_data + ebase + gate_w_off);
    device const uint16_t* gS = (device const uint16_t*)(layer_data + ebase + gate_s_off);
    device const uint16_t* gB = (device const uint16_t*)(layer_data + ebase + gate_b_off);

    device const uint32_t* uW = (device const uint32_t*)(layer_data + ebase + up_w_off);
    device const uint16_t* uS = (device const uint16_t*)(layer_data + ebase + up_s_off);
    device const uint16_t* uB = (device const uint16_t*)(layer_data + ebase + up_b_off);

    device const uint32_t* g_row = gW + row * packed_cols;
    device const uint16_t* gs_row = gS + row * num_groups;
    device const uint16_t* gb_row = gB + row * num_groups;

    device const uint32_t* u_row = uW + row * packed_cols;
    device const uint16_t* us_row = uS + row * num_groups;
    device const uint16_t* ub_row = uB + row * num_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        uint x_base = col * 8;

        float g_scale = bf16_to_f32(gs_row[g]);
        float g_bias  = bf16_to_f32(gb_row[g]);
        uint32_t g_packed = g_row[col];

        float u_scale = bf16_to_f32(us_row[g]);
        float u_bias  = bf16_to_f32(ub_row[g]);
        uint32_t u_packed = u_row[col];

        for (uint b = 0; b < 8; b++) {
            float xv = x_shared[x_base + b];
            float gsx = g_scale * xv;
            float gbx = g_bias * xv;
            float usx = u_scale * xv;
            float ubx = u_bias * xv;
            gate_acc += fma(float((g_packed >> (b * 4)) & 0xF), gsx, gbx);
            up_acc   += fma(float((u_packed >> (b * 4)) & 0xF), usx, ubx);
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum   = simd_sum(up_acc);

    if (simd_lane == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        act_out[expert_k * out_dim + row] = silu_gate * up_sum;
    }
}

// ============================================================================
// Dynamic batch expert matvec — reads expert indices from GPU buffer
// ============================================================================

kernel void batch_expert_mv_dyn(
    device const uint8_t*  layer_data    [[buffer(0)]],
    device const float*    x             [[buffer(1)]],
    device float*          out           [[buffer(2)]],
    device const uint32_t* expert_indices [[buffer(3)]],
    constant uint&         expert_size   [[buffer(4)]],
    constant uint&         proj_w_off    [[buffer(5)]],
    constant uint&         proj_s_off    [[buffer(6)]],
    constant uint&         proj_b_off    [[buffer(7)]],
    constant uint&         out_dim       [[buffer(8)]],
    constant uint&         in_dim        [[buffer(9)]],
    constant uint&         group_size    [[buffer(10)]],
    constant uint&         num_row_tgs   [[buffer(11)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    uint expert_id = expert_indices[expert_k];
    uint base = expert_id * expert_size;

    device const uint32_t* W = (device const uint32_t*)(layer_data + base + proj_w_off);
    device const uint16_t* S = (device const uint16_t*)(layer_data + base + proj_s_off);
    device const uint16_t* B = (device const uint16_t*)(layer_data + base + proj_b_off);

    device const uint32_t* w_row = W + row * packed_cols;
    device const uint16_t* s_row = S + row * num_groups;
    device const uint16_t* b_row = B + row * num_groups;

    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}

// ============================================================================
// Simple buffer copy kernel (replaces blit encoder to stay in compute encoder)
// ============================================================================

kernel void copy_buffer(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    dst[tid] = src[tid];
}

// Dynamic batch expert down matvec — per-expert packed input, GPU-driven indices
kernel void batch_expert_down_dyn(
    device const uint8_t*  layer_data    [[buffer(0)]],
    device const float*    x             [[buffer(1)]],
    device float*          out           [[buffer(2)]],
    device const uint32_t* expert_indices [[buffer(3)]],
    constant uint&         expert_size   [[buffer(4)]],
    constant uint&         proj_w_off    [[buffer(5)]],
    constant uint&         proj_s_off    [[buffer(6)]],
    constant uint&         proj_b_off    [[buffer(7)]],
    constant uint&         out_dim       [[buffer(8)]],
    constant uint&         in_dim        [[buffer(9)]],
    constant uint&         group_size    [[buffer(10)]],
    constant uint&         num_row_tgs   [[buffer(11)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    device const float* x_expert = x + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x_expert[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    uint expert_id = expert_indices[expert_k];
    uint base = expert_id * expert_size;

    device const uint32_t* W = (device const uint32_t*)(layer_data + base + proj_w_off);
    device const uint16_t* S = (device const uint16_t*)(layer_data + base + proj_s_off);
    device const uint16_t* B = (device const uint16_t*)(layer_data + base + proj_b_off);

    device const uint32_t* w_row = W + row * packed_cols;
    device const uint16_t* s_row = S + row * num_groups;
    device const uint16_t* b_row = B + row * num_groups;

    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}
