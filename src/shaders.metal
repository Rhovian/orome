/*
 * shaders.metal — GPU kernels for orome inference engine
 */

#include <metal_stdlib>
using namespace metal;

// Shared x staging tile for quantized matvec kernels.
// Large dense projections stream x in chunks so input dim no longer needs
// to fit entirely in threadgroup memory.
#define MATVEC_X_SHARED_SIZE 8192

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

#define ROWS_PER_TG 16

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
    device const float* gate     [[buffer(8)]],  // [num_heads, head_dim] sigmoid gate
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
    // Fused sigmoid gate: out = attn_value * sigmoid(gate)
    float g = 1.0f / (1.0f + exp(-gate[h * head_dim + d]));
    out[h * head_dim + d] = acc * g;
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
// Dispatch: one threadgroup per v-head, value_dim threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [n_v_heads * 128 * 128] float, persisted across tokens.
// GGUF keeps V-head tensors grouped by modulo of the shared K-head rather than
// sequentially. In that layout, head_id % num_k_heads selects the owning K-head.

kernel void gated_delta_net_step(
    device half *state,              // [n_v_heads * 128 * 128] persistent state (half)
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &num_k_heads,      // shared K-head set size
    constant float &qk_inv_scale,    // = 1/sqrt(key_dim)
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint kh = head_id % num_k_heads;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Load raw k and q into threadgroup memory, then apply RMS norm inline.
    // This fuses the rms_norm_qk dispatch into the delta_net kernel.
    threadgroup float k_shared[128];
    threadgroup float q_shared[128];
    float raw_k = k[k_base + vi];
    float raw_q = q[k_base + vi];
    k_shared[vi] = raw_k;
    q_shared[vi] = raw_q;

    // RMS norm reduction for Q and K (128 threads = 4 simdgroups)
    threadgroup float q_sums[4];
    threadgroup float k_sums[4];
    float q_sq = raw_q * raw_q;
    float k_sq = raw_k * raw_k;
    float q_simd = simd_sum(q_sq);
    float k_simd = simd_sum(k_sq);
    if (simd_lane == 0) {
        q_sums[simd_group] = q_simd;
        k_sums[simd_group] = k_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float q_broadcast;
    threadgroup float k_broadcast;
    if (simd_group == 0) {
        float qv = (simd_lane < 4) ? q_sums[simd_lane] : 0;
        float q_total = simd_sum(qv);
        float kv = (simd_lane < 4) ? k_sums[simd_lane] : 0;
        float k_total = simd_sum(kv);
        if (simd_lane == 0) {
            q_broadcast = q_total;
            k_broadcast = k_total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply QK RMS norm: Q scaled by inv_scale^2, K by inv_scale
    float q_inv_rms = rsqrt(q_broadcast / 128.0f + 1e-6f);
    float k_inv_rms = rsqrt(k_broadcast / 128.0f + 1e-6f);
    q_shared[vi] = raw_q * q_inv_rms * qk_inv_scale * qk_inv_scale;
    k_shared[vi] = raw_k * k_inv_rms * qk_inv_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1+2: Compute kv_mem from pre-decay state (read-only pass).
    // kv_mem_decayed = g * sum(old_state[ki] * k[ki]) since g is per-head scalar.
    // This avoids writing decayed state, halving state write traffic.
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        kv_mem += float(state[state_base + ki]) * k_shared[ki];
    }
    kv_mem *= g;  // apply decay to the dot product instead of to each state element

    // Step 3+4+5: Combined decay + delta update + output in one pass.
    // s_new = old_state * g + k * delta = decayed + update
    float delta = (v[v_base + vi] - kv_mem) * beta;
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = float(state[state_base + ki]) * g + k_shared[ki] * delta;
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
kernel void conv1d_step_f32(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,
    device const float *weights,      // [conv_dim * 4] F32
    device float *output,
    constant uint &conv_dim,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    uint w_base = idx * 4;
    float acc = 0.0f;
    acc += conv_state[0 * conv_dim + idx] * weights[w_base + 0];
    acc += conv_state[1 * conv_dim + idx] * weights[w_base + 1];
    acc += conv_state[2 * conv_dim + idx] * weights[w_base + 2];

    float inp = input[idx];
    acc += inp * weights[w_base + 3];

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history
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

    // RMS norm for q — simd reduction with broadcast
    float qval = (tid < key_dim) ? q[base + tid] : 0;
    float q_sq = qval * qval;
    float q_simd = simd_sum(q_sq);
    threadgroup float q_shared[4];
    if (simd_lane == 0) q_shared[simd_group] = q_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = 0;
    if (simd_group == 0) {
        float v = (simd_lane < 4) ? q_shared[simd_lane] : 0;
        q_total = simd_sum(v);
    }
    threadgroup float q_broadcast;
    if (tid == 0) q_broadcast = q_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_broadcast / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;
    }

    // RMS norm for k — simd reduction with broadcast
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    float k_sq = kval * kval;
    float k_simd = simd_sum(k_sq);
    threadgroup float k_shared[4];
    if (simd_lane == 0) k_shared[simd_group] = k_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_total = 0;
    if (simd_group == 0) {
        float v = (simd_lane < 4) ? k_shared[simd_lane] : 0;
        k_total = simd_sum(v);
    }
    threadgroup float k_broadcast;
    if (tid == 0) k_broadcast = k_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_broadcast / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}

// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
kernel void compute_decay_beta_f32(
    device const float *alpha_out,
    device const float *beta_out,
    device const float *A_log,
    device const float *dt_bias,      // F32 instead of BF16
    device float *g_decay,
    device float *beta_gate,
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = dt_bias[idx];
    // GGUF ssm_a stores -exp(A_log), not A_log itself
    float neg_A = A_log[idx];  // = -exp(A_log) = -A
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(neg_A * softplus_val);  // exp(-A * softplus)
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
    if (simd_group == 0) {
        float v2 = (simd_lane < 4) ? shared_sums[simd_lane] : 0;
        total = simd_sum(v2);
    }
    // Broadcast to all simdgroups via threadgroup memory
    threadgroup float total_broadcast;
    if (tid == 0) total_broadcast = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(total_broadcast / float(value_dim) + eps);

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
    if (head < num_q_heads) {
        uint base = head * head_dim;
        float qval = (tid < head_dim) ? q[base + tid] : 0;

        // simd reduction for sum of squares (256 threads = 8 simdgroups)
        float q_sq = qval * qval;
        float q_simd = simd_sum(q_sq);
        threadgroup float q_shared[8];
        if (simd_lane == 0) q_shared[simd_group] = q_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = 0;
        if (simd_group == 0) {
            float v = (simd_lane < 8) ? q_shared[simd_lane] : 0;
            q_total = simd_sum(v);
        }
        threadgroup float q_broadcast;
        if (tid == 0) q_broadcast = q_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            float inv_rms = rsqrt(q_broadcast / float(head_dim) + 1e-6f);
            float w = bf16_to_f32(q_weight[tid]);
            q[base + tid] = qval * inv_rms * w * inv_scale;
        }
    }

    // Process K head if in range
    if (head < num_kv_heads) {
        uint base = head * head_dim;
        float kval = (tid < head_dim) ? k[base + tid] : 0;

        float k_sq = kval * kval;
        float k_simd = simd_sum(k_sq);
        threadgroup float k_shared[8];
        if (simd_lane == 0) k_shared[simd_group] = k_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float k_total = 0;
        if (simd_group == 0) {
            float v = (simd_lane < 8) ? k_shared[simd_lane] : 0;
            k_total = simd_sum(v);
        }
        threadgroup float k_broadcast;
        if (tid == 0) k_broadcast = k_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            float inv_rms = rsqrt(k_broadcast / float(head_dim) + 1e-6f);
            float w = bf16_to_f32(k_weight[tid]);
            k[base + tid] = kval * inv_rms * w;
        }
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

    // Half-split RoPE pairing: (tid, tid+half_rot)
    // Qwen3.5 uses rotate_half which is half-split (not interleaved)

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
// Kernel 18b: Fused QK RMS norm + RoPE
// ============================================================================
// Combines weighted QK RMS norm and partial rotary embedding into one kernel.
// Eliminates 1 dispatch + 1 barrier per full-attention layer.
// Dispatch: num_q_heads threadgroups, head_dim threads each (256)

kernel void rms_norm_qk_rope(
    device float *q,                  // [num_q_heads * head_dim] in/out
    device float *k,                  // [num_kv_heads * head_dim] in/out
    device const uint16_t *q_weight,  // [head_dim] bf16 norm weight for Q
    device const uint16_t *k_weight,  // [head_dim] bf16 norm weight for K
    constant uint &head_dim,          // = 256
    constant uint &num_q_heads,       // = 16
    constant uint &num_kv_heads,      // = 2
    constant float &inv_scale,        // = 1/sqrt(head_dim)
    constant uint &rotary_dim,        // = 64
    constant uint &pos,               // current position
    constant float &theta,            // rope_theta
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint half_rot = rotary_dim / 2;

    // --- Q head: RMS norm + RoPE ---
    if (head < num_q_heads) {
        uint base = head * head_dim;
        float qval = (tid < head_dim) ? q[base + tid] : 0;

        float q_sq = qval * qval;
        float q_simd = simd_sum(q_sq);
        threadgroup float q_shared[8];
        if (simd_lane == 0) q_shared[simd_group] = q_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = 0;
        if (simd_group == 0) {
            float v = (simd_lane < 8) ? q_shared[simd_lane] : 0;
            q_total = simd_sum(v);
        }
        threadgroup float q_broadcast;
        if (tid == 0) q_broadcast = q_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            float inv_rms = rsqrt(q_broadcast / float(head_dim) + 1e-6f);
            float w = bf16_to_f32(q_weight[tid]);
            float normed = qval * inv_rms * w * inv_scale;

            // Apply RoPE to first rotary_dim elements via threadgroup sharing
            if (tid < rotary_dim) {
                threadgroup float q_rope[256]; // reuse for RoPE pairs
                q_rope[tid] = normed;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < half_rot) {
                    float freq = 1.0f / pow(theta, float(2 * tid) / float(rotary_dim));
                    float angle = float(pos) * freq;
                    float cos_a = cos(angle);
                    float sin_a = sin(angle);
                    float q0 = q_rope[tid];
                    float q1 = q_rope[tid + half_rot];
                    q[base + tid]            = q0 * cos_a - q1 * sin_a;
                    q[base + tid + half_rot] = q0 * sin_a + q1 * cos_a;
                }
            } else {
                q[base + tid] = normed;
            }
        }
    }

    // Need barrier before K reuses threadgroup memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- K head: RMS norm + RoPE ---
    if (head < num_kv_heads) {
        uint base = head * head_dim;
        float kval = (tid < head_dim) ? k[base + tid] : 0;

        float k_sq = kval * kval;
        float k_simd = simd_sum(k_sq);
        threadgroup float k_shared[8];
        if (simd_lane == 0) k_shared[simd_group] = k_simd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float k_total = 0;
        if (simd_group == 0) {
            float v = (simd_lane < 8) ? k_shared[simd_lane] : 0;
            k_total = simd_sum(v);
        }
        threadgroup float k_broadcast;
        if (tid == 0) k_broadcast = k_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < head_dim) {
            float inv_rms = rsqrt(k_broadcast / float(head_dim) + 1e-6f);
            float w = bf16_to_f32(k_weight[tid]);
            float normed = kval * inv_rms * w;

            if (tid < rotary_dim) {
                threadgroup float k_rope[256];
                k_rope[tid] = normed;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < half_rot) {
                    float freq = 1.0f / pow(theta, float(2 * tid) / float(rotary_dim));
                    float angle = float(pos) * freq;
                    float cos_a = cos(angle);
                    float sin_a = sin(angle);
                    float k0 = k_rope[tid];
                    float k1 = k_rope[tid + half_rot];
                    k[base + tid]            = k0 * cos_a - k1 * sin_a;
                    k[base + tid + half_rot] = k0 * sin_a + k1 * cos_a;
                }
            } else {
                k[base + tid] = normed;
            }
        }
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
        float shared_gate = 1.0f / (1.0f + exp(-params[K]));
        float moe = 0.0f;
        // Transposed layout: [dim][K] — contiguous reads per thread
        device const float* exp_base = expert_out + tid * K;
        for (uint k = 0; k < K && k < 16; k++) {
            moe += params[k] * exp_base[k];
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

kernel void moe_combine_copy_sq_k8(
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
    (void)K;

    float shared_gate = 0.0f;
    if (simd_lane == 0) {
        shared_gate = 1.0f / (1.0f + exp(-params[8]));
    }
    shared_gate = simd_broadcast_first(shared_gate);

    float val = 0.0f;
    if (tid < dim) {
        // Transposed layout: [dim][8] — 8 contiguous floats per element
        uint base = tid * 8;
        float moe =
            params[0] * expert_out[base] +
            params[1] * expert_out[base + 1] +
            params[2] * expert_out[base + 2] +
            params[3] * expert_out[base + 3] +
            params[4] * expert_out[base + 4] +
            params[5] * expert_out[base + 5] +
            params[6] * expert_out[base + 6] +
            params[7] * expert_out[base + 7];
        val = h_mid[tid] + moe + shared_gate * shared_out[tid];
        hidden_out[tid] = val;
        residual_out[tid] = val;
    }

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
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each thread holds its own logit value (256 threads for 256 experts)
    float my_val = (lid < n_experts) ? logits[lid] : -INFINITY;
    uint my_idx = lid;

    // Parallel top-K via repeated max-finding with simd reductions
    // Use threadgroup memory for cross-simdgroup max reduction
    threadgroup float sg_max_vals[8];
    threadgroup uint  sg_max_idxs[8];
    threadgroup float found_logits[8];
    threadgroup uint  found_indices[8];

    uint limit = (K < 8) ? K : 8;

    for (uint round = 0; round < limit; round++) {
        // Step 1: simdgroup-local max
        float local_max = my_val;
        uint local_idx = my_idx;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(local_max, offset);
            uint other_idx = simd_shuffle_down(local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Step 2: cross-simdgroup reduction via threadgroup memory
        if (simd_lane == 0) {
            sg_max_vals[simd_group] = local_max;
            sg_max_idxs[simd_group] = local_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: first simdgroup reduces across all simdgroups
        if (simd_group == 0) {
            uint num_sgs = (tg_size + 31) / 32;
            float val = (simd_lane < num_sgs) ? sg_max_vals[simd_lane] : -INFINITY;
            uint idx = (simd_lane < num_sgs) ? sg_max_idxs[simd_lane] : 0;
            for (uint offset = 16; offset > 0; offset >>= 1) {
                float other_val = simd_shuffle_down(val, offset);
                uint other_idx = simd_shuffle_down(idx, offset);
                if (other_val > val) {
                    val = other_val;
                    idx = other_idx;
                }
            }
            if (simd_lane == 0) {
                found_logits[round] = val;
                found_indices[round] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: winner invalidates its value for next round
        if (lid == found_indices[round]) {
            my_val = -INFINITY;
        }
    }

    // Softmax normalize the top-K logits and write output (lane 0 only)
    if (lid == 0) {
        float max_logit = found_logits[0];
        float weight_sum = 0.0f;
        float top_weights[8];
        for (uint k = 0; k < limit; k++) {
            float w = exp(found_logits[k] - max_logit);
            top_weights[k] = w;
            weight_sum += w;
        }
        float inv_weight_sum = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;

        for (uint k = 0; k < limit; k++) {
            out_indices[k] = found_indices[k];
            out_params[k] = top_weights[k] * inv_weight_sum;
        }
        for (uint k = limit; k < 8; k++) {
            out_indices[k] = 0;
            out_params[k] = 0.0f;
        }
        out_params[8] = logits[sgg_off_f];
    }
}

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

// GPU argmax — find index of maximum element. Single TG, 1024 threads.
// Avoids copying full logits (248320 floats = 993KB) back to CPU.
kernel void argmax_kernel(
    device const float* data    [[buffer(0)]],
    device uint32_t*    result  [[buffer(1)]],   // single uint32: argmax index
    constant uint&      count   [[buffer(2)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Phase 1: each thread finds local max across its strided range
    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = lid; i < count; i += tg_size) {
        float val = data[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Phase 2: simdgroup reduction (find max within each 32-thread simdgroup)
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(local_max, offset);
        uint other_idx = simd_shuffle_down(local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Phase 3: cross-simdgroup reduction via threadgroup memory
    threadgroup float tg_vals[32];
    threadgroup uint tg_idxs[32];
    if (simd_lane == 0) {
        tg_vals[simd_group] = local_max;
        tg_idxs[simd_group] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: first simdgroup reduces across all simdgroups
    uint num_simdgroups = (tg_size + 31) / 32;
    if (simd_group == 0) {
        float val = (simd_lane < num_simdgroups) ? tg_vals[simd_lane] : -INFINITY;
        uint idx = (simd_lane < num_simdgroups) ? tg_idxs[simd_lane] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(val, offset);
            uint other_idx = simd_shuffle_down(idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (simd_lane == 0) {
            result[0] = idx;
        }
    }
}
kernel void deinterleave_qgate(
    device float* buf           [[buffer(0)]],   // in-place
    device float* tmp           [[buffer(1)]],   // scratch [n_heads * hd * 2]
    constant uint& head_dim     [[buffer(2)]],   // 256
    constant uint& num_heads    [[buffer(3)]],   // 16
    uint tid [[thread_position_in_grid]]
) {
    uint total = num_heads * head_dim;
    if (tid >= total) return;

    uint head = tid / head_dim;
    uint d = tid % head_dim;

    // Read Q and gate from interleaved layout
    float q_val = buf[head * head_dim * 2 + d];
    float g_val = buf[head * head_dim * 2 + head_dim + d];

    // Write to de-interleaved layout via scratch
    tmp[head * head_dim + d] = q_val;           // Q part
    tmp[total + head * head_dim + d] = g_val;   // Gate part
}

kernel void copy_tmp_to_buf(
    device float* buf           [[buffer(0)]],
    device const float* tmp     [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    buf[tid] = tmp[tid];
}

// ============================================================================
// F32 matvec (no dequantization, for GGUF F32 tensors like routing gates)
// ============================================================================

kernel void matvec_f32(
    device const float*    W          [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    device const float* w_row = W + row * in_dim;
    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += w_row[col] * x[col];
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

kernel void matvec_f32_pair(
    device const float*    W0         [[buffer(0)]],
    device const float*    W1         [[buffer(1)]],
    device const float*    x          [[buffer(2)]],
    device float*          out        [[buffer(3)]],
    constant uint&         out0_dim   [[buffer(4)]],
    constant uint&         in_dim     [[buffer(5)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint total_out_dim = out0_dim + 1;
    if (row >= total_out_dim) return;

    device const float* w_row = (row < out0_dim) ? (W0 + row * in_dim) : W1;
    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += w_row[col] * x[col];
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// GGUF Q4_K dequant matvec
//
// Q4_K super-block: 256 weights in 144 bytes
//   [0..1]   float16 d     (super-block scale)
//   [2..3]   float16 dmin  (super-block min)
//   [4..15]  12 bytes: packed 6-bit scales + mins for 8 sub-blocks
//   [16..143] 128 bytes: 256 x 4-bit quantized weights
//
// Dequant: value = d * sc[sb] * q - dmin * m[sb]
//   where sb = sub-block index (0..7), q = 4-bit value (0..15)
//   sc[sb] and m[sb] are 6-bit values unpacked from the 12-byte block
// ============================================================================

// Unpack 6-bit sub-block scales and mins from the 12-byte packed region.
// The packing is: scales[0..3] in low 6 bits of bytes 0..3,
//                 scales[4..7] in low 4 bits of bytes 8..9 | high 2 bits of bytes 0..3,
//                 mins[0..3] in low 6 bits of bytes 4..7,
//                 mins[4..7] in low 4 bits of bytes 10..11 | high 2 bits of bytes 4..7.
inline void unpack_q4k_scale_min_pair(device const uint8_t* sc_data,
                                      uint g,
                                      thread float* sc_lo,
                                      thread float* sc_hi,
                                      thread float* mn_lo,
                                      thread float* mn_hi) {
    if (g < 2) {
        uint base = g * 2;
        *sc_lo = float(sc_data[base] & 63);
        *sc_hi = float(sc_data[base + 1] & 63);
        *mn_lo = float(sc_data[base + 4] & 63);
        *mn_hi = float(sc_data[base + 5] & 63);
        return;
    }

    uint base = (g - 2) * 2;
    uint upper = g - 2;
    uint8_t sc_pack = sc_data[8 + upper];
    uint8_t mn_pack = sc_data[10 + upper];
    *sc_lo = float((sc_pack & 0xF) | ((sc_data[base] >> 6) << 4));
    *sc_hi = float((sc_pack >> 4) | ((sc_data[base + 1] >> 6) << 4));
    *mn_lo = float((mn_pack & 0xF) | ((sc_data[base + 4] >> 6) << 4));
    *mn_hi = float((mn_pack >> 4) | ((sc_data[base + 5] >> 6) << 4));
}

kernel void dequant_matvec_q4k(
    device const uint8_t*  data       [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    if (!valid0) return;

    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 144;
    device const uint8_t* row0_data = data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    for (uint tile_base = 0; tile_base < in_dim; tile_base += MATVEC_X_SHARED_SIZE) {
        uint tile_elems = min((uint)MATVEC_X_SHARED_SIZE, in_dim - tile_base);
        for (uint i = lid; i < tile_elems; i += tg_size) {
            x_shared[i] = half(x[tile_base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_sb_base = tile_base / 256;
        uint tile_num_superblocks = tile_elems / 256;
        for (uint sb_rel = 0; sb_rel < tile_num_superblocks; sb_rel++) {
            uint sb_idx = tile_sb_base + sb_rel;
            device const uint8_t* sb0 = row0_data + sb_idx * 144;
            float sc_lo, sc_hi, mn_lo, mn_hi;
            uint w_base = sb_rel * 256 + g * 64;
            float d0    = float(as_type<half>(ushort(ushort(sb0[0]) | (ushort(sb0[1]) << 8))));
            float dmin0 = float(as_type<half>(ushort(ushort(sb0[2]) | (ushort(sb0[3]) << 8))));
            unpack_q4k_scale_min_pair(sb0 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            device const uint8_t* qs0 = sb0 + 16 + g * 32;
            float sc0_lo = d0 * sc_lo, mn0_lo = dmin0 * mn_lo;
            float sc0_hi = d0 * sc_hi, mn0_hi = dmin0 * mn_hi;
            for (uint j = 0; j < 4; j++) {
                uint l = l_start + j;
                uint8_t byte = qs0[l];
                acc0 += (sc0_lo * float(byte & 0xF) - mn0_lo) * float(x_shared[w_base + l]);
                acc0 += (sc0_hi * float(byte >> 4) - mn0_hi) * float(x_shared[w_base + 32 + l]);
            }

            if (valid1) {
                device const uint8_t* sb1 = row1_data + sb_idx * 144;
                float d1    = float(as_type<half>(ushort(ushort(sb1[0]) | (ushort(sb1[1]) << 8))));
                float dmin1 = float(as_type<half>(ushort(ushort(sb1[2]) | (ushort(sb1[3]) << 8))));
                unpack_q4k_scale_min_pair(sb1 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
                device const uint8_t* qs1 = sb1 + 16 + g * 32;
                float sc1_lo = d1 * sc_lo, mn1_lo = dmin1 * mn_lo;
                float sc1_hi = d1 * sc_hi, mn1_hi = dmin1 * mn_hi;
                for (uint j = 0; j < 4; j++) {
                    uint l = l_start + j;
                    uint8_t byte = qs1[l];
                    acc1 += (sc1_lo * float(byte & 0xF) - mn1_lo) * float(x_shared[w_base + l]);
                    acc1 += (sc1_hi * float(byte >> 4) - mn1_hi) * float(x_shared[w_base + 32 + l]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}

// ============================================================================
// GGUF Q8_0 dequant matvec
//
// Q8_0 block: 32 weights in 34 bytes
//   [0..1]   float16 d (scale)
//   [2..33]  32 x int8 quantized weights
//
// Dequant: value = d * q
// ============================================================================

kernel void dequant_matvec_q8_0(
    device const uint8_t*  data       [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    if (!valid0) return;

    // Q8_0: 32 weights per block, 34 bytes per block
    uint num_blocks = in_dim / 32;
    uint bytes_per_row = num_blocks * 34;

    device const uint8_t* row0_data = data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint tile_base = 0; tile_base < in_dim; tile_base += MATVEC_X_SHARED_SIZE) {
        uint tile_elems = min((uint)MATVEC_X_SHARED_SIZE, in_dim - tile_base);
        for (uint i = lid; i < tile_elems; i += tg_size) {
            x_shared[i] = half(x[tile_base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_block_base = tile_base / 32;
        uint tile_num_blocks = tile_elems / 32;
        for (uint blk_rel = simd_lane; blk_rel < tile_num_blocks; blk_rel += 32) {
            uint blk = tile_block_base + blk_rel;
            uint x_base = blk_rel * 32;

            device const uint8_t* block0 = row0_data + blk * 34;
            float d0 = float(as_type<half>(ushort(ushort(block0[0]) | (ushort(block0[1]) << 8))));
            device const int8_t* qs0 = (device const int8_t*)(block0 + 2);

            float local0 = 0.0f;
            for (uint j = 0; j < 32; j++) {
                local0 += float(qs0[j]) * float(x_shared[x_base + j]);
            }
            acc0 += d0 * local0;

            if (valid1) {
                device const uint8_t* block1 = row1_data + blk * 34;
                float d1 = float(as_type<half>(ushort(ushort(block1[0]) | (ushort(block1[1]) << 8))));
                device const int8_t* qs1 = (device const int8_t*)(block1 + 2);

                float local1 = 0.0f;
                for (uint j = 0; j < 32; j++) {
                    local1 += float(qs1[j]) * float(x_shared[x_base + j]);
                }
                acc1 += d1 * local1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}

kernel void dequant_matvec_q8_0_singletile(
    device const uint8_t*  data       [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    if (!valid0) return;

    uint num_blocks = in_dim / 32;
    uint bytes_per_row = num_blocks * 34;
    device const uint8_t* row0_data = data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? data + row1 * bytes_per_row : row0_data;

    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint blk = simd_lane; blk < num_blocks; blk += 32) {
        uint x_base = blk * 32;

        device const uint8_t* block0 = row0_data + blk * 34;
        float d0 = float(as_type<half>(ushort(ushort(block0[0]) | (ushort(block0[1]) << 8))));
        device const int8_t* qs0 = (device const int8_t*)(block0 + 2);

        float local0 = 0.0f;
        for (uint j = 0; j < 32; j++) {
            local0 += float(qs0[j]) * float(x_shared[x_base + j]);
        }
        acc0 += d0 * local0;

        if (valid1) {
            device const uint8_t* block1 = row1_data + blk * 34;
            float d1 = float(as_type<half>(ushort(ushort(block1[0]) | (ushort(block1[1]) << 8))));
            device const int8_t* qs1 = (device const int8_t*)(block1 + 2);

            float local1 = 0.0f;
            for (uint j = 0; j < 32; j++) {
                local1 += float(qs1[j]) * float(x_shared[x_base + j]);
            }
            acc1 += d1 * local1;
        }
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}

// ============================================================================
// ============================================================================
// GGUF Q5_K dequant matvec
//
// Q5_K super-block: 256 weights in 176 bytes
//   [0..1]   float16 d (super-block scale)
//   [2..3]   float16 dmin (super-block min)
//   [4..15]  12 bytes: packed 6-bit sub-block scales + mins (same as Q4_K)
//   [16..47] 32 bytes: 256 high bits (1 per weight, packed)
//   [48..175] 128 bytes: 256 x low 4 bits (packed as nibbles)
//
// Dequant: value = d * sc[sb] * q - dmin * m[sb]
//   where q = low4 | (high1 << 4), giving 5-bit value (0..31)
// ============================================================================

kernel void dequant_matvec_q5k(
    device const uint8_t*  data       [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    if (!valid0) return;

    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 176;
    device const uint8_t* row0_data = data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    uint8_t hm_lo = 1u << (g * 2);
    uint8_t hm_hi = 1u << (g * 2 + 1);
    for (uint tile_base = 0; tile_base < in_dim; tile_base += MATVEC_X_SHARED_SIZE) {
        uint tile_elems = min((uint)MATVEC_X_SHARED_SIZE, in_dim - tile_base);
        for (uint i = lid; i < tile_elems; i += tg_size) {
            x_shared[i] = half(x[tile_base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_sb_base = tile_base / 256;
        uint tile_num_superblocks = tile_elems / 256;
        for (uint sb_rel = 0; sb_rel < tile_num_superblocks; sb_rel++) {
            uint sb_idx = tile_sb_base + sb_rel;
            device const uint8_t* sb0 = row0_data + sb_idx * 176;
            float d0    = float(as_type<half>(ushort(ushort(sb0[0]) | (ushort(sb0[1]) << 8))));
            float dmin0 = float(as_type<half>(ushort(ushort(sb0[2]) | (ushort(sb0[3]) << 8))));
            float sc_lo, sc_hi, mn_lo, mn_hi;
            unpack_q4k_scale_min_pair(sb0 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            device const uint8_t* qh0 = sb0 + 16;
            device const uint8_t* ql0 = sb0 + 48 + g * 32;
            uint w_base = sb_rel * 256 + g * 64;
            float q_sc0_lo = d0 * sc_lo, q_mn0_lo = dmin0 * mn_lo;
            float q_sc0_hi = d0 * sc_hi, q_mn0_hi = dmin0 * mn_hi;
            for (uint j = 0; j < 4; j++) {
                uint l = l_start + j;
                uint8_t byte_val = ql0[l];
                uint8_t q5_lo = (byte_val & 0xF) | ((qh0[l] & hm_lo) ? 16 : 0);
                uint8_t q5_hi = (byte_val >> 4) | ((qh0[l] & hm_hi) ? 16 : 0);
                acc0 += (q_sc0_lo * float(q5_lo) - q_mn0_lo) * float(x_shared[w_base + l]);
                acc0 += (q_sc0_hi * float(q5_hi) - q_mn0_hi) * float(x_shared[w_base + 32 + l]);
            }

            if (valid1) {
                device const uint8_t* sb1 = row1_data + sb_idx * 176;
                float d1    = float(as_type<half>(ushort(ushort(sb1[0]) | (ushort(sb1[1]) << 8))));
                float dmin1 = float(as_type<half>(ushort(ushort(sb1[2]) | (ushort(sb1[3]) << 8))));
                unpack_q4k_scale_min_pair(sb1 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
                device const uint8_t* qh1 = sb1 + 16;
                device const uint8_t* ql1 = sb1 + 48 + g * 32;
                float q_sc1_lo = d1 * sc_lo, q_mn1_lo = dmin1 * mn_lo;
                float q_sc1_hi = d1 * sc_hi, q_mn1_hi = dmin1 * mn_hi;
                for (uint j = 0; j < 4; j++) {
                    uint l = l_start + j;
                    uint8_t byte_val = ql1[l];
                    uint8_t q5_lo = (byte_val & 0xF) | ((qh1[l] & hm_lo) ? 16 : 0);
                    uint8_t q5_hi = (byte_val >> 4) | ((qh1[l] & hm_hi) ? 16 : 0);
                    acc1 += (q_sc1_lo * float(q5_lo) - q_mn1_lo) * float(x_shared[w_base + l]);
                    acc1 += (q_sc1_hi * float(q5_hi) - q_mn1_hi) * float(x_shared[w_base + 32 + l]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}

// ============================================================================
// GGUF Q6_K dequant matvec
//
// Q6_K super-block: 256 weights in 210 bytes
//   [0..127]   128 bytes: 256 x low 4 bits (packed as nibbles)
//   [128..191]  64 bytes: 256 x high 2 bits (packed, 4 per byte)
//   [192..207]  16 bytes: 16 x int8 scales (one per 16 weights)
//   [208..209]   2 bytes: float16 d (super-block scale)
//
// Dequant: value = d * sc[j/16] * (q_lo | (q_hi << 4) - 32)
// where q is reconstructed as 6-bit value (0..63), centered at 32
// ============================================================================

kernel void dequant_matvec_q6k(
    device const uint8_t*  data       [[buffer(0)]],
    device const float*    x          [[buffer(1)]],
    device float*          out        [[buffer(2)]],
    constant uint&         out_dim    [[buffer(3)]],
    constant uint&         in_dim     [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row0 = tgid * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    if (!valid0) return;

    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 210;
    device const uint8_t* row0_data = data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    // Distribute l values across SIMD lanes — each lane handles one l (0..31)
    // per block, giving 4 weights per block × 2 blocks = 8 weights per superblock
    uint l = simd_lane;
    for (uint tile_base = 0; tile_base < in_dim; tile_base += MATVEC_X_SHARED_SIZE) {
        uint tile_elems = min((uint)MATVEC_X_SHARED_SIZE, in_dim - tile_base);
        for (uint i = lid; i < tile_elems; i += tg_size) {
            x_shared[i] = half(x[tile_base + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_sb_base = tile_base / 256;
        uint tile_num_superblocks = tile_elems / 256;
        for (uint sb_rel = 0; sb_rel < tile_num_superblocks; sb_rel++) {
            uint sb_idx = tile_sb_base + sb_rel;
            uint x_base = sb_rel * 256;
            uint is = l / 16;

            device const uint8_t* sb0 = row0_data + sb_idx * 210;
            device const uint8_t* ql0_base = sb0;
            device const uint8_t* qh0_base = sb0 + 128;
            device const int8_t* sc0_base = (device const int8_t*)(sb0 + 192);
            float d0 = float(as_type<half>(ushort(ushort(sb0[208]) | (ushort(sb0[209]) << 8))));
            for (uint blk = 0; blk < 2; blk++) {
                device const uint8_t* ql0 = ql0_base + blk * 64;
                device const uint8_t* qh0 = qh0_base + blk * 32;
                device const int8_t* sc0 = sc0_base + blk * 8;
                uint y_off = x_base + blk * 128;

                int q1 = (int)((ql0[l]      & 0xF) | (((qh0[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql0[l + 32] & 0xF) | (((qh0[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql0[l]      >> 4)  | (((qh0[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql0[l + 32] >> 4)  | (((qh0[l] >> 6) & 3) << 4)) - 32;

                acc0 += d0 * (float)sc0[is + 0] * (float)q1 * float(x_shared[y_off + l]);
                acc0 += d0 * (float)sc0[is + 2] * (float)q2 * float(x_shared[y_off + l + 32]);
                acc0 += d0 * (float)sc0[is + 4] * (float)q3 * float(x_shared[y_off + l + 64]);
                acc0 += d0 * (float)sc0[is + 6] * (float)q4 * float(x_shared[y_off + l + 96]);
            }

            if (valid1) {
                device const uint8_t* sb1 = row1_data + sb_idx * 210;
                device const uint8_t* ql1_base = sb1;
                device const uint8_t* qh1_base = sb1 + 128;
                device const int8_t* sc1_base = (device const int8_t*)(sb1 + 192);
                float d1 = float(as_type<half>(ushort(ushort(sb1[208]) | (ushort(sb1[209]) << 8))));

                for (uint blk = 0; blk < 2; blk++) {
                    device const uint8_t* ql1 = ql1_base + blk * 64;
                    device const uint8_t* qh1 = qh1_base + blk * 32;
                    device const int8_t* sc1 = sc1_base + blk * 8;
                    uint y_off = x_base + blk * 128;

                    int q1 = (int)((ql1[l]      & 0xF) | (((qh1[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql1[l + 32] & 0xF) | (((qh1[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql1[l]      >> 4)  | (((qh1[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql1[l + 32] >> 4)  | (((qh1[l] >> 6) & 3) << 4)) - 32;

                    acc1 += d1 * (float)sc1[is + 0] * (float)q1 * float(x_shared[y_off + l]);
                    acc1 += d1 * (float)sc1[is + 2] * (float)q2 * float(x_shared[y_off + l + 32]);
                    acc1 += d1 * (float)sc1[is + 4] * (float)q3 * float(x_shared[y_off + l + 64]);
                    acc1 += d1 * (float)sc1[is + 6] * (float)q4 * float(x_shared[y_off + l + 96]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        out[row0] = sum0;
        if (valid1) out[row1] = sum1;
    }
}
kernel void batch_expert_mv_q4k_dyn(
    device const uint8_t*       layer_data      [[buffer(0)]],
    device const float*         x               [[buffer(1)]],
    device float*               out             [[buffer(2)]],
    device const uint32_t*      expert_indices  [[buffer(3)]],
    constant uint&              expert_stride   [[buffer(4)]],
    constant uint&              out_dim         [[buffer(5)]],
    constant uint&              in_dim          [[buffer(6)]],
    constant uint&              num_row_tgs     [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert_id = expert_indices[expert_k];
    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 144;
    device const uint8_t* expert_data = layer_data + expert_id * expert_stride;
    device const uint8_t* row_data = expert_data + row * bytes_per_row;

    float acc = 0.0f;
    // Distribute 128 bytes per superblock across 32 SIMD lanes (4 bytes each)
    // → 100% utilization regardless of superblock count
    uint g = simd_lane / 8;           // group index (0-3)
    uint l_start = (simd_lane % 8) * 4;  // byte offset within group
    for (uint sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        device const uint8_t* sb = row_data + sb_idx * 144;
        float d    = float(as_type<half>(ushort(ushort(sb[0]) | (ushort(sb[1]) << 8))));
        float dmin = float(as_type<half>(ushort(ushort(sb[2]) | (ushort(sb[3]) << 8))));
        float sc_lo, sc_hi, mn_lo, mn_hi;
        unpack_q4k_scale_min_pair(sb + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* qs = sb + 16 + g * 32;
        uint w_base = sb_idx * 256 + g * 64;
        float q_sc_lo = d * sc_lo, q_mn_lo = dmin * mn_lo;
        float q_sc_hi = d * sc_hi, q_mn_hi = dmin * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = qs[l];
            acc += (q_sc_lo * float(byte & 0xF) - q_mn_lo) * float(x_shared[w_base + l]);
            acc += (q_sc_hi * float(byte >> 4) - q_mn_hi) * float(x_shared[w_base + 32 + l]);
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) { out[expert_k * out_dim + row] = sum; }
}

// ============================================================================
// Dynamic Q4_K batched expert gate+up+SwiGLU fusion
// ============================================================================

kernel void batch_expert_gate_up_swiglu_q4k_dyn(
    device const uint8_t*       gate_layer_data [[buffer(0)]],
    device const uint8_t*       up_layer_data   [[buffer(1)]],
    device const float*         x               [[buffer(2)]],
    device float*               out             [[buffer(3)]],
    device const uint32_t*      expert_indices  [[buffer(4)]],
    constant uint&              gate_stride     [[buffer(5)]],
    constant uint&              up_stride       [[buffer(6)]],
    constant uint&              out_dim         [[buffer(7)]],
    constant uint&              in_dim          [[buffer(8)]],
    constant uint&              num_row_tgs     [[buffer(9)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row = (tgid % num_row_tgs) * ROWS_PER_TG + simd_group;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    uint expert_id = expert_indices[expert_k];
    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 144;
    device const uint8_t* gate_expert = gate_layer_data + expert_id * gate_stride;
    device const uint8_t* up_expert = up_layer_data + expert_id * up_stride;
    device const uint8_t* gate_row_data = gate_expert + row * bytes_per_row;
    device const uint8_t* up_row_data = up_expert + row * bytes_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    for (uint sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        uint w_base = sb_idx * 256 + g * 64;
        float sc_lo, sc_hi, mn_lo, mn_hi;

        device const uint8_t* gate_sb = gate_row_data + sb_idx * 144;
        float gate_d = float(as_type<half>(ushort(ushort(gate_sb[0]) | (ushort(gate_sb[1]) << 8))));
        float gate_dmin = float(as_type<half>(ushort(ushort(gate_sb[2]) | (ushort(gate_sb[3]) << 8))));
        unpack_q4k_scale_min_pair(gate_sb + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* gate_qs = gate_sb + 16 + g * 32;
        float gate_sc_lo = gate_d * sc_lo, gate_mn_lo = gate_dmin * mn_lo;
        float gate_sc_hi = gate_d * sc_hi, gate_mn_hi = gate_dmin * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = gate_qs[l];
            gate_acc += (gate_sc_lo * float(byte & 0xF) - gate_mn_lo) * float(x_shared[w_base + l]);
            gate_acc += (gate_sc_hi * float(byte >> 4) - gate_mn_hi) * float(x_shared[w_base + 32 + l]);
        }

        device const uint8_t* up_sb = up_row_data + sb_idx * 144;
        float up_d = float(as_type<half>(ushort(ushort(up_sb[0]) | (ushort(up_sb[1]) << 8))));
        float up_dmin = float(as_type<half>(ushort(ushort(up_sb[2]) | (ushort(up_sb[3]) << 8))));
        unpack_q4k_scale_min_pair(up_sb + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* up_qs = up_sb + 16 + g * 32;
        float up_sc_lo = up_d * sc_lo, up_mn_lo = up_dmin * mn_lo;
        float up_sc_hi = up_d * sc_hi, up_mn_hi = up_dmin * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = up_qs[l];
            up_acc += (up_sc_lo * float(byte & 0xF) - up_mn_lo) * float(x_shared[w_base + l]);
            up_acc += (up_sc_hi * float(byte >> 4) - up_mn_hi) * float(x_shared[w_base + 32 + l]);
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        out[expert_k * out_dim + row] = silu_gate * up_sum;
    }
}

// ============================================================================
// Shared Q4_K gate+up+SwiGLU fusion
// ============================================================================

kernel void shared_gate_up_swiglu_q4k(
    device const uint8_t*       gate_data       [[buffer(0)]],
    device const uint8_t*       up_data         [[buffer(1)]],
    device const float*         x               [[buffer(2)]],
    device float*               out             [[buffer(3)]],
    constant uint&              out_dim         [[buffer(4)]],
    constant uint&              in_dim          [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 144;
    device const uint8_t* gate_row_data = gate_data + row * bytes_per_row;
    device const uint8_t* up_row_data = up_data + row * bytes_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    for (uint sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        uint w_base = sb_idx * 256 + g * 64;
        float sc_lo, sc_hi, mn_lo, mn_hi;

        device const uint8_t* gate_sb = gate_row_data + sb_idx * 144;
        float gate_d = float(as_type<half>(ushort(ushort(gate_sb[0]) | (ushort(gate_sb[1]) << 8))));
        float gate_dmin = float(as_type<half>(ushort(ushort(gate_sb[2]) | (ushort(gate_sb[3]) << 8))));
        unpack_q4k_scale_min_pair(gate_sb + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* gate_qs = gate_sb + 16 + g * 32;
        float gate_sc_lo = gate_d * sc_lo, gate_mn_lo = gate_dmin * mn_lo;
        float gate_sc_hi = gate_d * sc_hi, gate_mn_hi = gate_dmin * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = gate_qs[l];
            gate_acc += (gate_sc_lo * float(byte & 0xF) - gate_mn_lo) * float(x_shared[w_base + l]);
            gate_acc += (gate_sc_hi * float(byte >> 4) - gate_mn_hi) * float(x_shared[w_base + 32 + l]);
        }

        device const uint8_t* up_sb = up_row_data + sb_idx * 144;
        float up_d = float(as_type<half>(ushort(ushort(up_sb[0]) | (ushort(up_sb[1]) << 8))));
        float up_dmin = float(as_type<half>(ushort(ushort(up_sb[2]) | (ushort(up_sb[3]) << 8))));
        unpack_q4k_scale_min_pair(up_sb + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* up_qs = up_sb + 16 + g * 32;
        float up_sc_lo = up_d * sc_lo, up_mn_lo = up_dmin * mn_lo;
        float up_sc_hi = up_d * sc_hi, up_mn_hi = up_dmin * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = up_qs[l];
            up_acc += (up_sc_lo * float(byte & 0xF) - up_mn_lo) * float(x_shared[w_base + l]);
            up_acc += (up_sc_hi * float(byte >> 4) - up_mn_hi) * float(x_shared[w_base + 32 + l]);
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        out[row] = silu_gate * up_sum;
    }
}

kernel void shared_gate_up_swiglu_q8_0(
    device const uint8_t*       gate_data       [[buffer(0)]],
    device const uint8_t*       up_data         [[buffer(1)]],
    device const float*         x               [[buffer(2)]],
    device float*               out             [[buffer(3)]],
    constant uint&              out_dim         [[buffer(4)]],
    constant uint&              in_dim          [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    uint num_blocks = in_dim / 32;
    uint bytes_per_row = num_blocks * 34;
    device const uint8_t* gate_row_data = gate_data + row * bytes_per_row;
    device const uint8_t* up_row_data = up_data + row * bytes_per_row;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint blk = simd_lane; blk < num_blocks; blk += 32) {
        uint x_base = blk * 32;

        device const uint8_t* gate_block = gate_row_data + blk * 34;
        float gate_d = float(as_type<half>(ushort(ushort(gate_block[0]) | (ushort(gate_block[1]) << 8))));
        device const int8_t* gate_qs = (device const int8_t*)(gate_block + 2);
        float gate_local = 0.0f;
        for (uint j = 0; j < 32; j++) {
            gate_local += gate_d * float(gate_qs[j]) * float(x_shared[x_base + j]);
        }
        gate_acc += gate_local;

        device const uint8_t* up_block = up_row_data + blk * 34;
        float up_d = float(as_type<half>(ushort(ushort(up_block[0]) | (ushort(up_block[1]) << 8))));
        device const int8_t* up_qs = (device const int8_t*)(up_block + 2);
        float up_local = 0.0f;
        for (uint j = 0; j < 32; j++) {
            up_local += up_d * float(up_qs[j]) * float(x_shared[x_base + j]);
        }
        up_acc += up_local;
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        out[row] = silu_gate * up_sum;
    }
}

// ============================================================================
// Dynamic Q4_K batched expert down projection (per-expert packed input)
// ============================================================================

kernel void batch_expert_down_q4k_dyn(
    device const uint8_t*       layer_data      [[buffer(0)]],
    device const float*         x               [[buffer(1)]],
    device float*               out             [[buffer(2)]],
    device const uint32_t*      expert_indices  [[buffer(3)]],
    constant uint&              expert_stride   [[buffer(4)]],
    constant uint&              out_dim         [[buffer(5)]],
    constant uint&              in_dim          [[buffer(6)]],
    constant uint&              num_row_tgs     [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row0 = (tgid % num_row_tgs) * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    device const float* ex = x + expert_k * in_dim;
    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) { x_shared[i] = half(ex[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid0) return;

    uint expert_id = expert_indices[expert_k];
    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 144;
    device const uint8_t* expert_data = layer_data + expert_id * expert_stride;
    device const uint8_t* row0_data = expert_data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? expert_data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    for (uint sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        device const uint8_t* sb0 = row0_data + sb_idx * 144;
        float sc_lo, sc_hi, mn_lo, mn_hi;
        uint w_base = sb_idx * 256 + g * 64;
        float d0    = float(as_type<half>(ushort(ushort(sb0[0]) | (ushort(sb0[1]) << 8))));
        float dmin0 = float(as_type<half>(ushort(ushort(sb0[2]) | (ushort(sb0[3]) << 8))));
        unpack_q4k_scale_min_pair(sb0 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* qs0 = sb0 + 16 + g * 32;
        float sc0_lo = d0 * sc_lo, mn0_lo = dmin0 * mn_lo;
        float sc0_hi = d0 * sc_hi, mn0_hi = dmin0 * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte = qs0[l];
            acc0 += (sc0_lo * float(byte & 0xF) - mn0_lo) * float(x_shared[w_base + l]);
            acc0 += (sc0_hi * float(byte >> 4) - mn0_hi) * float(x_shared[w_base + 32 + l]);
        }

        if (valid1) {
            device const uint8_t* sb1 = row1_data + sb_idx * 144;
            float d1    = float(as_type<half>(ushort(ushort(sb1[0]) | (ushort(sb1[1]) << 8))));
            float dmin1 = float(as_type<half>(ushort(ushort(sb1[2]) | (ushort(sb1[3]) << 8))));
            unpack_q4k_scale_min_pair(sb1 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            device const uint8_t* qs1 = sb1 + 16 + g * 32;
            float sc1_lo = d1 * sc_lo, mn1_lo = dmin1 * mn_lo;
            float sc1_hi = d1 * sc_hi, mn1_hi = dmin1 * mn_hi;
            for (uint j = 0; j < 4; j++) {
                uint l = l_start + j;
                uint8_t byte = qs1[l];
                acc1 += (sc1_lo * float(byte & 0xF) - mn1_lo) * float(x_shared[w_base + l]);
                acc1 += (sc1_hi * float(byte >> 4) - mn1_hi) * float(x_shared[w_base + 32 + l]);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        // Transposed layout: [dim][K] instead of [K][dim]
        out[row0 * 8 + expert_k] = sum0;
        if (valid1) out[row1 * 8 + expert_k] = sum1;
    }
}

// ============================================================================
// Dynamic Q5_K batched expert down projection (per-expert packed input)
// ============================================================================

kernel void batch_expert_down_q5k_dyn(
    device const uint8_t*       layer_data      [[buffer(0)]],
    device const float*         x               [[buffer(1)]],
    device float*               out             [[buffer(2)]],
    device const uint32_t*      expert_indices  [[buffer(3)]],
    constant uint&              expert_stride   [[buffer(4)]],
    constant uint&              out_dim         [[buffer(5)]],
    constant uint&              in_dim          [[buffer(6)]],
    constant uint&              num_row_tgs     [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint expert_k = tgid / num_row_tgs;
    uint row0 = (tgid % num_row_tgs) * (ROWS_PER_TG * 2) + simd_group * 2;
    uint row1 = row0 + 1;
    bool valid0 = row0 < out_dim;
    bool valid1 = row1 < out_dim;

    device const float* ex = x + expert_k * in_dim;
    threadgroup half x_shared[MATVEC_X_SHARED_SIZE];
    for (uint i = lid; i < in_dim; i += tg_size) { x_shared[i] = half(ex[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid0) return;

    uint expert_id = expert_indices[expert_k];
    uint num_superblocks = in_dim / 256;
    uint bytes_per_row = num_superblocks * 176;
    device const uint8_t* expert_data = layer_data + expert_id * expert_stride;
    device const uint8_t* row0_data = expert_data + row0 * bytes_per_row;
    device const uint8_t* row1_data = valid1 ? expert_data + row1 * bytes_per_row : row0_data;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint g = simd_lane / 8;
    uint l_start = (simd_lane % 8) * 4;
    uint8_t hm_lo = 1u << (g * 2);
    uint8_t hm_hi = 1u << (g * 2 + 1);
    for (uint sb_idx = 0; sb_idx < num_superblocks; sb_idx++) {
        device const uint8_t* sb0 = row0_data + sb_idx * 176;
        float sc_lo, sc_hi, mn_lo, mn_hi;
        uint w_base = sb_idx * 256 + g * 64;
        float d0    = float(as_type<half>(ushort(ushort(sb0[0]) | (ushort(sb0[1]) << 8))));
        float dmin0 = float(as_type<half>(ushort(ushort(sb0[2]) | (ushort(sb0[3]) << 8))));
        unpack_q4k_scale_min_pair(sb0 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
        device const uint8_t* qh0 = sb0 + 16;
        device const uint8_t* ql0 = sb0 + 48 + g * 32;
        float sc0_lo = d0 * sc_lo, mn0_lo = dmin0 * mn_lo;
        float sc0_hi = d0 * sc_hi, mn0_hi = dmin0 * mn_hi;
        for (uint j = 0; j < 4; j++) {
            uint l = l_start + j;
            uint8_t byte_val = ql0[l];
            uint8_t q5_lo = (byte_val & 0xF) | ((qh0[l] & hm_lo) ? 16 : 0);
            uint8_t q5_hi = (byte_val >> 4) | ((qh0[l] & hm_hi) ? 16 : 0);
            acc0 += (sc0_lo * float(q5_lo) - mn0_lo) * float(x_shared[w_base + l]);
            acc0 += (sc0_hi * float(q5_hi) - mn0_hi) * float(x_shared[w_base + 32 + l]);
        }

        if (valid1) {
            device const uint8_t* sb1 = row1_data + sb_idx * 176;
            float d1    = float(as_type<half>(ushort(ushort(sb1[0]) | (ushort(sb1[1]) << 8))));
            float dmin1 = float(as_type<half>(ushort(ushort(sb1[2]) | (ushort(sb1[3]) << 8))));
            unpack_q4k_scale_min_pair(sb1 + 4, g, &sc_lo, &sc_hi, &mn_lo, &mn_hi);
            device const uint8_t* qh1 = sb1 + 16;
            device const uint8_t* ql1 = sb1 + 48 + g * 32;
            float sc1_lo = d1 * sc_lo, mn1_lo = dmin1 * mn_lo;
            float sc1_hi = d1 * sc_hi, mn1_hi = dmin1 * mn_hi;
            for (uint j = 0; j < 4; j++) {
                uint l = l_start + j;
                uint8_t byte_val = ql1[l];
                uint8_t q5_lo = (byte_val & 0xF) | ((qh1[l] & hm_lo) ? 16 : 0);
                uint8_t q5_hi = (byte_val >> 4) | ((qh1[l] & hm_hi) ? 16 : 0);
                acc1 += (sc1_lo * float(q5_lo) - mn1_lo) * float(x_shared[w_base + l]);
                acc1 += (sc1_hi * float(q5_hi) - mn1_hi) * float(x_shared[w_base + 32 + l]);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        // Transposed layout: [dim][K] instead of [K][dim]
        out[row0 * 8 + expert_k] = sum0;
        if (valid1) out[row1 * 8 + expert_k] = sum1;
    }
}
