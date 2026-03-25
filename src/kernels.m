/*
 * kernels.m — CPU compute kernels for dequantization, normalization, activations.
 *
 * These are the fallback path when Metal is unavailable, and also used for
 * operations that are cheaper on CPU (small reductions, topk, etc).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>

#include "orome.h"

// ============================================================================
// Timing
// ============================================================================

double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Dequantized matrix-vector multiply
// ============================================================================

void cpu_dequant_matvec(const uint32_t *W, const uint16_t *scales,
                        const uint16_t *biases, const float *x, float *out,
                        int out_dim, int in_dim, int group_size) {
    int packed_cols = in_dim / 8;
    int num_groups = in_dim / group_size;

    for (int r = 0; r < out_dim; r++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + r * packed_cols;
        const uint16_t *s_row = scales + r * num_groups;
        const uint16_t *b_row = biases + r * num_groups;

        for (int col = 0; col < packed_cols; col++) {
            int g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            uint32_t packed = w_row[col];
            int x_base = col * 8;

            for (int b = 0; b < 8; b++) {
                float nibble = (float)((packed >> (b * 4)) & 0xF);
                acc += nibble * scale * x[x_base + b] + bias * x[x_base + b];
            }
        }
        out[r] = acc;
    }
}

void cpu_dequant_matvec_2bit(const uint32_t *W, const uint16_t *scales,
                             const uint16_t *biases, const float *x, float *out,
                             int out_dim, int in_dim, int group_size) {
    int packed_cols = in_dim / 16;
    int num_groups = in_dim / group_size;

    for (int r = 0; r < out_dim; r++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + r * packed_cols;
        const uint16_t *s_row = scales + r * num_groups;
        const uint16_t *b_row = biases + r * num_groups;

        for (int col = 0; col < packed_cols; col++) {
            int g = col / (group_size / 16);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            uint32_t packed = w_row[col];
            int x_base = col * 16;

            for (int b = 0; b < 16; b++) {
                float crumb = (float)((packed >> (b * 2)) & 0x3);
                acc += crumb * scale * x[x_base + b] + bias * x[x_base + b];
            }
        }
        out[r] = acc;
    }
}

// ============================================================================
// Normalization
// ============================================================================

void cpu_rms_norm(const float *x, const uint16_t *weight, float *out,
                  int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms * bf16_to_f32(weight[i]);
    }
}

void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms;
    }
}

void cpu_rms_norm_gated(const float *values, const float *z,
                        const uint16_t *weight, float *out,
                        int num_heads, int value_dim, float eps) {
    for (int h = 0; h < num_heads; h++) {
        int base = h * value_dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < value_dim; i++) {
            float v = values[base + i];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / value_dim + eps);
        for (int i = 0; i < value_dim; i++) {
            float normed = values[base + i] * inv_rms;
            float zval = z[base + i];
            float gate = zval / (1.0f + expf(-zval));  // SiLU
            out[base + i] = normed * gate * bf16_to_f32(weight[i]);
        }
    }
}

// ============================================================================
// Activations & reductions
// ============================================================================

float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void cpu_softmax(float *x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) x[i] *= inv_sum;
}

void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

int cpu_argmax(const float *x, int len) {
    int best = 0;
    for (int i = 1; i < len; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

int cpu_sample_topk(const float *logits, int vocab_size, int top_k, float temperature) {
    if (temperature <= 0 || top_k <= 1) return cpu_argmax(logits, vocab_size);
    if (top_k > 64) top_k = 64;

    // Collect top-k values and indices via min-heap replacement
    float vals[64];
    int idxs[64];
    for (int j = 0; j < top_k; j++) { vals[j] = -1e30f; idxs[j] = 0; }
    for (int i = 0; i < vocab_size; i++) {
        int min_j = 0;
        for (int j = 1; j < top_k; j++) {
            if (vals[j] < vals[min_j]) min_j = j;
        }
        if (logits[i] > vals[min_j]) {
            vals[min_j] = logits[i];
            idxs[min_j] = i;
        }
    }

    // Softmax with temperature
    float maxl = vals[0];
    for (int j = 1; j < top_k; j++) {
        if (vals[j] > maxl) maxl = vals[j];
    }
    float probs[64], sum = 0;
    for (int j = 0; j < top_k; j++) {
        probs[j] = expf((vals[j] - maxl) / temperature);
        sum += probs[j];
    }
    for (int j = 0; j < top_k; j++) probs[j] /= sum;

    // Sample from distribution
    float r = (float)arc4random() / (float)UINT32_MAX;
    float cumsum = 0;
    for (int j = 0; j < top_k; j++) {
        cumsum += probs[j];
        if (r <= cumsum) return idxs[j];
    }
    return idxs[top_k - 1];
}

void cpu_topk(const float *scores, int n, int k, int *indices, float *weights) {
    for (int j = 0; j < k; j++) {
        int best = -1;
        float best_val = -1e30f;
        for (int i = 0; i < n; i++) {
            bool taken = false;
            for (int p = 0; p < j; p++) {
                if (indices[p] == i) { taken = true; break; }
            }
            if (!taken && scores[i] > best_val) {
                best_val = scores[i];
                best = i;
            }
        }
        indices[j] = best;
        weights[j] = best_val;
    }
}

void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv_sum;
    }
}

// ============================================================================
// Vector operations
// ============================================================================

void cpu_vec_add(float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) a[i] += b[i];
}

void cpu_vec_madd(float *out, const float *x, float scale, int len) {
    for (int i = 0; i < len; i++) out[i] += x[i] * scale;
}

// ============================================================================
// Positional encoding (RoPE)
// ============================================================================

void apply_rotary_emb(float *q, float *k, int pos, int num_heads,
                      int head_dim, int rotary_dim, float theta) {
    int half_rot = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        float *kh = k + h * head_dim;
        for (int i = 0; i < half_rot; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            // Half-split pairing: (i, i+half) — MLX traditional=False
            float q0 = qh[i], q1 = qh[i + half_rot];
            qh[i]            = q0 * cos_a - q1 * sin_a;
            qh[i + half_rot] = q0 * sin_a + q1 * cos_a;

            float k0 = kh[i], k1 = kh[i + half_rot];
            kh[i]            = k0 * cos_a - k1 * sin_a;
            kh[i + half_rot] = k0 * sin_a + k1 * cos_a;
        }
    }
}

// ============================================================================
// Conv1d step (for linear attention)
// ============================================================================

void cpu_conv1d_step(float *conv_state, const float *input,
                     const uint16_t *weights, float *output,
                     int channels, int kernel_size) {
    int history_slots = kernel_size - 1;
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // History slots
        for (int k = 0; k < history_slots; k++) {
            acc += conv_state[k * channels + c] * bf16_to_f32(weights[c * kernel_size + k]);
        }
        // Current input
        acc += input[c] * bf16_to_f32(weights[c * kernel_size + history_slots]);

        // SiLU activation
        output[c] = acc / (1.0f + expf(-acc));

        // Shift history
        for (int k = 0; k < history_slots - 1; k++) {
            conv_state[k * channels + c] = conv_state[(k + 1) * channels + c];
        }
        conv_state[(history_slots - 1) * channels + c] = input[c];
    }
}
