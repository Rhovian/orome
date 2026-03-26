/*
 * kernels.m — CPU utilities: timing, argmax, sampling.
 */

#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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


static int cpu_argmax(const float *x, int len) {
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

