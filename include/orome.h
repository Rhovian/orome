/*
 * orome.h — Core types and interfaces for multi-model MoE inference on Apple Silicon.
 *
 * Design principles:
 *   - No hardcoded model dimensions. Everything comes from ModelConfig.
 *   - Engine is parameterized: same code serves Qwen3.5-35B, 397B, or future models.
 *   - Clear ownership: each module has a well-defined interface.
 *   - Explicit state: no hidden globals in headers.
 */

#ifndef OROME_H
#define OROME_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ============================================================================
// Limits (allocation upper bounds, not model-specific)
// ============================================================================

#define OROME_MAX_EXPERTS       512     // largest MoE model we expect
#define OROME_MAX_ACTIVE        16      // max active experts per token
#define OROME_MAX_LAYERS        80      // enough for 397B (60 layers)
#define OROME_MAX_SEQ_LEN       1048576
#define OROME_GPU_KV_SEQ        4096    // GPU-side KV cache for attention offload

// ============================================================================
// Attention layer type
// ============================================================================

typedef enum {
    ATTN_FULL,          // Standard grouped-query attention (Q/K/V + RoPE + softmax)
    ATTN_LINEAR,        // GatedDeltaNet (linear attention with SSM-style state)
} AttnLayerType;

// ============================================================================
// Quantization
// ============================================================================

typedef enum {
    QUANT_NONE = 0,     // FP32 / BF16 (no packing)
    QUANT_4BIT = 4,     // 4-bit grouped quantization
    QUANT_2BIT = 2,     // 2-bit grouped quantization
} QuantType;

// ============================================================================
// Expert weight layout — computed from model config, not hardcoded
// ============================================================================

typedef struct {
    size_t expert_size;         // total bytes per expert (all 3 projections)
    // Offsets within a single expert's packed data:
    size_t gate_w_off, gate_s_off, gate_b_off;
    size_t up_w_off,   up_s_off,   up_b_off;
    size_t down_w_off, down_s_off, down_b_off;
} ExpertLayout;

// ============================================================================
// Model configuration — everything that varies between models
// ============================================================================

typedef struct {
    // --- Identity ---
    char name[64];              // e.g. "qwen3.5-35b-a3b"

    // --- Core dimensions ---
    int hidden_dim;             // e.g. 2048 (35B) or 4096 (397B)
    int num_layers;             // e.g. 40 (35B) or 60 (397B)
    int vocab_size;             // e.g. 248320
    float rms_norm_eps;         // e.g. 1e-6

    // --- Full attention ---
    int num_attn_heads;         // query heads, e.g. 16
    int num_kv_heads;           // KV heads (GQA), e.g. 2
    int head_dim;               // e.g. 256

    // --- Linear attention (GatedDeltaNet) ---
    int linear_num_v_heads;     // e.g. 32 (v-heads for delta-net)
    int linear_num_k_heads;     // e.g. 16
    int linear_key_dim;         // e.g. 128
    int linear_value_dim;       // e.g. 128
    int conv_kernel_size;       // e.g. 4

    // --- Layer layout ---
    int full_attn_interval;     // e.g. 4 means every 4th layer is full attn
    int full_attn_offset;       // e.g. 3 means full attn at layers 3,7,11,...
    int num_full_attn_layers;   // derived
    int num_linear_layers;      // derived
    AttnLayerType *layer_types; // [num_layers] — which attention type per layer

    // --- RoPE ---
    float rope_theta;           // e.g. 10000000.0
    float partial_rotary;       // fraction of head_dim that gets rotary, e.g. 0.25
    int rotary_dim;             // derived: head_dim * partial_rotary

    // --- MoE ---
    int num_experts;            // e.g. 256
    int num_experts_per_tok;    // default active experts, e.g. 8
    int moe_intermediate;       // expert FFN hidden dim, e.g. 512
    int shared_intermediate;    // shared expert FFN hidden dim, e.g. 512
    int group_size;             // quantization group size, e.g. 64

    // --- Expert weight layout (computed from above) ---
    ExpertLayout expert_4bit;
    ExpertLayout expert_2bit;

    // --- Special tokens ---
    int eos_tokens[4];          // EOS token IDs (up to 4, -1 terminated)
    int think_start_token;
    int think_end_token;

    // --- Derived (computed during init) ---
    int linear_total_key;       // linear_num_k_heads * linear_key_dim
    int linear_total_value;     // linear_num_v_heads * linear_value_dim
    int linear_conv_dim;        // total_key*2 + total_value
    int kv_dim;                 // num_kv_heads * head_dim
} ModelConfig;

// Compute derived fields and expert layouts from the core fields.
void model_config_init_derived(ModelConfig *cfg);

// Load config from a model directory (reads config.json from HF format).
// Returns 0 on success, -1 on failure.
int model_config_load(ModelConfig *cfg, const char *model_dir);

// Hardcoded config for known models (fallback if config.json not found).
void model_config_qwen35_35b(ModelConfig *cfg);
void model_config_qwen35_397b(ModelConfig *cfg);

// model_config_detect_layers declared below (after WeightFile typedef)

// ============================================================================
// Tensor manifest & weight file (mmap'd)
// ============================================================================

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
} TensorManifest;

typedef struct {
    void *data;
    size_t size;
    TensorManifest *manifest;
} WeightFile;

WeightFile *weights_open(const char *bin_path, const char *json_path);
void        weights_close(WeightFile *wf);
void       *weights_tensor_ptr(WeightFile *wf, const char *name);
TensorInfo *weights_tensor_info(WeightFile *wf, const char *name);
size_t      weights_tensor_offset(WeightFile *wf, const char *name);

// Layer-specific tensor helpers (constructs "model.layers.{layer}.{suffix}").
void       *weights_layer_ptr(WeightFile *wf, int layer, const char *suffix);
size_t      weights_layer_offset(WeightFile *wf, int layer, const char *suffix);

// Detect layer types (full vs linear attention) from weight manifest.
// Call after weights_open — overrides any hardcoded layer_types with reality.
void model_config_detect_layers(ModelConfig *cfg, WeightFile *wf);

// ============================================================================
// KV cache & linear attention state
// ============================================================================

typedef struct {
    float *k_cache;     // [gpu_kv_seq * kv_dim]
    float *v_cache;     // [gpu_kv_seq * kv_dim]
    int len;
} KVCache;

typedef struct {
    float *conv_state;  // [(conv_kernel_size - 1) * linear_conv_dim]
    float *ssm_state;   // [linear_num_v_heads * linear_key_dim * linear_value_dim]
} LinearAttnState;

KVCache         *kv_cache_new(const ModelConfig *cfg);
void             kv_cache_free(KVCache *kv);
LinearAttnState *linear_state_new(const ModelConfig *cfg);
void             linear_state_free(LinearAttnState *s);

// ============================================================================
// Metal GPU context
// ============================================================================

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;

    // Pipelines (created from shaders.metal)
    id<MTLComputePipelineState> matvec_4bit;
    id<MTLComputePipelineState> matvec_2bit;
    id<MTLComputePipelineState> norm_sum_sq;
    id<MTLComputePipelineState> norm_apply;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> attn_softmax;
    id<MTLComputePipelineState> attn_values;
    id<MTLComputePipelineState> sigmoid_gate;
    id<MTLComputePipelineState> swiglu;
    id<MTLComputePipelineState> moe_combine;
    id<MTLComputePipelineState> delta_net;
    id<MTLComputePipelineState> conv1d;
    id<MTLComputePipelineState> rms_norm_qk;
    id<MTLComputePipelineState> decay_beta;
    id<MTLComputePipelineState> gated_rms_norm;
    id<MTLComputePipelineState> batch_expert_mv;
    id<MTLComputePipelineState> batch_swiglu;
    id<MTLComputePipelineState> batch_expert_down;
    id<MTLComputePipelineState> moe_combine_packed;
    id<MTLComputePipelineState> rms_norm_qk_w;
    id<MTLComputePipelineState> rope_apply;
    id<MTLComputePipelineState> kv_cache_write;
    id<MTLComputePipelineState> softmax_topk;
    id<MTLComputePipelineState> batch_expert_mv_dyn;
    id<MTLComputePipelineState> batch_expert_down_dyn;
    id<MTLComputePipelineState> expert_gate_up_swiglu;
    id<MTLComputePipelineState> copy_buffer;
    id<MTLComputePipelineState> residual_add_sq;
    id<MTLComputePipelineState> norm_apply_partial;
    id<MTLComputePipelineState> moe_combine_copy_sq;
    id<MTLComputePipelineState> matvec_4bit_2row;
    id<MTLComputePipelineState> batch_expert_down_dyn_2row;
    id<MTLComputePipelineState> argmax;

    // Shared buffers (allocated based on ModelConfig)
    id<MTLBuffer> buf_input;
    id<MTLBuffer> buf_output;
    id<MTLBuffer> buf_argmax_result;   // single uint32 for GPU argmax
    id<MTLBuffer> buf_sum_sq;
    id<MTLBuffer> buf_residual;
    id<MTLBuffer> buf_h_mid;
    id<MTLBuffer> buf_moe_hidden;
    id<MTLBuffer> buf_combine_params;
    id<MTLBuffer> buf_weights;          // mmap'd weight file as MTL buffer

    // Per-expert GPU buffers
    id<MTLBuffer> buf_multi_expert_input;
    id<MTLBuffer> buf_multi_expert_data[OROME_MAX_ACTIVE];
    id<MTLBuffer> buf_multi_expert_gate[OROME_MAX_ACTIVE];
    id<MTLBuffer> buf_multi_expert_up[OROME_MAX_ACTIVE];
    id<MTLBuffer> buf_multi_expert_act[OROME_MAX_ACTIVE];
    id<MTLBuffer> buf_multi_expert_out[OROME_MAX_ACTIVE];

    // Packed batch expert buffers (K × dim)
    id<MTLBuffer> buf_batch_expert_gate;
    id<MTLBuffer> buf_batch_expert_up;
    id<MTLBuffer> buf_batch_expert_act;
    id<MTLBuffer> buf_batch_expert_out;
    id<MTLBuffer> buf_expert_offsets;
    id<MTLBuffer> buf_topk_indices;  // [K] uint32_t expert indices from GPU routing

    // Shared expert buffers
    id<MTLBuffer> buf_shared_gate;
    id<MTLBuffer> buf_shared_up;
    id<MTLBuffer> buf_shared_act;
    id<MTLBuffer> buf_shared_out;

    // KV cache GPU mirrors (per layer, only for full attention)
    id<MTLBuffer> __strong *buf_kv_k;   // [num_full_attn_layers]
    id<MTLBuffer> __strong *buf_kv_v;   // [num_full_attn_layers]

    // Attention scratch
    id<MTLBuffer> buf_attn_scores;
    id<MTLBuffer> buf_attn_output;

    // Expert layer data wrapped as Metal buffers (per layer)
    id<MTLBuffer> __strong *buf_expert_layers;  // [num_layers] - mmap'd expert data
    int num_expert_layers;

    // Linear attention GPU state (per layer)
    id<MTLBuffer> __strong *buf_linear_state;   // [num_linear_layers]
    id<MTLBuffer> __strong *buf_conv_state;     // [num_linear_layers]
    id<MTLBuffer> buf_linear_q;
    id<MTLBuffer> buf_linear_k;
    id<MTLBuffer> buf_linear_v;
    id<MTLBuffer> buf_linear_decay;
    id<MTLBuffer> buf_linear_beta;
    id<MTLBuffer> buf_linear_output;
    id<MTLBuffer> buf_conv_input;
    id<MTLBuffer> buf_conv_output;
} MetalCtx;

// Initialize Metal: create device, compile shaders, allocate buffers.
// Returns NULL if Metal is unavailable (CPU-only fallback).
MetalCtx *metal_setup(const ModelConfig *cfg);

// Wrap mmap'd weights as a Metal buffer for zero-copy GPU access.
void metal_set_weights(MetalCtx *ctx, WeightFile *wf);

// metal_set_expert_weights declared after ExpertFiles typedef below

void metal_free(MetalCtx *ctx);

// ============================================================================
// GPU matvec dispatch
// ============================================================================

typedef struct {
    id<MTLBuffer> w_buf;    size_t w_off;
    id<MTLBuffer> s_buf;    size_t s_off;
    id<MTLBuffer> b_buf;    size_t b_off;
    id<MTLBuffer> in_buf;   size_t in_off;  // input buffer (NULL → ctx->buf_input)
    id<MTLBuffer> out_buf;  size_t out_off;
    float *out_ptr;         // CPU pointer for readback (or NULL)
    int out_dim;
    int in_dim;
    int group_size;
    bool is_2bit;
} GpuMatvecJob;

// Batch multiple matvecs in one command buffer. Returns after GPU completion.
void gpu_run_matvec_batch(MetalCtx *ctx, GpuMatvecJob *jobs, int count);

// Encode a single matvec into an existing command encoder (no commit).
void gpu_encode_matvec_job(id<MTLComputeCommandEncoder> enc, MetalCtx *ctx,
                           GpuMatvecJob *job);

// Single matvec (copies input to buf_input, runs, copies result out).
void gpu_dequant_matvec(MetalCtx *ctx, const ModelConfig *cfg,
                        uint32_t *W, uint16_t *scales, uint16_t *biases,
                        float *x, float *out, int out_dim, int in_dim,
                        QuantType quant);

// Dispatch to GPU if available, else CPU fallback.
void fast_dequant_matvec(MetalCtx *ctx, const ModelConfig *cfg,
                         uint32_t *W, uint16_t *scales, uint16_t *biases,
                         float *x, float *out, int out_dim, int in_dim,
                         QuantType quant);

// ============================================================================
// CPU compute kernels
// ============================================================================

// Dequantized matrix-vector multiply (4-bit and 2-bit)
void cpu_dequant_matvec(const uint32_t *W, const uint16_t *scales,
                        const uint16_t *biases, const float *x, float *out,
                        int out_dim, int in_dim, int group_size);
void cpu_dequant_matvec_2bit(const uint32_t *W, const uint16_t *scales,
                             const uint16_t *biases, const float *x, float *out,
                             int out_dim, int in_dim, int group_size);

// Normalization
void cpu_rms_norm(const float *x, const uint16_t *weight, float *out,
                  int dim, float eps);
void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps);
void cpu_rms_norm_gated(const float *values, const float *z,
                        const uint16_t *weight, float *out,
                        int num_heads, int value_dim, float eps);

// Activations & reductions
void cpu_softmax(float *x, int len);
float cpu_sigmoid(float x);
void cpu_swiglu(const float *gate, const float *up, float *out, int dim);
int cpu_argmax(const float *x, int len);
void cpu_topk(const float *scores, int n, int k, int *indices, float *weights);
void cpu_normalize_weights(float *weights, int K);

// Vector ops
void cpu_vec_add(float *a, const float *b, int len);
void cpu_vec_madd(float *out, const float *x, float scale, int len);

// Positional encoding
void apply_rotary_emb(float *q, float *k, int pos, int num_heads,
                      int head_dim, int rotary_dim, float theta);

// Conv1d step (for linear attention)
void cpu_conv1d_step(float *conv_state, const float *input,
                     const uint16_t *weights, float *output,
                     int channels, int kernel_size);

// BF16 conversion
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// Timing
double now_ms(void);

// ============================================================================
// Attention layers
// ============================================================================

// Attention forward — returns pre-O-proj output via attn_out/attn_out_dim.
// Does NOT perform O projection or residual add (caller handles these).
void full_attention_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                            int layer_idx, int pos, float *hidden, float *residual,
                            float *h_post, KVCache *kv,
                            float **attn_out, int *attn_out_dim);

void linear_attention_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                              int layer_idx, int pos, float *hidden, float *residual,
                              float *h_post, LinearAttnState *state,
                              float **attn_out, int *attn_out_dim);

// ============================================================================
// MoE (Mixture of Experts)
// ============================================================================

// Expert weight management — mmap'd for machines with enough RAM
typedef struct {
    void **layer_data;      // [num_layers] mmap'd expert data (NULL if not loaded)
    size_t *layer_size;     // [num_layers] size of each mmap
    int *layer_fds;         // [num_layers] fd (kept open for mmap lifetime)
    bool pread_mode;        // true when experts are loaded via pread into staging buffers
    int *layer_fds_2bit;    // [num_layers] fd for 2-bit expert files
    int num_experts;        // cached for hot mask checks
    int num_layers;         // cached for cleanup
    uint32_t *hot_mask;     // [num_layers * (max_experts/32)] bitmask of hot experts
    bool tiered_quant;      // using tiered quantization?
    bool all_mmaped;        // true if all layers are mmap'd in memory
} ExpertFiles;

ExpertFiles *expert_files_open(const ModelConfig *cfg, const char *model_dir,
                               const char *hot_mask_path);
void         expert_files_close(ExpertFiles *ef, const ModelConfig *cfg);
bool         expert_is_hot(const ExpertFiles *ef, int layer, int expert_id);
void         moe_set_profile_experts(bool enabled);
bool         moe_get_profile_experts(void);

// Wrap mmap'd expert layer data as Metal buffers for GPU expert forward.
void metal_set_expert_weights(MetalCtx *ctx, ExpertFiles *ef, const ModelConfig *cfg);

void moe_forward(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                 int layer_idx, float *hidden, float *h_post,
                 ExpertFiles *ef, int K, QuantType quant);

// MoE forward with pre-computed routing (skips routing GPU batch).
// gate_scores and shared_gate_score must already be computed.
// If gpu_combine=true: adds moe_combine kernel on GPU, writes result to
// ctx->buf_moe_hidden, does NOT wait or read back (caller must sync).
// If gpu_combine=false: reads back expert results, combines on CPU into hidden.
void moe_forward_routed(WeightFile *wf, MetalCtx *ctx, const ModelConfig *cfg,
                        int layer_idx, float *hidden, float *h_post,
                        float *gate_scores, float shared_gate_score,
                        ExpertFiles *ef, int K, QuantType quant,
                        bool gpu_combine);

// ============================================================================
// Engine — full forward pass
// ============================================================================

typedef struct {
    bool enabled;
    int hot_k;                  // reduced K when engaged
    int min_gen;                // minimum generated tokens before engaging
    double proj_threshold_ms;   // EMA threshold for engagement
    double proj_ema_ms;         // current EMA
    int generated;              // number of timed tokens seen
    bool engaged;               // latched once throttling engages
    bool have_proj;             // true after first timing sample
} ThermalKState;

typedef struct {
    ModelConfig     *cfg;
    WeightFile      *wf;
    MetalCtx        *ctx;       // NULL if CPU-only
    ExpertFiles     *ef;

    // Per-token state
    float *hidden;
    float *residual;
    float *h_post;
    float *logits;

    // Per-layer state
    KVCache         **kv_caches;       // [num_full_attn_layers]
    LinearAttnState **linear_states;   // [num_linear_layers]

    // Generation state
    int pos;
    QuantType quant;
    int active_experts;     // runtime K (may differ from cfg default)
    ThermalKState thermal;

    // Precomputed weight offsets (opaque, owned by engine.m)
    void *weight_cache;
} Engine;

Engine *engine_create(ModelConfig *cfg, WeightFile *wf, MetalCtx *ctx,
                      ExpertFiles *ef, QuantType quant, int active_experts);
void    engine_free(Engine *eng);
void    engine_reset(Engine *eng);  // clear caches, reset pos

// Run one token through the model. Returns the next token ID.
int engine_step(Engine *eng, int token_id);

// ============================================================================
// Tokenizer
// ============================================================================

typedef struct {
    char **tokens;
    int *lengths;
    int num_tokens;
    int max_id;
} Vocabulary;

typedef struct {
    int *ids;
    int count;
} PromptTokens;

Vocabulary   *vocab_load(const char *path);
void          vocab_free(Vocabulary *v);

int           tokenizer_init(const char *model_dir);
PromptTokens *tokenizer_encode(const char *text);
const char   *tokenizer_decode(int token_id);
void          prompt_tokens_free(PromptTokens *pt);

bool          is_eos_token(const ModelConfig *cfg, int token_id);

// ============================================================================
// Server (HTTP/SSE, OpenAI-compatible API)
// ============================================================================

void serve_loop(Engine *eng, Vocabulary *vocab, int port);

#endif // OROME_H
