/*
 * orome.h — Core types and interfaces for GGUF MoE inference on Apple Silicon.
 *
 * Design principles:
 *   - No hardcoded model dimensions. Everything comes from ModelConfig.
 *   - Engine is parameterized: the same code can serve current and future models.
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

// Forward declarations
typedef struct GGUFFile GGUFFile;
typedef struct GGUFTensorInfo GGUFTensorInfo;

// ============================================================================
// Limits (allocation upper bounds, not model-specific)
// ============================================================================

#define OROME_MAX_ACTIVE        16      // max active experts per token
#define OROME_GPU_KV_SEQ        4096    // GPU-side KV cache for attention offload

// ============================================================================
// Attention layer type
// ============================================================================

typedef enum {
    ATTN_FULL,          // Standard grouped-query attention (Q/K/V + RoPE + softmax)
    ATTN_LINEAR,        // GatedDeltaNet (linear attention with SSM-style state)
} AttnLayerType;

// ============================================================================
// Model configuration — everything that varies between models
// ============================================================================

typedef struct {
    // --- Identity ---
    char name[64];              // e.g. "qwen3.5-35b-a3b"

    // --- Core dimensions ---
    int hidden_dim;             // e.g. 2048
    int num_layers;             // e.g. 40
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
    // --- Special tokens ---
    int eos_tokens[4];          // EOS token IDs (up to 4, -1 terminated)
    int think_end_token;

    // --- Derived (computed during init) ---
    int linear_total_key;       // linear_num_k_heads * linear_key_dim
    int linear_total_value;     // linear_num_v_heads * linear_value_dim
    int linear_conv_dim;        // total_key*2 + total_value
    int kv_dim;                 // num_kv_heads * head_dim
} ModelConfig;

// Compute derived fields and expert layouts from the core fields.
void model_config_init_derived(ModelConfig *cfg);

// ============================================================================
// Metal GPU context
// ============================================================================

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;

    // Pipelines
    id<MTLComputePipelineState> norm_sum_sq;
    id<MTLComputePipelineState> norm_apply;
    id<MTLComputePipelineState> attn_scores;
    id<MTLComputePipelineState> attn_softmax;
    id<MTLComputePipelineState> attn_values;
    id<MTLComputePipelineState> sigmoid_gate;
    id<MTLComputePipelineState> swiglu;
    id<MTLComputePipelineState> delta_net;
    id<MTLComputePipelineState> rms_norm_qk;
    id<MTLComputePipelineState> gated_rms_norm;
    id<MTLComputePipelineState> batch_swiglu;
    id<MTLComputePipelineState> rms_norm_qk_w;
    id<MTLComputePipelineState> rope_apply;
    id<MTLComputePipelineState> kv_cache_write;
    id<MTLComputePipelineState> softmax_topk;
    id<MTLComputePipelineState> copy_buffer;
    id<MTLComputePipelineState> residual_add_sq;
    id<MTLComputePipelineState> norm_apply_partial;
    id<MTLComputePipelineState> moe_combine_copy_sq;
    id<MTLComputePipelineState> argmax;
    id<MTLComputePipelineState> deinterleave_qgate;
    id<MTLComputePipelineState> copy_tmp_to_buf;
    id<MTLComputePipelineState> conv1d_f32;
    id<MTLComputePipelineState> decay_beta_f32;
    id<MTLComputePipelineState> matvec_f32;
    id<MTLComputePipelineState> matvec_q4k;
    id<MTLComputePipelineState> matvec_q5k;
    id<MTLComputePipelineState> matvec_q8_0;
    id<MTLComputePipelineState> matvec_q6k;
    id<MTLComputePipelineState> batch_expert_mv_q4k_dyn;
    id<MTLComputePipelineState> batch_expert_gate_up_swiglu_q4k_dyn;
    id<MTLComputePipelineState> shared_gate_up_swiglu_q4k;
    id<MTLComputePipelineState> batch_expert_down_q4k_dyn;
    id<MTLComputePipelineState> batch_expert_down_q5k_dyn;

    // Shared buffers (allocated based on ModelConfig)
    id<MTLBuffer> buf_input;
    id<MTLBuffer> buf_output;
    id<MTLBuffer> buf_argmax_result;   // single uint32 for GPU argmax
    id<MTLBuffer> buf_sum_sq;
    id<MTLBuffer> buf_residual;
    id<MTLBuffer> buf_h_mid;
    id<MTLBuffer> buf_moe_hidden;
    id<MTLBuffer> buf_combine_params;

    // Batched expert buffers (K × dim)
    id<MTLBuffer> buf_batch_expert_gate;
    id<MTLBuffer> buf_batch_expert_up;
    id<MTLBuffer> buf_batch_expert_act;
    id<MTLBuffer> buf_batch_expert_out;
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

    // Linear attention GPU state (per layer)
    id<MTLBuffer> __strong *buf_linear_state;   // [num_linear_layers]
    id<MTLBuffer> __strong *buf_conv_state;     // [num_linear_layers]
    id<MTLBuffer> buf_linear_q;
    id<MTLBuffer> buf_linear_v;
    id<MTLBuffer> buf_linear_decay;
    id<MTLBuffer> buf_linear_beta;
    id<MTLBuffer> buf_linear_output;
    id<MTLBuffer> buf_conv_input;
    id<MTLBuffer> buf_conv_output;
} MetalCtx;

MetalCtx *metal_setup(const ModelConfig *cfg);
void metal_free(MetalCtx *ctx);

// ============================================================================
// Weight format abstraction
// ============================================================================

typedef enum {
    QFMT_GGUF_Q4_K,     // GGUF: 256 weights in 144-byte super-block
    QFMT_GGUF_Q5_K,     // GGUF: 256 weights in 176-byte super-block
    QFMT_GGUF_Q6_K,     // GGUF: 256 weights in 210-byte super-block
    QFMT_GGUF_Q8_0,     // GGUF: 32 int8 weights + fp16 scale per block
    QFMT_F16,           // unquantized float16
    QFMT_BF16,          // unquantized bfloat16
    QFMT_F32,           // unquantized float32
} QuantFormat;

// A fully-resolved reference to a quantized weight tensor, ready for GPU dispatch.
// The engine doesn't need to know the underlying file format — it just dispatches.
typedef struct {
    id<MTLBuffer> buffer;       // Metal buffer containing the data
    size_t offset;              // byte offset within buffer
    id<MTLComputePipelineState> pipeline;  // which dequant kernel to use
    QuantFormat format;
    uint32_t out_dim;
    uint32_t in_dim;
} TensorRef;

// Per-projection info within an ExpertLayerRef
typedef struct {
    size_t offset;              // byte offset to start of 3D tensor in buffer
    size_t expert_stride;       // bytes between consecutive experts
    QuantFormat format;
} ExpertProjRef;

// All expert weight references for one layer, handles mixed quant
typedef struct {
    id<MTLBuffer> buffer;       // single Metal buffer (GGUF mmap)
    ExpertProjRef gate, up, down;
} ExpertLayerRef;

// Opaque provider that resolves tensor names to dispatch-ready refs
typedef struct {
    GGUFFile *gguf;             // non-NULL for GGUF models
    id<MTLBuffer> model_buf;    // Metal buffer wrapping the model data
} FormatProvider;

// Format-agnostic per-layer weight cache.
// Engine uses these TensorRefs directly — never sees byte offsets or quant formats.
typedef struct {
    TensorRef input_norm;       // F32 norm weights (not matvec'd, used by RMS norm kernel)
    TensorRef post_norm;
    TensorRef routing_gate;     // MoE routing: [hidden, num_experts]
    TensorRef shared_gate;      // shared expert gate projection
    TensorRef shared_up;        // shared expert up projection
    TensorRef shared_down;      // shared expert down projection
    TensorRef shared_expert_gate; // scalar gate for shared expert
    union {
        struct {
            TensorRef qkv;      // fused Q+K+V projection
            TensorRef z;        // Z gate (in_proj_z)
            TensorRef a;        // alpha projection
            TensorRef b;        // beta projection
            TensorRef o;        // output projection
            TensorRef conv;     // conv1d weights (F32)
            TensorRef A_log;    // SSM decay (F32)
            TensorRef dt_bias;  // SSM time step bias (F32)
            TensorRef o_norm;   // output norm (F32)
        } lin;
        struct {
            TensorRef q, k, v;  // separate Q, K, V projections
            TensorRef o;        // output projection
            TensorRef q_norm;   // query norm (F32)
            TensorRef k_norm;   // key norm (F32)
        } full;
    };
} LayerTensorCache;

// Global (non-per-layer) tensor refs
typedef struct {
    TensorRef lm_head;          // final projection
    TensorRef final_norm;       // final RMS norm
} GlobalTensorCache;


// Build tensor cache from GGUF file
LayerTensorCache *build_tensor_cache_gguf(GGUFFile *gf, id<MTLBuffer> model_buf,
                                           MetalCtx *ctx,
                                           const ModelConfig *cfg,
                                           GlobalTensorCache *globals);

FormatProvider   *format_provider_open_gguf(GGUFFile *gf, MetalCtx *ctx);
void              format_provider_close(FormatProvider *fp);
ExpertLayerRef    format_resolve_expert_layer(FormatProvider *fp, int layer_idx,
                                              uint32_t num_experts);
void              format_dispatch_matvec(id<MTLComputeCommandEncoder> enc,
                                         MetalCtx *ctx, TensorRef *ref,
                                         id<MTLBuffer> in_buf, size_t in_off,
                                         id<MTLBuffer> out_buf, size_t out_off);
id<MTLComputePipelineState> format_pipeline_for(MetalCtx *ctx, QuantFormat fmt);
QuantFormat       format_from_ggml_type(uint32_t ggml_type);

// Timing
double now_ms(void);
int cpu_sample_topk(const float *logits, int vocab_size, int top_k, float temperature);

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
    MetalCtx        *ctx;

    float *hidden;

    // Generation state
    int pos;
    int active_experts;     // runtime K (may differ from cfg default)
    ThermalKState thermal;

    // Format-agnostic tensor cache (built via format.m)
    LayerTensorCache *tensor_cache;
    GlobalTensorCache globals;

    // GGUF format support
    GGUFFile *gf;
    ExpertLayerRef *expert_layer_cache;  // [num_layers], pre-resolved from FormatProvider
} Engine;

Engine *engine_create(ModelConfig *cfg, MetalCtx *ctx, int active_experts);
void    engine_free(Engine *eng);
void    engine_reset(Engine *eng);  // clear caches, reset pos

// Run one token through the model. Returns the next token ID.
int engine_step(Engine *eng, int token_id);

// ============================================================================
// Tokenizer
// ============================================================================

typedef struct {
    int *ids;
    int count;
} PromptTokens;

int           tokenizer_init(const char *model_dir);
PromptTokens *tokenizer_encode(const char *text);
const char   *tokenizer_decode(int token_id);
void          prompt_tokens_free(PromptTokens *pt);

bool          is_eos_token(const ModelConfig *cfg, int token_id);

// ============================================================================
// Server (HTTP/SSE, OpenAI-compatible API)
// ============================================================================

void serve_loop(Engine *eng, int port);

// ============================================================================
// GGUF file format support
// ============================================================================

struct GGUFTensorInfo {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;          // GGML quantization type
    uint64_t offset;        // relative to data section start
};

struct GGUFFile {
    int fd;
    void *mmap_base;
    size_t file_size;
    size_t data_offset;     // absolute offset where tensor data begins
    GGUFTensorInfo *tensors;
    uint64_t num_tensors;
    uint32_t alignment;
    // Extracted model config from metadata
    char arch[32];
    int num_layers, hidden_dim, num_experts, num_experts_per_tok;
    int num_attn_heads, num_kv_heads, moe_intermediate;
    float rope_theta;
    int vocab_size;
};

GGUFFile       *gguf_open(const char *path);
void            gguf_close(GGUFFile *gf);
GGUFTensorInfo *gguf_find_tensor(GGUFFile *gf, const char *name);
size_t          gguf_tensor_size(GGUFTensorInfo *ti);
const char     *ggml_type_name(uint32_t type);
void            gguf_print_summary(GGUFFile *gf);

#endif // OROME_H
