/*
 * weights.m — Tensor manifest, weight file loading, and model config.
 *
 * Handles mmap'd weight files with JSON manifests for O(1) tensor lookup.
 * Also provides model config loading from HF config.json.
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
// Tensor hash table (FNV-1a, open addressing)
// ============================================================================

#define TENSOR_HT_SIZE 8192

typedef struct {
    const char *key;
    TensorInfo *value;
} TensorHTEntry;

static TensorHTEntry s_tensor_ht[TENSOR_HT_SIZE];
static int s_tensor_ht_built = 0;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(TensorManifest *m) {
    if (s_tensor_ht_built) return;
    memset(s_tensor_ht, 0, sizeof(s_tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (s_tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        s_tensor_ht[idx].key = m->tensors[i].name;
        s_tensor_ht[idx].value = &m->tensors[i];
    }
    s_tensor_ht_built = 1;
}

static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!s_tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (s_tensor_ht[idx].key) {
        if (strcmp(s_tensor_ht[idx].key, name) == 0) {
            return s_tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Tensor name normalization
// ============================================================================
//
// The engine uses canonical tensor names. The weight loader normalizes
// whatever naming convention the model files use into canonical form.
//
// Canonical names (what engine code uses):
//   model.layers.{N}.self_attn.q_proj.weight     — full attention
//   model.layers.{N}.linear_attn.in_proj_qkv.weight  — linear attention
//   model.layers.{N}.mlp.gate.weight              — MoE routing
//   model.embed_tokens.weight                     — embedding
//   lm_head.weight                                — output head
//   model.norm.weight                             — final norm
//
// Known source conventions that get normalized:
//   "language_model.model.layers.N.*"  → strip "language_model." prefix
//   "language_model.lm_head.*"         → strip "language_model." prefix
//   (future: GPTQ qweight→weight, etc.)

static char *normalize_tensor_name(const char *raw) {
    const char *name = raw;

    // Strip "language_model." prefix (HF/MLX convention)
    if (strncmp(name, "language_model.", 15) == 0) {
        name = name + 15;
    }

    return strdup(name);
}

// ============================================================================
// Manifest loading (JSON) — normalizes tensor names on load
// ============================================================================

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];
            t->name = normalize_tensor_name([key UTF8String]);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }
            strncpy(t->dtype, [info[@"dtype"] UTF8String], 7);
            m->num_tensors++;
        }

        printf("[weights] Loaded %d tensors from %s (names normalized)\n",
               m->num_tensors, json_path);
        return m;
    }
}

// ============================================================================
// Weight file (mmap'd binary blob)
// ============================================================================

WeightFile *weights_open(const char *bin_path, const char *json_path) {
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }
    // Pre-fault all weight pages into RAM and lock them
    madvise(data, size, MADV_WILLNEED);
    mlock(data, size);

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        munmap(data, size);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;

    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
    return wf;
}

void weights_close(WeightFile *wf) {
    if (!wf) return;
    if (wf->manifest) {
        for (int i = 0; i < wf->manifest->num_tensors; i++) {
            free((void *)wf->manifest->tensors[i].name);
        }
        free(wf->manifest->tensors);
        free(wf->manifest);
    }
    if (wf->data && wf->size > 0) {
        munmap(wf->data, wf->size);
    }
    free(wf);
}

TensorInfo *weights_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}

void *weights_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    return (uint8_t *)wf->data + t->offset;
}

size_t weights_tensor_offset(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    return t ? t->offset : 0;
}

// Layer-specific helpers
void *weights_layer_ptr(WeightFile *wf, int layer, const char *suffix) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    return weights_tensor_ptr(wf, name);
}

size_t weights_layer_offset(WeightFile *wf, int layer, const char *suffix) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    return weights_tensor_offset(wf, name);
}

// ============================================================================
// Expert layout computation
// ============================================================================

static ExpertLayout compute_expert_layout(int moe_intermediate, int hidden_dim,
                                          int group_size, QuantType quant) {
    ExpertLayout layout = {0};
    int bits = (quant == QUANT_2BIT) ? 2 : 4;
    int vals_per_u32 = 32 / bits;

    // gate_proj: [moe_intermediate, hidden_dim] quantized
    size_t gate_w_size = (size_t)moe_intermediate * hidden_dim / vals_per_u32 * sizeof(uint32_t);
    int gate_groups = hidden_dim / group_size;
    size_t gate_s_size = (size_t)moe_intermediate * gate_groups * sizeof(uint16_t);
    size_t gate_b_size = gate_s_size;

    // up_proj: same shape as gate
    size_t up_w_size = gate_w_size;
    size_t up_s_size = gate_s_size;
    size_t up_b_size = gate_b_size;

    // down_proj: [hidden_dim, moe_intermediate] quantized
    size_t down_w_size = (size_t)hidden_dim * moe_intermediate / vals_per_u32 * sizeof(uint32_t);
    int down_groups = moe_intermediate / group_size;
    size_t down_s_size = (size_t)hidden_dim * down_groups * sizeof(uint16_t);
    size_t down_b_size = down_s_size;

    size_t off = 0;
    layout.gate_w_off = off; off += gate_w_size;
    layout.gate_s_off = off; off += gate_s_size;
    layout.gate_b_off = off; off += gate_b_size;
    layout.up_w_off   = off; off += up_w_size;
    layout.up_s_off   = off; off += up_s_size;
    layout.up_b_off   = off; off += up_b_size;
    layout.down_w_off = off; off += down_w_size;
    layout.down_s_off = off; off += down_s_size;
    layout.down_b_off = off; off += down_b_size;
    layout.expert_size = off;

    return layout;
}

// ============================================================================
// Model config
// ============================================================================

void model_config_init_derived(ModelConfig *cfg) {
    // Attention layout
    cfg->num_full_attn_layers = cfg->num_layers / cfg->full_attn_interval;
    cfg->num_linear_layers = cfg->num_layers - cfg->num_full_attn_layers;
    cfg->rotary_dim = (int)(cfg->head_dim * cfg->partial_rotary);

    // Linear attention derived
    cfg->linear_total_key = cfg->linear_num_k_heads * cfg->linear_key_dim;
    cfg->linear_total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
    cfg->linear_conv_dim = cfg->linear_total_key * 2 + cfg->linear_total_value;

    // KV dim
    cfg->kv_dim = cfg->num_kv_heads * cfg->head_dim;

    // Layer type array — detect from full_attn_interval and full_attn_offset
    // Qwen3.5: full attn at layers 3,7,11,...  (offset=3, interval=4)
    // Pattern: layer % interval == offset
    if (cfg->layer_types) free(cfg->layer_types);
    cfg->layer_types = calloc(cfg->num_layers, sizeof(AttnLayerType));
    int full_count = 0, lin_count = 0;
    for (int i = 0; i < cfg->num_layers; i++) {
        cfg->layer_types[i] = (i % cfg->full_attn_interval == cfg->full_attn_offset)
            ? ATTN_FULL : ATTN_LINEAR;
        if (cfg->layer_types[i] == ATTN_FULL) full_count++;
        else lin_count++;
    }
    cfg->num_full_attn_layers = full_count;
    cfg->num_linear_layers = lin_count;

    // Expert layouts
    cfg->expert_4bit = compute_expert_layout(
        cfg->moe_intermediate, cfg->hidden_dim, cfg->group_size, QUANT_4BIT);
    cfg->expert_2bit = compute_expert_layout(
        cfg->moe_intermediate, cfg->hidden_dim, cfg->group_size, QUANT_2BIT);
}

void model_config_qwen35_35b(ModelConfig *cfg) {
    memset(cfg, 0, sizeof(ModelConfig));
    strncpy(cfg->name, "qwen3.5-35b-a3b", sizeof(cfg->name) - 1);

    cfg->hidden_dim = 2048;
    cfg->num_layers = 40;
    cfg->vocab_size = 248320;
    cfg->rms_norm_eps = 1e-6f;

    cfg->num_attn_heads = 16;
    cfg->num_kv_heads = 2;
    cfg->head_dim = 256;

    cfg->linear_num_v_heads = 32;
    cfg->linear_num_k_heads = 16;
    cfg->linear_key_dim = 128;
    cfg->linear_value_dim = 128;
    cfg->conv_kernel_size = 4;

    cfg->full_attn_interval = 4;
    cfg->full_attn_offset = 3;      // full attn at layers 3,7,11,...,39
    cfg->rope_theta = 10000000.0f;
    cfg->partial_rotary = 0.25f;

    cfg->num_experts = 256;
    cfg->num_experts_per_tok = 8;
    cfg->moe_intermediate = 512;
    cfg->shared_intermediate = 512;
    cfg->group_size = 64;

    cfg->eos_tokens[0] = 248046;
    cfg->eos_tokens[1] = 248044;
    cfg->eos_tokens[2] = -1;
    cfg->think_start_token = 248068;
    cfg->think_end_token = 248069;

    model_config_init_derived(cfg);
}

void model_config_qwen35_397b(ModelConfig *cfg) {
    memset(cfg, 0, sizeof(ModelConfig));
    strncpy(cfg->name, "qwen3.5-397b-a17b", sizeof(cfg->name) - 1);

    cfg->hidden_dim = 4096;
    cfg->num_layers = 60;
    cfg->vocab_size = 248320;
    cfg->rms_norm_eps = 1e-6f;

    cfg->num_attn_heads = 32;
    cfg->num_kv_heads = 2;
    cfg->head_dim = 256;

    cfg->linear_num_v_heads = 64;
    cfg->linear_num_k_heads = 16;
    cfg->linear_key_dim = 128;
    cfg->linear_value_dim = 128;
    cfg->conv_kernel_size = 4;

    // 397B layout: 15 × (3 × DeltaNet + 1 × Global Attention)
    cfg->full_attn_interval = 4;
    cfg->full_attn_offset = 3;      // same pattern as 35B
    cfg->rope_theta = 10000000.0f;
    cfg->partial_rotary = 0.25f;

    cfg->num_experts = 512;
    cfg->num_experts_per_tok = 10;  // 10 routed + 1 shared
    cfg->moe_intermediate = 1024;
    cfg->shared_intermediate = 1024;
    cfg->group_size = 64;

    cfg->eos_tokens[0] = 248046;
    cfg->eos_tokens[1] = 248044;
    cfg->eos_tokens[2] = -1;
    cfg->think_start_token = 248068;
    cfg->think_end_token = 248069;

    model_config_init_derived(cfg);
}

int model_config_load(ModelConfig *cfg, const char *model_dir) {
    @autoreleasepool {
        char path[512];
        snprintf(path, sizeof(path), "%s/config.json", model_dir);

        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:path]];
        if (!data) {
            fprintf(stderr, "[config] No config.json at %s, using hardcoded config\n", path);
            return -1;
        }

        NSError *error = nil;
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data
                                                              options:0
                                                                error:&error];
        if (!json) {
            fprintf(stderr, "[config] Failed to parse config.json: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        memset(cfg, 0, sizeof(ModelConfig));

        // Map HF config fields to our config
        if (json[@"hidden_size"])
            cfg->hidden_dim = [json[@"hidden_size"] intValue];
        if (json[@"num_hidden_layers"])
            cfg->num_layers = [json[@"num_hidden_layers"] intValue];
        if (json[@"vocab_size"])
            cfg->vocab_size = [json[@"vocab_size"] intValue];
        if (json[@"rms_norm_eps"])
            cfg->rms_norm_eps = [json[@"rms_norm_eps"] floatValue];
        if (json[@"num_attention_heads"])
            cfg->num_attn_heads = [json[@"num_attention_heads"] intValue];
        if (json[@"num_key_value_heads"])
            cfg->num_kv_heads = [json[@"num_key_value_heads"] intValue];
        if (json[@"head_dim"])
            cfg->head_dim = [json[@"head_dim"] intValue];
        if (json[@"num_experts"])
            cfg->num_experts = [json[@"num_experts"] intValue];
        if (json[@"num_experts_per_tok"])
            cfg->num_experts_per_tok = [json[@"num_experts_per_tok"] intValue];
        if (json[@"moe_intermediate_size"])
            cfg->moe_intermediate = [json[@"moe_intermediate_size"] intValue];
        if (json[@"shared_expert_intermediate_size"])
            cfg->shared_intermediate = [json[@"shared_expert_intermediate_size"] intValue];
        if (json[@"rope_theta"])
            cfg->rope_theta = [json[@"rope_theta"] floatValue];

        // Defaults for fields not always in config.json
        if (cfg->rms_norm_eps == 0) cfg->rms_norm_eps = 1e-6f;
        if (cfg->rope_theta == 0) cfg->rope_theta = 10000000.0f;
        cfg->partial_rotary = 0.25f;
        cfg->full_attn_interval = 4;
        cfg->full_attn_offset = 3;
        cfg->group_size = 64;
        cfg->conv_kernel_size = 4;

        // Linear attention defaults (Qwen3.5 family)
        if (cfg->linear_num_v_heads == 0) cfg->linear_num_v_heads = 32;
        if (cfg->linear_num_k_heads == 0) cfg->linear_num_k_heads = 16;
        if (cfg->linear_key_dim == 0) cfg->linear_key_dim = 128;
        if (cfg->linear_value_dim == 0) cfg->linear_value_dim = 128;

        // EOS tokens (Qwen family)
        cfg->eos_tokens[0] = 248046;
        cfg->eos_tokens[1] = 248044;
        cfg->eos_tokens[2] = -1;
        cfg->think_start_token = 248068;
        cfg->think_end_token = 248069;

        model_config_init_derived(cfg);

        printf("[config] Loaded: %s (%d layers, %d hidden, %d experts)\n",
               cfg->name[0] ? cfg->name : "unknown",
               cfg->num_layers, cfg->hidden_dim, cfg->num_experts);
        return 0;
    }
}

// ============================================================================
// KV cache & linear attention state
// ============================================================================

KVCache *kv_cache_new(const ModelConfig *cfg) {
    KVCache *kv = calloc(1, sizeof(KVCache));
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    kv->k_cache = calloc(OROME_GPU_KV_SEQ * kv_dim, sizeof(float));
    kv->v_cache = calloc(OROME_GPU_KV_SEQ * kv_dim, sizeof(float));
    kv->len = 0;
    return kv;
}

void kv_cache_free(KVCache *kv) {
    if (!kv) return;
    free(kv->k_cache);
    free(kv->v_cache);
    free(kv);
}

LinearAttnState *linear_state_new(const ModelConfig *cfg) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    int conv_size = (cfg->conv_kernel_size - 1) * cfg->linear_conv_dim;
    int ssm_size = cfg->linear_num_v_heads * cfg->linear_key_dim * cfg->linear_value_dim;
    s->conv_state = calloc(conv_size, sizeof(float));
    s->ssm_state = calloc(ssm_size, sizeof(float));
    return s;
}

void linear_state_free(LinearAttnState *s) {
    if (!s) return;
    free(s->conv_state);
    free(s->ssm_state);
    free(s);
}

// ============================================================================
// Layer type auto-detection from weight manifest
// ============================================================================

void model_config_detect_layers(ModelConfig *cfg, WeightFile *wf) {
    if (!wf || !wf->manifest) return;

    if (!cfg->layer_types) {
        cfg->layer_types = calloc(cfg->num_layers, sizeof(AttnLayerType));
    }

    int full_count = 0, linear_count = 0;
    for (int i = 0; i < cfg->num_layers; i++) {
        // Check for self_attn (full attention) signature tensor
        char probe[256];
        snprintf(probe, sizeof(probe), "model.layers.%d.self_attn.q_proj.weight", i);
        bool has_full = (find_tensor(wf->manifest, probe) != NULL);

        // Check for linear_attn (GatedDeltaNet) signature tensor
        snprintf(probe, sizeof(probe), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
        bool has_linear = (find_tensor(wf->manifest, probe) != NULL);

        if (has_full && !has_linear) {
            cfg->layer_types[i] = ATTN_FULL;
            full_count++;
        } else if (has_linear && !has_full) {
            cfg->layer_types[i] = ATTN_LINEAR;
            linear_count++;
        } else if (has_full && has_linear) {
            // Both present — unusual, default to full
            cfg->layer_types[i] = ATTN_FULL;
            full_count++;
        } else {
            // Neither found — keep whatever was set by init_derived
            if (cfg->layer_types[i] == ATTN_FULL) full_count++;
            else linear_count++;
        }
    }

    cfg->num_full_attn_layers = full_count;
    cfg->num_linear_layers = linear_count;

    printf("[config] Detected layer types from manifest: %d full, %d linear\n",
           full_count, linear_count);
}

bool is_eos_token(const ModelConfig *cfg, int token_id) {
    for (int i = 0; i < 4 && cfg->eos_tokens[i] >= 0; i++) {
        if (token_id == cfg->eos_tokens[i]) return true;
    }
    return false;
}
