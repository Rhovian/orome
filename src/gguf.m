// GGUF file format parser for orome
// Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

#import <Foundation/Foundation.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import "orome.h"

// --- GGUF constants ---

#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian

// Metadata value types
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGML tensor types (quantization formats)
enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
};

// Block sizes and byte sizes per quantization type
static size_t ggml_type_block_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return 1;
        case GGML_TYPE_F16:     return 1;
        case GGML_TYPE_BF16:    return 1;
        case GGML_TYPE_Q4_0:    return 32;
        case GGML_TYPE_Q4_1:    return 32;
        case GGML_TYPE_Q5_0:    return 32;
        case GGML_TYPE_Q5_1:    return 32;
        case GGML_TYPE_Q8_0:    return 32;
        case GGML_TYPE_Q8_1:    return 32;
        case GGML_TYPE_Q2_K:    return 256;
        case GGML_TYPE_Q3_K:    return 256;
        case GGML_TYPE_Q4_K:    return 256;
        case GGML_TYPE_Q5_K:    return 256;
        case GGML_TYPE_Q6_K:    return 256;
        case GGML_TYPE_Q8_K:    return 256;
        default:                return 1;
    }
}

static size_t ggml_type_bytes_per_block(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_Q4_0:    return 18;    // 2 (scale) + 16 (32 nibbles)
        case GGML_TYPE_Q4_1:    return 20;    // 2+2 (scale+min) + 16
        case GGML_TYPE_Q5_0:    return 22;    // 2 + 4 (high bits) + 16
        case GGML_TYPE_Q5_1:    return 24;    // 2+2 + 4 + 16
        case GGML_TYPE_Q8_0:    return 34;    // 2 (scale) + 32 (bytes)
        case GGML_TYPE_Q8_1:    return 36;    // 2+2 + 32
        case GGML_TYPE_Q2_K:    return 84;    // 256 weights
        case GGML_TYPE_Q3_K:    return 110;   // 256 weights
        case GGML_TYPE_Q4_K:    return 144;   // 256 weights
        case GGML_TYPE_Q5_K:    return 176;   // 256 weights
        case GGML_TYPE_Q6_K:    return 210;   // 256 weights
        case GGML_TYPE_Q8_K:    return 292;   // 256 weights
        default:                return 0;
    }
}

// --- Binary reader helpers ---

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} GGUFReader;

static bool reader_has(GGUFReader *r, size_t n) {
    return r->pos + n <= r->size;
}

static uint8_t read_u8(GGUFReader *r) {
    if (!reader_has(r, 1)) return 0;
    return r->data[r->pos++];
}

static uint16_t read_u16(GGUFReader *r) {
    if (!reader_has(r, 2)) return 0;
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(GGUFReader *r) {
    if (!reader_has(r, 4)) return 0;
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(GGUFReader *r) {
    if (!reader_has(r, 8)) return 0;
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static int32_t read_i32(GGUFReader *r) {
    return (int32_t)read_u32(r);
}

static float read_f32(GGUFReader *r) {
    if (!reader_has(r, 4)) return 0;
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

// GGUF strings: uint64_t length + bytes (NOT null-terminated in file)
static char *read_string(GGUFReader *r) {
    uint64_t len = read_u64(r);
    if (!reader_has(r, len) || len > 1024*1024) return NULL;
    char *s = malloc(len + 1);
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

// Skip a metadata value (used when we don't need it)
static void skip_value(GGUFReader *r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    r->pos += 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   r->pos += 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: r->pos += 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            char *s = read_string(r);
            free(s);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = read_u32(r);
            uint64_t count = read_u64(r);
            for (uint64_t i = 0; i < count; i++) {
                skip_value(r, elem_type);
            }
            break;
        }
    }
}

// Read a metadata value as uint64 (coercing numeric types)
static uint64_t read_value_uint(GGUFReader *r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return read_u8(r);
        case GGUF_TYPE_INT8:    return (uint64_t)(int8_t)read_u8(r);
        case GGUF_TYPE_UINT16:  return read_u16(r);
        case GGUF_TYPE_INT16:   return (uint64_t)(int16_t)read_u16(r);
        case GGUF_TYPE_UINT32:  return read_u32(r);
        case GGUF_TYPE_INT32:   return (uint64_t)read_i32(r);
        case GGUF_TYPE_UINT64:  return read_u64(r);
        case GGUF_TYPE_INT64:   return read_u64(r);
        case GGUF_TYPE_BOOL:    return read_u8(r);
        default:                skip_value(r, type); return 0;
    }
}

static float read_value_float(GGUFReader *r, uint32_t type) {
    if (type == GGUF_TYPE_FLOAT32) return read_f32(r);
    if (type == GGUF_TYPE_FLOAT64) { r->pos += 8; return 0; } // skip double
    return (float)read_value_uint(r, type);
}

// --- Tensor hash table ---

#define GGUF_HT_SIZE 65536

typedef struct {
    const char *name;
    uint64_t index;
    bool occupied;
} GGUFHTEntry;

static uint32_t gguf_hash(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static GGUFHTEntry *s_gguf_ht = NULL;

static void gguf_ht_insert(const char *name, uint64_t index) {
    if (!s_gguf_ht) {
        s_gguf_ht = calloc(GGUF_HT_SIZE, sizeof(GGUFHTEntry));
    }
    uint32_t slot = gguf_hash(name) & (GGUF_HT_SIZE - 1);
    for (uint32_t i = 0; i < GGUF_HT_SIZE; i++) {
        uint32_t idx = (slot + i) & (GGUF_HT_SIZE - 1);
        if (!s_gguf_ht[idx].occupied) {
            s_gguf_ht[idx].name = name;
            s_gguf_ht[idx].index = index;
            s_gguf_ht[idx].occupied = true;
            return;
        }
    }
    fprintf(stderr, "[gguf] hash table full\n");
}

static int64_t gguf_ht_find(const char *name) {
    if (!s_gguf_ht) return -1;
    uint32_t slot = gguf_hash(name) & (GGUF_HT_SIZE - 1);
    for (uint32_t i = 0; i < GGUF_HT_SIZE; i++) {
        uint32_t idx = (slot + i) & (GGUF_HT_SIZE - 1);
        if (!s_gguf_ht[idx].occupied) return -1;
        if (strcmp(s_gguf_ht[idx].name, name) == 0) return (int64_t)s_gguf_ht[idx].index;
    }
    return -1;
}

// --- Public API ---

GGUFFile *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[gguf] cannot open: %s\n", path);
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[gguf] mmap failed: %s\n", path);
        close(fd);
        return NULL;
    }

    GGUFReader r = { .data = mapped, .size = file_size, .pos = 0 };

    // Read header
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[gguf] bad magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        munmap(mapped, file_size);
        close(fd);
        return NULL;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "[gguf] unsupported version: %u\n", version);
        munmap(mapped, file_size);
        close(fd);
        return NULL;
    }

    uint64_t tensor_count = read_u64(&r);
    uint64_t metadata_kv_count = read_u64(&r);

    fprintf(stderr, "[gguf] version=%u tensors=%llu metadata=%llu\n",
            version, tensor_count, metadata_kv_count);

    GGUFFile *gf = calloc(1, sizeof(GGUFFile));
    gf->fd = fd;
    gf->mmap_base = mapped;
    gf->file_size = file_size;
    gf->num_tensors = tensor_count;
    gf->alignment = 32; // default
    gf->tensors = calloc(tensor_count, sizeof(GGUFTensorInfo));

    // Parse metadata KVs
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        char *key = read_string(&r);
        uint32_t vtype = read_u32(&r);

        if (!key) {
            fprintf(stderr, "[gguf] failed to read metadata key %llu\n", i);
            break;
        }

        // Extract fields we care about
        if (strcmp(key, "general.architecture") == 0) {
            char *val = read_string(&r);
            if (val) {
                snprintf(gf->arch, sizeof(gf->arch), "%s", val);
                free(val);
            }
        } else if (strcmp(key, "general.alignment") == 0) {
            gf->alignment = (uint32_t)read_value_uint(&r, vtype);
        } else {
            // Check for architecture-specific keys
            // Build expected prefixes dynamically
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "%s.", gf->arch[0] ? gf->arch : "llama");
            size_t plen = strlen(prefix);

            if (strncmp(key, prefix, plen) == 0) {
                const char *field = key + plen;
                if (strcmp(field, "block_count") == 0) {
                    gf->num_layers = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "embedding_length") == 0) {
                    gf->hidden_dim = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "expert_count") == 0) {
                    gf->num_experts = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "expert_used_count") == 0) {
                    gf->num_experts_per_tok = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "feed_forward_length") == 0) {
                    gf->moe_intermediate = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "attention.head_count") == 0) {
                    gf->num_attn_heads = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "attention.head_count_kv") == 0) {
                    gf->num_kv_heads = (int)read_value_uint(&r, vtype);
                } else if (strcmp(field, "rope.freq_base") == 0) {
                    gf->rope_theta = read_value_float(&r, vtype);
                } else if (strcmp(field, "vocab_size") == 0) {
                    gf->vocab_size = (int)read_value_uint(&r, vtype);
                } else {
                    skip_value(&r, vtype);
                }
            } else if (strncmp(key, "tokenizer.", 10) == 0) {
                // Tokenizer metadata — skip for now
                skip_value(&r, vtype);
            } else if (strncmp(key, "general.", 8) == 0) {
                skip_value(&r, vtype);
            } else {
                skip_value(&r, vtype);
            }
        }
        free(key);
    }

    // Parse tensor info
    // Reset hash table for new file
    if (s_gguf_ht) { free(s_gguf_ht); s_gguf_ht = NULL; }

    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo *ti = &gf->tensors[i];
        ti->name = read_string(&r);
        ti->n_dims = read_u32(&r);
        for (uint32_t d = 0; d < ti->n_dims && d < 4; d++) {
            ti->dims[d] = read_u64(&r);
        }
        // Skip extra dims if n_dims > 4 (shouldn't happen)
        for (uint32_t d = 4; d < ti->n_dims; d++) {
            read_u64(&r);
        }
        ti->type = read_u32(&r);
        ti->offset = read_u64(&r);  // relative to data section start

        if (ti->name) {
            gguf_ht_insert(ti->name, i);
        }
    }

    // Infer vocab_size from embedding tensor shape if not in metadata
    if (gf->vocab_size == 0) {
        for (uint64_t i = 0; i < tensor_count; i++) {
            if (gf->tensors[i].name && strcmp(gf->tensors[i].name, "token_embd.weight") == 0) {
                // Shape is [hidden_dim, vocab_size]
                for (uint32_t d = 0; d < gf->tensors[i].n_dims; d++) {
                    if (gf->tensors[i].dims[d] != (uint64_t)gf->hidden_dim) {
                        gf->vocab_size = (int)gf->tensors[i].dims[d];
                        break;
                    }
                }
                break;
            }
        }
    }

    // Infer moe_intermediate from tensor shapes if not in metadata
    if (gf->moe_intermediate == 0) {
        for (uint64_t i = 0; i < tensor_count; i++) {
            if (gf->tensors[i].name && strstr(gf->tensors[i].name, "ffn_gate_exps")) {
                for (uint32_t d = 0; d < gf->tensors[i].n_dims; d++) {
                    uint64_t dim = gf->tensors[i].dims[d];
                    if (dim != (uint64_t)gf->hidden_dim && dim != (uint64_t)gf->num_experts) {
                        gf->moe_intermediate = (int)dim;
                        break;
                    }
                }
                break;
            }
        }
    }

    fprintf(stderr, "[gguf] arch=%s layers=%d hidden=%d experts=%d K=%d intermediate=%d vocab=%d\n",
            gf->arch, gf->num_layers, gf->hidden_dim,
            gf->num_experts, gf->num_experts_per_tok, gf->moe_intermediate, gf->vocab_size);

    // Compute data_offset: current position, aligned up to alignment
    size_t align = gf->alignment;
    gf->data_offset = (r.pos + align - 1) & ~(align - 1);

    fprintf(stderr, "[gguf] data_offset=%zu (%.1f MB into file)\n",
            gf->data_offset, gf->data_offset / (1024.0 * 1024.0));

    return gf;
}

void gguf_close(GGUFFile *gf) {
    if (!gf) return;
    if (gf->mmap_base) munmap(gf->mmap_base, gf->file_size);
    if (gf->fd >= 0) close(gf->fd);
    for (uint64_t i = 0; i < gf->num_tensors; i++) {
        free(gf->tensors[i].name);
    }
    free(gf->tensors);
    if (s_gguf_ht) { free(s_gguf_ht); s_gguf_ht = NULL; }
    free(gf);
}

GGUFTensorInfo *gguf_find_tensor(GGUFFile *gf, const char *name) {
    int64_t idx = gguf_ht_find(name);
    if (idx < 0) return NULL;
    return &gf->tensors[idx];
}


size_t gguf_tensor_size(GGUFTensorInfo *ti) {
    if (!ti) return 0;
    uint64_t num_elements = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) {
        num_elements *= ti->dims[d];
    }
    size_t block_size = ggml_type_block_size(ti->type);
    size_t bytes_per_block = ggml_type_bytes_per_block(ti->type);
    if (block_size == 0) return 0;
    return (num_elements / block_size) * bytes_per_block;
}

const char *ggml_type_name(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        default:             return "unknown";
    }
}

// Debug: print summary of tensor types
void gguf_print_summary(GGUFFile *gf) {
    if (!gf) return;

    // Count tensors by type
    int type_counts[64] = {0};
    size_t type_bytes[64] = {0};
    int expert_tensors = 0;
    int attn_tensors = 0;
    int other_tensors = 0;

    for (uint64_t i = 0; i < gf->num_tensors; i++) {
        GGUFTensorInfo *ti = &gf->tensors[i];
        if (ti->type < 64) {
            type_counts[ti->type]++;
            type_bytes[ti->type] += gguf_tensor_size(ti);
        }
        if (ti->name) {
            if (strstr(ti->name, "ffn_gate_exps") || strstr(ti->name, "ffn_up_exps") ||
                strstr(ti->name, "ffn_down_exps") || strstr(ti->name, "ffn_gate_shexp") ||
                strstr(ti->name, "ffn_up_shexp") || strstr(ti->name, "ffn_down_shexp")) {
                expert_tensors++;
            } else if (strstr(ti->name, "attn")) {
                attn_tensors++;
            } else {
                other_tensors++;
            }
        }
    }

    fprintf(stderr, "[gguf] Tensor summary:\n");
    for (int t = 0; t < 64; t++) {
        if (type_counts[t] > 0) {
            fprintf(stderr, "[gguf]   %s: %d tensors, %.1f MB\n",
                    ggml_type_name(t), type_counts[t],
                    type_bytes[t] / (1024.0 * 1024.0));
        }
    }
    fprintf(stderr, "[gguf]   expert=%d attn=%d other=%d\n",
            expert_tensors, attn_tensors, other_tensors);

    // Print first few tensor names for debugging
    fprintf(stderr, "[gguf] First 20 tensors:\n");
    for (uint64_t i = 0; i < gf->num_tensors && i < 20; i++) {
        GGUFTensorInfo *ti = &gf->tensors[i];
        fprintf(stderr, "[gguf]   [%llu] %s  type=%s  dims=[", i, ti->name, ggml_type_name(ti->type));
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            fprintf(stderr, "%llu%s", ti->dims[d], d + 1 < ti->n_dims ? "," : "");
        }
        fprintf(stderr, "]  size=%.2f MB  offset=%llu\n",
                gguf_tensor_size(ti) / (1024.0 * 1024.0), ti->offset);
    }
}

// ============================================================================
// Model config utilities (moved from weights.m)
// ============================================================================

void model_config_init_derived(ModelConfig *cfg) {
    cfg->rotary_dim = (int)(cfg->head_dim * cfg->partial_rotary);
    cfg->linear_total_key = cfg->linear_num_k_heads * cfg->linear_key_dim;
    cfg->linear_total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
    cfg->linear_conv_dim = cfg->linear_total_key * 2 + cfg->linear_total_value;
    cfg->kv_dim = cfg->num_kv_heads * cfg->head_dim;

    // Layer type array from full_attn_interval and full_attn_offset
    if (cfg->full_attn_interval > 0) {
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
    }
}

bool is_eos_token(const ModelConfig *cfg, int token_id) {
    for (int i = 0; i < 4 && cfg->eos_tokens[i] >= 0; i++) {
        if (cfg->eos_tokens[i] == token_id) return true;
    }
    return false;
}
