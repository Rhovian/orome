/*
 * tokenizer.m — BPE tokenizer wrapper.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOKENIZER_IMPL
#include "tokenizer.h"
#include "orome.h"

// ============================================================================
// BPE tokenizer (global singleton)
// ============================================================================

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

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

static bool has_suffix(const char *path, const char *suffix) {
    if (!path || !suffix) return false;
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);
    return path_len >= suffix_len &&
           strcmp(path + path_len - suffix_len, suffix) == 0;
}

static int read_u64_gguf(FILE *f, uint64_t *v) {
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static int read_i32_gguf(FILE *f, int32_t *v) {
    return fread(v, sizeof(*v), 1, f) == 1 ? 0 : -1;
}

static int read_string_gguf(FILE *f, char **out) {
    uint64_t len = 0;
    if (read_u64_gguf(f, &len)) return -1;
    if (len > SIZE_MAX - 1 || len > UINT16_MAX) return -1;

    char *buf = malloc((size_t)len + 1);
    if (!buf) return -1;
    if (len > 0 && fread(buf, 1, (size_t)len, f) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *out = buf;
    return 0;
}

static int skip_bytes_gguf(FILE *f, uint64_t len) {
    if (len == 0) return 0;
    return fseeko(f, (off_t)len, SEEK_CUR) == 0 ? 0 : -1;
}

static int skip_value_gguf(FILE *f, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return skip_bytes_gguf(f, 1);
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return skip_bytes_gguf(f, 2);
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return skip_bytes_gguf(f, 4);
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return skip_bytes_gguf(f, 8);
        case GGUF_TYPE_STRING: {
            char *tmp = NULL;
            int rc = read_string_gguf(f, &tmp);
            free(tmp);
            return rc;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (read_u32(f, &elem_type) || read_u64_gguf(f, &count)) return -1;
            for (uint64_t i = 0; i < count; i++) {
                if (skip_value_gguf(f, elem_type)) return -1;
            }
            return 0;
        }
        default:
            return -1;
    }
}

static int read_string_array_gguf(FILE *f, char ***items_out, uint32_t *count_out) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    char **items = NULL;

    if (read_u32(f, &elem_type) || read_u64_gguf(f, &count)) return -1;
    if (elem_type != GGUF_TYPE_STRING || count > UINT32_MAX) return -1;

    items = calloc((size_t)count, sizeof(char *));
    if (!items && count > 0) return -1;
    for (uint64_t i = 0; i < count; i++) {
        if (read_string_gguf(f, &items[i])) {
            for (uint64_t j = 0; j < i; j++) free(items[j]);
            free(items);
            return -1;
        }
    }

    *items_out = items;
    *count_out = (uint32_t)count;
    return 0;
}

static int read_i32_array_gguf(FILE *f, int32_t **items_out, uint32_t *count_out) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    int32_t *items = NULL;

    if (read_u32(f, &elem_type) || read_u64_gguf(f, &count)) return -1;
    if (elem_type != GGUF_TYPE_INT32 || count > UINT32_MAX) return -1;

    items = calloc((size_t)count, sizeof(int32_t));
    if (!items && count > 0) return -1;
    for (uint64_t i = 0; i < count; i++) {
        if (read_i32_gguf(f, &items[i])) {
            free(items);
            return -1;
        }
    }

    *items_out = items;
    *count_out = (uint32_t)count;
    return 0;
}

static void free_string_array(char **items, uint32_t count) {
    if (!items) return;
    for (uint32_t i = 0; i < count; i++) free(items[i]);
    free(items);
}

static int bpe_build_tables(bpe_tokenizer *tok) {
    uint32_t ht_size = next_pow2((tok->vocab_size > 0 ? tok->vocab_size : 1) * 2);
    tok->ht_mask = ht_size - 1;
    tok->ht_ids = malloc(ht_size * sizeof(uint32_t));
    tok->ht_keys = calloc(ht_size, sizeof(char *));
    tok->ht_klens = calloc(ht_size, sizeof(uint16_t));
    if (!tok->ht_ids || !tok->ht_keys || !tok->ht_klens) return -1;
    memset(tok->ht_ids, 0xFF, ht_size * sizeof(uint32_t));
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        ht_insert(tok->ht_ids, tok->ht_keys, tok->ht_klens, tok->ht_mask,
                  tok->vocab[i].str, tok->vocab[i].len, tok->vocab[i].id);
    }

    uint32_t mt_size = next_pow2((tok->num_merges > 0 ? tok->num_merges : 1) * 2);
    tok->mt_mask = mt_size - 1;
    tok->mt_prio = malloc(mt_size * sizeof(uint32_t));
    tok->mt_keys = calloc(mt_size, sizeof(char *));
    tok->mt_klens = calloc(mt_size, sizeof(uint16_t));
    if (!tok->mt_prio || !tok->mt_keys || !tok->mt_klens) return -1;
    memset(tok->mt_prio, 0xFF, mt_size * sizeof(uint32_t));
    for (uint32_t i = 0; i < tok->num_merges; i++) {
        uint16_t klen = tok->merges[i].len_a + 1 + tok->merges[i].len_b;
        char *key = malloc(klen);
        if (!key) return -1;
        memcpy(key, tok->merges[i].a, tok->merges[i].len_a);
        key[tok->merges[i].len_a] = '\xff';
        memcpy(key + tok->merges[i].len_a + 1, tok->merges[i].b, tok->merges[i].len_b);
        ht_insert(tok->mt_prio, tok->mt_keys, tok->mt_klens, tok->mt_mask,
                  key, klen, i);
    }

    return 0;
}

static int bpe_load_gguf(bpe_tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    memset(tok, 0, sizeof(*tok));
    build_byte_unicode_table(tok);

    char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    char **tokens = NULL;
    char **merges = NULL;
    int32_t *token_types = NULL;
    uint32_t token_count = 0;
    uint32_t merge_count = 0;
    uint32_t token_type_count = 0;
    int rc = -1;

    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, "GGUF", sizeof(magic)) != 0 ||
        read_u32(f, &version) ||
        (version != 2 && version != 3) ||
        read_u64_gguf(f, &tensor_count) ||
        read_u64_gguf(f, &metadata_count)) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < metadata_count; i++) {
        char *key = NULL;
        uint32_t value_type = 0;
        if (read_string_gguf(f, &key) || read_u32(f, &value_type)) {
            free(key);
            goto cleanup;
        }

        if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            free_string_array(tokens, token_count);
            tokens = NULL;
            token_count = 0;
            if (read_string_array_gguf(f, &tokens, &token_count)) {
                free(key);
                goto cleanup;
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0) {
            free_string_array(merges, merge_count);
            merges = NULL;
            merge_count = 0;
            if (read_string_array_gguf(f, &merges, &merge_count)) {
                free(key);
                goto cleanup;
            }
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0) {
            free(token_types);
            token_types = NULL;
            token_type_count = 0;
            if (read_i32_array_gguf(f, &token_types, &token_type_count)) {
                free(key);
                goto cleanup;
            }
        } else if (skip_value_gguf(f, value_type)) {
            free(key);
            goto cleanup;
        }

        free(key);
    }

    if (!tokens || !merges || !token_types ||
        token_count == 0 || token_count != token_type_count) {
        goto cleanup;
    }

    tok->vocab_size = token_count;
    tok->vocab = calloc(tok->vocab_size, sizeof(bpe_vocab_entry));
    if (!tok->vocab) goto cleanup;
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        size_t len = strlen(tokens[i]);
        if (len > UINT16_MAX) goto cleanup;
        tok->vocab[i].id = i;
        tok->vocab[i].len = (uint16_t)len;
        tok->vocab[i].str = tokens[i];
        tokens[i] = NULL;
    }
    free_string_array(tokens, token_count);
    tokens = NULL;

    tok->num_merges = merge_count;
    tok->merges = calloc(tok->num_merges, sizeof(bpe_merge));
    if (!tok->merges && tok->num_merges > 0) goto cleanup;
    for (uint32_t i = 0; i < tok->num_merges; i++) {
        char *sep = strchr(merges[i], ' ');
        size_t len_a = sep ? (size_t)(sep - merges[i]) : strlen(merges[i]);
        const char *rhs = sep ? sep + 1 : "";
        size_t len_b = strlen(rhs);
        if (!sep || len_a > UINT16_MAX || len_b > UINT16_MAX) goto cleanup;

        *sep = '\0';
        tok->merges[i].a = merges[i];
        tok->merges[i].len_a = (uint16_t)len_a;
        tok->merges[i].b = strdup(rhs);
        tok->merges[i].len_b = (uint16_t)len_b;
        merges[i] = NULL;
        if (!tok->merges[i].b) goto cleanup;
    }
    free_string_array(merges, merge_count);
    merges = NULL;

    for (uint32_t i = 0; i < token_type_count; i++) {
        if (token_types[i] != 1) tok->num_added++;
    }
    tok->added = calloc(tok->num_added, sizeof(bpe_added_token));
    if (!tok->added && tok->num_added > 0) goto cleanup;
    for (uint32_t i = 0, j = 0; i < token_type_count; i++) {
        if (token_types[i] == 1) continue;
        tok->added[j].id = i;
        tok->added[j].len = tok->vocab[i].len;
        tok->added[j].str = strdup(tok->vocab[i].str);
        if (!tok->added[j].str) goto cleanup;
        j++;
    }

    if (bpe_build_tables(tok)) goto cleanup;

    fprintf(stderr, "bpe_load_gguf: %u vocab, %u merges, %u added tokens\n",
            tok->vocab_size, tok->num_merges, tok->num_added);
    rc = 0;

cleanup:
    free(token_types);
    free_string_array(tokens, token_count);
    free_string_array(merges, merge_count);
    fclose(f);
    if (rc != 0) {
        bpe_free(tok);
    }
    return rc;
}

int tokenizer_init(const char *model_dir) {
    if (g_tokenizer_loaded) return 0;

    struct stat st;
    if (model_dir &&
        has_suffix(model_dir, ".gguf") &&
        stat(model_dir, &st) == 0 &&
        S_ISREG(st.st_mode) &&
        bpe_load_gguf(&g_tokenizer, model_dir) == 0) {
        printf("[tokenizer] Loaded from GGUF metadata: %s\n", model_dir);
        g_tokenizer_loaded = 1;
        return 0;
    }

    const char *search_paths[] = { NULL, "vocab.bin", NULL };
    char base_dir[512];
    char vocab_path[512];
    if (model_dir) {
        const char *dir = model_dir;
        if (stat(model_dir, &st) == 0 && S_ISREG(st.st_mode)) {
            strncpy(base_dir, model_dir, sizeof(base_dir));
            base_dir[sizeof(base_dir) - 1] = '\0';
            char *slash = strrchr(base_dir, '/');
            if (slash) {
                *slash = '\0';
                dir = base_dir;
            } else {
                dir = ".";
            }
        }

        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", dir);
        search_paths[0] = vocab_path;
    }

    for (int i = 0; search_paths[i]; i++) {
        if (access(search_paths[i], R_OK) == 0) {
            if (bpe_load(&g_tokenizer, search_paths[i]) == 0) {
                printf("[tokenizer] Loaded from %s\n", search_paths[i]);
                g_tokenizer_loaded = 1;
                return 0;
            }
        }
    }

    fprintf(stderr, "ERROR: Could not load tokenizer from any search path\n");
    fprintf(stderr, "Hint: build vocab.bin from GGUF or Hugging Face tokenizer assets with:\n");
    fprintf(stderr, "  python3 tools/build_vocab_bin.py /path/to/model.gguf\n");
    fprintf(stderr, "  python3 tools/build_vocab_bin.py /path/to/tokenizer_dir\n");
    return -1;
}

int tokenizer_find_token(const char *text) {
    if (!g_tokenizer_loaded || !text) return -1;

    size_t len = strlen(text);
    for (uint32_t i = 0; i < g_tokenizer.vocab_size; i++) {
        if (g_tokenizer.vocab[i].len == len &&
            memcmp(g_tokenizer.vocab[i].str, text, len) == 0) {
            return (int)g_tokenizer.vocab[i].id;
        }
    }
    for (uint32_t i = 0; i < g_tokenizer.num_added; i++) {
        if (g_tokenizer.added[i].len == len &&
            memcmp(g_tokenizer.added[i].str, text, len) == 0) {
            return (int)g_tokenizer.added[i].id;
        }
    }
    return -1;
}

int tokenizer_find_added_tokens_in_text(const char *text, int *ids, int max_ids) {
    if (!g_tokenizer_loaded || !text || !ids || max_ids <= 0) return 0;

    int count = 0;
    const char *p = text;
    while (*p) {
        int best_match = -1;
        uint16_t best_len = 0;
        for (uint32_t i = 0; i < g_tokenizer.num_added; i++) {
            uint16_t len = g_tokenizer.added[i].len;
            if (len == 0 || len < best_len) continue;
            if (strncmp(p, g_tokenizer.added[i].str, len) != 0) continue;
            best_match = (int)i;
            best_len = len;
        }

        if (best_match >= 0) {
            ids[count++] = (int)g_tokenizer.added[best_match].id;
            if (count >= max_ids) break;
            p += best_len;
            continue;
        }
        p++;
    }

    return count;
}

PromptTokens *tokenizer_encode(const char *text) {
    if (!g_tokenizer_loaded) {
        fprintf(stderr, "ERROR: Tokenizer not initialized\n");
        return NULL;
    }

    // bpe_encode takes a fixed output buffer
    uint32_t id_buf[16384];
    int count = bpe_encode(&g_tokenizer, text, id_buf, 16384);
    if (count <= 0) {
        fprintf(stderr, "ERROR: Tokenizer encode failed\n");
        return NULL;
    }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = calloc(count, sizeof(int));
    pt->count = count;
    for (int i = 0; i < count; i++) {
        pt->ids[i] = (int)id_buf[i];
    }
    return pt;
}

const char *tokenizer_decode(int token_id) {
    if (!g_tokenizer_loaded) return "<no_tokenizer>";

    static char ring[8][BPE_MAX_TOKEN_LEN * 4];
    static int ring_idx = 0;
    char *buf = ring[ring_idx++ & 7];

    if (token_id < 0 || (uint32_t)token_id >= g_tokenizer.vocab_size ||
        !g_tokenizer.vocab[token_id].str) {
        snprintf(buf, sizeof(ring[0]), "<unk_%d>", token_id);
        return buf;
    }

    // GPT-2 BPE stores tokens as unicode codepoints that map to bytes.
    // We need to decode: BPE string → UTF-8 codepoints → char_byte → raw bytes.
    const char *src = g_tokenizer.vocab[token_id].str;
    int src_len = g_tokenizer.vocab[token_id].len;
    int out = 0;

    for (int i = 0; i < src_len && out < (int)sizeof(ring[0]) - 4;) {
        unsigned char c0 = (unsigned char)src[i];
        uint32_t cp = 0;
        int consumed = 1;

        // Decode UTF-8 codepoint
        if (c0 < 0x80) {
            cp = c0;
        } else if ((c0 & 0xE0) == 0xC0 && i + 1 < src_len) {
            cp = ((uint32_t)(c0 & 0x1F) << 6) |
                 (uint32_t)(src[i + 1] & 0x3F);
            consumed = 2;
        } else if ((c0 & 0xF0) == 0xE0 && i + 2 < src_len) {
            cp = ((uint32_t)(c0 & 0x0F) << 12) |
                 ((uint32_t)(src[i + 1] & 0x3F) << 6) |
                 (uint32_t)(src[i + 2] & 0x3F);
            consumed = 3;
        } else if ((c0 & 0xF8) == 0xF0 && i + 3 < src_len) {
            cp = ((uint32_t)(c0 & 0x07) << 18) |
                 ((uint32_t)(src[i + 1] & 0x3F) << 12) |
                 ((uint32_t)(src[i + 2] & 0x3F) << 6) |
                 (uint32_t)(src[i + 3] & 0x3F);
            consumed = 4;
        } else {
            cp = c0;
        }
        i += consumed;

        // Map through GPT-2 char→byte table
        if (cp < 512 && (g_tokenizer.char_byte[cp] != 0 || cp == 256)) {
            buf[out++] = (char)g_tokenizer.char_byte[cp];
            continue;
        }

        // Pass through as UTF-8
        if (cp < 0x80) {
            buf[out++] = (char)cp;
        } else if (cp < 0x800) {
            buf[out++] = (char)(0xC0 | (cp >> 6));
            buf[out++] = (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            buf[out++] = (char)(0xE0 | (cp >> 12));
            buf[out++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[out++] = (char)(0x80 | (cp & 0x3F));
        } else {
            buf[out++] = (char)(0xF0 | (cp >> 18));
            buf[out++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            buf[out++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[out++] = (char)(0x80 | (cp & 0x3F));
        }
    }

    buf[out] = '\0';
    return buf;
}

void prompt_tokens_free(PromptTokens *pt) {
    if (!pt) return;
    free(pt->ids);
    free(pt);
}
