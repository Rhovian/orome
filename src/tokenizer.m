/*
 * tokenizer.m — Vocabulary loading and BPE tokenizer wrapper.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TOKENIZER_IMPL
#include "tokenizer.h"
#include "orome.h"

// ============================================================================
// BPE tokenizer (global singleton)
// ============================================================================

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

int tokenizer_init(const char *model_dir) {
    if (g_tokenizer_loaded) return 0;

    const char *search_paths[] = {
        "vocab.bin",
        NULL, // filled from model_dir below
        NULL
    };
    char dir_path[512];
    if (model_dir) {
        snprintf(dir_path, sizeof(dir_path), "%s/vocab.bin", model_dir);
        search_paths[1] = dir_path;
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
    return -1;
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

// ============================================================================
// Vocabulary (binary format: count, max_id, then per-token: len + bytes)
// ============================================================================

Vocabulary *vocab_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "WARNING: Cannot open vocab %s\n", path);
        return NULL;
    }

    uint32_t count, max_id;
    if (fread(&count, 4, 1, f) != 1 || fread(&max_id, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = (int)count;
    v->tokens = calloc(count, sizeof(char *));
    v->lengths = calloc(count, sizeof(int));

    for (uint32_t i = 0; i < count; i++) {
        uint16_t len;
        if (fread(&len, 2, 1, f) != 1) break;
        v->tokens[i] = calloc(len + 1, 1);
        v->lengths[i] = len;
        if (len > 0 && fread(v->tokens[i], 1, len, f) != len) break;
    }

    fclose(f);
    printf("[vocab] Loaded %d tokens from %s\n", v->num_tokens, path);
    return v;
}

void vocab_free(Vocabulary *v) {
    if (!v) return;
    for (int i = 0; i < v->num_tokens; i++) free(v->tokens[i]);
    free(v->tokens);
    free(v->lengths);
    free(v);
}
