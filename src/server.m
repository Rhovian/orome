/*
 * server.m — HTTP/SSE server (OpenAI-compatible API).
 *
 * Implements:
 *   POST /v1/chat/completions — streaming inference with multi-turn chat
 *   GET  /v1/models           — model list
 *   GET  /health              — health check
 *
 * Chat completions parses the full OpenAI messages array and formats it
 * with model-resolved chat markers when available.
 * Each request resets the engine and re-prefills the entire conversation.
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>

#include "orome.h"

// ============================================================================
// HTTP helpers
// ============================================================================

static ssize_t http_write(int fd, const void *buf, size_t len) {
    return write(fd, buf, len);
}

static void http_write_str(int fd, const char *str) {
    http_write(fd, str, strlen(str));
}

static char *read_http_request(int fd, size_t *out_len) {
    size_t cap = 65536, len = 0;
    char *buf = malloc(cap);
    while (len < cap - 1) {
        ssize_t n = read(fd, buf + len, cap - len - 1);
        if (n <= 0) break;
        len += n;
        buf[len] = '\0';
        // Check for end of headers
        if (strstr(buf, "\r\n\r\n")) {
            // Check Content-Length for body
            char *cl = strstr(buf, "Content-Length:");
            if (cl) {
                int body_len = atoi(cl + 15);
                char *body_start = strstr(buf, "\r\n\r\n") + 4;
                int header_len = (int)(body_start - buf);
                size_t total = (size_t)header_len + body_len;
                if (total >= cap) {
                    cap = total + 1;
                    buf = realloc(buf, cap);
                }
                while (len < total && len < cap - 1) {
                    ssize_t n2 = read(fd, buf + len, cap - len - 1);
                    if (n2 <= 0) break;
                    len += n2;
                    buf[len] = '\0';
                }
            }
            break;
        }
    }
    *out_len = len;
    return buf;
}

// ============================================================================
// SSE helpers
// ============================================================================

static void sse_send_delta(int fd, const char *token_text, const char *req_id) {
    char chunk[1024];
    // JSON-escape the token text
    char escaped[512];
    int j = 0;
    for (int i = 0; token_text[i] && j < 500; i++) {
        if (token_text[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
        else if (token_text[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
        else if (token_text[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else if (token_text[i] == '\t') { escaped[j++] = '\\'; escaped[j++] = 't'; }
        else escaped[j++] = token_text[i];
    }
    escaped[j] = '\0';

    snprintf(chunk, sizeof(chunk),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"delta\":{\"content\":\"%s\"},\"index\":0}]}\n\n",
             req_id, escaped);
    http_write_str(fd, chunk);
}

static void sse_send_done(int fd, const char *req_id) {
    char chunk[512];
    snprintf(chunk, sizeof(chunk),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n"
             "data: [DONE]\n\n",
             req_id);
    http_write_str(fd, chunk);
}

// ============================================================================
// Prompt building helpers
// ============================================================================

typedef struct {
    int *ids;
    int count;
    int cap;
} PromptBuilder;

static bool prompt_builder_reserve(PromptBuilder *pb, int extra) {
    if (!pb || extra < 0) return false;
    if (pb->count + extra <= pb->cap) return true;

    int new_cap = pb->cap > 0 ? pb->cap : 128;
    while (new_cap < pb->count + extra) {
        new_cap *= 2;
    }

    int *new_ids = realloc(pb->ids, (size_t)new_cap * sizeof(int));
    if (!new_ids) return false;
    pb->ids = new_ids;
    pb->cap = new_cap;
    return true;
}

static bool prompt_builder_append_token(PromptBuilder *pb, int token_id) {
    if (token_id < 0) return true;
    if (!prompt_builder_reserve(pb, 1)) return false;
    pb->ids[pb->count++] = token_id;
    return true;
}

static bool prompt_builder_append_text(PromptBuilder *pb, const char *text) {
    if (!text || text[0] == '\0') return true;
    PromptTokens *frag = tokenizer_encode(text);
    if (!frag) return false;
    bool ok = prompt_builder_reserve(pb, frag->count);
    if (ok) {
        memcpy(pb->ids + pb->count, frag->ids, (size_t)frag->count * sizeof(int));
        pb->count += frag->count;
    }
    prompt_tokens_free(frag);
    return ok;
}

static bool string_has_token_prefix(NSString *text, int token_id) {
    if (token_id < 0 || ![text isKindOfClass:[NSString class]]) return false;
    const char *tok_text = tokenizer_decode(token_id);
    if (!tok_text || tok_text[0] == '\0') return false;
    NSString *prefix = [NSString stringWithUTF8String:tok_text];
    return prefix ? [text hasPrefix:prefix] : false;
}

static bool append_chat_turn(PromptBuilder *pb, const ModelConfig *cfg,
                             const char *role, const char *content,
                             bool inject_empty_think) {
    bool use_chat_tokens = cfg->chat_start_token >= 0 && cfg->chat_end_token >= 0;

    if (use_chat_tokens && !prompt_builder_append_token(pb, cfg->chat_start_token)) return false;
    if (!prompt_builder_append_text(pb, role)) return false;
    if (!prompt_builder_append_text(pb, "\n")) return false;

    if (inject_empty_think && cfg->think_start_token >= 0 && cfg->think_end_token >= 0) {
        if (!prompt_builder_append_token(pb, cfg->think_start_token)) return false;
        if (!prompt_builder_append_text(pb, "\n\n")) return false;
        if (!prompt_builder_append_token(pb, cfg->think_end_token)) return false;
        if (!prompt_builder_append_text(pb, "\n\n")) return false;
    }

    if (!prompt_builder_append_text(pb, content)) return false;

    if (use_chat_tokens) {
        if (!prompt_builder_append_token(pb, cfg->chat_end_token)) return false;
        return prompt_builder_append_text(pb, "\n");
    }
    return prompt_builder_append_text(pb, "\n\n");
}

static bool append_generation_prefix(PromptBuilder *pb, const ModelConfig *cfg) {
    if (cfg->chat_start_token >= 0 && !prompt_builder_append_token(pb, cfg->chat_start_token)) {
        return false;
    }
    if (!prompt_builder_append_text(pb, "assistant\n")) return false;
    if (cfg->chat_prefill_think &&
        cfg->think_start_token >= 0 &&
        cfg->think_end_token >= 0) {
        if (!prompt_builder_append_token(pb, cfg->think_start_token)) return false;
        if (!prompt_builder_append_text(pb, "\n\n")) return false;
        if (!prompt_builder_append_token(pb, cfg->think_end_token)) return false;
        if (!prompt_builder_append_text(pb, "\n\n")) return false;
    }
    return true;
}

static PromptTokens *build_chat_prompt_tokens(const ModelConfig *cfg, NSArray *messages) {
    PromptBuilder pb = {0};

    if ([messages isKindOfClass:[NSArray class]] && messages.count > 0) {
        for (NSDictionary *msg in messages) {
            NSString *role = msg[@"role"];
            NSString *content = msg[@"content"];
            if (![role isKindOfClass:[NSString class]] || ![content isKindOfClass:[NSString class]]) {
                continue;
            }

            bool inject_empty_think =
                cfg->chat_prefill_think &&
                [role isEqualToString:@"assistant"] &&
                !string_has_token_prefix(content, cfg->think_start_token);
            if (!append_chat_turn(&pb, cfg, [role UTF8String], [content UTF8String], inject_empty_think)) {
                free(pb.ids);
                return NULL;
            }
        }
    } else {
        if (!append_chat_turn(&pb, cfg, "user", "Hello", false)) {
            free(pb.ids);
            return NULL;
        }
    }

    if (!append_generation_prefix(&pb, cfg)) {
        free(pb.ids);
        return NULL;
    }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = pb.ids;
    pt->count = pb.count;
    return pt;
}

// ============================================================================
// Sampling helper — runs engine_step then samples from logits
// ============================================================================

static inline int model_step(Engine *eng, int token_id) {
    return model_uses_qwen35_dense_hybrid(eng->cfg)
        ? engine_step_qwen35_dense_hybrid(eng, token_id)
        : engine_step(eng, token_id);
}

static int sample_next(Engine *eng, int token_id, float temperature, int top_k) {
    model_step(eng, token_id);
    const float *logits = (const float *)[eng->ctx->buf_output contents];
    return cpu_sample_topk(logits, eng->cfg->vocab_size, top_k, temperature);
}

// ============================================================================
// Main server loop
// ============================================================================

void serve_loop(Engine *eng, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return;
    }

    listen(server_fd, 8);
    printf("[server] Listening on port %d\n", port);
    printf("[server] Model: %s\n", eng->cfg->name);

    signal(SIGPIPE, SIG_IGN);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        size_t req_len;
        char *req = read_http_request(client_fd, &req_len);
        if (!req) { close(client_fd); continue; }

        // Parse method and path
        char method[8] = {0}, path[256] = {0};
        sscanf(req, "%7s %255s", method, path);

        // CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd,
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Connection: close\r\n\r\n");
            free(req); close(client_fd); continue;
        }

        // GET /health
        if (strcmp(path, "/health") == 0) {
            http_write_str(client_fd,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n"
                "{\"status\":\"ok\"}\n");
            free(req); close(client_fd); continue;
        }

        // GET /v1/models
        if (strcmp(path, "/v1/models") == 0) {
            char body[1024];
            snprintf(body, sizeof(body),
                "{\"object\":\"list\",\"data\":["
                "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"orome\"}"
                "]}\n", eng->cfg->name);
            char resp[2048];
            snprintf(resp, sizeof(resp),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Content-Length: %zu\r\n"
                "Connection: close\r\n\r\n%s",
                strlen(body), body);
            http_write_str(client_fd, resp);
            free(req); close(client_fd); continue;
        }

        // POST /v1/chat/completions
        if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
            @autoreleasepool {
            // Parse JSON body
            char *body_start = strstr(req, "\r\n\r\n");
            PromptTokens *pt = NULL;
            int max_tokens = 256;
            float temperature = 0.6f;
            int top_k = 20;

            if (body_start) {
                body_start += 4;
                NSData *body_data = [NSData dataWithBytes:body_start
                                                   length:req_len - (body_start - req)];
                NSDictionary *json = [NSJSONSerialization JSONObjectWithData:body_data
                                                                    options:0 error:nil];
                if (json) {
                    NSNumber *mt = json[@"max_tokens"];
                    if (mt) max_tokens = [mt intValue];
                    NSNumber *temp = json[@"temperature"];
                    if (temp) temperature = [temp floatValue];
                    pt = build_chat_prompt_tokens(eng->cfg, json[@"messages"]);
                }
            }

            if (!pt) {
                pt = build_chat_prompt_tokens(eng->cfg, nil);
            }
            if (!pt) {
                http_write_str(client_fd,
                    "HTTP/1.1 500 Internal Server Error\r\n"
                    "Content-Type: application/json\r\n"
                    "Connection: close\r\n\r\n"
                    "{\"error\":\"failed to build prompt\"}\n");
                free(req);
                close(client_fd);
                continue;
            }
            int needed_seq = pt->count + (max_tokens > 0 ? max_tokens : 0);
            if (!metal_ensure_kv_capacity(eng->ctx, eng->cfg, needed_seq > 0 ? needed_seq : 1)) {
                prompt_tokens_free(pt);
                http_write_str(client_fd,
                    "HTTP/1.1 500 Internal Server Error\r\n"
                    "Content-Type: application/json\r\n"
                    "Connection: close\r\n\r\n"
                    "{\"error\":\"failed to size KV cache\"}\n");
                free(req);
                close(client_fd);
                continue;
            }

            // SSE headers
            http_write_str(client_fd,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: keep-alive\r\n\r\n");

            // Tokenize and prefill full conversation (greedy — no sampling)
            engine_reset(eng);
            int next_token = 0;
            for (int i = 0; i < pt->count; i++) {
                next_token = model_step(eng, pt->ids[i]);
            }
            prompt_tokens_free(pt);

            // Re-sample the first token after prefill
            if (temperature > 0) {
                const float *logits = (const float *)[eng->ctx->buf_output contents];
                next_token = cpu_sample_topk(logits, eng->cfg->vocab_size,
                                             top_k, temperature);
            }

            // Generate — detect and filter think blocks dynamically
            char req_id[32];
            snprintf(req_id, sizeof(req_id), "cmpl-%d", (int)eng->pos);
            int think_start_token = eng->cfg->think_start_token;
            int think_end_token = eng->cfg->think_end_token;
            bool in_think = false;
            for (int i = 0; i < max_tokens; i++) {
                if (is_eos_token(eng->cfg, next_token)) break;

                if (think_start_token >= 0 && next_token == think_start_token) {
                    in_think = true;
                    next_token = sample_next(eng, next_token, temperature, top_k);
                    continue;
                }
                if (think_end_token >= 0 && next_token == think_end_token) {
                    in_think = false;
                    next_token = sample_next(eng, next_token, temperature, top_k);
                    // Skip newline after </think>
                    if (!is_eos_token(eng->cfg, next_token)) {
                        const char *t = tokenizer_decode(next_token);
                        if (t[0] == '\n' && t[1] == '\0') {
                            next_token = sample_next(eng, next_token,
                                                     temperature, top_k);
                        }
                    }
                    continue;
                }

                if (!in_think) {
                    const char *text = tokenizer_decode(next_token);
                    sse_send_delta(client_fd, text, req_id);
                }

                next_token = sample_next(eng, next_token, temperature, top_k);
            }
            sse_send_done(client_fd, req_id);
            } // @autoreleasepool
        } else {
            http_write_str(client_fd,
                "HTTP/1.1 404 Not Found\r\n"
                "Content-Type: application/json\r\n"
                "Connection: close\r\n\r\n"
                "{\"error\":\"not found\"}\n");
        }

        free(req);
        close(client_fd);
    }
}
