/*
 * server.m — HTTP/SSE server (OpenAI-compatible API).
 *
 * Implements:
 *   POST /v1/chat/completions — streaming inference with multi-turn chat
 *   GET  /v1/models           — model list
 *   GET  /health              — health check
 *
 * Chat completions parses the full OpenAI messages array and formats it
 * with Qwen chat template (<|im_start|>role\ncontent<|im_end|>).
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
// Sampling helper — runs engine_step then samples from logits
// ============================================================================

static int sample_next(Engine *eng, int token_id, float temperature, int top_k) {
    engine_step(eng, token_id);
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
            char *prompt = NULL;
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

                    // Build chat-templated prompt from messages array
                    NSArray *messages = json[@"messages"];
                    if ([messages isKindOfClass:[NSArray class]] && messages.count > 0) {
                        NSMutableString *chat = [NSMutableString new];
                        for (NSDictionary *msg in messages) {
                            NSString *role = msg[@"role"];
                            NSString *content = msg[@"content"];
                            if (!role || !content) continue;
                            // For assistant messages, wrap in think block if not present
                            if ([role isEqualToString:@"assistant"]
                                && ![content hasPrefix:@"<think>"]) {
                                [chat appendFormat:@"<|im_start|>assistant\n<think>\n</think>\n%@<|im_end|>\n",
                                 content];
                            } else {
                                [chat appendFormat:@"<|im_start|>%@\n%@<|im_end|>\n",
                                 role, content];
                            }
                        }
                        // Prefill an empty think block so visible generation starts
                        // in the answer channel rather than hidden reasoning.
                        [chat appendString:@"<|im_start|>assistant\n<think>\n</think>\n"];
                        prompt = strdup([chat UTF8String]);
                    }
                }
            }

            if (!prompt) prompt = strdup("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n");

            // SSE headers
            http_write_str(client_fd,
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: keep-alive\r\n\r\n");

            // Tokenize and prefill full conversation (greedy — no sampling)
            engine_reset(eng);
            PromptTokens *pt = tokenizer_encode(prompt);
            free(prompt);
            int next_token = 0;
            if (pt) {
                for (int i = 0; i < pt->count; i++) {
                    next_token = engine_step(eng, pt->ids[i]);
                }
                prompt_tokens_free(pt);
            }

            // Re-sample the first token after prefill
            if (temperature > 0) {
                const float *logits = (const float *)[eng->ctx->buf_output contents];
                next_token = cpu_sample_topk(logits, eng->cfg->vocab_size,
                                             top_k, temperature);
            }

            // Generate — detect and filter think blocks dynamically
            char req_id[32];
            snprintf(req_id, sizeof(req_id), "cmpl-%d", (int)eng->pos);
            int think_start_token = 248068;  // <think>
            int think_end_token = eng->cfg->think_end_token;  // </think> = 248069
            bool in_think = false;
            for (int i = 0; i < max_tokens; i++) {
                if (is_eos_token(eng->cfg, next_token)) break;

                if (next_token == think_start_token) {
                    in_think = true;
                    next_token = sample_next(eng, next_token, temperature, top_k);
                    continue;
                }
                if (next_token == think_end_token) {
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
