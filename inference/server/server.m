/*
 * server.m — HTTP/SSE server (OpenAI-compatible API).
 *
 * Implements:
 *   POST /v1/chat/completions — streaming/non-streaming chat with tool calling
 *   GET  /v1/models           — model list (all known + loaded indicator)
 *   GET  /health              — health check with model metadata
 *
 * Conforms closely to the OpenAI chat completions API spec:
 *   - Proper SSE chunk format with model, created, system_fingerprint
 *   - First chunk sends role: "assistant" in delta
 *   - Final chunk includes usage object
 *   - Tool-calling payloads in function_call / tool_calls format
 *   - OpenAI-style error objects
 *   - Non-streaming mode (stream: false)
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <time.h>

#include "orome.h"

// ============================================================================
// Known models — all Qwen3.5 variants supported by this engine
// ============================================================================

typedef struct {
    const char *id;
    const char *owned_by;
    int64_t created;
} KnownModel;

static const KnownModel known_models[] = {
    { "qwen3.5-35b-a3b", "orome", 1750000000 },
    { "qwen3.5-27b",     "orome", 1750000000 },
    { "qwen3.5-9b",      "orome", 1750000000 },
};
static const int num_known_models = sizeof(known_models) / sizeof(known_models[0]);

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
        if (strstr(buf, "\r\n\r\n")) {
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
// JSON escape helper
// ============================================================================

static int json_escape(const char *src, char *dst, int dst_cap) {
    int j = 0;
    for (int i = 0; src[i] && j < dst_cap - 1; i++) {
        switch (src[i]) {
            case '"':  if (j + 2 > dst_cap - 1) goto done; dst[j++] = '\\'; dst[j++] = '"'; break;
            case '\\': if (j + 2 > dst_cap - 1) goto done; dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\n': if (j + 2 > dst_cap - 1) goto done; dst[j++] = '\\'; dst[j++] = 'n'; break;
            case '\r': if (j + 2 > dst_cap - 1) goto done; dst[j++] = '\\'; dst[j++] = 'r'; break;
            case '\t': if (j + 2 > dst_cap - 1) goto done; dst[j++] = '\\'; dst[j++] = 't'; break;
            default:
                if ((unsigned char)src[i] < 0x20) {
                    if (j + 6 > dst_cap - 1) goto done;
                    j += snprintf(dst + j, dst_cap - j, "\\u%04x", (unsigned char)src[i]);
                } else {
                    dst[j++] = src[i];
                }
                break;
        }
    }
done:
    dst[j] = '\0';
    return j;
}

// ============================================================================
// Error response helper (OpenAI format)
// ============================================================================

static void http_send_error(int fd, int status_code, const char *status_text,
                            const char *message, const char *error_type, const char *code) {
    char body[1024];
    char escaped_msg[512];
    json_escape(message, escaped_msg, sizeof(escaped_msg));
    snprintf(body, sizeof(body),
             "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"param\":null,\"code\":\"%s\"}}",
             escaped_msg, error_type, code);
    char resp[2048];
    snprintf(resp, sizeof(resp),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: application/json\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             status_code, status_text, strlen(body), body);
    http_write_str(fd, resp);
}

// ============================================================================
// SSE helpers — OpenAI-conformant chunk format
// ============================================================================

static void sse_send_role_chunk(int fd, const char *req_id, const char *model,
                                int64_t created) {
    char chunk[1024];
    snprintf(chunk, sizeof(chunk),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"system_fingerprint\":\"orome-v0\","
             "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
             "\"logprobs\":null,\"finish_reason\":null}]}\n\n",
             req_id, (long long)created, model);
    http_write_str(fd, chunk);
}

static void sse_send_content_delta(int fd, const char *token_text, const char *req_id,
                                   const char *model, int64_t created) {
    char escaped[512];
    json_escape(token_text, escaped, sizeof(escaped));
    char chunk[1024];
    snprintf(chunk, sizeof(chunk),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"system_fingerprint\":\"orome-v0\","
             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
             "\"logprobs\":null,\"finish_reason\":null}]}\n\n",
             req_id, (long long)created, model, escaped);
    http_write_str(fd, chunk);
}

static void sse_send_tool_call_delta(int fd, const char *req_id, const char *model,
                                     int64_t created, int tool_idx,
                                     const char *call_id, const char *fn_name,
                                     const char *args_fragment) {
    char escaped_args[512];
    json_escape(args_fragment, escaped_args, sizeof(escaped_args));
    char escaped_name[128] = "";
    if (fn_name) json_escape(fn_name, escaped_name, sizeof(escaped_name));

    char chunk[2048];
    if (fn_name) {
        // First chunk for this tool call — includes id and function name
        char escaped_id[64];
        json_escape(call_id, escaped_id, sizeof(escaped_id));
        snprintf(chunk, sizeof(chunk),
                 "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"system_fingerprint\":\"orome-v0\","
                 "\"choices\":[{\"index\":0,\"delta\":{"
                 "\"tool_calls\":[{\"index\":%d,\"id\":\"%s\","
                 "\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},"
                 "\"logprobs\":null,\"finish_reason\":null}]}\n\n",
                 req_id, (long long)created, model,
                 tool_idx, escaped_id, escaped_name, escaped_args);
    } else {
        // Continuation chunk — arguments only
        snprintf(chunk, sizeof(chunk),
                 "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"system_fingerprint\":\"orome-v0\","
                 "\"choices\":[{\"index\":0,\"delta\":{"
                 "\"tool_calls\":[{\"index\":%d,\"function\":{\"arguments\":\"%s\"}}]},"
                 "\"logprobs\":null,\"finish_reason\":null}]}\n\n",
                 req_id, (long long)created, model,
                 tool_idx, escaped_args);
    }
    http_write_str(fd, chunk);
}

static void sse_send_done(int fd, const char *req_id, const char *model,
                          int64_t created, const char *finish_reason,
                          int prompt_tokens, int completion_tokens,
                          double prefill_ms, double decode_ms) {
    char chunk[2048];
    snprintf(chunk, sizeof(chunk),
             "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"system_fingerprint\":\"orome-v0\","
             "\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,"
             "\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
             "\"total_tokens\":%d},"
             "\"x_orome\":{\"prefill_ms\":%.1f,\"decode_ms\":%.1f,"
             "\"tokens_per_sec\":%.2f}}\n\n"
             "data: [DONE]\n\n",
             req_id, (long long)created, model,
             finish_reason,
             prompt_tokens, completion_tokens,
             prompt_tokens + completion_tokens,
             prefill_ms, decode_ms,
             decode_ms > 0 ? completion_tokens * 1000.0 / decode_ms : 0);
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

static bool append_tool_result_turn(PromptBuilder *pb, const ModelConfig *cfg,
                                    const char *tool_call_id, const char *content) {
    bool use_chat_tokens = cfg->chat_start_token >= 0 && cfg->chat_end_token >= 0;
    if (use_chat_tokens && !prompt_builder_append_token(pb, cfg->chat_start_token)) return false;
    if (!prompt_builder_append_text(pb, "tool\n")) return false;

    // Format: tool_call_id followed by content
    char header[256];
    snprintf(header, sizeof(header), "<tool_response>\n%s\n", tool_call_id ? tool_call_id : "");
    if (!prompt_builder_append_text(pb, header)) return false;
    if (!prompt_builder_append_text(pb, content)) return false;
    if (!prompt_builder_append_text(pb, "\n</tool_response>")) return false;

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
        cfg->think_start_token >= 0) {
        if (!prompt_builder_append_token(pb, cfg->think_start_token)) return false;
        if (!prompt_builder_append_text(pb, "\n")) return false;
    }
    return true;
}

static PromptTokens *build_chat_prompt_tokens(const ModelConfig *cfg, NSArray *messages,
                                              bool has_tools) {
    PromptBuilder pb = {0};

    if ([messages isKindOfClass:[NSArray class]] && messages.count > 0) {
        for (NSDictionary *msg in messages) {
            NSString *role = msg[@"role"];
            if (![role isKindOfClass:[NSString class]]) continue;

            // Handle tool result messages
            if ([role isEqualToString:@"tool"]) {
                NSString *content = msg[@"content"];
                NSString *tool_call_id = msg[@"tool_call_id"];
                if (![content isKindOfClass:[NSString class]]) continue;
                const char *tcid = [tool_call_id isKindOfClass:[NSString class]]
                    ? [tool_call_id UTF8String] : NULL;
                if (!append_tool_result_turn(&pb, cfg, tcid, [content UTF8String])) {
                    free(pb.ids);
                    return NULL;
                }
                continue;
            }

            // Handle assistant messages with tool_calls
            NSArray *tool_calls = msg[@"tool_calls"];
            if ([role isEqualToString:@"assistant"] && [tool_calls isKindOfClass:[NSArray class]]) {
                // Serialize the tool calls as the assistant's content
                NSMutableString *tc_content = [NSMutableString string];
                NSString *content = msg[@"content"];
                if ([content isKindOfClass:[NSString class]] && content.length > 0) {
                    [tc_content appendString:content];
                    [tc_content appendString:@"\n"];
                }
                for (NSDictionary *tc in tool_calls) {
                    NSDictionary *fn = tc[@"function"];
                    if (![fn isKindOfClass:[NSDictionary class]]) continue;
                    NSString *fn_name = fn[@"name"];
                    NSString *fn_args = fn[@"arguments"];
                    NSString *tc_id = tc[@"id"];
                    [tc_content appendFormat:@"<tool_call>\n{\"name\": \"%@\", \"arguments\": %@",
                     fn_name ? fn_name : @"", fn_args ? fn_args : @"{}"];
                    if (tc_id) [tc_content appendFormat:@", \"id\": \"%@\"", tc_id];
                    [tc_content appendString:@"}\n</tool_call>\n"];
                }
                if (!append_chat_turn(&pb, cfg, "assistant", [tc_content UTF8String], false)) {
                    free(pb.ids);
                    return NULL;
                }
                continue;
            }

            NSString *content = msg[@"content"];
            if (![content isKindOfClass:[NSString class]]) continue;

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

    // If tools are available, inject a system hint so the model knows it can call them
    if (has_tools) {
        // The generation prefix follows; tool-calling format is model-native
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
// Tool-call detection — parse <tool_call> blocks from generated text
// ============================================================================

typedef struct {
    char name[128];
    char arguments[4096];
    char id[64];
} ParsedToolCall;

static int parse_tool_calls(const char *text, ParsedToolCall *out, int max_calls) {
    int count = 0;
    const char *p = text;
    while (count < max_calls) {
        const char *start = strstr(p, "<tool_call>");
        if (!start) break;
        const char *end = strstr(start, "</tool_call>");
        if (!end) break;

        start += strlen("<tool_call>");
        size_t block_len = (size_t)(end - start);
        char *block = malloc(block_len + 1);
        memcpy(block, start, block_len);
        block[block_len] = '\0';

        // Parse JSON from the block
        NSData *data = [NSData dataWithBytes:block length:block_len];
        NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (json) {
            ParsedToolCall *tc = &out[count];
            memset(tc, 0, sizeof(*tc));

            NSString *name = json[@"name"];
            if ([name isKindOfClass:[NSString class]]) {
                strlcpy(tc->name, [name UTF8String], sizeof(tc->name));
            }

            // Re-serialize arguments as JSON string
            id args = json[@"arguments"];
            if (args) {
                NSData *args_data = [NSJSONSerialization dataWithJSONObject:args options:0 error:nil];
                if (args_data) {
                    NSString *args_str = [[NSString alloc] initWithData:args_data encoding:NSUTF8StringEncoding];
                    if (args_str) strlcpy(tc->arguments, [args_str UTF8String], sizeof(tc->arguments));
                }
            }

            snprintf(tc->id, sizeof(tc->id), "call_%d_%ld", count, (long)time(NULL));
            count++;
        }
        free(block);
        p = end + strlen("</tool_call>");
    }
    return count;
}

// ============================================================================
// Sampling helper — runs engine_step then samples from logits
// ============================================================================

static inline int model_step_dispatch(Engine *eng, int token_id) {
    return model_uses_qwen35_dense_hybrid(eng->cfg)
        ? engine_step_qwen35_dense_hybrid(eng, token_id)
        : engine_step(eng, token_id);
}

static int sample_next(Engine *eng, int token_id, float temperature, int top_k) {
    model_step_dispatch(eng, token_id);
    const float *logits = (const float *)[eng->ctx->buf_output contents];
    return cpu_sample_topk(logits, eng->cfg->vocab_size, top_k, temperature);
}

// ============================================================================
// Endpoint handlers
// ============================================================================

static void handle_health(int fd, Engine *eng, time_t server_start) {
    char body[1024];
    snprintf(body, sizeof(body),
             "{\"status\":\"ok\","
             "\"model\":\"%s\","
             "\"uptime_seconds\":%lld,"
             "\"engine\":{\"layers\":%d,\"hidden_dim\":%d,"
             "\"ffn_type\":\"%s\",\"context_length\":%d}}",
             eng->cfg->name,
             (long long)(time(NULL) - server_start),
             eng->cfg->num_layers, eng->cfg->hidden_dim,
             eng->cfg->ffn_type == FFN_MOE ? "moe" : "dense",
             eng->cfg->context_length);
    char resp[2048];
    snprintf(resp, sizeof(resp),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: application/json\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(body), body);
    http_write_str(fd, resp);
}

static void handle_models(int fd, Engine *eng) {
    // Build JSON array of all known models, marking the loaded one
    char body[4096];
    int pos = 0;
    pos += snprintf(body + pos, sizeof(body) - pos, "{\"object\":\"list\",\"data\":[");

    for (int i = 0; i < num_known_models; i++) {
        if (i > 0) body[pos++] = ',';
        bool is_loaded = (strcasestr(eng->cfg->name, known_models[i].id) != NULL);
        pos += snprintf(body + pos, sizeof(body) - pos,
                        "{\"id\":\"%s\",\"object\":\"model\","
                        "\"created\":%lld,\"owned_by\":\"%s\""
                        "%s}",
                        known_models[i].id,
                        (long long)known_models[i].created,
                        known_models[i].owned_by,
                        is_loaded ? ",\"x_orome_loaded\":true" : "");
    }

    pos += snprintf(body + pos, sizeof(body) - pos, "]}");

    char resp[8192];
    snprintf(resp, sizeof(resp),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: application/json\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Content-Length: %zu\r\n"
             "Connection: close\r\n\r\n%s",
             strlen(body), body);
    http_write_str(fd, resp);
}

static void handle_chat_completions(int fd, Engine *eng, const char *req, size_t req_len) {
    @autoreleasepool {
    char *body_start = strstr(req, "\r\n\r\n");
    if (!body_start) {
        http_send_error(fd, 400, "Bad Request", "Missing request body",
                        "invalid_request_error", "missing_body");
        return;
    }
    body_start += 4;

    NSData *body_data = [NSData dataWithBytes:body_start
                                       length:req_len - (body_start - req)];
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:body_data
                                                        options:0 error:nil];
    if (!json) {
        http_send_error(fd, 400, "Bad Request", "Invalid JSON in request body",
                        "invalid_request_error", "invalid_json");
        return;
    }

    // Parse request parameters
    int max_tokens = 256;
    float temperature = 0.6f;
    int top_k = 20;
    bool stream = true;

    NSNumber *mt = json[@"max_tokens"];
    if (!mt) mt = json[@"max_completion_tokens"];
    if (mt) max_tokens = [mt intValue];
    NSNumber *temp = json[@"temperature"];
    if (temp) temperature = [temp floatValue];
    NSNumber *stream_val = json[@"stream"];
    if (stream_val) stream = [stream_val boolValue];

    // Check for tools
    NSArray *tools = json[@"tools"];
    bool has_tools = [tools isKindOfClass:[NSArray class]] && tools.count > 0;

    // Build prompt
    PromptTokens *pt = build_chat_prompt_tokens(eng->cfg, json[@"messages"], has_tools);
    if (!pt) {
        http_send_error(fd, 400, "Bad Request", "Failed to build prompt from messages",
                        "invalid_request_error", "invalid_messages");
        return;
    }

    int prompt_tokens = pt->count;
    int needed_seq = pt->count + (max_tokens > 0 ? max_tokens : 0);
    if (!metal_ensure_kv_capacity(eng->ctx, eng->cfg, needed_seq > 0 ? needed_seq : 1)) {
        prompt_tokens_free(pt);
        http_send_error(fd, 500, "Internal Server Error",
                        "Failed to allocate KV cache for request",
                        "server_error", "kv_cache_error");
        return;
    }

    // Prefill
    double prefill_start = now_ms();
    engine_reset(eng);
    int next_token = 0;
    for (int i = 0; i < pt->count; i++) {
        next_token = model_step_dispatch(eng, pt->ids[i]);
    }
    prompt_tokens_free(pt);
    double prefill_ms = now_ms() - prefill_start;

    // Re-sample the first token after prefill
    if (temperature > 0) {
        const float *logits = (const float *)[eng->ctx->buf_output contents];
        next_token = cpu_sample_topk(logits, eng->cfg->vocab_size, top_k, temperature);
    }

    // Generate
    int64_t created = (int64_t)time(NULL);
    char req_id[48];
    snprintf(req_id, sizeof(req_id), "chatcmpl-%lld-%d",
             (long long)created, (int)eng->pos);
    const char *model_name = eng->cfg->name;
    int think_start_token = eng->cfg->think_start_token;
    int think_end_token = eng->cfg->think_end_token;

    if (stream) {
        // SSE streaming response
        http_write_str(fd,
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n"
            "X-Request-Id: ");
        http_write_str(fd, req_id);
        http_write_str(fd, "\r\n\r\n");

        // First chunk: role
        sse_send_role_chunk(fd, req_id, model_name, created);

        double decode_start = now_ms();
        bool in_think = false;
        int completion_tokens = 0;
        const char *finish_reason = "stop";

        // When tools are present, accumulate text to detect tool_call blocks
        size_t accum_cap = has_tools ? 4096 : 0;
        size_t accum_len = 0;
        char *accum_buf = has_tools ? malloc(accum_cap) : NULL;
        if (accum_buf) accum_buf[0] = '\0';

        for (int i = 0; i < max_tokens; i++) {
            if (is_eos_token(eng->cfg, next_token)) break;

            if (think_start_token >= 0 && next_token == think_start_token) {
                in_think = true;
                next_token = sample_next(eng, next_token, temperature, top_k);
                completion_tokens++;
                continue;
            }
            if (think_end_token >= 0 && next_token == think_end_token) {
                in_think = false;
                next_token = sample_next(eng, next_token, temperature, top_k);
                completion_tokens++;
                // Skip newline after </think>
                if (!is_eos_token(eng->cfg, next_token)) {
                    const char *t = tokenizer_decode(next_token);
                    if (t[0] == '\n' && t[1] == '\0') {
                        next_token = sample_next(eng, next_token, temperature, top_k);
                        completion_tokens++;
                    }
                }
                continue;
            }

            if (!in_think) {
                const char *text = tokenizer_decode(next_token);
                if (has_tools) {
                    // Accumulate for tool-call detection
                    size_t tlen = strlen(text);
                    while (accum_len + tlen + 1 > accum_cap) {
                        accum_cap *= 2;
                        accum_buf = realloc(accum_buf, accum_cap);
                    }
                    memcpy(accum_buf + accum_len, text, tlen);
                    accum_len += tlen;
                    accum_buf[accum_len] = '\0';
                } else {
                    sse_send_content_delta(fd, text, req_id, model_name, created);
                }
            }

            next_token = sample_next(eng, next_token, temperature, top_k);
            completion_tokens++;

            if (i == max_tokens - 1) {
                finish_reason = "length";
            }
        }

        // If tools were available, check for tool_call blocks; otherwise stream as content
        if (has_tools && accum_buf && accum_len > 0) {
            ParsedToolCall stream_tcs[8];
            int num_stream_tcs = parse_tool_calls(accum_buf, stream_tcs, 8);
            if (num_stream_tcs > 0) {
                // Emit any content before the first <tool_call>
                char *tc_marker = strstr(accum_buf, "<tool_call>");
                if (tc_marker && tc_marker > accum_buf) {
                    size_t prefix_len = (size_t)(tc_marker - accum_buf);
                    while (prefix_len > 0 && (accum_buf[prefix_len-1] == '\n' ||
                           accum_buf[prefix_len-1] == ' ')) prefix_len--;
                    if (prefix_len > 0) {
                        char saved = accum_buf[prefix_len];
                        accum_buf[prefix_len] = '\0';
                        sse_send_content_delta(fd, accum_buf, req_id, model_name, created);
                        accum_buf[prefix_len] = saved;
                    }
                }
                // Emit tool calls
                for (int t = 0; t < num_stream_tcs; t++) {
                    sse_send_tool_call_delta(fd, req_id, model_name, created, t,
                                             stream_tcs[t].id, stream_tcs[t].name,
                                             stream_tcs[t].arguments);
                }
                finish_reason = "tool_calls";
            } else {
                // No tool calls detected — flush accumulated text as content
                sse_send_content_delta(fd, accum_buf, req_id, model_name, created);
            }
        }
        free(accum_buf);

        double decode_ms = now_ms() - decode_start;
        sse_send_done(fd, req_id, model_name, created, finish_reason,
                      prompt_tokens, completion_tokens, prefill_ms, decode_ms);
    } else {
        // Non-streaming response — collect all output
        double decode_start = now_ms();
        bool in_think = false;
        int completion_tokens = 0;
        const char *finish_reason = "stop";

        // Accumulate generated text
        size_t text_cap = 4096;
        size_t text_len = 0;
        char *text_buf = malloc(text_cap);
        text_buf[0] = '\0';

        for (int i = 0; i < max_tokens; i++) {
            if (is_eos_token(eng->cfg, next_token)) break;

            if (think_start_token >= 0 && next_token == think_start_token) {
                in_think = true;
                next_token = sample_next(eng, next_token, temperature, top_k);
                completion_tokens++;
                continue;
            }
            if (think_end_token >= 0 && next_token == think_end_token) {
                in_think = false;
                next_token = sample_next(eng, next_token, temperature, top_k);
                completion_tokens++;
                if (!is_eos_token(eng->cfg, next_token)) {
                    const char *t = tokenizer_decode(next_token);
                    if (t[0] == '\n' && t[1] == '\0') {
                        next_token = sample_next(eng, next_token, temperature, top_k);
                        completion_tokens++;
                    }
                }
                continue;
            }

            if (!in_think) {
                const char *tok_text = tokenizer_decode(next_token);
                size_t tok_len = strlen(tok_text);
                while (text_len + tok_len + 1 > text_cap) {
                    text_cap *= 2;
                    text_buf = realloc(text_buf, text_cap);
                }
                memcpy(text_buf + text_len, tok_text, tok_len);
                text_len += tok_len;
                text_buf[text_len] = '\0';
            }

            next_token = sample_next(eng, next_token, temperature, top_k);
            completion_tokens++;

            if (i == max_tokens - 1) finish_reason = "length";
        }

        double decode_ms = now_ms() - decode_start;

        // Check for tool calls in generated text
        ParsedToolCall tool_calls[8];
        int num_tool_calls = 0;
        if (has_tools) {
            num_tool_calls = parse_tool_calls(text_buf, tool_calls, 8);
        }

        // Build response JSON
        char *escaped_content = malloc(text_len * 2 + 1);
        json_escape(text_buf, escaped_content, (int)(text_len * 2 + 1));

        // Build tool_calls JSON if present
        char tc_json[8192] = "";
        if (num_tool_calls > 0) {
            finish_reason = "tool_calls";
            int tc_pos = 0;
            tc_pos += snprintf(tc_json + tc_pos, sizeof(tc_json) - tc_pos, ",\"tool_calls\":[");
            for (int i = 0; i < num_tool_calls; i++) {
                if (i > 0) tc_json[tc_pos++] = ',';
                char esc_name[256], esc_args[8192], esc_id[128];
                json_escape(tool_calls[i].name, esc_name, sizeof(esc_name));
                json_escape(tool_calls[i].arguments, esc_args, sizeof(esc_args));
                json_escape(tool_calls[i].id, esc_id, sizeof(esc_id));
                tc_pos += snprintf(tc_json + tc_pos, sizeof(tc_json) - tc_pos,
                                   "{\"id\":\"%s\",\"type\":\"function\","
                                   "\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}",
                                   esc_id, esc_name, esc_args);
            }
            tc_pos += snprintf(tc_json + tc_pos, sizeof(tc_json) - tc_pos, "]");
        }

        // Strip tool_call blocks from visible content when tool calls were parsed
        char *visible_content = escaped_content;
        if (num_tool_calls > 0) {
            // For tool-call responses, content before the first <tool_call> is the text content
            char *tc_start = strstr(text_buf, "<tool_call>");
            if (tc_start && tc_start > text_buf) {
                size_t prefix_len = (size_t)(tc_start - text_buf);
                // Trim trailing whitespace
                while (prefix_len > 0 && (text_buf[prefix_len - 1] == '\n' ||
                       text_buf[prefix_len - 1] == ' ')) prefix_len--;
                char *trimmed = malloc(prefix_len + 1);
                memcpy(trimmed, text_buf, prefix_len);
                trimmed[prefix_len] = '\0';
                visible_content = malloc(prefix_len * 2 + 1);
                json_escape(trimmed, visible_content, (int)(prefix_len * 2 + 1));
                free(trimmed);
                free(escaped_content);
                escaped_content = NULL;
            } else if (tc_start == text_buf) {
                visible_content = NULL;
                free(escaped_content);
                escaped_content = NULL;
            }
        }

        size_t resp_cap = text_len * 2 + sizeof(tc_json) + 2048;
        char *body = malloc(resp_cap);
        snprintf(body, resp_cap,
                 "{\"id\":\"%s\",\"object\":\"chat.completion\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"system_fingerprint\":\"orome-v0\","
                 "\"choices\":[{\"index\":0,"
                 "\"message\":{\"role\":\"assistant\"%s%s%s%s},"
                 "\"logprobs\":null,\"finish_reason\":\"%s\"}],"
                 "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                 "\"total_tokens\":%d},"
                 "\"x_orome\":{\"prefill_ms\":%.1f,\"decode_ms\":%.1f,"
                 "\"tokens_per_sec\":%.2f}}",
                 req_id, (long long)created, model_name,
                 visible_content ? ",\"content\":\"" : "",
                 visible_content ? visible_content : "",
                 visible_content ? "\"" : ",\"content\":null",
                 tc_json,
                 finish_reason,
                 prompt_tokens, completion_tokens,
                 prompt_tokens + completion_tokens,
                 prefill_ms, decode_ms,
                 decode_ms > 0 ? completion_tokens * 1000.0 / decode_ms : 0);

        char *resp_header = malloc(512);
        snprintf(resp_header, 512,
                 "HTTP/1.1 200 OK\r\n"
                 "Content-Type: application/json\r\n"
                 "Access-Control-Allow-Origin: *\r\n"
                 "Content-Length: %zu\r\n"
                 "Connection: close\r\n\r\n",
                 strlen(body));
        http_write_str(fd, resp_header);
        http_write_str(fd, body);

        if (escaped_content) free(escaped_content);
        if (visible_content && visible_content != escaped_content) free(visible_content);
        free(text_buf);
        free(body);
        free(resp_header);
    }

    } // @autoreleasepool
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
    time_t server_start = time(NULL);
    printf("[server] Listening on port %d\n", port);
    printf("[server] Model: %s (%d layers, %s)\n",
           eng->cfg->name, eng->cfg->num_layers,
           eng->cfg->ffn_type == FFN_MOE ? "MoE" : "dense");
    printf("[server] Endpoints: /v1/chat/completions, /v1/models, /health\n");

    signal(SIGPIPE, SIG_IGN);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        size_t req_len;
        char *req = read_http_request(client_fd, &req_len);
        if (!req) { close(client_fd); continue; }

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
        } else if (strcmp(path, "/health") == 0 && strcmp(method, "GET") == 0) {
            handle_health(client_fd, eng, server_start);
        } else if (strcmp(path, "/v1/models") == 0 && strcmp(method, "GET") == 0) {
            handle_models(client_fd, eng);
        } else if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
            handle_chat_completions(client_fd, eng, req, req_len);
        } else {
            http_send_error(client_fd, 404, "Not Found",
                            "The requested endpoint does not exist",
                            "invalid_request_error", "not_found");
        }

        free(req);
        close(client_fd);
    }
}
