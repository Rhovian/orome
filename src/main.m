/*
 * main.m — CLI entry point for orome inference engine.
 *
 * Usage:
 *   ./orome --model FILE.gguf --prompt "Hello" --tokens 20
 *   ./orome --model FILE.gguf --serve 8080
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include "orome.h"

static void add_unique_token(int *tokens, int max_tokens, int *count, int token_id) {
    if (!tokens || !count || token_id < 0 || *count >= max_tokens) return;
    for (int i = 0; i < *count; i++) {
        if (tokens[i] == token_id) return;
    }
    tokens[(*count)++] = token_id;
}

static GGUFTensorInfo *find_first_layer_tensor(GGUFFile *gf, const ModelConfig *cfg,
                                               AttnLayerType layer_type, const char *suffix) {
    char name[128];
    for (int i = 0; i < cfg->num_layers; i++) {
        if (cfg->layer_types[i] != layer_type) continue;
        snprintf(name, sizeof(name), "blk.%d.%s", i, suffix);
        GGUFTensorInfo *ti = gguf_find_tensor(gf, name);
        if (ti) return ti;
    }
    return NULL;
}

static int template_collect_added_tokens_from_text(const char *text, int *ids, int max_ids) {
    if (!text || !ids || max_ids <= 0) return 0;

    int count = 0;
    const char *p = text;
    while (*p && count < max_ids) {
        const char *lt = strchr(p, '<');
        if (!lt) break;
        const char *gt = strchr(lt + 1, '>');
        if (!gt) break;

        size_t len = (size_t)(gt - lt + 1);
        if (len > 1 && len < 256) {
            char token[256];
            memcpy(token, lt, len);
            token[len] = '\0';

            int token_id = tokenizer_find_token(token);
            if (token_id >= 0) {
                ids[count++] = token_id;
            }
        }

        p = gt + 1;
    }

    if (count == 0) {
        count = tokenizer_find_added_tokens_in_text(text, ids, max_ids);
    }
    return count;
}

static int template_collect_added_tokens_from_literals(const char *expr, int *ids, int max_ids) {
    if (!expr || !ids || max_ids <= 0) return 0;

    int count = 0;
    const char *p = expr;
    while (*p && count < max_ids) {
        if (*p != '\'' && *p != '"') {
            p++;
            continue;
        }

        char quote = *p++;
        const char *start = p;
        bool escape = false;
        while (*p) {
            if (escape) {
                escape = false;
            } else if (*p == '\\') {
                escape = true;
            } else if (*p == quote) {
                break;
            }
            p++;
        }

        size_t len = (size_t)(p - start);
        char *literal = malloc(len + 1);
        if (!literal) break;
        memcpy(literal, start, len);
        literal[len] = '\0';

        int tmp[16];
        int tmp_count = template_collect_added_tokens_from_text(literal, tmp, 16);
        for (int i = 0; i < tmp_count && count < max_ids; i++) {
            ids[count++] = tmp[i];
        }
        free(literal);

        if (*p == quote) p++;
    }

    return count;
}

static const char *skip_jinja_tag_prefix(const char *tag) {
    while (*tag == ' ' || *tag == '\t' || *tag == '\r' || *tag == '\n' || *tag == '-') {
        tag++;
    }
    return tag;
}

static bool jinja_tag_starts_with(const char *tag, const char *prefix) {
    if (!tag || !prefix) return false;
    tag = skip_jinja_tag_prefix(tag);
    size_t prefix_len = strlen(prefix);
    return strncmp(tag, prefix, prefix_len) == 0;
}

typedef enum {
    TEMPLATE_ROLE_NONE = 0,
    TEMPLATE_ROLE_SYSTEM,
    TEMPLATE_ROLE_USER,
    TEMPLATE_ROLE_ASSISTANT,
    TEMPLATE_ROLE_TOOL,
} TemplateRole;

static TemplateRole template_role_from_tag(const char *tag) {
    if (!tag) return TEMPLATE_ROLE_NONE;
    if (strstr(tag, "message.role == \"system\"") || strstr(tag, "message.role == 'system'")) {
        return TEMPLATE_ROLE_SYSTEM;
    }
    if (strstr(tag, "message.role == \"user\"") || strstr(tag, "message.role == 'user'")) {
        return TEMPLATE_ROLE_USER;
    }
    if (strstr(tag, "message.role == \"assistant\"") || strstr(tag, "message.role == 'assistant'")) {
        return TEMPLATE_ROLE_ASSISTANT;
    }
    if (strstr(tag, "message.role == \"tool\"") || strstr(tag, "message.role == 'tool'")) {
        return TEMPLATE_ROLE_TOOL;
    }
    return TEMPLATE_ROLE_NONE;
}

static void resolve_chat_tokens_from_template(ModelConfig *cfg, const char *chat_template) {
    if (!cfg || !chat_template || chat_template[0] == '\0') return;

    enum { MAX_TEMPLATE_TOKENS = 16 };
    int generation_tokens[MAX_TEMPLATE_TOKENS];
    int generation_count = 0;
    int generation_if_depth = 0;
    int message_loop_depth = 0;
    TemplateRole current_role = TEMPLATE_ROLE_NONE;

    const char *p = chat_template;
    while (*p) {
        if (p[0] == '{' && p[1] == '%') {
            const char *end = strstr(p + 2, "%}");
            if (!end) break;

            size_t len = (size_t)(end - (p + 2));
            char *tag = malloc(len + 1);
            if (tag) {
                memcpy(tag, p + 2, len);
                tag[len] = '\0';

                if (jinja_tag_starts_with(tag, "if") && strstr(tag, "add_generation_prompt")) {
                    generation_if_depth = generation_if_depth > 0 ? generation_if_depth + 1 : 1;
                } else if (generation_if_depth > 0 && jinja_tag_starts_with(tag, "if")) {
                    generation_if_depth++;
                } else if (generation_if_depth > 0 && jinja_tag_starts_with(tag, "endif")) {
                    generation_if_depth--;
                }

                if (jinja_tag_starts_with(tag, "for") && strstr(tag, "message in messages")) {
                    message_loop_depth++;
                    current_role = TEMPLATE_ROLE_NONE;
                } else if (message_loop_depth > 0 && jinja_tag_starts_with(tag, "endfor")) {
                    message_loop_depth--;
                    if (message_loop_depth == 0) current_role = TEMPLATE_ROLE_NONE;
                } else if (message_loop_depth > 0) {
                    TemplateRole tagged_role = template_role_from_tag(tag);
                    if (tagged_role != TEMPLATE_ROLE_NONE) {
                        current_role = tagged_role;
                    }
                }

                free(tag);
            }

            p = end + 2;
            continue;
        }

        if (p[0] == '{' && p[1] == '{') {
            const char *end = strstr(p + 2, "}}");
            if (!end) break;

            size_t len = (size_t)(end - (p + 2));
            char *expr = malloc(len + 1);
            if (expr) {
                memcpy(expr, p + 2, len);
                expr[len] = '\0';

                int ids[MAX_TEMPLATE_TOKENS];
                int id_count = template_collect_added_tokens_from_literals(expr, ids, MAX_TEMPLATE_TOKENS);
                bool has_message_role = strstr(expr, "message.role") != NULL;
                bool has_reasoning = strstr(expr, "reasoning_content") != NULL;

                if (message_loop_depth > 0 &&
                    current_role == TEMPLATE_ROLE_USER &&
                    has_message_role &&
                    id_count > 0) {
                    cfg->chat_start_token = ids[0];
                    if (id_count >= 2) cfg->chat_end_token = ids[id_count - 1];
                } else if (message_loop_depth > 0 &&
                           current_role == TEMPLATE_ROLE_ASSISTANT &&
                           has_message_role &&
                           id_count > 0 &&
                           cfg->chat_start_token < 0) {
                    cfg->chat_start_token = ids[0];
                }
                if (message_loop_depth > 0 &&
                    current_role == TEMPLATE_ROLE_ASSISTANT &&
                    has_reasoning &&
                    id_count >= 3) {
                    cfg->think_start_token = ids[1];
                    cfg->think_end_token = ids[id_count - 1];
                }
                if (generation_if_depth > 0 && id_count > 0) {
                    for (int i = 0; i < id_count && generation_count < MAX_TEMPLATE_TOKENS; i++) {
                        generation_tokens[generation_count++] = ids[i];
                    }
                }

                free(expr);
            }

            p = end + 2;
            continue;
        }

        p++;
    }

    if (cfg->chat_start_token < 0 && generation_count > 0) {
        cfg->chat_start_token = generation_tokens[0];
    }

    if ((cfg->think_start_token < 0 || cfg->think_end_token < 0) && generation_count >= 3) {
        int inner_start = 0;
        if (cfg->chat_start_token >= 0 && generation_tokens[0] == cfg->chat_start_token) {
            inner_start = 1;
        }
        if (generation_count - inner_start >= 2) {
            if (cfg->think_start_token < 0) cfg->think_start_token = generation_tokens[inner_start];
            if (cfg->think_end_token < 0) cfg->think_end_token = generation_tokens[generation_count - 1];
        }
    }

    if (cfg->think_start_token >= 0 && cfg->think_end_token >= 0) {
        bool saw_think_start = false;
        for (int i = 0; i < generation_count; i++) {
            if (!saw_think_start && generation_tokens[i] == cfg->think_start_token) {
                saw_think_start = true;
                continue;
            }
            if (saw_think_start && generation_tokens[i] == cfg->think_end_token) {
                cfg->chat_prefill_think = true;
                break;
            }
        }
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model FILE     GGUF model file\n"
        "  --prompt TEXT     Prompt text\n"
        "  --tokens N        Max tokens to generate (default: 20)\n"
        "  --k N             Active experts per layer (default: from config)\n"
        "  --thermal-k N     Reduce K to N when thermal throttling engages\n"
        "  --thermal-proj-ms F  Engage thermal-K when step EMA exceeds F ms\n"
        "  --thermal-gen N   Minimum tokens before thermal-K can engage\n"
        "  --serve PORT      Run HTTP/SSE server on PORT\n"
        "  --timing          Print per-token timing\n"
        "  --gguf-info       Print GGUF tensor/metadata summary and exit\n"
        "  --layers N        Limit inference to the first N layers\n"
        "  --help            Show this help\n",
        prog);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_dir = NULL;
        const char *prompt_text = NULL;
        int max_tokens = 20;
        int active_k = 0;  // 0 = use config default
        int serve_port = 0;
        int timing = 0;
        int thermal_k = 0;
        double thermal_proj_ms = 85.0;
        int thermal_gen = 16;
        int max_layers = 0; // 0 = all layers
        static struct option long_options[] = {
            {"model",   required_argument, 0, 'm'},
            {"prompt",  required_argument, 0, 'p'},
            {"tokens",  required_argument, 0, 't'},
            {"k",       required_argument, 0, 'k'},
            {"thermal-k", required_argument, 0, 'K'},
            {"thermal-proj-ms", required_argument, 0, 'P'},
            {"thermal-gen", required_argument, 0, 'G'},
            {"serve",   required_argument, 0, 'S'},
            {"timing",  no_argument,       0, 'T'},
            {"gguf-info", no_argument,     0, 'I'},
            {"layers",   required_argument, 0, 'L'},
            {"help",    no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };
        int gguf_info = 0;

        int c;
        while ((c = getopt_long(argc, argv, "m:p:t:k:K:P:G:S:TI:L:h", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_dir = optarg; break;
                case 'p': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': active_k = atoi(optarg); break;
                case 'K': thermal_k = atoi(optarg); break;
                case 'L': max_layers = atoi(optarg); break;
                case 'P': thermal_proj_ms = atof(optarg); break;
                case 'G': thermal_gen = atoi(optarg); break;
                case 'S': serve_port = atoi(optarg); break;
                case 'T': timing = 1; break;
                case 'I': gguf_info = 1; break;

                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        if (!prompt_text && serve_port == 0 && !gguf_info) {
            prompt_text = "Hello, what is";
        }

        // ---- GGUF info mode (parse and dump, no inference) ----
        if (gguf_info && model_dir) {
            GGUFFile *gf = gguf_open(model_dir);
            if (!gf) { fprintf(stderr, "Failed to open GGUF: %s\n", model_dir); return 1; }
            gguf_print_summary(gf);
            gguf_close(gf);
            return 0;
        }

        ModelConfig cfg;
        MetalCtx *ctx = NULL;
        GGUFFile *gf = NULL;
        FormatProvider *fp = NULL;
        Engine *eng = NULL;

        {
            // ==== GGUF loading path ====
            if (!model_dir || !strstr(model_dir, ".gguf")) {
                fprintf(stderr, "ERROR: Model must be a .gguf file\n");
                return 1;
            }
            gf = gguf_open(model_dir);
            if (!gf) { fprintf(stderr, "ERROR: Cannot open GGUF: %s\n", model_dir); return 1; }

            // Build config from GGUF metadata
            memset(&cfg, 0, sizeof(cfg));
            snprintf(cfg.name, sizeof(cfg.name), "%s", gf->arch);
            cfg.hidden_dim = gf->hidden_dim;
            cfg.num_layers = gf->num_layers;
            cfg.ffn_type = gf->ffn_type;
            cfg.num_experts = gf->num_experts;
            cfg.num_experts_per_tok = gf->num_experts_per_tok;
            cfg.moe_intermediate = gf->moe_intermediate;
            cfg.vocab_size = gf->vocab_size;
            cfg.context_length = gf->context_length;

            // Attention config — detect from GGUF metadata
            cfg.num_attn_heads = gf->num_attn_heads;
            cfg.num_kv_heads = gf->num_kv_heads;
            cfg.rope_theta = gf->rope_theta;
            cfg.rms_norm_eps = gf->rms_norm_eps;

            // Detect layer types from GGUF tensors
            cfg.layer_types = calloc(cfg.num_layers, sizeof(AttnLayerType));
            cfg.num_full_attn_layers = 0;
            cfg.num_linear_layers = 0;
            char tname[128];
            for (int i = 0; i < cfg.num_layers; i++) {
                snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", i);
                GGUFTensorInfo *ti = gguf_find_tensor(gf, tname);
                if (ti) {
                    cfg.layer_types[i] = ATTN_FULL;
                    cfg.num_full_attn_layers++;
                } else {
                    cfg.layer_types[i] = ATTN_LINEAR;
                    cfg.num_linear_layers++;
                }
            }
            fprintf(stderr, "[main] GGUF layer detection: %d full + %d linear\n",
                    cfg.num_full_attn_layers, cfg.num_linear_layers);

            // Prefer GGUF metadata for attention geometry, then infer from tensor shapes.
            cfg.head_dim = gf->attn_key_length;
            if (cfg.head_dim == 0 && cfg.num_full_attn_layers > 0) {
                GGUFTensorInfo *ki = find_first_layer_tensor(gf, &cfg, ATTN_FULL, "attn_k.weight");
                if (ki && ki->n_dims >= 2 && cfg.num_kv_heads > 0) {
                    cfg.head_dim = (int)ki->dims[1] / cfg.num_kv_heads;
                    fprintf(stderr, "[main] K tensor shape: out=%d / kv_heads=%d -> head_dim=%d\n",
                            (int)ki->dims[1], cfg.num_kv_heads, cfg.head_dim);
                }
            }
            if (cfg.head_dim == 0 && cfg.num_attn_heads > 0) {
                cfg.head_dim = cfg.hidden_dim / cfg.num_attn_heads;
            }
            if (cfg.head_dim <= 0) {
                fprintf(stderr, "ERROR: Could not infer attention head_dim from GGUF metadata or tensors\n");
                return 1;
            }
            if (gf->attn_value_length > 0 && gf->attn_value_length != cfg.head_dim) {
                fprintf(stderr,
                        "ERROR: attention.value_length=%d differs from key/head_dim=%d; "
                        "variable full-attention V dims are not yet supported\n",
                        gf->attn_value_length, cfg.head_dim);
                return 1;
            }
            if (cfg.rms_norm_eps <= 0.0f) {
                fprintf(stderr, "ERROR: Missing RMS norm epsilon in GGUF metadata\n");
                return 1;
            }

            if (gf->rope_dimension_count > 0) {
                cfg.partial_rotary = (float)gf->rope_dimension_count / (float)cfg.head_dim;
            } else if (cfg.num_full_attn_layers == 0) {
                cfg.partial_rotary = 0.0f;
            } else {
                fprintf(stderr, "ERROR: Missing rope.dimension_count for model with full-attention layers\n");
                return 1;
            }
            if (cfg.partial_rotary > 0.0f && cfg.rope_theta <= 0.0f) {
                fprintf(stderr, "ERROR: Missing rope.freq_base for model with rotary attention\n");
                return 1;
            }

            cfg.linear_num_k_heads = gf->ssm_group_count;
            cfg.linear_num_v_heads = gf->ssm_time_step_rank;
            cfg.linear_key_dim = gf->ssm_state_size;
            cfg.conv_kernel_size = gf->ssm_conv_kernel;

            int linear_total_value = gf->ssm_inner_size;
            if (cfg.num_linear_layers > 0) {
                GGUFTensorInfo *norm_ti = find_first_layer_tensor(gf, &cfg, ATTN_LINEAR, "ssm_norm.weight");
                GGUFTensorInfo *alpha_ti = find_first_layer_tensor(gf, &cfg, ATTN_LINEAR, "ssm_alpha.weight");
                GGUFTensorInfo *qkv_ti = find_first_layer_tensor(gf, &cfg, ATTN_LINEAR, "attn_qkv.weight");
                GGUFTensorInfo *out_ti = find_first_layer_tensor(gf, &cfg, ATTN_LINEAR, "ssm_out.weight");
                GGUFTensorInfo *conv_ti = find_first_layer_tensor(gf, &cfg, ATTN_LINEAR, "ssm_conv1d.weight");

                if (cfg.linear_key_dim <= 0 && norm_ti && norm_ti->n_dims >= 1) {
                    cfg.linear_key_dim = (int)norm_ti->dims[0];
                }
                if (cfg.linear_num_v_heads <= 0 && alpha_ti && alpha_ti->n_dims >= 2) {
                    cfg.linear_num_v_heads = (int)alpha_ti->dims[1];
                }
                if (linear_total_value <= 0 && out_ti && out_ti->n_dims >= 2) {
                    linear_total_value = (int)out_ti->dims[0];
                }
                if (cfg.conv_kernel_size <= 0 && conv_ti && conv_ti->n_dims >= 1) {
                    cfg.conv_kernel_size = (int)conv_ti->dims[0];
                }
                if (cfg.linear_num_k_heads <= 0 &&
                    qkv_ti && qkv_ti->n_dims >= 2 &&
                    cfg.linear_key_dim > 0 &&
                    cfg.linear_num_v_heads > 0 &&
                    linear_total_value > 0) {
                    int qkv_dim = (int)qkv_ti->dims[1];
                    cfg.linear_num_k_heads = (qkv_dim - linear_total_value) /
                                             (2 * cfg.linear_key_dim);
                    fprintf(stderr, "[main] GGUF linear attn: qkv_dim=%d v_heads=%d k_heads=%d\n",
                            qkv_dim, cfg.linear_num_v_heads, cfg.linear_num_k_heads);
                }
            }

            if (cfg.linear_num_v_heads > 0 && linear_total_value > 0) {
                cfg.linear_value_dim = linear_total_value / cfg.linear_num_v_heads;
            }

            if (cfg.num_linear_layers > 0 &&
                (cfg.linear_num_v_heads <= 0 || cfg.linear_num_k_heads <= 0 ||
                 cfg.linear_key_dim <= 0 || cfg.linear_value_dim <= 0 ||
                 cfg.conv_kernel_size <= 0)) {
                fprintf(stderr,
                        "ERROR: Could not infer linear-attention config "
                        "(k_heads=%d v_heads=%d key_dim=%d value_dim=%d conv=%d)\n",
                        cfg.linear_num_k_heads, cfg.linear_num_v_heads,
                        cfg.linear_key_dim, cfg.linear_value_dim, cfg.conv_kernel_size);
                return 1;
            }

            // Derive shared_intermediate from shared expert gate tensor shape
            if (cfg.ffn_type == FFN_MOE) {
                GGUFTensorInfo *sg = gguf_find_tensor(gf, "blk.0.ffn_gate_shexp.weight");
                if (sg && sg->n_dims >= 2) {
                    cfg.shared_intermediate = (int)sg->dims[1];
                    fprintf(stderr, "[main] GGUF shared_intermediate=%d\n", cfg.shared_intermediate);
                } else {
                    cfg.shared_intermediate = cfg.moe_intermediate; // fallback
                }
            } else {
                cfg.num_experts = 0;
                cfg.num_experts_per_tok = 0;
                cfg.shared_intermediate = 0;
            }

            for (int i = 0; i < 4; i++) cfg.eos_tokens[i] = -1;
            cfg.chat_start_token = -1;
            cfg.chat_end_token = -1;
            cfg.think_start_token = -1;
            cfg.think_end_token = -1;
            cfg.chat_prefill_think = false;

            if (cfg.ffn_type == FFN_MOE && active_k > 0) {
                cfg.num_experts_per_tok = active_k;
            } else if (cfg.ffn_type == FFN_DENSE && active_k > 0) {
                fprintf(stderr, "[main] Ignoring --k for dense FFN model\n");
            }

            model_config_init_derived(&cfg);
            if (cfg.num_full_attn_layers > 0 && cfg.q_heads_per_kv <= 0) {
                fprintf(stderr, "ERROR: Invalid grouped-query ratio: q_heads=%d kv_heads=%d\n",
                        cfg.num_attn_heads, cfg.num_kv_heads);
                return 1;
            }
            if (cfg.num_linear_layers > 0 && cfg.linear_v_heads_per_k <= 0) {
                fprintf(stderr, "ERROR: Invalid linear-attention head ratio: v_heads=%d k_heads=%d\n",
                        cfg.linear_num_v_heads, cfg.linear_num_k_heads);
                return 1;
            }

            if (max_layers > 0 && max_layers < cfg.num_layers) {
                fprintf(stderr, "[main] Limiting to %d layers (was %d)\n", max_layers, cfg.num_layers);
                cfg.num_layers = max_layers;
                // Recount layer types
                cfg.num_full_attn_layers = 0;
                cfg.num_linear_layers = 0;
                for (int i = 0; i < cfg.num_layers; i++) {
                    if (cfg.layer_types[i] == ATTN_FULL) cfg.num_full_attn_layers++;
                    else cfg.num_linear_layers++;
                }
            }

            fprintf(stderr, "[main] GGUF config: %s, %d layers (%d full + %d linear), "
                    "hidden=%d, ffn=%s, experts=%d, K=%d\n",
                    cfg.name, cfg.num_layers, cfg.num_full_attn_layers,
                    cfg.num_linear_layers, cfg.hidden_dim,
                    cfg.ffn_type == FFN_MOE ? "moe" : "dense",
                    cfg.num_experts,
                    cfg.num_experts_per_tok);

            // Initialize Metal
            ctx = metal_setup(&cfg);
            if (!ctx) {
                fprintf(stderr, "ERROR: Metal required for GGUF inference\n");
                return 1;
            }

            // Create format provider (wraps GGUF mmap as Metal buffer)
            fp = format_provider_open_gguf(gf, ctx);
            if (!fp) {
                fprintf(stderr, "ERROR: Failed to create format provider\n");
                return 1;
            }

            // Load tokenizer from the model directory when available.
            // Text tokenization still uses vocab.bin today, but it can be
            // generated from either GGUF metadata or Hugging Face assets.
            int tok_loaded = tokenizer_init(model_dir);
            if (tok_loaded != 0) {
                fprintf(stderr, "ERROR: Could not load tokenizer\n");
                return 1;
            }

            // Resolve special tokens from tokenizer.chat_template structure when available.
            int eos_count = 0;
            resolve_chat_tokens_from_template(&cfg, gf->chat_template);

            add_unique_token(cfg.eos_tokens, 4, &eos_count, gf->eos_token_id);
            add_unique_token(cfg.eos_tokens, 4, &eos_count, cfg.chat_end_token);
            add_unique_token(cfg.eos_tokens, 4, &eos_count, tokenizer_find_token("<|endoftext|>"));

            fprintf(stderr, "[main] GGUF: creating engine...\n");
            eng = engine_create(&cfg, ctx,
                                active_k > 0 ? active_k : 0);
            eng->gf = gf;

            // Build format-agnostic tensor cache from GGUF
            eng->tensor_cache = build_tensor_cache_gguf(gf, fp->model_buf,
                                                        ctx, &cfg, &eng->globals);

            // Pre-resolve expert layer refs (avoids per-token GGUF hash lookups)
            if (cfg.ffn_type == FFN_MOE) {
                eng->expert_layer_cache = calloc(cfg.num_layers, sizeof(ExpertLayerRef));
                for (int i = 0; i < cfg.num_layers; i++) {
                    eng->expert_layer_cache[i] =
                        format_resolve_expert_layer(fp, i, cfg.num_experts);
                }
                fprintf(stderr, "[main] Pre-resolved %d expert layer refs\n", cfg.num_layers);
            }

        }
        if (thermal_k > 0 && cfg.ffn_type == FFN_MOE) {
            eng->thermal.enabled = true;
            eng->thermal.hot_k = thermal_k;
            eng->thermal.proj_threshold_ms = thermal_proj_ms;
            eng->thermal.min_gen = thermal_gen;
            printf("[main] Thermal-K enabled: K→%d when proj EMA > %.0fms (after %d tokens)\n",
                   thermal_k, thermal_proj_ms, thermal_gen);
        } else if (thermal_k > 0) {
            fprintf(stderr, "[main] Ignoring --thermal-k for dense FFN model\n");
        }
        if (cfg.ffn_type == FFN_MOE) {
            printf("[main] Engine ready: %s, %d layers, MoE K=%d\n",
                   cfg.name, cfg.num_layers, eng->active_experts);
        } else {
            printf("[main] Engine ready: %s, %d layers, dense FFN\n",
                   cfg.name, cfg.num_layers);
        }

        // ---- Run ----
        if (serve_port > 0) {
            serve_loop(eng, serve_port);
        } else {
            // Tokenize prompt (or use raw token IDs if prompt starts with "[")
            PromptTokens *pt = NULL;
            if (prompt_text[0] == '[') {
                // Parse raw token IDs: "[248045,846,198]"
                pt = calloc(1, sizeof(PromptTokens));
                int cap = 64;
                pt->ids = calloc(cap, sizeof(int));
                const char *p = prompt_text + 1;
                while (*p && *p != ']') {
                    if (pt->count == cap) {
                        cap *= 2;
                        pt->ids = realloc(pt->ids, (size_t)cap * sizeof(int));
                    }
                    pt->ids[pt->count++] = atoi(p);
                    while (*p && *p != ',' && *p != ']') p++;
                    if (*p == ',') p++;
                }
            } else {
                pt = tokenizer_encode(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to tokenize prompt\n");
                    return 1;
                }
            }

            printf("[main] Prompt: \"%s\" (%d tokens)\n", prompt_text, pt->count);
            int needed_seq = pt->count + (max_tokens > 0 ? max_tokens : 0);
            if (!metal_ensure_kv_capacity(ctx, &cfg, needed_seq > 0 ? needed_seq : 1)) {
                fprintf(stderr, "ERROR: Failed to size KV cache for %d tokens\n", needed_seq);
                prompt_tokens_free(pt);
                return 1;
            }

            // Prefill
            double t_start = now_ms();
            int next_token = 0;
            for (int i = 0; i < pt->count; i++) {
                next_token = engine_step(eng, pt->ids[i]);
            }
            double ttft = now_ms() - t_start;
            printf("TTFT: %.1f ms\n", ttft);

            // Generate
            double gen_start = now_ms();
            int generated = 0;
            for (int i = 0; i < max_tokens; i++) {
                if (is_eos_token(&cfg, next_token)) break;

                const char *text = tokenizer_decode(next_token);
                printf("%s", text);
                fflush(stdout);

                double tok_start = timing ? now_ms() : 0;
                next_token = engine_step(eng, next_token);
                generated++;

                if (timing) {
                    fprintf(stderr, "[tok %d] %.1f ms\n", generated, now_ms() - tok_start);
                }
            }
            printf("\n");

            double gen_ms = now_ms() - gen_start;
            printf("Generation: %.3f s (%.2f tok/s)\n",
                   gen_ms / 1000.0,
                   gen_ms > 0 ? generated * 1000.0 / gen_ms : 0);

            prompt_tokens_free(pt);
        }

        // ---- Cleanup ----
        engine_free(eng);
        if (fp) format_provider_close(fp);
        if (gf) gguf_close(gf);
        metal_free(ctx);
        free(cfg.layer_types);

        return 0;
    }
}
