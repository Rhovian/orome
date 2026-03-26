/*
 * main.m — CLI entry point for orome inference engine.
 *
 * Usage:
 *   ./orome --model DIR --prompt "Hello" --tokens 20
 *   ./orome --model DIR --serve 8080
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "orome.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model DIR      Model directory (weights + config)\n"
        "  --prompt TEXT     Prompt text\n"
        "  --tokens N        Max tokens to generate (default: 20)\n"
        "  --k N             Active experts per layer (default: from config)\n"
        "  --thermal-k N     Reduce K to N when thermal throttling engages\n"
        "  --thermal-proj-ms F  Engage thermal-K when step EMA exceeds F ms\n"
        "  --thermal-gen N   Minimum tokens before thermal-K can engage\n"
        "  --serve PORT      Run HTTP/SSE server on PORT\n"
        "  --timing          Print per-token timing\n"
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
        while ((c = getopt_long(argc, argv, "m:p:t:k:2K:P:G:H:ES:Th", long_options, NULL)) != -1) {
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
            cfg.num_experts = gf->num_experts;
            cfg.num_experts_per_tok = gf->num_experts_per_tok;
            cfg.moe_intermediate = gf->moe_intermediate;
            cfg.vocab_size = gf->vocab_size;

            // Attention config — detect from GGUF metadata
            cfg.num_attn_heads = gf->num_attn_heads;
            cfg.num_kv_heads = gf->num_kv_heads;
            cfg.rope_theta = gf->rope_theta > 0 ? gf->rope_theta : 1000000.0f;

            // Derive head_dim from K tensor shape (more reliable than Q since Q may be gated)
            cfg.head_dim = 0;
            for (int i = 0; i < cfg.num_layers && cfg.head_dim == 0; i++) {
                char kname[128];
                snprintf(kname, sizeof(kname), "blk.%d.attn_k.weight", i);
                GGUFTensorInfo *ki = gguf_find_tensor(gf, kname);
                if (ki && ki->n_dims >= 2 && cfg.num_kv_heads > 0) {
                    cfg.head_dim = (int)ki->dims[1] / cfg.num_kv_heads;
                    fprintf(stderr, "[main] K tensor shape: out=%d / kv_heads=%d → head_dim=%d\n",
                            (int)ki->dims[1], cfg.num_kv_heads, cfg.head_dim);
                }
            }
            if (cfg.head_dim == 0) cfg.head_dim = cfg.hidden_dim / cfg.num_attn_heads;
            if (cfg.head_dim == 0) cfg.head_dim = 256; // fallback

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

            // Linear attention params — infer from GGUF tensor shapes
            cfg.linear_key_dim = 128;
            cfg.linear_value_dim = 128;
            cfg.partial_rotary = 0.25;
            cfg.conv_kernel_size = 4;
            cfg.rms_norm_eps = 1e-6f;
            // attn_qkv output dim = v_heads*(key+value) + k_heads*key
            // For Qwen3.5: qkv_dim = v_heads*256 + k_heads*128
            // We know k_heads from metadata (num_kv_heads for linear layers can differ)
            // Infer from attn_qkv tensor shape
            {
                char tname[128];
                // Find first linear attention layer — derive v_heads and k_heads from tensor shapes
                for (int i = 0; i < cfg.num_layers; i++) {
                    if (cfg.layer_types[i] == ATTN_LINEAR) {
                        // alpha tensor has dims[1] = v_heads
                        snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", i);
                        GGUFTensorInfo *alpha_ti = gguf_find_tensor(gf, tname);
                        snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", i);
                        GGUFTensorInfo *qkv_ti = gguf_find_tensor(gf, tname);
                        if (alpha_ti && alpha_ti->n_dims >= 2 && qkv_ti && qkv_ti->n_dims >= 2) {
                            int qkv_dim = (int)qkv_ti->dims[1];
                            cfg.linear_num_v_heads = (int)alpha_ti->dims[1];
                            // conv_dim = 2 * k_heads * key_dim + v_heads * value_dim
                            cfg.linear_num_k_heads = (qkv_dim - cfg.linear_num_v_heads * cfg.linear_value_dim)
                                                   / (2 * cfg.linear_key_dim);
                            fprintf(stderr, "[main] GGUF linear attn: qkv_dim=%d v_heads=%d k_heads=%d\n",
                                    qkv_dim, cfg.linear_num_v_heads, cfg.linear_num_k_heads);
                        }
                        break;
                    }
                }
            }

            // Derive shared_intermediate from shared expert gate tensor shape
            {
                GGUFTensorInfo *sg = gguf_find_tensor(gf, "blk.0.ffn_gate_shexp.weight");
                if (sg && sg->n_dims >= 2) {
                    cfg.shared_intermediate = (int)sg->dims[1];
                    fprintf(stderr, "[main] GGUF shared_intermediate=%d\n", cfg.shared_intermediate);
                } else {
                    cfg.shared_intermediate = cfg.moe_intermediate; // fallback
                }
            }

            // EOS tokens (Qwen3.5)
            cfg.eos_tokens[0] = 248046;
            cfg.eos_tokens[1] = 248044;
            cfg.eos_tokens[2] = -1;
            cfg.think_end_token = 248069;

            if (active_k > 0) cfg.num_experts_per_tok = active_k;

            // Set full_attn_interval and offset so init_derived generates correct layer types
            if (cfg.num_full_attn_layers > 0) {
                cfg.full_attn_interval = cfg.num_layers / cfg.num_full_attn_layers;
                // Find the first full attention layer to determine offset
                for (int i = 0; i < cfg.num_layers; i++) {
                    if (cfg.layer_types[i] == ATTN_FULL) {
                        cfg.full_attn_offset = i % cfg.full_attn_interval;
                        break;
                    }
                }
            }
            model_config_init_derived(&cfg);

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
                    "hidden=%d, experts=%d, K=%d\n",
                    cfg.name, cfg.num_layers, cfg.num_full_attn_layers,
                    cfg.num_linear_layers, cfg.hidden_dim, cfg.num_experts,
                    cfg.num_experts_per_tok);

            // Initialize Metal
            ctx = metal_setup(&cfg);
            if (!ctx) {
                fprintf(stderr, "ERROR: Metal required for GGUF inference\n");
                return 1;
            }

            // Create format provider (wraps GGUF mmap as Metal buffer)
            fp = format_provider_open_gguf(gf, ctx);
            if (fp) ctx->buf_weights = fp->model_buf;  // share Metal buffer with ctx
            if (!fp) {
                fprintf(stderr, "ERROR: Failed to create format provider\n");
                return 1;
            }

            // Load tokenizer (try model directory, then fallback paths)
            // GGUF embeds vocab but we use our external tokenizer for now
            // Strip .gguf filename to get directory
            char tok_dir[512];
            strncpy(tok_dir, model_dir, sizeof(tok_dir));
            char *last_slash = strrchr(tok_dir, '/');
            if (last_slash) *last_slash = '\0';
            if (tokenizer_init(tok_dir) != 0) {
                // Try the legacy 35B model dir for tokenizer
                tokenizer_init("/Users/j/models/Qwen3.5-35B-A3B-4bit");
            }

            fprintf(stderr, "[main] GGUF: creating engine...\n");
            eng = engine_create(&cfg, ctx,
                                active_k > 0 ? active_k : 0);
            eng->gf = gf;

            // Build format-agnostic tensor cache from GGUF
            eng->tensor_cache = build_tensor_cache_gguf(gf, ctx, &cfg, &eng->globals);

            // Pre-resolve expert layer refs (avoids per-token GGUF hash lookups)
            eng->expert_layer_cache = calloc(cfg.num_layers, sizeof(ExpertLayerRef));
            for (int i = 0; i < cfg.num_layers; i++) {
                eng->expert_layer_cache[i] =
                    format_resolve_expert_layer(fp, i, cfg.num_experts);
            }
            fprintf(stderr, "[main] Pre-resolved %d expert layer refs\n", cfg.num_layers);

        }
        if (thermal_k > 0) {
            eng->thermal.enabled = true;
            eng->thermal.hot_k = thermal_k;
            eng->thermal.proj_threshold_ms = thermal_proj_ms;
            eng->thermal.min_gen = thermal_gen;
            printf("[main] Thermal-K enabled: K→%d when proj EMA > %.0fms (after %d tokens)\n",
                   thermal_k, thermal_proj_ms, thermal_gen);
        }
        printf("[main] Engine ready: %s, %d layers, K=%d\n",
               cfg.name, cfg.num_layers, eng->active_experts);

        // ---- Run ----
        if (serve_port > 0) {
            serve_loop(eng, serve_port);
        } else {
            // Tokenize prompt (or use raw token IDs if prompt starts with "[")
            PromptTokens *pt = NULL;
            if (prompt_text[0] == '[') {
                // Parse raw token IDs: "[248045,846,198]"
                pt = calloc(1, sizeof(PromptTokens));
                pt->ids = calloc(256, sizeof(int));
                const char *p = prompt_text + 1;
                while (*p && *p != ']' && pt->count < 256) {
                    pt->ids[pt->count++] = atoi(p);
                    while (*p && *p != ',' && *p != ']') p++;
                    if (*p == ',') p++;
                }
                fprintf(stderr, "[main] Raw token IDs: %d tokens\n", pt->count);
            } else {
                pt = tokenizer_encode(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to tokenize prompt\n");
                    return 1;
                }
            }

            printf("[main] Prompt: \"%s\" (%d tokens)\n", prompt_text, pt->count);

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
            fprintf(stderr, "[gguf] first predicted token: %d '%s' (eos=%d)\n",
                    next_token, tokenizer_decode(next_token), is_eos_token(&cfg, next_token));
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
