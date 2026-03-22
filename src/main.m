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
#include <sys/time.h>

#include "orome.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model DIR      Model directory (weights + config)\n"
        "  --prompt TEXT     Prompt text\n"
        "  --tokens N        Max tokens to generate (default: 20)\n"
        "  --k N             Active experts per layer (default: from config)\n"
        "  --2bit            Use 2-bit quantized experts\n"
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
        int use_2bit = 0;
        int serve_port = 0;
        int timing = 0;
        static struct option long_options[] = {
            {"model",   required_argument, 0, 'm'},
            {"prompt",  required_argument, 0, 'p'},
            {"tokens",  required_argument, 0, 't'},
            {"k",       required_argument, 0, 'k'},
            {"2bit",    no_argument,       0, '2'},
            {"serve",   required_argument, 0, 'S'},
            {"timing",  no_argument,       0, 'T'},
            {"help",    no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "m:p:t:k:2S:Th", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_dir = optarg; break;
                case 'p': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': active_k = atoi(optarg); break;
                case '2': use_2bit = 1; break;
                case 'S': serve_port = atoi(optarg); break;
                case 'T': timing = 1; break;

                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        if (!prompt_text && serve_port == 0) {
            prompt_text = "Hello, what is";
        }

        // ---- Load model config ----
        ModelConfig cfg;
        if (model_dir && model_config_load(&cfg, model_dir) == 0) {
            printf("[main] Config loaded from %s\n", model_dir);
        } else {
            printf("[main] Using default Qwen3.5-35B config\n");
            model_config_qwen35_35b(&cfg);
        }

        if (active_k > 0) cfg.num_experts_per_tok = active_k;

        // ---- Initialize Metal ----
        MetalCtx *ctx = metal_setup(&cfg);
        if (!ctx) {
            fprintf(stderr, "WARNING: Metal unavailable, running CPU-only\n");
        }

        // ---- Load weights ----
        // Try to find weight files
        char weights_path[512], manifest_path[512], vocab_path[512];
        if (model_dir) {
            snprintf(weights_path, sizeof(weights_path), "%s/model_weights.bin", model_dir);
            snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json", model_dir);
            snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.bin", model_dir);
        } else {
            strncpy(weights_path, "model_weights.bin", sizeof(weights_path));
            strncpy(manifest_path, "model_weights.json", sizeof(manifest_path));
            strncpy(vocab_path, "vocab.bin", sizeof(vocab_path));
        }

        WeightFile *wf = weights_open(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Cannot load weights from %s\n", weights_path);
            return 1;
        }

        if (ctx) metal_set_weights(ctx, wf);

        // ---- Detect layer types from actual weights ----
        model_config_detect_layers(&cfg, wf);

        // ---- Load tokenizer ----
        if (tokenizer_init(model_dir) != 0) {
            fprintf(stderr, "WARNING: Tokenizer not loaded, decode will show token IDs\n");
        }

        Vocabulary *vocab = vocab_load(vocab_path);

        // ---- Open expert files ----
        ExpertFiles *ef = expert_files_open(&cfg, model_dir ? model_dir : ".", NULL);

        // ---- Create engine ----
        QuantType quant = use_2bit ? QUANT_2BIT : QUANT_4BIT;
        Engine *eng = engine_create(&cfg, wf, ctx, ef, quant,
                                    active_k > 0 ? active_k : 0);

        printf("[main] Engine ready: %s, %d layers, K=%d, %s\n",
               cfg.name, cfg.num_layers, eng->active_experts,
               use_2bit ? "2-bit" : "4-bit");

        // ---- Run ----
        if (serve_port > 0) {
            serve_loop(eng, vocab, serve_port);
        } else {
            // Tokenize prompt
            PromptTokens *pt = tokenizer_encode(prompt_text);
            if (!pt) {
                fprintf(stderr, "ERROR: Failed to tokenize prompt\n");
                return 1;
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
        expert_files_close(ef, &cfg);
        if (vocab) vocab_free(vocab);
        weights_close(wf);
        metal_free(ctx);
        free(cfg.layer_types);

        return 0;
    }
}
