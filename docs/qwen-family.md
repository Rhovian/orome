# Qwen3.5 Family Notes

## Model Variants

| Model | Type | FFN | Layers | Hidden | Quantization |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | Dense | SwiGLU | 64 | 4096 | Q8_0 |
| Qwen3.5-27B | Dense (hybrid attn) | SwiGLU | 64 (16 full + 48 linear) | 5120 | Q4_K_M |
| Qwen3.5-35B-A3B | MoE | Routed experts | 64 | 4096 | Q4_K_S |

## Architecture

Orome distinguishes MoE vs dense FFN from GGUF tensor names. The 35B-A3B uses routed MoE experts; the 9B and 27B use the dense SwiGLU FFN path.

The 27B uses a hybrid attention architecture: 16 full attention layers (standard QKV + RoPE + KV cache + softmax) interleaved with 48 linear attention layers (GatedDeltaNet with conv1d + L2-normed QK + recurrence state). This is handled by the `engine_qwen35_hybrid` backend.

## Tokenizer

All Qwen3.5 models share the same BPE tokenizer (248320 vocab). Orome loads it directly from GGUF metadata — no separate tokenizer files needed.
