# 27B Performance Gap Plan

## Why This Exists

We should not keep running blind 27B autoresearch sessions while `llama.cpp`
is materially faster on that model and we do not yet understand why.

The fixed-token comparison harness currently shows:

- `9B`: Orome `35.32 tok/s` vs `llama.cpp` `31.22 tok/s`
- `27B`: Orome `9.61 tok/s` vs `llama.cpp` `14.38 tok/s`
- `35B`: Orome `65.15 tok/s` vs `llama.cpp` `51.34 tok/s`

Comparison basis:

- Orome: `ceb9d82`
- llama.cpp: `c46758d`
- Method: same machine, same GGUF files, prompt `Hello`, `100` generated
  tokens, context size `256`, `5`-trial median

This means the problem is not "Orome is generally slower than llama.cpp."
It is much more likely that `27B` is exposing one or two path-specific gaps.

## What We Know

### 1. 27B is not just "another dense model"

`Qwen3.5-27B-Q4_K_M.gguf` is a hybrid dense model with:

- `64` layers
- `16` full-attention layers
- `48` linear / recurrent gated-delta-net layers

That is a very different hot path mix from the simpler all-`Q8_0`-leaning `9B`
case.

Relevant code:

- `src/main.m`
- `src/engine.m`

### 2. 27B heavily exercises Q5_K and Q6_K in expensive tensors

The `27B` tensor mix is:

- `Q4_K`: `263` tensors
- `Q5_K`: `96` tensors
- `Q6_K`: `43` tensors
- `Q8_0`: `96` tensors

Important hot tensors include:

- `attn_qkv.weight = Q5_K`
- `ssm_out.weight = Q5_K`
- `ffn_down.weight = Q6_K`
- `output.weight = Q6_K`
- `token_embd.weight = Q4_K`

That is very different from `9B`, which is overwhelmingly `Q8_0`, and from
`35B`, whose main wins come from a different MoE-heavy mix.

Relevant code:

- `src/gguf.m`
- `src/format.m`
- `src/shaders.metal`

### 3. Orome still uses a one-size-fits-all quantized matvec dispatch

Today Orome dispatches `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` matvecs with the
same high-level threadgroup geometry:

- `FORMAT_ROWS_PER_TG = 16`
- `FORMAT_TWO_ROW_MULTIPLIER = 2`
- fixed `512` threads per threadgroup

The only format-specific special case is the `Q8_0` single-tile fast path.

Relevant code:

- `src/format.m`

### 4. llama.cpp does not use the same geometry

`llama.cpp`'s Metal backend uses per-format tuning:

- `Q4_K`: `N_R0 = 2`, `N_SG = 2`
- `Q5_K`: `N_R0 = 1`, `N_SG = 2`
- `Q6_K`: `N_R0 = 2`, `N_SG = 2`
- `Q8_0`: `N_R0 = 2`, `N_SG = 4`

That `Q5_K = 1 row per simdgroup` choice is especially interesting because
`27B` leans heavily on `Q5_K` in its recurrent and attention projections.

Relevant code:

- `../llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h`
- `../llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`

### 5. Orome's Q5_K and Q6_K kernels are simpler than llama.cpp's

Orome's current `Q5_K` and `Q6_K` kernels:

- always stage `x` into threadgroup memory
- always use the same broad 2-row pattern
- do not appear to have format-specific geometry or work partitioning

`llama.cpp`'s Metal kernels are more specialized and more obviously tuned for
the format layout.

Relevant code:

- Orome:
  - `src/shaders.metal`
- llama.cpp:
  - `../llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`

### 6. llama.cpp is taking a fused Gated Delta Net path on 27B

The `llama.cpp` run logs explicitly report:

- `fused Gated Delta Net (autoregressive) enabled`
- `fused Gated Delta Net (chunked) enabled`

Orome's linear-attention path is still orchestrated as multiple explicit
phases:

- input norm
- QKV/Z/A/B projections
- conv1d
- decay/beta
- delta-net
- gated RMS norm
- out projection

This is a plausible secondary gap after the quant kernels.

Relevant code:

- Orome:
  - `src/engine.m`
- llama.cpp:
  - `../llama.cpp/src/models/qwen3next.cpp`

## Working Hypothesis

Current likely ranking of causes:

1. Primary gap: `Q5_K/Q6_K` Metal matvec performance on 27B-sized shapes
2. Secondary gap: hybrid linear-attention / recurrent Gated Delta Net fusion
3. Unlikely main issue: tokenizer, dense FFN logic, or generic output quality

## Plan

### Phase 1: Lock the measurement loop

Use the existing comparison harness as the outer truth source:

```bash
python3 tools/compare_orome_llama.py --models 9B 27B 35B --tokens 100 --trials 5 --json
```

Keep `9B` and `35B` in the loop as guardrails while working on `27B`.

Do not resume blind 27B-only autoresearch until the major gap is explained.

### Phase 2: Build targeted 27B microbenches

We need shape-specific measurements for the expensive kernels rather than only
whole-model tok/s.

Initial target shapes:

- `attn_qkv`: `Q5_K`, `5120 x 10240`
- `ssm_out`: `Q5_K`, `6144 x 5120`
- `ffn_down`: `Q6_K`, `17408 x 5120`
- `output.weight`: `Q6_K`, `5120 x 248320` if practical

Goal:

- determine how much of the whole-model gap is explained by `Q5_K/Q6_K`
  matvec throughput alone

### Phase 3: Make quant dispatch genuinely per-format

Replace the current one-size-fits-all matvec dispatch policy with
format-specific tuning.

Start here:

- teach `src/format.m` to choose geometry per format
- match llama.cpp's `Q5_K` and `Q6_K` row/simdgroup choices first
- keep the existing `Q8_0` single-tile specialization

Constraints:

- no 27B-specific tensor-shape hacks in shared logic
- tuning should be driven by format and shape class, not model name

### Phase 4: Re-test whole-model 27B after quant work

After each meaningful quant-kernel change:

1. rebuild
2. rerun the fixed-token comparison
3. rerun the standard model smoke/canary path

If the bulk of the 27B gap closes here, resume normal autoresearch from the
new baseline.

### Phase 5: If needed, investigate the linear / recurrent path

If `Q5_K/Q6_K` tuning does not explain most of the gap, then profile the
linear-attention stack directly.

Focus areas:

- projection scheduling
- conv1d cost
- decay/beta path
- delta-net implementation
- gated RMS norm + out projection

Questions to answer:

- how much time is spent in linear layers vs full-attention layers?
- how much does `llama.cpp` gain from fused Gated Delta Net vs our phased path?
- can Orome reduce barriers / dispatches here without reintroducing
  model-shaped assumptions?

## Guardrails

- Preserve generic shared logic. Do not hardcode a `27B` shape into the engine.
- Do not regress `9B` dense or `35B` MoE while chasing `27B`.
- Keep inference runs serialized. Never overlap heavy Orome and llama.cpp runs.
- Prefer small, explainable changes over broad speculative rewrites.

## Success Criteria

This plan is successful if we can do both of these:

1. explain where the majority of the `27B` gap to `llama.cpp` comes from
2. recover meaningful `27B` tok/s without sacrificing the current `9B` and
   `35B` advantages
