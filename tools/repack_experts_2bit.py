#!/usr/bin/env python3
"""
repack_experts_2bit.py — Requantize 4-bit packed expert files to 2-bit format.

Reads packed_experts/layer_XX.bin and writes packed_experts_2bit/layer_XX.bin.

Supports both 35B (moe_intermediate=512, hidden_dim=2048, 256 experts, 40 layers)
and 397B (moe_intermediate=1024, hidden_dim=4096, 512 experts, 60 layers).

Requantization per group of 64 values:
  1. Dequantize: f[i] = uint4[i] * scale + bias  (range 0-15 mapped affinely)
  2. Compute optimal 2-bit params: S2 = (max(f) - min(f)) / 3, B2 = min(f)
  3. Quantize:   uint2[i] = clamp(round((f[i] - B2) / S2), 0, 3)
  4. Repack:     16 x 2-bit values per uint32 (vs 8 x 4-bit per uint32)

Usage:
    python repack_experts_2bit.py --model /path/to/model [--layer N] [--verify]
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path


GROUP_SIZE = 64


# ============================================================================
# Expert layout computation (matches compute_expert_layout in weights.m)
# ============================================================================

def compute_expert_layout(moe_intermediate, hidden_dim, group_size, bits):
    """Compute expert layout for a given quantization bitwidth."""
    vals_per_u32 = 32 // bits

    # gate_proj: [moe_intermediate, hidden_dim]
    gate_out, gate_in = moe_intermediate, hidden_dim
    gate_w_size = gate_out * gate_in // vals_per_u32 * 4
    gate_groups = gate_in // group_size
    gate_s_size = gate_out * gate_groups * 2
    gate_b_size = gate_s_size

    # up_proj: same shape as gate
    up_w_size = gate_w_size
    up_s_size = gate_s_size
    up_b_size = gate_b_size

    # down_proj: [hidden_dim, moe_intermediate]
    down_out, down_in = hidden_dim, moe_intermediate
    down_w_size = down_out * down_in // vals_per_u32 * 4
    down_groups = down_in // group_size
    down_s_size = down_out * down_groups * 2
    down_b_size = down_s_size

    off = 0
    layout = {}
    layout['gate_w_off'] = off; off += gate_w_size
    layout['gate_s_off'] = off; off += gate_s_size
    layout['gate_b_off'] = off; off += gate_b_size
    layout['up_w_off'] = off; off += up_w_size
    layout['up_s_off'] = off; off += up_s_size
    layout['up_b_off'] = off; off += up_b_size
    layout['down_w_off'] = off; off += down_w_size
    layout['down_s_off'] = off; off += down_s_size
    layout['down_b_off'] = off; off += down_b_size
    layout['expert_size'] = off

    layout['gate_w_size'] = gate_w_size
    layout['gate_s_size'] = gate_s_size
    layout['gate_b_size'] = gate_b_size
    layout['up_w_size'] = up_w_size
    layout['up_s_size'] = up_s_size
    layout['up_b_size'] = up_b_size
    layout['down_w_size'] = down_w_size
    layout['down_s_size'] = down_s_size
    layout['down_b_size'] = down_b_size

    layout['projs'] = [
        ("gate", moe_intermediate, hidden_dim,
         layout['gate_w_off'], layout['gate_s_off'], layout['gate_b_off']),
        ("up", moe_intermediate, hidden_dim,
         layout['up_w_off'], layout['up_s_off'], layout['up_b_off']),
        ("down", hidden_dim, moe_intermediate,
         layout['down_w_off'], layout['down_s_off'], layout['down_b_off']),
    ]

    return layout


def load_model_config(model_path):
    """Load model config from HF config.json (handles nested text_config)."""
    config_path = model_path / 'config.json'
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    # HF configs may nest under text_config
    tc = cfg.get('text_config', cfg)
    return {
        'moe_intermediate': tc.get('moe_intermediate_size', 512),
        'hidden_dim': tc.get('hidden_size', 2048),
        'num_experts': tc.get('num_experts', 256),
        'num_layers': tc.get('num_hidden_layers', 40),
    }


# ============================================================================
# bf16 <-> f32 conversion helpers
# ============================================================================

def bf16_to_f32(bf16_u16):
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


# ============================================================================
# Unpack/pack bit-packed values
# ============================================================================

def unpack_4bit(packed):
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size
    out = np.empty(n * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))

def pack_2bit(vals):
    shape = vals.shape
    assert shape[-1] % 16 == 0
    n_packed = shape[-1] // 16
    flat = vals.reshape(-1, shape[-1])
    rows = flat.shape[0]
    out = np.zeros((rows, n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(shape[:-1] + (n_packed,))


# ============================================================================
# Requantize one projection: 4-bit -> dequant -> optimal 2-bit
# ============================================================================

def requantize_projection(packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim):
    num_groups = in_dim // GROUP_SIZE

    vals_4bit = unpack_4bit(packed_4bit)
    assert vals_4bit.shape == (out_dim, in_dim)

    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)

    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    s = scales_f32[:, :, np.newaxis]
    b = biases_f32[:, :, np.newaxis]

    dequant = vals_grouped * s + b

    f_min = dequant.min(axis=2, keepdims=True)
    f_max = dequant.max(axis=2, keepdims=True)

    s2 = (f_max - f_min) / 3.0
    b2 = f_min

    degenerate = (s2 == 0.0)
    s2_safe = np.where(degenerate, 1.0, s2)

    vals_2bit_f = (dequant - b2) / s2_safe
    vals_2bit = np.clip(np.round(vals_2bit_f), 0, 3).astype(np.uint8)

    recon = vals_2bit.astype(np.float32) * s2 + b2
    error = dequant - recon
    rmse = float(np.sqrt(np.mean(error ** 2)))

    vals_2bit_flat = vals_2bit.reshape(out_dim, in_dim)
    packed_2bit = pack_2bit(vals_2bit_flat)

    new_scales_bf16 = f32_to_bf16(s2.squeeze(axis=2).astype(np.float32))
    new_biases_bf16 = f32_to_bf16(b2.squeeze(axis=2).astype(np.float32))

    return packed_2bit, new_scales_bf16, new_biases_bf16, rmse


# ============================================================================
# Process one expert
# ============================================================================

def requantize_expert(expert_blob, layout_4bit, layout_2bit):
    expert_size_4 = layout_4bit['expert_size']
    expert_size_2 = layout_2bit['expert_size']
    assert len(expert_blob) == expert_size_4

    output = bytearray(expert_size_2)
    proj_rmses = {}

    for name, out_dim, in_dim, w_off, s_off, b_off in layout_4bit['projs']:
        packed_cols_4 = in_dim // 8
        num_groups = in_dim // GROUP_SIZE

        w_end = w_off + out_dim * packed_cols_4 * 4
        s_end = s_off + out_dim * num_groups * 2
        b_end = b_off + out_dim * num_groups * 2

        packed_4bit = np.frombuffer(
            expert_blob[w_off:w_end], dtype=np.uint32
        ).reshape(out_dim, packed_cols_4)
        scales_bf16 = np.frombuffer(
            expert_blob[s_off:s_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)
        biases_bf16 = np.frombuffer(
            expert_blob[b_off:b_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)

        packed_2bit, new_scales, new_biases, rmse = requantize_projection(
            packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim
        )
        proj_rmses[name] = rmse

        # Find the 2-bit offsets for this projection
        for n2, _, _, w2, s2, b2 in layout_2bit['projs']:
            if n2 == name:
                w_off_2, s_off_2, b_off_2 = w2, s2, b2
                break

        w_data = packed_2bit.tobytes()
        s_data = new_scales.tobytes()
        b_data = new_biases.tobytes()

        output[w_off_2 : w_off_2 + len(w_data)] = w_data
        output[s_off_2 : s_off_2 + len(s_data)] = s_data
        output[b_off_2 : b_off_2 + len(b_data)] = b_data

    return bytes(output), proj_rmses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Requantize 4-bit packed experts to 2-bit')
    parser.add_argument('--model', type=str, default='.',
                        help='Path to model directory (containing packed_experts/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: MODEL/packed_experts_2bit)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Process only this layer. Default: all layers.')
    parser.add_argument('--verify', action='store_true',
                        help='Verify by dequantizing 2-bit output and comparing to 4-bit')
    parser.add_argument('--moe-intermediate', type=int, default=None,
                        help='MoE intermediate size (auto-detected from config.json)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dimension (auto-detected from config.json)')
    parser.add_argument('--experts', type=int, default=None,
                        help='Number of experts per layer (auto-detected from config.json)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers (auto-detected from config.json)')
    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = model_path / 'packed_experts'
    output_dir = Path(args.output) if args.output else model_path / 'packed_experts_2bit'

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Load config from HF config.json, override with CLI args
    cfg = load_model_config(model_path) or {}
    moe_intermediate = args.moe_intermediate or cfg.get('moe_intermediate', 512)
    hidden_dim = args.hidden_dim or cfg.get('hidden_dim', 2048)
    num_experts = args.experts or cfg.get('num_experts', 256)
    num_layers = args.num_layers or cfg.get('num_layers', 40)

    layout_4bit = compute_expert_layout(moe_intermediate, hidden_dim, GROUP_SIZE, 4)
    layout_2bit = compute_expert_layout(moe_intermediate, hidden_dim, GROUP_SIZE, 2)

    expert_size_4 = layout_4bit['expert_size']
    expert_size_2 = layout_2bit['expert_size']

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover layers
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = []
        for i in range(num_layers):
            if (input_dir / f'layer_{i:02d}.bin').exists():
                layers.append(i)
        if not layers:
            print(f"ERROR: No layer_XX.bin files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Model:       {model_path}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Dimensions:  moe_intermediate={moe_intermediate}, hidden_dim={hidden_dim}")
    print(f"Layers:      {len(layers)} ({layers[0]}-{layers[-1]})")
    print(f"Experts:     {num_experts}")
    print(f"4-bit size:  {expert_size_4:,} bytes/expert  "
          f"({num_experts * expert_size_4 / 1e9:.2f} GB/layer)")
    print(f"2-bit size:  {expert_size_2:,} bytes/expert  "
          f"({num_experts * expert_size_2 / 1e9:.2f} GB/layer)")
    print(f"Savings:     {1 - expert_size_2 / expert_size_4:.1%}")
    print()

    total_t0 = time.time()

    for layer_idx in layers:
        input_path = input_dir / f'layer_{layer_idx:02d}.bin'
        output_path = output_dir / f'layer_{layer_idx:02d}.bin'

        actual_size = input_path.stat().st_size
        num_experts_actual = actual_size // expert_size_4
        if actual_size % expert_size_4 != 0:
            print(f"ERROR: layer_{layer_idx:02d}.bin size {actual_size:,} "
                  f"not divisible by expert_size {expert_size_4:,}", file=sys.stderr)
            continue

        if num_experts_actual != num_experts:
            print(f"  Adjusted to {num_experts_actual} experts based on file size")

        print(f"=== Layer {layer_idx:02d} ({num_experts_actual} experts, "
              f"{actual_size / 1e9:.2f} GB -> "
              f"{num_experts_actual * expert_size_2 / 1e9:.2f} GB) ===")

        layer_t0 = time.time()
        rmse_accum = {"gate": 0.0, "up": 0.0, "down": 0.0}

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for eidx in range(num_experts_actual):
                fin.seek(eidx * expert_size_4)
                expert_4bit = fin.read(expert_size_4)
                if len(expert_4bit) != expert_size_4:
                    print(f"  ERROR: Short read for expert {eidx}", file=sys.stderr)
                    break

                expert_2bit, proj_rmses = requantize_expert(
                    expert_4bit, layout_4bit, layout_2bit)
                assert len(expert_2bit) == expert_size_2

                for p in ("gate", "up", "down"):
                    rmse_accum[p] += proj_rmses[p]

                fout.write(expert_2bit)

                if (eidx + 1) % 64 == 0 or eidx == num_experts_actual - 1:
                    elapsed = time.time() - layer_t0
                    rate = (eidx + 1) / elapsed if elapsed > 0 else 0
                    eta = (num_experts_actual - eidx - 1) / rate if rate > 0 else 0
                    print(f"  [{eidx+1:3d}/{num_experts_actual}] "
                          f"{elapsed:.1f}s elapsed, {rate:.1f} experts/s, "
                          f"ETA {eta:.0f}s")

        layer_elapsed = time.time() - layer_t0
        avg_rmse = {p: rmse_accum[p] / num_experts_actual for p in rmse_accum}
        print(f"\n  Layer {layer_idx:02d} done in {layer_elapsed:.1f}s "
              f"({num_experts_actual / layer_elapsed:.1f} experts/s)")
        print(f"  Avg RMSE:  gate={avg_rmse['gate']:.6f}  "
              f"up={avg_rmse['up']:.6f}  down={avg_rmse['down']:.6f}")

        out_size = output_path.stat().st_size
        print(f"  Output: {output_path} ({out_size / 1e9:.2f} GB)")
        print()

    total_elapsed = time.time() - total_t0
    total_out = sum(
        (model_path / 'packed_experts_2bit' / f'layer_{i:02d}.bin').stat().st_size
        for i in layers
        if (model_path / 'packed_experts_2bit' / f'layer_{i:02d}.bin').exists()
    )
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Total output: {total_out / 1e9:.2f} GB")
    print(f"Expert size 2-bit: {expert_size_2:,} bytes")


if __name__ == '__main__':
    main()
