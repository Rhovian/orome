#!/usr/bin/env python3
"""Profile expert activation frequency by parsing --profile-experts output."""

import argparse
import json
import subprocess

import numpy as np


PROMPTS = [
    "Explain quantum computing in simple terms",
    "Write a Python function to sort a list",
    "What is the capital of France and why is it important?",
    "The quick brown fox jumps over the lazy dog",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "In mathematics, a prime number is",
    '{"name": "Alice", "age": 30, "scores": [95, 87, 92]}',
    "Translate the following to Spanish: Hello, how are you?",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./orome", help="Path to the orome binary")
    ap.add_argument("--model", default=None, help="Optional model directory")
    ap.add_argument("--tokens", type=int, default=50)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--k", type=int, default=8, help="Experts per token during profiling")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--expert-size-4bit", type=int, default=1769472)
    ap.add_argument("--expert-size-2bit", type=int, default=983040)
    return ap.parse_args()


def run_profile(args):
    counts = np.zeros((args.layers, args.experts), dtype=np.int64)

    for i, prompt in enumerate(PROMPTS):
        print(f"Profiling prompt {i + 1}/{len(PROMPTS)}: {prompt[:40]}...")
        cmd = [args.binary, "--prompt", prompt, "--tokens", str(args.tokens), "--profile-experts"]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.k > 0:
            cmd.extend(["--k", str(args.k)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or f"command failed: {' '.join(cmd)}")

        for line in result.stderr.splitlines():
            if not line.startswith("EXPERT_ROUTE"):
                continue
            parts = dict(p.split("=", 1) for p in line.split()[1:])
            layer = int(parts["layer"])
            experts = [int(e) for e in parts["experts"].split(",") if e]
            if layer < 0 or layer >= args.layers:
                continue
            for expert in experts:
                if 0 <= expert < args.experts:
                    counts[layer][expert] += 1

    total_tokens = len(PROMPTS) * args.tokens
    print(f"\nTotal tokens profiled: {total_tokens}")
    print(f"Total routing decisions: {total_tokens * args.layers}")

    for layer in range(args.layers):
        layer_counts = counts[layer]
        total_activations = layer_counts.sum()
        if total_activations == 0:
            continue
        sorted_counts = np.sort(layer_counts)[::-1]
        nonzero = np.count_nonzero(layer_counts)
        top25_count = sorted_counts[: args.experts * 25 // 100].sum()
        top25_pct = 100.0 * top25_count / total_activations
        print(f"  Layer {layer:2d}: {nonzero:3d}/{args.experts} experts active, "
              f"top 25% cover {top25_pct:.1f}% of activations")

    np.save("expert_counts.npy", counts)

    for hot_pct in [10, 20, 25, 33, 50]:
        n_hot = args.experts * hot_pct // 100
        hot_experts = {}
        for layer in range(args.layers):
            top_indices = np.argsort(counts[layer])[::-1][:n_hot]
            hot_experts[str(layer)] = sorted(top_indices.tolist())

        fname = f"hot_experts_{hot_pct}pct.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(hot_experts, f)

        total_reads = total_tokens * args.layers * args.k
        hot_reads = sum(counts[layer][expert] for layer in range(args.layers)
                        for expert in hot_experts[str(layer)])
        cold_reads = sum(counts[layer].sum() - sum(counts[layer][expert]
                        for expert in hot_experts[str(layer)])
                        for layer in range(args.layers))
        hot_io = hot_reads * args.expert_size_4bit
        cold_io = cold_reads * args.expert_size_2bit
        all_4bit_io = (hot_reads + cold_reads) * args.expert_size_4bit
        savings_pct = 100.0 * (1 - (hot_io + cold_io) / all_4bit_io) if all_4bit_io else 0.0

        disk_4bit = n_hot * args.layers * args.expert_size_4bit
        disk_2bit = (args.experts - n_hot) * args.layers * args.expert_size_2bit
        total_disk_gb = (disk_4bit + disk_2bit) / 1e9
        all_4bit_gb = (args.experts * args.layers * args.expert_size_4bit) / 1e9
        all_2bit_gb = (args.experts * args.layers * args.expert_size_2bit) / 1e9

        print(f"\n  {hot_pct}% hot ({n_hot} experts/layer):")
        print(f"    IO savings: {savings_pct:.1f}% less bytes read")
        print(f"    Disk: {total_disk_gb:.1f} GB (vs {all_4bit_gb:.1f} GB all-4bit, "
              f"{all_2bit_gb:.1f} GB all-2bit)")


if __name__ == "__main__":
    run_profile(parse_args())
