"""
progress.py — Visualize orome experiment results.
Reads results.tsv and generates progress.png showing optimization trajectory.

Usage:
    pip install pandas matplotlib
    python3 progress.py
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    results_path = "results.tsv"
    if not os.path.exists(results_path):
        print("No results.tsv found.")
        sys.exit(1)

    cols = ["commit", "tok_sec", "ttft_ms", "proj_avg_ms", "status", "description"]
    df = pd.read_csv(results_path, sep="\t", names=cols, skiprows=1)
    df["tok_sec"] = pd.to_numeric(df["tok_sec"], errors="coerce")
    df["ttft_ms"] = pd.to_numeric(df["ttft_ms"], errors="coerce")
    df["experiment_num"] = range(1, len(df) + 1)

    kept = df[df["status"] == "keep"].copy()
    discarded = df[df["status"] == "discard"].copy()
    crashed = df[df["status"] == "crash"].copy()

    # Compute running best
    if len(kept) > 0:
        kept = kept.sort_values("experiment_num")
        kept["running_best"] = kept["tok_sec"].cummax()

    print(f"\n=== Orome Experiment Results (M2 Max 96GB) ===")
    print(f"Total experiments: {len(df)}")
    print(f"  Kept:      {len(kept)}")
    print(f"  Discarded: {len(discarded)}")
    print(f"  Crashed:   {len(crashed)}")
    if len(kept) > 0:
        print(f"  Best tok/s: {kept['tok_sec'].max():.2f}")
        print(f"  Baseline:   {kept['tok_sec'].iloc[0]:.2f}")
        improvement = kept['tok_sec'].max() - kept['tok_sec'].iloc[0]
        pct = 100 * improvement / kept['tok_sec'].iloc[0] if kept['tok_sec'].iloc[0] > 0 else 0
        print(f"  Improvement: +{improvement:.2f} tok/s ({pct:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: tok/s over experiments
    if len(discarded) > 0:
        ax1.scatter(discarded["experiment_num"], discarded["tok_sec"],
                    c="#FF5252", marker="x", s=40, alpha=0.6, label="Discarded", zorder=2)
    if len(crashed) > 0:
        ax1.scatter(crashed["experiment_num"], [0] * len(crashed),
                    c="#757575", marker="v", s=40, alpha=0.6, label="Crash", zorder=2)
    if len(kept) > 0:
        ax1.scatter(kept["experiment_num"], kept["tok_sec"],
                    c="#4CAF50", marker="o", s=60, edgecolors="black", linewidth=0.5,
                    label="Kept", zorder=3)
        ax1.step(kept["experiment_num"], kept["running_best"],
                 where="post", color="#2196F3", linewidth=2, label="Running Best", zorder=4)

    ax1.set_xlabel("Experiment #", fontsize=11)
    ax1.set_ylabel("Tokens/second (100-token sustained)", fontsize=11)
    ax1.set_title("Orome: Inference Optimization on M2 Max 96GB\nQwen3.5-35B-A3B, K=8, 4-bit",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Bottom: TTFT over experiments (kept only)
    if len(kept) > 0 and kept["ttft_ms"].notna().any():
        ax2.bar(kept["experiment_num"], kept["ttft_ms"], color="#FF9800", alpha=0.7, width=0.8)
        ax2.set_ylabel("TTFT (ms)", fontsize=11)
        ax2.set_xlabel("Experiment #", fontsize=11)
        ax2.grid(True, alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "No TTFT data yet", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    print("\nSaved progress.png")

    # Also print a summary table of kept experiments
    if len(kept) > 0:
        print("\n--- Kept experiments (chronological) ---")
        for _, row in kept.iterrows():
            print(f"  #{int(row['experiment_num']):3d}  {row['tok_sec']:6.2f} tok/s  "
                  f"ttft={row['ttft_ms']:.0f}ms  {row['commit']}  {row['description']}")


if __name__ == "__main__":
    main()
