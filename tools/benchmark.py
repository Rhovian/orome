#!/usr/bin/env python3
"""
benchmark.py — Reproducible benchmark runner for orome inference engine.

Adapted from flash-moe for Mac Studio M2 Max (actively cooled, no thermal throttling).
Shorter cooldowns since the machine has a fan.

Protocol per trial:
1. Run warm-up generation(s) to warm the OS page cache / Metal pipeline.
2. Brief pause (no extended cooldown needed — Mac Studio has active cooling).
3. Run the timed benchmark and parse tok/s, TTFT, and projection timings.

Output is machine-parseable for the autoresearch experiment loop.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


GEN_RE = re.compile(r"Generation:\s+([0-9.]+) s \(([0-9.]+) tok/s\)")
TTFT_RE = re.compile(r"TTFT:\s+([0-9.]+) ms")
PROJ_RE = re.compile(r"proj=([0-9.]+)")


def parse_metrics(output):
    gen_match = GEN_RE.search(output)
    ttft_match = TTFT_RE.search(output)
    proj_vals = [float(v) for v in PROJ_RE.findall(output)]
    if not gen_match or not ttft_match:
        raise ValueError(f"Failed to parse benchmark output:\n{output[-500:]}")
    tok_sec = float(gen_match.group(2))
    ttft_ms = float(ttft_match.group(1))
    proj_avg = statistics.fmean(proj_vals) if proj_vals else 0.0
    proj_min = min(proj_vals) if proj_vals else 0.0
    proj_max = max(proj_vals) if proj_vals else 0.0
    return {
        "tok_sec": tok_sec,
        "ttft_ms": ttft_ms,
        "proj_avg_ms": proj_avg,
        "proj_min_ms": proj_min,
        "proj_max_ms": proj_max,
    }


def run_infer(cwd, infer_path, prompt, tokens, k, use_2bit, timing, model=None, extra_args=None):
    cmd = [str(infer_path)]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--prompt", prompt, "--tokens", str(tokens)])
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"infer exited with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n{output[-1000:]}"
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Run a repeatable orome benchmark")
    parser.add_argument("--infer", default="./orome", help="Path to orome binary")
    parser.add_argument("--model", default="/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf", help="Path to GGUF model file")
    parser.add_argument("--prompt", default="[248045,846,198,9419,248046,198,248045,74455,198]", help="Prompt (raw token IDs or text)")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate (default: 100 for sustained throughput)")
    parser.add_argument("--k", type=int, default=8, help="Active experts")
    parser.add_argument("--2bit", dest="use_2bit", action="store_true", help="Use 2-bit experts (legacy)")
    parser.add_argument("--trials", type=int, default=3, help="Number of timed trials")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warm-up runs before first trial")
    parser.add_argument("--cooldown-sec", type=float, default=5.0, help="Sleep between trials (short — Mac Studio has a fan)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON (for autoresearch)")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args to pass to infer")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent.parent  # tools/ -> project root
    infer_path = (cwd / args.infer).resolve() if not Path(args.infer).is_absolute() else Path(args.infer)
    if not infer_path.exists():
        print(f"ERROR: infer binary not found at {infer_path}", file=sys.stderr)
        sys.exit(1)

    quant = "2-bit" if args.use_2bit else "4-bit"
    if not args.json:
        print("=== orome benchmark (M2 Max 96GB) ===")
        print(f"infer:        {infer_path}")
        print(f"prompt:       {args.prompt!r}")
        print(f"tokens:       {args.tokens}")
        print(f"experts:      {args.k}")
        print(f"quant:        {quant}")
        print(f"trials:       {args.trials}")
        print(f"warm-up runs: {args.warmup_runs}")
        print(f"cool-down:    {args.cooldown_sec:.0f}s")
        if args.extra:
            print(f"extra args:   {args.extra}")
        print()

    # Warm up once before all trials
    if not args.json:
        print("[warmup] warming cache and Metal pipeline")
    for warmup_idx in range(1, args.warmup_runs + 1):
        run_infer(cwd, infer_path, args.prompt, min(args.tokens, 20), args.k,
                  args.use_2bit, timing=False, model=args.model, extra_args=args.extra)
        if not args.json:
            print(f"  warm-up {warmup_idx}/{args.warmup_runs} complete")

    trials = []
    for trial_idx in range(1, args.trials + 1):
        if args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)

        if not args.json:
            print(f"[trial {trial_idx}/{args.trials}] running timed benchmark")
        output = run_infer(
            cwd, infer_path, args.prompt, args.tokens, args.k,
            args.use_2bit, timing=True, model=args.model, extra_args=args.extra
        )
        metrics = parse_metrics(output)
        trials.append(metrics)
        if not args.json:
            print(
                "  result:"
                f" tok/s={metrics['tok_sec']:.2f}"
                f" ttft={metrics['ttft_ms']:.0f}ms"
                f" proj_avg={metrics['proj_avg_ms']:.1f}ms"
                f" proj_range={metrics['proj_min_ms']:.1f}-{metrics['proj_max_ms']:.1f}ms"
            )

    tok_vals = [t["tok_sec"] for t in trials]
    ttft_vals = [t["ttft_ms"] for t in trials]
    proj_vals = [t["proj_avg_ms"] for t in trials]

    result = {
        "tok_sec_median": round(statistics.median(tok_vals), 2),
        "tok_sec_mean": round(statistics.fmean(tok_vals), 2),
        "tok_sec_best": round(max(tok_vals), 2),
        "ttft_ms_median": round(statistics.median(ttft_vals), 0),
        "proj_avg_ms": round(statistics.fmean(proj_vals), 1),
        "trials": trials,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print()
        print("=== summary ===")
        print(f"tok/s median: {result['tok_sec_median']:.2f}")
        print(f"tok/s mean:   {result['tok_sec_mean']:.2f}")
        print(f"tok/s best:   {result['tok_sec_best']:.2f}")
        print(f"TTFT median:  {result['ttft_ms_median']:.0f} ms")
        print(f"proj avg:     {result['proj_avg_ms']:.1f} ms")

    # Machine-parseable summary line (always printed to stderr for autoresearch)
    print(f"---\ntok_sec_median: {result['tok_sec_median']}", file=sys.stderr)


if __name__ == "__main__":
    main()
