#!/usr/bin/env python3
"""
compare_orome_llama.py — compare fixed-token generation throughput between
Orome and llama.cpp on the same GGUF models.

The script runs engines strictly one at a time to avoid memory pressure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROMPT = "Hello"
DEFAULT_TOKENS = 100
DEFAULT_CTX = 256

EVAL_RE = re.compile(
    r"(?:common_perf_print|llama_perf_context_print):\s+eval time =\s*"
    r"([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs\s*\(\s*([0-9.]+)\s*ms per token,\s*"
    r"([0-9.]+)\s*tokens per second\)"
)
PROMPT_RE = re.compile(
    r"(?:common_perf_print|llama_perf_context_print):\s+prompt eval time =\s*"
    r"([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens"
)
LOAD_RE = re.compile(
    r"(?:common_perf_print|llama_perf_context_print):\s+load time =\s*([0-9.]+)\s*ms"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def model_presets(root: Path) -> dict[str, dict[str, object]]:
    models_dir = root.parent / "models"
    return {
        "9B": {
            "model": models_dir / "Qwen3.5-9B-Q8_0.gguf",
            "k": 0,
            "ctx": DEFAULT_CTX,
        },
        "27B": {
            "model": models_dir / "Qwen3.5-27B-Q4_K_M.gguf",
            "k": 0,
            "ctx": DEFAULT_CTX,
        },
        "35B": {
            "model": models_dir / "Qwen3.5-35B-A3B-Q4_K_S.gguf",
            "k": 8,
            "ctx": DEFAULT_CTX,
        },
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Compare Orome and llama.cpp generation throughput on shared GGUF models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["9B", "27B", "35B"],
        help="Model aliases to compare (default: 9B 27B 35B)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="Generated tokens per run")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX, help="Context size for llama.cpp runs")
    parser.add_argument("--trials", type=int, default=1, help="Timed trials per engine")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Untimed warm-up runs before the first trial")
    parser.add_argument("--cooldown-sec", type=float, default=0.0, help="Sleep between timed trials")
    parser.add_argument(
        "--orome-benchmark",
        default=str(root / "tools" / "benchmark.py"),
        help="Path to tools/benchmark.py",
    )
    parser.add_argument(
        "--orome-binary",
        default=str(root / "orome"),
        help="Path to Orome binary",
    )
    parser.add_argument(
        "--llama-completion",
        default=str(root.parent / "llama.cpp" / "build" / "bin" / "llama-completion"),
        help="Path to llama.cpp llama-completion binary",
    )
    parser.add_argument(
        "--llama-repo",
        default=str(root.parent / "llama.cpp"),
        help="Path to llama.cpp git repository",
    )
    parser.add_argument(
        "--llama-gpu-layers",
        default="99",
        help="Value for llama.cpp -ngl / --n-gpu-layers (default: 99)",
    )
    parser.add_argument(
        "--allow-active",
        action="store_true",
        help="Do not abort when existing orome/llama benchmark processes are already running",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")


def git_capture(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git command failed in {repo}: git {' '.join(args)}\n"
            f"{(proc.stdout + proc.stderr).strip()}"
        )
    return proc.stdout.strip()


def git_metadata(repo: Path) -> dict[str, object]:
    ensure_exists(repo, "git repo")
    commit = git_capture(repo, "rev-parse", "HEAD")
    branch = git_capture(repo, "branch", "--show-current")
    status = git_capture(repo, "status", "--porcelain")
    return {
        "repo": str(repo),
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
    }


def check_for_active_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    current_pid = str(os.getpid())
    offenders: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        if pid == str(current_pid):
            continue
        if "compare_orome_llama.py" in cmd:
            continue
        if any(token in cmd for token in (
            "/orome",
            " tools/benchmark.py",
            "llama-completion",
            "llama-cli",
            "llama-simple",
            "llama-bench",
        )):
            offenders.append(f"{pid} {cmd}")
    return offenders


def run_capture(cmd: list[str], cwd: Path, timeout: float) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"{output[-2000:]}"
        )
    return output


def parse_orome_json(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"could not find benchmark JSON in output:\n{output[-1000:]}")


def run_orome(args: argparse.Namespace, preset: dict[str, object], root: Path) -> dict[str, float]:
    cmd = [
        sys.executable,
        str(Path(args.orome_benchmark).resolve()),
        "--infer",
        str(Path(args.orome_binary).resolve()),
        "--model",
        str(Path(preset["model"]).resolve()),
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.tokens),
        "--k",
        str(preset["k"]),
        "--trials",
        str(args.trials),
        "--warmup-runs",
        str(args.warmup_runs),
        "--cooldown-sec",
        str(args.cooldown_sec),
        "--skip-quality-check",
        "--json",
    ]
    output = run_capture(cmd, root, timeout=1800)
    parsed = parse_orome_json(output)
    return {
        "tok_sec": float(parsed["tok_sec_median"]),
        "ttft_ms": float(parsed["ttft_ms_median"]),
    }


def parse_llama_metrics(output: str) -> dict[str, float]:
    eval_matches = EVAL_RE.findall(output)
    if not eval_matches:
        raise ValueError(f"could not parse llama.cpp eval metrics:\n{output[-2000:]}")

    prompt_matches = PROMPT_RE.findall(output)
    load_matches = LOAD_RE.findall(output)

    eval_ms, eval_runs, ms_per_tok, tok_per_sec = eval_matches[-1]
    prompt_ms = float(prompt_matches[-1][0]) if prompt_matches else 0.0
    prompt_tokens = float(prompt_matches[-1][1]) if prompt_matches else 0.0
    load_ms = float(load_matches[-1]) if load_matches else 0.0

    return {
        "tok_sec": float(tok_per_sec),
        "eval_ms": float(eval_ms),
        "eval_runs": float(eval_runs),
        "ms_per_tok": float(ms_per_tok),
        "prompt_ms": prompt_ms,
        "prompt_tokens": prompt_tokens,
        "load_ms": load_ms,
    }


def build_llama_cmd(args: argparse.Namespace, preset: dict[str, object], tokens: int) -> list[str]:
    return [
        str(Path(args.llama_completion).resolve()),
        "-m",
        str(Path(preset["model"]).resolve()),
        "-p",
        args.prompt,
        "-n",
        str(tokens),
        "-c",
        str(args.ctx_size or preset["ctx"]),
        "-ngl",
        str(args.llama_gpu_layers),
        "-fa",
        "on",
        "-no-cnv",
        "--no-warmup",
        "--ignore-eos",
        "--simple-io",
        "--perf",
        "--seed",
        "123",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1.0",
        "--min-p",
        "0.0",
    ]


def run_llama(args: argparse.Namespace, preset: dict[str, object], root: Path) -> dict[str, float]:
    warmup_tokens = min(args.tokens, 20)
    for _ in range(args.warmup_runs):
        run_capture(build_llama_cmd(args, preset, warmup_tokens), root, timeout=1800)

    trials: list[dict[str, float]] = []
    for trial_idx in range(args.trials):
        if trial_idx > 0 and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)
        output = run_capture(build_llama_cmd(args, preset, args.tokens), root, timeout=1800)
        trials.append(parse_llama_metrics(output))

    tok_vals = [trial["tok_sec"] for trial in trials]
    ms_vals = [trial["ms_per_tok"] for trial in trials]
    prompt_vals = [trial["prompt_ms"] for trial in trials]
    load_vals = [trial["load_ms"] for trial in trials]
    return {
        "tok_sec": round(statistics.median(tok_vals), 2),
        "ms_per_tok": round(statistics.median(ms_vals), 2),
        "prompt_ms": round(statistics.median(prompt_vals), 2),
        "load_ms": round(statistics.median(load_vals), 2),
    }


def compare_one(args: argparse.Namespace, alias: str, preset: dict[str, object], root: Path) -> dict[str, object]:
    orome = run_orome(args, preset, root)
    llama = run_llama(args, preset, root)
    delta = round(orome["tok_sec"] - llama["tok_sec"], 2)
    ratio = round(orome["tok_sec"] / llama["tok_sec"], 3) if llama["tok_sec"] else None
    winner = "tie"
    if delta > 0:
        winner = "orome"
    elif delta < 0:
        winner = "llama.cpp"
    return {
        "model": alias,
        "model_path": str(preset["model"]),
        "orome": orome,
        "llama": llama,
        "delta_tok_sec": delta,
        "ratio_orome_over_llama": ratio,
        "winner": winner,
    }


def print_table(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'model':<5}  {'orome tok/s':>11}  {'llama tok/s':>11}  "
        f"{'delta':>7}  {'winner':<9}  {'orome ttft':>10}  {'llama prompt':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<5}  "
            f"{row['orome']['tok_sec']:>11.2f}  "
            f"{row['llama']['tok_sec']:>11.2f}  "
            f"{row['delta_tok_sec']:>7.2f}  "
            f"{row['winner']:<9}  "
            f"{row['orome']['ttft_ms']:>10.1f}  "
            f"{row['llama']['prompt_ms']:>12.2f}"
        )


def main() -> int:
    args = parse_args()
    root = repo_root()
    presets = model_presets(root)

    ensure_exists(Path(args.orome_benchmark), "benchmark.py")
    ensure_exists(Path(args.orome_binary), "orome binary")
    ensure_exists(Path(args.llama_completion), "llama-completion binary")
    ensure_exists(Path(args.llama_repo), "llama.cpp repo")

    chosen: list[tuple[str, dict[str, object]]] = []
    for alias in args.models:
        key = alias.upper()
        if key not in presets:
            raise SystemExit(f"unknown model alias '{alias}', expected one of: {', '.join(presets)}")
        ensure_exists(Path(presets[key]["model"]), f"{key} model")
        chosen.append((key, presets[key]))

    if not args.allow_active:
        offenders = check_for_active_processes()
        if offenders:
            joined = "\n".join(f"  {line}" for line in offenders)
            raise SystemExit(
                "refusing to start while other heavy inference processes are active:\n"
                f"{joined}\n"
                "Re-run after those processes exit, or pass --allow-active if you really mean it."
            )

    results = []
    for alias, preset in chosen:
        results.append(compare_one(args, alias, preset, root))

    orome_git = git_metadata(root)
    llama_git = git_metadata(Path(args.llama_repo).resolve())
    payload = {
        "prompt": args.prompt,
        "tokens": args.tokens,
        "ctx_size": args.ctx_size,
        "trials": args.trials,
        "warmup_runs": args.warmup_runs,
        "cooldown_sec": args.cooldown_sec,
        "orome_git": orome_git,
        "llama_git": llama_git,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"orome: {orome_git['branch']} {orome_git['commit']}"
            f"{' (dirty)' if orome_git['dirty'] else ''}"
        )
        print(
            f"llama.cpp: {llama_git['branch']} {llama_git['commit']}"
            f"{' (dirty)' if llama_git['dirty'] else ''}"
        )
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
