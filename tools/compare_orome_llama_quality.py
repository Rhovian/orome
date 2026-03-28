#!/usr/bin/env python3
"""
compare_orome_llama_quality.py — compare greedy completion quality between
Orome and llama.cpp on the same GGUF models.

The script runs engines strictly one at a time to avoid memory pressure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from benchmark import evaluate_quality_reply


DEFAULT_CTX = 256
PROFILE_RE = re.compile(r"\[profile\][^\n]*")
WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_CASES = [
    {
        "name": "capital",
        "prompt": "The capital of France is",
        "must_contain": ["paris"],
        "tokens": 8,
    },
    {
        "name": "opposite",
        "prompt": "The opposite of hot is",
        "must_contain": ["cold"],
        "tokens": 8,
    },
    {
        "name": "sky",
        "prompt": "The sky is",
        "must_contain": ["blue"],
        "tokens": 16,
    },
]


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
        description="Compare Orome and llama.cpp greedy completion quality on shared GGUF models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["9B", "27B", "35B"],
        help="Model aliases to compare (default: 9B 27B 35B)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Subset of built-in case names to run (default: all built-in cases)",
    )
    parser.add_argument(
        "--cases-file",
        help="Path to JSON array of cases: {name, prompt, must_contain?, tokens?}",
    )
    parser.add_argument(
        "--default-tokens",
        type=int,
        default=16,
        help="Fallback max tokens when a case does not define one",
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
        help="Do not abort when existing orome/llama inference processes are already running",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary")
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
        if pid == current_pid:
            continue
        if "compare_orome_llama.py" in cmd or "compare_orome_llama_quality.py" in cmd:
            continue
        if any(token in cmd for token in (
            "/orome",
            " tools/benchmark.py",
            "llama-completion",
            "llama-cli",
            "llama-simple",
            "llama-bench",
            "llama-server",
        )):
            offenders.append(f"{pid} {cmd}")
    return offenders


def run_process(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"{(proc.stdout + proc.stderr)[-2000:]}"
        )
    return proc


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_orome_reply(output: str) -> str:
    lines = output.splitlines()
    ttft_line = -1
    gen_line = -1
    for idx, line in enumerate(lines):
        if ttft_line < 0 and line.startswith("TTFT:"):
            ttft_line = idx
        elif ttft_line >= 0 and line.startswith("Generation:"):
            gen_line = idx
            break
    if ttft_line < 0 or gen_line < 0 or gen_line <= ttft_line:
        raise ValueError(f"could not parse Orome completion output:\n{output[-1000:]}")
    text = "\n".join(lines[ttft_line + 1:gen_line])
    text = PROFILE_RE.sub("", text)
    return normalize_text(text)


def extract_llama_reply(output: str) -> str:
    marker = "generate:"
    gen_idx = output.find(marker)
    if gen_idx < 0:
        text = normalize_text(output)
        if text:
            return text
        raise ValueError(f"could not parse llama.cpp completion output:\n{output[-1000:]}")
    lines = output[gen_idx:].splitlines()[1:]
    started = False
    chunks: list[str] = []
    for line in lines:
        if line.startswith("common_perf_print:"):
            break
        if not started and not line.strip():
            continue
        started = True
        chunks.append(line.rstrip())
    return normalize_text("\n".join(chunks))


def load_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.cases_file:
        raw = json.loads(Path(args.cases_file).read_text())
    else:
        raw = DEFAULT_CASES

    cases: list[dict[str, object]] = []
    wanted = {name.lower() for name in args.cases} if args.cases else None
    for item in raw:
        name = str(item["name"])
        if wanted is not None and name.lower() not in wanted:
            continue
        cases.append({
            "name": name,
            "prompt": str(item["prompt"]),
            "must_contain": [str(x) for x in item.get("must_contain", [])],
            "tokens": int(item.get("tokens", args.default_tokens)),
        })

    if not cases:
        raise SystemExit("no cases selected")
    return cases


def run_orome_case(root: Path, orome_binary: Path, preset: dict[str, object], case: dict[str, object]) -> dict[str, object]:
    cmd = [
        str(orome_binary.resolve()),
        "--model",
        str(Path(preset["model"]).resolve()),
        "--prompt",
        str(case["prompt"]),
        "--tokens",
        str(case["tokens"]),
    ]
    k = int(preset["k"])
    if k > 0:
        cmd.extend(["--k", str(k)])
    proc = run_process(cmd, root, timeout=1800)
    reply = extract_orome_reply(proc.stdout)
    result = evaluate_quality_reply(reply, case["must_contain"])
    result["reply"] = result.pop("quality_reply")
    result["reasons"] = result.pop("quality_reasons")
    return result


def run_llama_case(root: Path, llama_completion: Path, preset: dict[str, object],
                   case: dict[str, object], llama_gpu_layers: str) -> dict[str, object]:
    cmd = [
        str(llama_completion.resolve()),
        "-m",
        str(Path(preset["model"]).resolve()),
        "-p",
        str(case["prompt"]),
        "-n",
        str(case["tokens"]),
        "-c",
        str(preset["ctx"]),
        "-ngl",
        str(llama_gpu_layers),
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
    proc = run_process(cmd, root, timeout=1800)
    reply = extract_llama_reply(proc.stdout)
    result = evaluate_quality_reply(reply, case["must_contain"])
    result["reply"] = result.pop("quality_reply")
    result["reasons"] = result.pop("quality_reasons")
    return result


def compare_model(root: Path, args: argparse.Namespace, alias: str, preset: dict[str, object],
                  cases: list[dict[str, object]]) -> dict[str, object]:
    results = []
    orome_passes = 0
    llama_passes = 0
    for case in cases:
        orome = run_orome_case(root, Path(args.orome_binary), preset, case)
        llama = run_llama_case(root, Path(args.llama_completion), preset, case, args.llama_gpu_layers)
        if orome["quality_pass"]:
            orome_passes += 1
        if llama["quality_pass"]:
            llama_passes += 1
        results.append({
            "name": case["name"],
            "prompt": case["prompt"],
            "must_contain": case["must_contain"],
            "tokens": case["tokens"],
            "orome": orome,
            "llama": llama,
        })
    return {
        "model": alias,
        "model_path": str(preset["model"]),
        "orome_passes": orome_passes,
        "llama_passes": llama_passes,
        "case_count": len(cases),
        "cases": results,
    }


def print_summary(payload: dict[str, object]) -> None:
    orome_git = payload["orome_git"]
    llama_git = payload["llama_git"]
    print(
        f"orome: {orome_git['branch']} {orome_git['commit']}"
        f"{' (dirty)' if orome_git['dirty'] else ''}"
    )
    print(
        f"llama.cpp: {llama_git['branch']} {llama_git['commit']}"
        f"{' (dirty)' if llama_git['dirty'] else ''}"
    )
    print()
    for model in payload["results"]:
        print(
            f"{model['model']}: "
            f"Orome {model['orome_passes']}/{model['case_count']} pass, "
            f"llama.cpp {model['llama_passes']}/{model['case_count']} pass"
        )
        for case in model["cases"]:
            print(f"  [{case['name']}] prompt={case['prompt']!r}")
            print(
                f"    Orome: {'PASS' if case['orome']['quality_pass'] else 'FAIL'} "
                f"{case['orome']['reply']!r}"
            )
            if case["orome"]["reasons"]:
                print(f"      reasons: {', '.join(case['orome']['reasons'])}")
            print(
                f"    llama.cpp: {'PASS' if case['llama']['quality_pass'] else 'FAIL'} "
                f"{case['llama']['reply']!r}"
            )
            if case["llama"]["reasons"]:
                print(f"      reasons: {', '.join(case['llama']['reasons'])}")
        print()


def main() -> int:
    args = parse_args()
    root = repo_root()
    presets = model_presets(root)

    ensure_exists(Path(args.orome_binary), "orome binary")
    ensure_exists(Path(args.llama_completion), "llama-completion binary")
    ensure_exists(Path(args.llama_repo), "llama.cpp repo")

    if not args.allow_active:
        offenders = check_for_active_processes()
        if offenders:
            joined = "\n".join(f"  {line}" for line in offenders)
            raise SystemExit(
                "refusing to start while other heavy inference processes are active:\n"
                f"{joined}\n"
                "Re-run after those processes exit, or pass --allow-active if you really mean it."
            )

    cases = load_cases(args)
    chosen: list[tuple[str, dict[str, object]]] = []
    for alias in args.models:
        key = alias.upper()
        if key not in presets:
            raise SystemExit(f"unknown model alias '{alias}', expected one of: {', '.join(presets)}")
        ensure_exists(Path(presets[key]["model"]), f"{key} model")
        chosen.append((key, presets[key]))

    payload = {
        "cases": cases,
        "orome_git": git_metadata(root),
        "llama_git": git_metadata(Path(args.llama_repo).resolve()),
        "results": [compare_model(root, args, alias, preset, cases) for alias, preset in chosen],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
