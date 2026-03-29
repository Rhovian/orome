#!/usr/bin/env python3
"""
compare_orome_llama_tokens.py — localize the first visible continuation-token
divergence between Orome and llama.cpp on the same GGUF model.

This script runs engines strictly one at a time to avoid memory pressure.
It reconstructs continuation token IDs by re-tokenizing `prompt + reply` with
llama.cpp's tokenizer and slicing off the prompt prefix.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_CTX = 256
TOKEN_LINE_RE = re.compile(r"^\s*(-?\d+)\s+->\s+'(.*)'(?: \(utf-8 decode failure\))?$")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
        description="Compare visible continuation token traces between Orome and llama.cpp.",
    )
    parser.add_argument("--model", choices=["9B", "27B", "35B"], default="27B")
    parser.add_argument("--prompt", default="The sky is")
    parser.add_argument("--tokens", type=int, default=8)
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
        "--llama-tokenize",
        default=str(root.parent / "llama.cpp" / "build" / "bin" / "llama-tokenize"),
        help="Path to llama.cpp llama-tokenize binary",
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
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
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
        if "compare_orome_llama_tokens.py" in cmd:
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
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return proc


def extract_orome_reply_raw(output: str) -> str:
    lines = output.splitlines(keepends=True)
    ttft_line = -1
    gen_line = -1
    for idx, line in enumerate(lines):
        if ttft_line < 0 and line.startswith("TTFT:"):
            ttft_line = idx
            continue
        if ttft_line >= 0 and line.startswith("Generation:"):
            gen_line = idx
            break
    if ttft_line < 0 or gen_line < 0 or gen_line <= ttft_line:
        raise ValueError(f"could not parse Orome completion output:\n{output[-1000:]}")
    reply = "".join(lines[ttft_line + 1:gen_line])
    return reply.rstrip("\n")


def extract_llama_reply_raw(stdout: str, prompt: str) -> str:
    if "common_perf_print:" in stdout:
        stdout = stdout.split("common_perf_print:", 1)[0]
    if "generate:" in stdout:
        stdout = stdout.split("generate:", 1)[1]
        stdout = stdout.lstrip("\n")
    stdout = stdout.rstrip("\n")
    if stdout.startswith(prompt):
        return stdout[len(prompt):]
    return stdout


def tokenize_ids(llama_tokenize: Path, model_path: Path, text: str) -> list[int]:
    proc = run_process(
        [
            str(llama_tokenize.resolve()),
            "--log-disable",
            "--ids",
            "-m",
            str(model_path.resolve()),
            "-p",
            text,
            "--no-escape",
        ],
        model_path.parent,
        timeout=1800,
    )
    return list(ast.literal_eval(proc.stdout.strip()))


def tokenize_pieces(llama_tokenize: Path, model_path: Path, text: str) -> list[dict[str, object]]:
    proc = run_process(
        [
            str(llama_tokenize.resolve()),
            "--log-disable",
            "-m",
            str(model_path.resolve()),
            "-p",
            text,
            "--no-escape",
        ],
        model_path.parent,
        timeout=1800,
    )
    pieces: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        match = TOKEN_LINE_RE.match(line)
        if not match:
            raise ValueError(f"could not parse token line: {line!r}")
        token_id, piece = match.groups()
        pieces.append({"id": int(token_id), "piece": piece})
    return pieces


def continuation_trace(llama_tokenize: Path, model_path: Path, prompt: str, reply: str) -> dict[str, object]:
    prompt_ids = tokenize_ids(llama_tokenize, model_path, prompt)
    full_text = prompt + reply
    full_ids = tokenize_ids(llama_tokenize, model_path, full_text)
    try:
        full_pieces = tokenize_pieces(llama_tokenize, model_path, full_text)
    except ValueError:
        full_pieces = [{"id": token_id, "piece": None} for token_id in full_ids]

    shared_prefix = 0
    max_prefix = min(len(prompt_ids), len(full_ids))
    while shared_prefix < max_prefix and prompt_ids[shared_prefix] == full_ids[shared_prefix]:
        shared_prefix += 1

    continuation = full_pieces[shared_prefix:]
    return {
        "prompt_token_count": len(prompt_ids),
        "full_token_count": len(full_ids),
        "shared_prefix_token_count": shared_prefix,
        "prompt_prefix_match": shared_prefix == len(prompt_ids),
        "continuation": continuation,
    }


def build_orome_cmd(args: argparse.Namespace, preset: dict[str, object]) -> list[str]:
    cmd = [
        str(Path(args.orome_binary).resolve()),
        "--model",
        str(Path(preset["model"]).resolve()),
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.tokens),
    ]
    k = int(preset["k"])
    if k > 0:
        cmd.extend(["--k", str(k)])
    return cmd


def build_llama_cmd(args: argparse.Namespace, preset: dict[str, object]) -> list[str]:
    return [
        str(Path(args.llama_completion).resolve()),
        "-m",
        str(Path(preset["model"]).resolve()),
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "-c",
        str(preset["ctx"]),
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


def first_divergence(orome_tokens: list[dict[str, object]], llama_tokens: list[dict[str, object]]) -> dict[str, object]:
    limit = min(len(orome_tokens), len(llama_tokens))
    for idx in range(limit):
        if orome_tokens[idx]["id"] != llama_tokens[idx]["id"]:
            return {
                "index0": idx,
                "index1": idx + 1,
                "orome": orome_tokens[idx],
                "llama": llama_tokens[idx],
            }
    if len(orome_tokens) != len(llama_tokens):
        idx = limit
        return {
            "index0": idx,
            "index1": idx + 1,
            "orome": orome_tokens[idx] if idx < len(orome_tokens) else None,
            "llama": llama_tokens[idx] if idx < len(llama_tokens) else None,
        }
    return {"index0": None, "index1": None, "orome": None, "llama": None}


def main() -> int:
    args = parse_args()
    root = repo_root()
    presets = model_presets(root)
    preset = presets[args.model]

    ensure_exists(Path(args.orome_binary), "orome binary")
    ensure_exists(Path(args.llama_completion), "llama-completion binary")
    ensure_exists(Path(args.llama_tokenize), "llama-tokenize binary")
    ensure_exists(Path(args.llama_repo), "llama.cpp repo")

    if not args.allow_active:
        offenders = check_for_active_processes()
        if offenders:
            joined = "\n".join(f"  {line}" for line in offenders)
            raise SystemExit(
                "refusing to start while other heavy inference processes are active:\n"
                f"{joined}\n"
                "rerun after they exit, or pass --allow-active if you are sure"
            )

    orome_proc = run_process(build_orome_cmd(args, preset), root, timeout=1800)
    orome_reply = extract_orome_reply_raw(orome_proc.stdout)
    orome_trace = continuation_trace(
        Path(args.llama_tokenize),
        Path(preset["model"]),
        args.prompt,
        orome_reply,
    )

    llama_proc = run_process(build_llama_cmd(args, preset), root, timeout=1800)
    llama_reply = extract_llama_reply_raw(llama_proc.stdout, args.prompt)
    llama_trace = continuation_trace(
        Path(args.llama_tokenize),
        Path(preset["model"]),
        args.prompt,
        llama_reply,
    )

    divergence = first_divergence(orome_trace["continuation"], llama_trace["continuation"])
    payload = {
        "model": args.model,
        "model_path": str(preset["model"]),
        "prompt": args.prompt,
        "tokens_requested": args.tokens,
        "orome_git": git_metadata(root),
        "llama_git": git_metadata(Path(args.llama_repo)),
        "orome": {
            "reply_raw": orome_reply,
            **orome_trace,
        },
        "llama": {
            "reply_raw": llama_reply,
            **llama_trace,
        },
        "first_divergence": divergence,
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"model: {args.model}")
    print(f"prompt: {args.prompt!r}")
    print(f"orome reply: {orome_reply!r}")
    print(f"llama reply: {llama_reply!r}")
    print()
    print("orome continuation tokens:")
    for idx, tok in enumerate(orome_trace["continuation"], start=1):
        print(f"  {idx:2d}. {tok['id']:6d} {tok['piece']!r}")
    print("llama continuation tokens:")
    for idx, tok in enumerate(llama_trace["continuation"], start=1):
        print(f"  {idx:2d}. {tok['id']:6d} {tok['piece']!r}")
    print()
    if divergence["index1"] is None:
        print("first divergence: none within reconstructed continuation")
    else:
        print(f"first divergence: token {divergence['index1']}")
        print(f"  orome: {divergence['orome']}")
        print(f"  llama: {divergence['llama']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
