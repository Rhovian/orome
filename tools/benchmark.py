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
import contextlib
import http.client
import json
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


GEN_RE = re.compile(r"Generation:\s+([0-9.]+) s \(([0-9.]+) tok/s\)")
TTFT_RE = re.compile(r"TTFT:\s+([0-9.]+) ms")
PROJ_RE = re.compile(r"proj=([0-9.]+)")
DEFAULT_QUALITY_SYSTEM = ""
DEFAULT_QUALITY_PROMPT = "Hello"
DEFAULT_QUALITY_MUST_CONTAIN = []
DEFAULT_QUALITY_RUNS = 1
DEFAULT_QUALITY_MIN_PASSES = 1
DEFAULT_QUALITY_CASES = None
DEFAULT_QUALITY_CASE_MIN_PASSES = 1


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


def has_cli_option(args, flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in (args or []))


def normalize_quality_needles(value):
    if value is None:
        return list(DEFAULT_QUALITY_MUST_CONTAIN)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def normalize_quality_cases(value):
    if value is None:
        return None
    cases = []
    for idx, case in enumerate(value):
        if not isinstance(case, dict):
            raise ValueError(f"quality_cases[{idx}] must be an object")
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"quality_cases[{idx}] is missing a prompt")
        cases.append({
            "name": str(case.get("name", f"case_{idx+1}")),
            "prompt": prompt,
            "must_contain": normalize_quality_needles(case.get("must_contain", [])),
            "tokens": int(case.get("tokens", 64)),
            "system": str(case.get("system", DEFAULT_QUALITY_SYSTEM)),
            "temperature": float(case.get("temperature", case.get("quality_temperature", 0.2))),
        })
    return cases


def build_infer_cmd(infer_path, model=None, k=None, extra_args=None):
    cmd = [str(infer_path)]
    if model:
        cmd.extend(["--model", model])
    if k and not has_cli_option(extra_args, "--k"):
        cmd.extend(["--k", str(k)])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_infer(cwd, infer_path, prompt, tokens, k, timing, model=None, extra_args=None):
    cmd = build_infer_cmd(infer_path, model=model, k=k, extra_args=extra_args)
    cmd.extend(["--prompt", prompt, "--tokens", str(tokens)])

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


def max_repeated_char_run(text):
    best = run = 0
    prev = None
    for ch in text:
        if ch == prev:
            run += 1
        else:
            prev = ch
            run = 1
        if run > best:
            best = run
    return best


def max_repeated_word_run(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    best = run = 0
    prev = None
    for word in words:
        if word == prev:
            run += 1
        else:
            prev = word
            run = 1
        if run > best:
            best = run
    return best


def parse_sse_chat_body(body):
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def extract_chat_reply(body):
    stripped = body.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return ""
        for choice in payload.get("choices") or []:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                return content
        return ""
    return parse_sse_chat_body(body)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(port, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    last_error = "timeout"
    while time.monotonic() < deadline:
        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200:
                return
            last_error = f"health returned {resp.status}: {body[:200]!r}"
        except OSError as exc:
            last_error = str(exc)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        time.sleep(0.1)
    raise TimeoutError(f"server on port {port} did not become ready: {last_error}")


def strip_think_blocks(text):
    """Remove <think>...</think> blocks (and truncated trailing <think>...) from raw completion output."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def evaluate_quality_reply(reply, must_contain):
    text = strip_think_blocks(reply.strip())
    lowered = text.lower()
    reasons = []

    if not text:
        reasons.append("empty visible reply")
    if "<unk_" in lowered:
        reasons.append("contains unknown-token marker")
    if "\ufffd" in text:
        reasons.append("contains replacement characters")
    if max_repeated_char_run(text) >= 6:
        reasons.append("contains repeated-character spam")
    if max_repeated_word_run(text) >= 4:
        reasons.append("contains repeated-word spam")

    alpha_count = sum(ch.isalpha() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    punct_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    if text and alpha_count == 0 and digit_count == 0:
        reasons.append("contains no visible lexical content")
    if punct_count >= 4 and punct_count >= alpha_count + digit_count:
        reasons.append("reply is punctuation-heavy")

    missing = [needle for needle in must_contain if needle.lower() not in lowered]
    if missing:
        reasons.append("missing required content: " + ", ".join(missing))

    return {
        "quality_pass": not reasons,
        "quality_reply": text,
        "quality_reasons": reasons,
    }


def run_quality_probe(cwd, infer_path, model, k, extra_args, *, system_prompt,
                      user_prompt, must_contain, max_tokens, temperature,
                      timeout_sec):
    port = find_free_port()
    cmd = build_infer_cmd(infer_path, model=model, k=k, extra_args=extra_args)
    cmd.extend(["--serve", str(port)])
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    reply = ""
    raw_response = ""
    server_output = ""
    try:
        wait_for_server(port, timeout_sec)

        body = json.dumps({
            "model": "orome",
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        })
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_sec)
        resp = None
        try:
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw_response = resp.read().decode("utf-8", errors="replace")
        finally:
            with contextlib.suppress(Exception):
                conn.close()

        if resp.status != 200:
            raise RuntimeError(f"quality probe HTTP {resp.status}: {raw_response[:400]}")

        reply = extract_chat_reply(raw_response)
    except Exception as exc:
        result = {
            "quality_pass": False,
            "quality_reply": reply.strip(),
            "quality_reasons": [f"quality probe failed: {exc}"],
            "quality_raw_response": raw_response[-500:],
        }
    else:
        result = evaluate_quality_reply(reply, must_contain)
        result["quality_raw_response"] = raw_response[-500:]
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                server_output = proc.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                server_output = proc.communicate()[0]
        else:
            server_output = proc.communicate()[0]

    server_lines = [line for line in server_output.splitlines() if line.strip()]
    if server_lines:
        result["quality_server_log_tail"] = server_lines[-10:]
    else:
        result["quality_server_log_tail"] = []
    return result


def run_quality_probes(cwd, infer_path, model, k, extra_args, *, system_prompt,
                       user_prompt, must_contain, max_tokens, temperature,
                       timeout_sec, runs, min_passes):
    runs = max(1, int(runs))
    min_passes = max(1, int(min_passes))
    min_passes = min(min_passes, runs)

    probes = []
    pass_count = 0
    for _ in range(runs):
        probe = run_quality_probe(
            cwd, infer_path, model, k, extra_args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            must_contain=must_contain,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )
        probes.append(probe)
        if probe.get("quality_pass"):
            pass_count += 1

    aggregate = {
        "quality_pass": pass_count >= min_passes,
        "quality_reply": probes[-1]["quality_reply"] if probes else "",
        "quality_reasons": [],
        "quality_runs": runs,
        "quality_pass_count": pass_count,
        "quality_min_passes": min_passes,
        "quality_probe_results": probes,
    }

    if probes:
        aggregate["quality_server_log_tail"] = probes[-1].get("quality_server_log_tail", [])
        aggregate["quality_raw_response"] = probes[-1].get("quality_raw_response", "")
    else:
        aggregate["quality_server_log_tail"] = []
        aggregate["quality_raw_response"] = ""

    if aggregate["quality_pass"]:
        if runs > 1:
            aggregate["quality_reasons"] = [f"quality passed {pass_count}/{runs} runs"]
    else:
        reasons = [f"quality passed {pass_count}/{runs} runs (need {min_passes})"]
        first_fail = next((p for p in probes if not p.get("quality_pass")), None)
        if first_fail:
            reasons.extend(first_fail.get("quality_reasons") or [])
            aggregate["quality_reply"] = first_fail.get("quality_reply", aggregate["quality_reply"])
        aggregate["quality_reasons"] = reasons

    return aggregate


def run_quality_case_suite(cwd, infer_path, model, k, extra_args, *, cases,
                           timeout_sec, runs, min_passes, min_case_passes):
    case_results = []
    case_pass_count = 0
    for case in cases:
        result = run_quality_probes(
            cwd, infer_path, model, k, extra_args,
            system_prompt=case.get("system", DEFAULT_QUALITY_SYSTEM),
            user_prompt=case["prompt"],
            must_contain=case["must_contain"],
            max_tokens=case["tokens"],
            temperature=case["temperature"],
            timeout_sec=timeout_sec,
            runs=runs,
            min_passes=min_passes,
        )
        case_results.append({
            "name": case["name"],
            "prompt": case["prompt"],
            "must_contain": case["must_contain"],
            "tokens": case["tokens"],
            "quality_pass": result["quality_pass"],
            "quality_reply": result["quality_reply"],
            "quality_reasons": result["quality_reasons"],
            "quality_runs": result["quality_runs"],
            "quality_pass_count": result["quality_pass_count"],
            "quality_min_passes": result["quality_min_passes"],
            "quality_probe_results": result["quality_probe_results"],
        })
        if result["quality_pass"]:
            case_pass_count += 1

    min_case_passes = max(1, int(min_case_passes))
    min_case_passes = min(min_case_passes, len(case_results))
    suite_pass = case_pass_count >= min_case_passes
    first_fail = next((case for case in case_results if not case["quality_pass"]), None)
    last_case = case_results[-1] if case_results else None

    reasons = []
    if not suite_pass:
        reasons.append(
            f"quality cases passed {case_pass_count}/{len(case_results)} (need {min_case_passes})"
        )
        if first_fail:
            reasons.extend(first_fail.get("quality_reasons") or [])

    return {
        "quality_pass": suite_pass,
        "quality_reply": (first_fail or last_case or {}).get("quality_reply", ""),
        "quality_reasons": reasons,
        "quality_case_results": case_results,
        "quality_case_pass_count": case_pass_count,
        "quality_case_count": len(case_results),
        "quality_case_min_passes": min_case_passes,
        "quality_runs": runs,
        "quality_pass_count": case_pass_count,
        "quality_min_passes": min_case_passes,
        "quality_probe_results": [],
        "quality_server_log_tail": [],
        "quality_raw_response": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Run a repeatable orome benchmark")
    parser.add_argument("--infer", default="./orome", help="Path to orome binary")
    parser.add_argument("--model", default="/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf", help="Path to GGUF model file")
    parser.add_argument("--prompt", default="[248045,846,198,9419,248046,198,248045,74455,198]", help="Prompt (raw token IDs or text)")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate (default: 100 for sustained throughput)")
    parser.add_argument("--k", type=int, default=8, help="Active experts")
    parser.add_argument("--trials", type=int, default=3, help="Number of timed trials")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warm-up runs before first trial")
    parser.add_argument("--cooldown-sec", type=float, default=5.0, help="Sleep between trials (short — Mac Studio has a fan)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON (for autoresearch)")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args to pass to infer")
    parser.add_argument("--skip-quality-check", action="store_true", help="Skip the chat quality canary")
    parser.add_argument("--quality-config", help="Optional JSON file with quality_* overrides")
    parser.add_argument("--quality-system", default=DEFAULT_QUALITY_SYSTEM, help="System prompt for the quality canary")
    parser.add_argument("--quality-prompt", default=DEFAULT_QUALITY_PROMPT, help="User prompt for the quality canary")
    parser.add_argument("--quality-must-contain", nargs="*", default=None, help="Substrings that must appear in the quality reply")
    parser.add_argument("--quality-max-tokens", type=int, default=64, help="Max tokens for the quality canary reply")
    parser.add_argument("--quality-temperature", type=float, default=0.2, help="Sampling temperature for the quality canary")
    parser.add_argument("--quality-timeout-sec", type=float, default=30.0, help="Timeout for server startup and quality probe")
    parser.add_argument("--quality-runs", type=int, default=DEFAULT_QUALITY_RUNS, help="Number of sequential quality probes to run")
    parser.add_argument("--quality-min-passes", type=int, default=DEFAULT_QUALITY_MIN_PASSES, help="Minimum number of passing probes required")
    parser.add_argument("--quality-cases", help="Optional JSON array of quality cases")
    parser.add_argument("--quality-case-min-passes", type=int, default=DEFAULT_QUALITY_CASE_MIN_PASSES, help="Minimum number of passing quality cases required")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent.parent  # tools/ -> project root
    infer_path = (cwd / args.infer).resolve() if not Path(args.infer).is_absolute() else Path(args.infer)
    if not infer_path.exists():
        print(f"ERROR: infer binary not found at {infer_path}", file=sys.stderr)
        sys.exit(1)

    quality_enabled = not args.skip_quality_check
    quality_must_contain = normalize_quality_needles(args.quality_must_contain)
    quality_system = args.quality_system
    quality_prompt = args.quality_prompt
    quality_max_tokens = args.quality_max_tokens
    quality_temperature = args.quality_temperature
    quality_timeout_sec = args.quality_timeout_sec
    quality_runs = args.quality_runs
    quality_min_passes = args.quality_min_passes
    quality_cases = normalize_quality_cases(json.loads(args.quality_cases)) if args.quality_cases else DEFAULT_QUALITY_CASES
    quality_case_min_passes = args.quality_case_min_passes

    if args.quality_config:
        with open(args.quality_config) as f:
            quality_cfg = json.load(f)
        quality_enabled = quality_cfg.get("quality_check", quality_enabled)
        quality_system = quality_cfg.get("quality_system", quality_system)
        quality_prompt = quality_cfg.get("quality_prompt", quality_prompt)
        quality_must_contain = normalize_quality_needles(
            quality_cfg.get("quality_must_contain", quality_must_contain)
        )
        quality_max_tokens = quality_cfg.get("quality_max_tokens", quality_max_tokens)
        quality_temperature = quality_cfg.get("quality_temperature", quality_temperature)
        quality_timeout_sec = quality_cfg.get("quality_timeout_sec", quality_timeout_sec)
        quality_runs = quality_cfg.get("quality_runs", quality_runs)
        quality_min_passes = quality_cfg.get("quality_min_passes", quality_min_passes)
        quality_cases = normalize_quality_cases(quality_cfg.get("quality_cases", quality_cases))
        quality_case_min_passes = quality_cfg.get("quality_case_min_passes", quality_case_min_passes)

    if not args.json:
        print("=== orome benchmark (M2 Max 96GB) ===")
        print(f"infer:        {infer_path}")
        print(f"model:        {args.model}")
        print(f"prompt:       {args.prompt!r}")
        print(f"tokens:       {args.tokens}")
        print(f"experts:      {args.k}")
        print(f"trials:       {args.trials}")
        print(f"warm-up runs: {args.warmup_runs}")
        print(f"cool-down:    {args.cooldown_sec:.0f}s")
        print(f"quality gate: {'on' if quality_enabled else 'off'}")
        if quality_enabled:
            if quality_cases:
                print(f"quality cases: {len(quality_cases)} (min passing cases: {quality_case_min_passes})")
                print(f"per-case runs: {quality_runs} (min passes per case: {quality_min_passes})")
            else:
                print(f"quality runs: {quality_runs} (min passes: {quality_min_passes})")
        if args.extra:
            print(f"extra args:   {args.extra}")
        print()

    # Warm up once before all trials
    if not args.json:
        print("[warmup] warming cache and Metal pipeline")
    for warmup_idx in range(1, args.warmup_runs + 1):
        run_infer(cwd, infer_path, args.prompt, min(args.tokens, 20), args.k,
                  timing=False, model=args.model, extra_args=args.extra)
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
            timing=True, model=args.model, extra_args=args.extra
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

    quality_result = None
    exit_code = 0
    if quality_enabled:
        if not args.json:
            if quality_cases:
                print(f"[quality] running {len(quality_cases)}-case quality suite")
            elif quality_runs > 1:
                print(f"[quality] running chat coherence canary ({quality_runs} sequential runs)")
            else:
                print("[quality] running chat coherence canary")
        if quality_cases:
            quality_result = run_quality_case_suite(
                cwd, infer_path, args.model, args.k, args.extra,
                cases=quality_cases,
                timeout_sec=quality_timeout_sec,
                runs=quality_runs,
                min_passes=quality_min_passes,
                min_case_passes=quality_case_min_passes,
            )
        else:
            quality_result = run_quality_probes(
                cwd, infer_path, args.model, args.k, args.extra,
                system_prompt=quality_system,
                user_prompt=quality_prompt,
                must_contain=quality_must_contain,
                max_tokens=quality_max_tokens,
                temperature=quality_temperature,
                timeout_sec=quality_timeout_sec,
                runs=quality_runs,
                min_passes=quality_min_passes,
            )
        result.update({
            "quality_pass": quality_result["quality_pass"],
            "quality_reply": quality_result["quality_reply"],
            "quality_reasons": quality_result["quality_reasons"],
            "quality_runs": quality_result["quality_runs"],
            "quality_pass_count": quality_result["quality_pass_count"],
            "quality_min_passes": quality_result["quality_min_passes"],
            "quality_probe_results": quality_result["quality_probe_results"],
            "quality_case_results": quality_result.get("quality_case_results", []),
            "quality_case_pass_count": quality_result.get("quality_case_pass_count", 0),
            "quality_case_count": quality_result.get("quality_case_count", 0),
            "quality_case_min_passes": quality_result.get("quality_case_min_passes", 0),
        })
        if quality_result.get("quality_server_log_tail"):
            result["quality_server_log_tail"] = quality_result["quality_server_log_tail"]
        if quality_result.get("quality_raw_response"):
            result["quality_raw_response"] = quality_result["quality_raw_response"]
        if not args.json:
            if quality_result["quality_pass"]:
                if quality_cases:
                    print(f"[quality] PASS: {quality_result['quality_case_pass_count']}/{quality_result['quality_case_count']} cases")
                    print(f"[quality] final reply: {quality_result['quality_reply']!r}")
                elif quality_runs > 1:
                    print(f"[quality] PASS: {quality_result['quality_pass_count']}/{quality_result['quality_runs']} runs")
                    print(f"[quality] final reply: {quality_result['quality_reply']!r}")
                else:
                    print(f"[quality] PASS: {quality_result['quality_reply']!r}")
            else:
                print("[quality] FAIL: " + "; ".join(quality_result["quality_reasons"]))
                if quality_result["quality_reply"]:
                    print(f"[quality] reply: {quality_result['quality_reply']!r}")
        if not quality_result["quality_pass"]:
            exit_code = 2
    else:
        result["quality_pass"] = None
        result["quality_reply"] = ""
        result["quality_reasons"] = []
        result["quality_runs"] = 0
        result["quality_pass_count"] = 0
        result["quality_min_passes"] = 0
        result["quality_probe_results"] = []
        result["quality_case_results"] = []
        result["quality_case_pass_count"] = 0
        result["quality_case_count"] = 0
        result["quality_case_min_passes"] = 0

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
        if quality_enabled:
            print(f"quality:      {'pass' if result['quality_pass'] else 'FAIL'}")

    # Machine-parseable summary line (always printed to stderr for autoresearch)
    print(f"---\ntok_sec_median: {result['tok_sec_median']}", file=sys.stderr)
    if quality_enabled and not result["quality_pass"]:
        print("[quality] FAIL: " + "; ".join(result["quality_reasons"]), file=sys.stderr)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
