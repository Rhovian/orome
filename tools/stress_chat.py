#!/usr/bin/env python3
"""
stress_chat.py — repeatedly chat with the orome server to reproduce and
diagnose intermittent generation collapse (repeated-character spam).

Sends "debug": true so the server logs per-token diagnostics to stderr.
When collapse is detected, prints the full conversation context that
triggered it.

Usage:
    # Terminal 1 — start server (stderr shows debug traces on collapse)
    make serve MODEL=/path/to/model.gguf

    # Terminal 2 — run stress test
    python3 tools/stress_chat.py --port 8080 --rounds 50
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Prompt bank
# ---------------------------------------------------------------------------

STARTERS = [
    # Factual
    "What causes tides?",
    "Explain how TCP works.",
    "What is the Krebs cycle?",
    "How do magnets work?",
    "Why is the sky blue?",
    "What is a quasar?",
    "How does photosynthesis work?",
    "Explain plate tectonics.",
    # Coding
    "Write a Python function to check if a number is prime.",
    "How do you write a for loop in Rust?",
    "Explain recursion with an example in C.",
    "What is tokio in Rust?",
    "Show me a bubble sort in JavaScript.",
    "How do you read a file in Go?",
    "Write a linked list in C.",
    "What is the difference between a mutex and a semaphore?",
    # Creative
    "Write a short poem about rain.",
    "Tell me a story about a lost cat.",
    "Describe a sunset over the ocean.",
    "Write a haiku about programming.",
    # Short / edge-case
    "Hello",
    "Hi",
    "!",
    "???",
    "The capital of France is",
    "The opposite of hot is",
    "1 + 1 =",
    "What about in zig?",
    "What about in C?",
    "What about in Haskell?",
]

FOLLOW_UPS = [
    "Can you elaborate?",
    "Give me another example.",
    "Now explain it to a 5 year old.",
    "What about in a different programming language?",
    "Why?",
    "Can you show me the code for that?",
    "What are the downsides?",
    "How does that compare to the alternative?",
    "Tell me more.",
    "Interesting. What else should I know?",
    "What about in zig?",
    "What about in C?",
    "Now do it in Python.",
    "Summarize that in one sentence.",
]

SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a concise technical expert.",
    "You are a friendly tutor.",
]

# ---------------------------------------------------------------------------
# Collapse detection (mirrors benchmark.py heuristics)
# ---------------------------------------------------------------------------


def max_repeated_char_run(text: str) -> int:
    best = run = 0
    prev = None
    for ch in text:
        if ch == prev:
            run += 1
        else:
            best = max(best, run)
            run = 1
            prev = ch
    return max(best, run)


def max_repeated_word_run(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    best = run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def max_repeated_nonws_char_run(text: str) -> tuple[int, str]:
    """Like max_repeated_char_run but ignores whitespace."""
    best = run = 0
    best_ch = ""
    prev = None
    for ch in text:
        if ch.isspace():
            prev = None
            run = 0
            continue
        if ch == prev:
            run += 1
            if run > best:
                best = run
                best_ch = ch
        else:
            run = 1
            prev = ch
    return best, best_ch


def detect_collapse(text: str) -> str | None:
    run, ch = max_repeated_nonws_char_run(text)
    if run >= 20:
        return f"repeated char {repr(ch)} x{run}+"
    if max_repeated_word_run(text) >= 4:
        return "repeated-word spam"
    return None


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------


def stream_response(host: str, port: int, messages: list[dict],
                    max_tokens: int, debug: bool = True) -> str:
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "debug": debug,
    }).encode()
    conn = http.client.HTTPConnection(host, port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()

    full_text: list[str] = []
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        if buf.endswith(b"\n\n"):
            for raw_line in buf.decode(errors="replace").splitlines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data == "[DONE]":
                    conn.close()
                    return "".join(full_text)
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    text = delta.get("content", "") or ""
                    reasoning = delta.get("reasoning_content", "") or ""
                    if text:
                        full_text.append(text)
                    if reasoning:
                        full_text.append(reasoning)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
            buf = b""

    conn.close()
    return "".join(full_text)


def server_is_up(host: str, port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Conversation generation
# ---------------------------------------------------------------------------


def run_conversation(host: str, port: int, max_tokens: int,
                     conv_id: int, verbose: bool) -> dict | None:
    """Run a single multi-turn conversation. Returns collapse info or None."""
    system = random.choice(SYSTEM_PROMPTS)
    messages = [{"role": "system", "content": system}]
    num_turns = random.randint(1, 8)
    starter = random.choice(STARTERS)

    for turn in range(num_turns):
        if turn == 0:
            user_msg = starter
        else:
            user_msg = random.choice(FOLLOW_UPS)

        messages.append({"role": "user", "content": user_msg})

        try:
            reply = stream_response(host, port, messages, max_tokens)
        except Exception as e:
            if verbose:
                print(f"  [conv {conv_id} turn {turn}] error: {e}")
            return None

        collapse = detect_collapse(reply)
        if collapse:
            return {
                "conv_id": conv_id,
                "turn": turn,
                "num_turns": num_turns,
                "system": system,
                "messages": messages,
                "reply": reply[:500],
                "collapse_reason": collapse,
            }

        messages.append({"role": "assistant", "content": reply})

        if verbose:
            label = reply[:60].replace("\n", " ")
            print(f"  [conv {conv_id} turn {turn}] ok: {label}...")

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress-test orome chat server to reproduce generation collapse.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tokens", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of conversations to run (default: 100)")
    parser.add_argument("--stop-on-collapse", action="store_true",
                        help="Stop after first collapse detected")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each turn's result")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not server_is_up(args.host, args.port):
        print(f"Cannot reach server at {args.host}:{args.port}")
        print("Start it with: make serve MODEL=/path/to/model.gguf")
        return 1

    collapses: list[dict] = []
    total_turns = 0

    print(f"Running {args.rounds} conversations against {args.host}:{args.port}")
    print(f"Server stderr will show per-token debug traces on collapse.\n")

    for i in range(args.rounds):
        result = run_conversation(args.host, args.port, args.tokens, i, args.verbose)
        if result:
            collapses.append(result)
            total_turns += result["turn"] + 1
            print(f"\n{'='*70}")
            print(f"COLLAPSE #{len(collapses)} in conv {result['conv_id']} "
                  f"turn {result['turn']}/{result['num_turns']}")
            print(f"Reason: {result['collapse_reason']}")
            print(f"System: {result['system']}")
            print(f"Conversation:")
            for msg in result["messages"]:
                role = msg["role"]
                content = msg["content"][:200].replace("\n", " ")
                print(f"  [{role}] {content}")
            print(f"Collapsed reply: {result['reply'][:200]}")
            print(f"{'='*70}\n")

            if args.stop_on_collapse:
                break
        else:
            # estimate turns from a full conversation
            total_turns += random.randint(1, 8)  # rough estimate

        if not args.verbose and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{args.rounds} conversations, "
                  f"{len(collapses)} collapses so far")

    print(f"\nDone. {args.rounds} conversations, {len(collapses)} collapses detected.")
    if collapses:
        print(f"\nCollapse summary:")
        for c in collapses:
            starter = next(
                (m["content"] for m in c["messages"] if m["role"] == "user"), "?"
            )
            print(f"  conv {c['conv_id']} turn {c['turn']}: "
                  f"{c['collapse_reason']} (starter: {starter[:50]})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
