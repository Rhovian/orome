#!/usr/bin/env python3
"""Terminal chat client for orome server."""

import argparse
import http.client
import json
import sys
import urllib.request


def stream_response(host, port, messages, max_tokens):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens}).encode()
    conn = http.client.HTTPConnection(host, port)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()

    full_text = []
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
                    text = obj["choices"][0].get("delta", {}).get("content", "")
                    if text:
                        print(text, end="", flush=True)
                        full_text.append(text)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
            buf = b""

    conn.close()
    return "".join(full_text)


def main():
    parser = argparse.ArgumentParser(description="Chat with orome")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--system", default="You are a helpful assistant.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Check server is up
    try:
        urllib.request.urlopen(f"{url}/health", timeout=2)
    except Exception:
        print(f"Cannot reach orome server at {url}. Start it with:")
        print(f"  ./orome --model <MODEL_DIR> --serve {args.port}")
        sys.exit(1)

    messages = [{"role": "system", "content": args.system}]
    print(f"Connected to {url}. Type /reset to clear, /quit to exit.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input.strip():
            continue
        if user_input.strip() == "/quit":
            break
        if user_input.strip() == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("Conversation reset.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = stream_response(args.host, args.port, messages, args.tokens)
            # Server already filters think blocks from SSE, so reply is clean
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()

        print()


if __name__ == "__main__":
    main()
