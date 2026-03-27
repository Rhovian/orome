#!/usr/bin/env python3
"""Build Orome's vocab.bin from Hugging Face GPT-BPE tokenizer assets.

Expected inputs in TOKENIZER_DIR:
  - vocab.json
  - merges.txt
  - tokenizer_config.json (optional, for added tokens)
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def load_vocab(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab: dict[int, str] = {}
    for token, token_id in raw.items():
        vocab[int(token_id)] = token
    return vocab


def load_added_tokens(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    added: dict[int, str] = {}

    decoder = raw.get("added_tokens_decoder", {})
    if isinstance(decoder, dict):
        for token_id, entry in decoder.items():
            if isinstance(entry, dict) and "content" in entry:
                added[int(token_id)] = entry["content"]

    entries = raw.get("added_tokens", [])
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and "id" in entry and "content" in entry:
                added[int(entry["id"])] = entry["content"]

    return added


def load_merges(path: Path) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Bad merge line: {line!r}")
        merges.append((parts[0], parts[1]))
    return merges


def write_string(f, text: str) -> None:
    data = text.encode("utf-8")
    if len(data) > 0xFFFF:
        raise ValueError(f"Token too long for vocab.bin: {text[:64]!r}")
    f.write(struct.pack("<H", len(data)))
    f.write(data)


def build_vocab_entries(base_vocab: dict[int, str], added_tokens: dict[int, str]) -> list[str]:
    max_id = max(base_vocab.keys() | added_tokens.keys())
    entries: list[str] = []
    for token_id in range(max_id + 1):
        token = base_vocab.get(token_id)
        if token is None:
            token = added_tokens.get(token_id, f"<reserved_{token_id}>")
        entries.append(token)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_dir", help="Directory containing vocab.json and merges.txt")
    parser.add_argument("--out", help="Output vocab.bin path (default: TOKENIZER_DIR/vocab.bin)")
    args = parser.parse_args()

    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    config_path = tokenizer_dir / "tokenizer_config.json"
    out_path = Path(args.out).expanduser().resolve() if args.out else tokenizer_dir / "vocab.bin"

    if not vocab_path.exists():
        raise SystemExit(f"Missing vocab.json: {vocab_path}")
    if not merges_path.exists():
        raise SystemExit(f"Missing merges.txt: {merges_path}")

    base_vocab = load_vocab(vocab_path)
    added_tokens = load_added_tokens(config_path)
    merges = load_merges(merges_path)
    vocab_entries = build_vocab_entries(base_vocab, added_tokens)

    with out_path.open("wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(vocab_entries)))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(added_tokens)))

        for token_id, token in enumerate(vocab_entries):
            f.write(struct.pack("<I", token_id))
            write_string(f, token)

        for left, right in merges:
            write_string(f, left)
            write_string(f, right)

        for token_id, token in sorted(added_tokens.items()):
            f.write(struct.pack("<I", token_id))
            write_string(f, token)

    print(
        f"Wrote {out_path} with {len(vocab_entries)} vocab entries, "
        f"{len(merges)} merges, and {len(added_tokens)} added tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
