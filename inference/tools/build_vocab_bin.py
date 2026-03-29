#!/usr/bin/env python3
"""Build Orome's vocab.bin from Hugging Face GPT-BPE tokenizer assets or GGUF.

Tokenizer-dir mode expects inputs in TOKENIZER_DIR:
  - vocab.json
  - merges.txt
  - tokenizer_config.json (optional, for added tokens)
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_INT32 = 5


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


def parse_gguf_tokenizer(path: Path) -> tuple[list[str], list[tuple[str, str]], dict[int, str]]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        _tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_count = struct.unpack("<Q", f.read(8))[0]

        def read_u32() -> int:
            return struct.unpack("<I", f.read(4))[0]

        def read_u64() -> int:
            return struct.unpack("<Q", f.read(8))[0]

        def read_i32() -> int:
            return struct.unpack("<i", f.read(4))[0]

        def read_string() -> str:
            return f.read(read_u64()).decode("utf-8", "replace")

        def skip_value(value_type: int) -> None:
            if value_type in (0, 1, 7):
                f.read(1)
            elif value_type in (2, 3):
                f.read(2)
            elif value_type in (4, 5, 6):
                f.read(4)
            elif value_type in (10, 11, 12):
                f.read(8)
            elif value_type == GGUF_TYPE_STRING:
                read_string()
            elif value_type == GGUF_TYPE_ARRAY:
                elem_type = read_u32()
                length = read_u64()
                for _ in range(length):
                    skip_value(elem_type)
            else:
                raise ValueError(f"Unsupported GGUF metadata type: {value_type}")

        tokens: list[str] | None = None
        merges: list[str] | None = None
        token_types: list[int] | None = None

        for _ in range(metadata_count):
            key = read_string()
            value_type = read_u32()

            if key == "tokenizer.ggml.tokens":
                if value_type != GGUF_TYPE_ARRAY:
                    raise ValueError("tokenizer.ggml.tokens is not an array")
                elem_type = read_u32()
                length = read_u64()
                if elem_type != GGUF_TYPE_STRING:
                    raise ValueError("tokenizer.ggml.tokens is not a string array")
                tokens = [read_string() for _ in range(length)]
            elif key == "tokenizer.ggml.merges":
                if value_type != GGUF_TYPE_ARRAY:
                    raise ValueError("tokenizer.ggml.merges is not an array")
                elem_type = read_u32()
                length = read_u64()
                if elem_type != GGUF_TYPE_STRING:
                    raise ValueError("tokenizer.ggml.merges is not a string array")
                merges = [read_string() for _ in range(length)]
            elif key == "tokenizer.ggml.token_type":
                if value_type != GGUF_TYPE_ARRAY:
                    raise ValueError("tokenizer.ggml.token_type is not an array")
                elem_type = read_u32()
                length = read_u64()
                if elem_type != GGUF_TYPE_INT32:
                    raise ValueError("tokenizer.ggml.token_type is not an int32 array")
                token_types = [read_i32() for _ in range(length)]
            else:
                skip_value(value_type)

    if tokens is None or merges is None or token_types is None:
        raise ValueError("GGUF did not contain complete tokenizer metadata")
    if len(tokens) != len(token_types):
        raise ValueError("GGUF token/token_type length mismatch")

    added_tokens = {
        token_id: token
        for token_id, (token, token_type) in enumerate(zip(tokens, token_types))
        if token_type != 1
    }
    merge_pairs = []
    for merge in merges:
        parts = merge.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"Bad GGUF merge entry: {merge!r}")
        merge_pairs.append((parts[0], parts[1]))
    return tokens, merge_pairs, added_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Tokenizer directory containing vocab.json/merges.txt, or a GGUF file with embedded tokenizer metadata",
    )
    parser.add_argument("--out", help="Output vocab.bin path")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Missing source path: {source}")

    if source.is_file() and source.suffix.lower() == ".gguf":
        vocab_entries, merges, added_tokens = parse_gguf_tokenizer(source)
        out_path = Path(args.out).expanduser().resolve() if args.out else source.with_name("vocab.bin")
    else:
        tokenizer_dir = source
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
