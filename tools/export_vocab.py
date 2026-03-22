#!/usr/bin/env python3
"""Export vocab.bin from tokenizer.json for the C inference engine's decode_token().

Binary format:
  uint32 num_entries
  uint32 max_id
  For each entry (0..num_entries-1):
    uint16 byte_len
    char[byte_len] UTF-8 string (no null terminator)
"""
import json
import struct
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_vocab.py tokenizer.json [vocab.bin]", file=sys.stderr)
        sys.exit(1)

    tok_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']  # str -> int
    added = {tok['content']: tok['id'] for tok in t.get('added_tokens', [])}

    # Merge added tokens into vocab
    all_tokens = dict(vocab)
    all_tokens.update(added)

    max_id = max(all_tokens.values())
    num_entries = max_id + 1

    # Build id -> string mapping
    id_to_str = [''] * num_entries
    for token_str, token_id in all_tokens.items():
        if token_id < num_entries:
            id_to_str[token_id] = token_str

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', num_entries))
        f.write(struct.pack('<I', max_id))

        for i in range(num_entries):
            b = id_to_str[i].encode('utf-8')
            f.write(struct.pack('<H', len(b)))
            if len(b) > 0:
                f.write(b)

    import os
    sz = os.path.getsize(out_path)
    print(f"Exported {out_path}:")
    print(f"  Entries: {num_entries}")
    print(f"  Max ID:  {max_id}")
    print(f"  File:    {sz / 1024:.1f} KB")

if __name__ == '__main__':
    main()
