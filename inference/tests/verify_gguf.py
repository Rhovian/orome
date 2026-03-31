"""Verify GGUF inference against Python reference.
Computes embedding + RMS norm + QKV matvec for layer 0 and prints
reference values to compare against orome's GPU output.
"""
import struct
import numpy as np

GGUF_PATH = "/Users/j/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"

def parse_gguf(path):
    """Parse GGUF header, metadata, tensor info."""
    tensors = {}
    with open(path, 'rb') as f:
        magic, ver = struct.unpack('<II', f.read(8))
        tc, mc = struct.unpack('<QQ', f.read(16))

        # Skip metadata
        for _ in range(mc):
            kl = struct.unpack('<Q', f.read(8))[0]; f.read(kl)
            vt = struct.unpack('<I', f.read(4))[0]
            if vt == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
            elif vt == 9:
                et = struct.unpack('<I', f.read(4))[0]; al = struct.unpack('<Q', f.read(8))[0]
                for _ in range(al):
                    if et == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
                    elif et in (0,1,7): f.read(1)
                    elif et in (2,3): f.read(2)
                    elif et in (4,5,6): f.read(4)
                    elif et in (10,11,12): f.read(8)
            elif vt in (0,1,7): f.read(1)
            elif vt in (2,3): f.read(2)
            elif vt in (4,5,6): f.read(4)
            elif vt in (10,11,12): f.read(8)

        # Read tensor info
        for i in range(tc):
            nl = struct.unpack('<Q', f.read(8))[0]; name = f.read(nl).decode()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'dims': dims, 'type': tt, 'offset': off}

        align = 32
        data_off = (f.tell() + align - 1) & ~(align - 1)

    return tensors, data_off

def dequant_q8_0_row(data, row, in_dim):
    """Dequant one row of Q8_0 data."""
    blocks_per_row = in_dim // 32
    bpr = blocks_per_row * 34
    row_data = data[row * bpr : (row + 1) * bpr]
    result = np.zeros(in_dim, dtype=np.float32)
    for blk in range(blocks_per_row):
        d = np.frombuffer(row_data[blk*34:blk*34+2], dtype=np.float16).astype(np.float32)[0]
        qs = np.frombuffer(row_data[blk*34+2:blk*34+34], dtype=np.int8).astype(np.float32)
        result[blk*32:(blk+1)*32] = d * qs
    return result

def q8_0_matvec(data, x, out_dim, in_dim):
    """Q8_0 matrix-vector multiply."""
    result = np.zeros(out_dim, dtype=np.float32)
    for row in range(out_dim):
        w_row = dequant_q8_0_row(data, row, in_dim)
        result[row] = np.dot(w_row, x)
    return result

def q8_0_embed(data, token_id, hidden_dim):
    """Q8_0 embedding lookup."""
    return dequant_q8_0_row(data, token_id, hidden_dim)

def rms_norm(x, weight, eps=1e-6):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x**2) + eps)
    return x / rms * weight

def main():
    print("Parsing GGUF...")
    tensors, data_off = parse_gguf(GGUF_PATH)

    with open(GGUF_PATH, 'rb') as f:
        f.seek(0)
        raw = f.read()

    def tensor_data(name):
        t = tensors[name]
        start = data_off + t['offset']
        return start

    H = 2048

    # Token IDs for "Hello" (from orome: tokens 39, 300, ...)
    # Actually let's use token 39 (first token after BOS)
    token_id = 39

    # 1. Embedding lookup (Q8_0)
    emb_off = tensor_data('token_embd.weight')
    emb = q8_0_embed(raw[emb_off:], token_id, H)
    print(f"Embedding[{token_id}] first 4: {emb[:4]}")

    # 2. RMS norm (layer 0 input norm)
    norm_off = tensor_data('blk.0.attn_norm.weight')
    norm_w = np.frombuffer(raw[norm_off:norm_off + H*4], dtype=np.float32)
    normed = rms_norm(emb, norm_w)
    print(f"After RMS norm first 4: {normed[:4]}")

    # 3. QKV matvec (layer 0, Q8_0)
    qkv_off = tensor_data('blk.0.attn_qkv.weight')
    qkv_ti = tensors['blk.0.attn_qkv.weight']
    qkv_out_dim = qkv_ti['dims'][1]  # ne1 = 8192
    qkv_in_dim = qkv_ti['dims'][0]   # ne0 = 2048
    print(f"QKV dims: [{qkv_in_dim}, {qkv_out_dim}]")

    # Q8_0 matvec: out[row] = sum(W_row[col] * x[col])
    # Only compute first 4 output values for speed
    qkv_result = np.zeros(min(8, qkv_out_dim), dtype=np.float32)
    blocks_per_row = qkv_in_dim // 32
    bpr = blocks_per_row * 34
    qkv_data = raw[qkv_off:]

    for row in range(len(qkv_result)):
        w_row = dequant_q8_0_row(qkv_data, row, qkv_in_dim)
        qkv_result[row] = np.dot(w_row, normed)

    print(f"QKV output first 8: {qkv_result}")

    # 4. Z gate matvec (layer 0, Q8_0)
    z_off = tensor_data('blk.0.attn_gate.weight')
    z_ti = tensors['blk.0.attn_gate.weight']
    z_out_dim = z_ti['dims'][1]
    z_in_dim = z_ti['dims'][0]
    z_result = np.zeros(min(4, z_out_dim), dtype=np.float32)
    z_data = raw[z_off:]
    for row in range(len(z_result)):
        w_row = dequant_q8_0_row(z_data, row, z_in_dim)
        z_result[row] = np.dot(w_row, normed)
    print(f"Z gate output first 4: {z_result}")

    # 5. Conv1d weights check
    conv_off = tensor_data('blk.0.ssm_conv1d.weight')
    conv_ti = tensors['blk.0.ssm_conv1d.weight']
    conv_data = np.frombuffer(raw[conv_off:conv_off + conv_ti['dims'][0]*conv_ti['dims'][1]*4],
                               dtype=np.float32).reshape(conv_ti['dims'][1], conv_ti['dims'][0])
    # GGUF: [4, 8192] = [kernel_size, conv_dim], stored as 8192 rows of 4
    # Wait: dims=[4, 8192], ne0=4 contiguous. So it's 8192 rows of 4 values
    # Actually ne0=4 means each row has 4 elements, ne1=8192 means 8192 rows
    conv_raw = np.frombuffer(raw[conv_off:conv_off + 4*8192*4], dtype=np.float32).reshape(8192, 4)
    print(f"Conv1d weight shape: {conv_raw.shape}")
    print(f"Conv1d[0,:] = {conv_raw[0,:]}")
    print(f"Conv1d[1,:] = {conv_raw[1,:]}")

if __name__ == '__main__':
    main()
