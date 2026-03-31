#!/usr/bin/env python3
"""Verify layer 0 forward pass against C implementation.

Opens the GGUF, dequantizes relevant tensors, and computes layer 0's
linear attention + MoE forward pass for token 39 (the first prompt token
when "Hello" is tokenized).
"""
import struct
import numpy as np
import sys

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf"

# ---- GGUF Parser ----
class GGUF:
    def __init__(self, path):
        self.f = open(path, 'rb')
        magic = self.f.read(4)
        assert magic == b'GGUF', f"Bad magic: {magic}"
        self.version, = struct.unpack('<I', self.f.read(4))
        self.n_tensors, = struct.unpack('<Q', self.f.read(8))
        self.n_kv, = struct.unpack('<Q', self.f.read(8))

        # Parse metadata
        self.metadata = {}
        for _ in range(self.n_kv):
            key = self._read_string()
            vtype, = struct.unpack('<I', self.f.read(4))
            val = self._read_value(vtype)
            self.metadata[key] = val

        # Parse tensor infos
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._read_string()
            ndims, = struct.unpack('<I', self.f.read(4))
            dims = [struct.unpack('<Q', self.f.read(8))[0] for _ in range(ndims)]
            ttype, = struct.unpack('<I', self.f.read(4))
            offset, = struct.unpack('<Q', self.f.read(8))
            self.tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

        # Compute data offset
        alignment = self.metadata.get('general.alignment', 32)
        pos = self.f.tell()
        self.data_offset = (pos + alignment - 1) // alignment * alignment

        # mmap the file
        import mmap
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def _read_string(self):
        slen, = struct.unpack('<Q', self.f.read(8))
        return self.f.read(slen).decode('utf-8')

    def _read_value(self, vtype):
        if vtype == 0: return struct.unpack('<B', self.f.read(1))[0]  # uint8
        elif vtype == 1: return struct.unpack('<b', self.f.read(1))[0]  # int8
        elif vtype == 2: return struct.unpack('<H', self.f.read(2))[0]  # uint16
        elif vtype == 3: return struct.unpack('<h', self.f.read(2))[0]  # int16
        elif vtype == 4: return struct.unpack('<I', self.f.read(4))[0]  # uint32
        elif vtype == 5: return struct.unpack('<i', self.f.read(4))[0]  # int32
        elif vtype == 6: return struct.unpack('<f', self.f.read(4))[0]  # float32
        elif vtype == 7: return struct.unpack('<?', self.f.read(1))[0]  # bool
        elif vtype == 8: return self._read_string()  # string
        elif vtype == 9:  # array
            atype, = struct.unpack('<I', self.f.read(4))
            alen, = struct.unpack('<Q', self.f.read(8))
            return [self._read_value(atype) for _ in range(alen)]
        elif vtype == 10: return struct.unpack('<Q', self.f.read(8))[0]  # uint64
        elif vtype == 11: return struct.unpack('<q', self.f.read(8))[0]  # int64
        elif vtype == 12: return struct.unpack('<d', self.f.read(8))[0]  # float64
        else: raise ValueError(f"Unknown vtype {vtype}")

    def tensor_data(self, name):
        ti = self.tensors[name]
        off = self.data_offset + ti['offset']
        return ti, off

    def dequant_q8_0(self, name):
        """Dequantize Q8_0 tensor to float32."""
        ti, off = self.tensor_data(name)
        dims = ti['dims']
        ne0, ne1 = dims[0], dims[1] if len(dims) > 1 else 1
        result = np.zeros((ne1, ne0), dtype=np.float32)

        num_blocks_per_row = ne0 // 32
        bytes_per_row = num_blocks_per_row * 34

        for row in range(ne1):
            row_off = off + row * bytes_per_row
            for blk in range(num_blocks_per_row):
                blk_off = row_off + blk * 34
                d = np.frombuffer(self.mm[blk_off:blk_off+2], dtype=np.float16)[0]
                qs = np.frombuffer(self.mm[blk_off+2:blk_off+34], dtype=np.int8)
                col_start = blk * 32
                result[row, col_start:col_start+32] = float(d) * qs.astype(np.float32)

        return result

    def read_f32(self, name):
        """Read F32 tensor."""
        ti, off = self.tensor_data(name)
        n = 1
        for d in ti['dims']:
            n *= d
        return np.frombuffer(self.mm[off:off+n*4], dtype=np.float32).copy()

# ---- Load model ----
print("Loading GGUF...")
gf = GGUF(GGUF_PATH)

H = 2048
token_id = 39  # First token from "Hello" tokenization

# ---- Embedding (Q8_0) ----
print("Dequantizing embedding for token", token_id, "...")
emb_ti, emb_off = gf.tensor_data('token_embd.weight')
# Just dequant one row
num_blocks = H // 32
bytes_per_row = num_blocks * 34
row_off = emb_off + token_id * bytes_per_row
hidden = np.zeros(H, dtype=np.float32)
for blk in range(num_blocks):
    blk_off = row_off + blk * 34
    d = np.frombuffer(gf.mm[blk_off:blk_off+2], dtype=np.float16)[0]
    qs = np.frombuffer(gf.mm[blk_off+2:blk_off+34], dtype=np.int8)
    hidden[blk*32:(blk+1)*32] = float(d) * qs.astype(np.float32)

print(f"Embedding: {hidden[:4]}")

# ---- RMS Norm (layer 0 input norm) ----
norm_w = gf.read_f32('blk.0.attn_norm.weight')
sq = np.mean(hidden**2)
inv_rms = 1.0 / np.sqrt(sq + 1e-6)
normed = hidden * inv_rms * norm_w
print(f"After norm: {normed[:4]}")

# ---- QKV projection (Q8_0 matvec) ----
print("Computing QKV projection...")
qkv_w = gf.dequant_q8_0('blk.0.attn_qkv.weight')  # [8192, 2048]
qkv_out = qkv_w @ normed  # [8192]
print(f"QKV out[0:4]: {qkv_out[:4]}")
print(f"QKV out[2048:2052]: {qkv_out[2048:2052]}")  # K start
print(f"QKV out[4096:4100]: {qkv_out[4096:4100]}")  # V start

# ---- Alpha/Beta projection ----
alpha_w = gf.dequant_q8_0('blk.0.ssm_alpha.weight')  # [32, 2048]
beta_w = gf.dequant_q8_0('blk.0.ssm_beta.weight')    # [32, 2048]
alpha = alpha_w @ normed  # [32]
beta = beta_w @ normed    # [32]
print(f"Alpha[0:4]: {alpha[:4]}")
print(f"Beta[0:4]: {beta[:4]}")

# ---- Z (gate) projection ----
z_w = gf.dequant_q8_0('blk.0.attn_gate.weight')  # [4096, 2048]
z_out = z_w @ normed  # [4096]
print(f"Z[0:4]: {z_out[:4]}")

# ---- Conv1d ----
conv_w = gf.read_f32('blk.0.ssm_conv1d.weight')  # [4, 8192] → stored as [8192, 4] in memory
conv_w = conv_w.reshape(8192, 4)  # ne0=4 contiguous per channel
# First token: conv state is zero, so only current input contributes
# acc = state[0]*w[0] + state[1]*w[1] + state[2]*w[2] + input*w[3]
# With zero state: acc = input * w[3]
conv_out = qkv_out * conv_w[:, 3]  # element-wise multiply with w[3]
# SiLU activation
conv_out = conv_out / (1.0 + np.exp(-conv_out))
print(f"Conv1d out[0:4]: {conv_out[:4]}")

# ---- QK RMS norm ----
total_key = 16 * 128  # 2048
key_dim = 128
n_k_heads = 16
inv_scale = 1.0 / np.sqrt(key_dim)

q = conv_out[:total_key].copy()
k = conv_out[total_key:2*total_key].copy()

for h in range(n_k_heads):
    base = h * key_dim
    q_head = q[base:base+key_dim]
    k_head = k[base:base+key_dim]
    q_rms = np.sqrt(np.mean(q_head**2) + 1e-6)
    k_rms = np.sqrt(np.mean(k_head**2) + 1e-6)
    q[base:base+key_dim] = q_head / q_rms * inv_scale * inv_scale
    k[base:base+key_dim] = k_head / k_rms * inv_scale

print(f"Q normed[0:4]: {q[:4]}")
print(f"K normed[0:4]: {k[:4]}")

# ---- Decay + Beta gate ----
A_log = gf.read_f32('blk.0.ssm_a')  # [32]
dt_bias = gf.read_f32('blk.0.ssm_dt.bias')  # [32]

A_val = np.exp(A_log)
softplus = np.log(1.0 + np.exp(alpha + dt_bias))
g_decay = np.exp(-A_val * softplus)
beta_gate = 1.0 / (1.0 + np.exp(-beta))

print(f"g_decay[0:4]: {g_decay[:4]}")
print(f"beta_gate[0:4]: {beta_gate[:4]}")

# ---- Delta net recurrence ----
v = conv_out[2*total_key:]  # [4096]
n_v_heads = 32
value_dim = 128
k_per_v = n_v_heads // n_k_heads  # 2

# State is zero at init
state = np.zeros((n_v_heads, value_dim, key_dim), dtype=np.float32)
output = np.zeros(n_v_heads * value_dim, dtype=np.float32)

for vh in range(n_v_heads):
    kh = vh // k_per_v
    g = g_decay[vh]
    b = beta_gate[vh]

    q_head = q[kh*key_dim:(kh+1)*key_dim]
    k_head = k[kh*key_dim:(kh+1)*key_dim]
    v_head = v[vh*value_dim:(vh+1)*value_dim]

    # Decay state (initially zero, so no effect)
    state[vh] *= g

    for vi in range(value_dim):
        # kv_mem = dot(state[vi,:], k)
        kv_mem = np.dot(state[vh, vi, :], k_head)
        # delta
        delta = (v_head[vi] - kv_mem) * b
        # Update state
        state[vh, vi, :] += k_head * delta
        # Output
        output[vh*value_dim + vi] = np.dot(state[vh, vi, :], q_head)

print(f"Delta net out[0:4]: {output[:4]}")
print(f"Delta net out mag: min={output.min():.6f} max={output.max():.6f} mean_abs={np.abs(output).mean():.6f}")

# ---- Gated RMS norm ----
ssm_norm_data = gf.read_f32('blk.0.ssm_norm.weight')  # [128]
print(f"ssm_norm shape: {ssm_norm_data.shape}")

gated_out = np.zeros(n_v_heads * value_dim, dtype=np.float32)
for vh in range(n_v_heads):
    base = vh * value_dim
    vals = output[base:base+value_dim]
    z_vals = z_out[base:base+value_dim]

    # RMS norm
    rms = np.sqrt(np.mean(vals**2) + 1e-6)
    normed_vals = vals / rms

    # SiLU gate
    gate = z_vals / (1.0 + np.exp(-z_vals))

    gated_out[base:base+value_dim] = normed_vals * gate * ssm_norm_data

print(f"Gated RMS norm out[0:4]: {gated_out[:4]}")

# ---- O projection (ssm_out) ----
print("Computing O projection...")
o_w = gf.dequant_q8_0('blk.0.ssm_out.weight')  # [2048, 4096]
o_out = o_w @ gated_out  # [2048]
print(f"O proj out[0:4]: {o_out[:4]}")

# ---- Residual ----
h_after_attn = hidden + o_out
print(f"After residual[0:4]: {h_after_attn[:4]}")
print(f"Residual change: min={o_out.min():.6f} max={o_out.max():.6f}")

print("\nDone! Compare these values with C output.")
