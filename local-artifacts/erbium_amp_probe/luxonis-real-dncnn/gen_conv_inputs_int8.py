#!/usr/bin/env python3
"""INT8 TFMA conv test.

Quantize weights and activations symmetrically to int8 (per-tensor scale).
TFMA tensor_ima8a32: int8 inputs, int32 accumulator.

ACOLS field=3 → actual=16 (same as fp32 here for fair comparison; INT8 path
multiplies field+1 by 4, but field=3 gives 16, matching IC=16).

INT8 scratchpad layout per sysemu:
  A line: 64 int8 lanes (16 used: lanes 0..15 = A[oc, 0..15])
  B line per k-quad k0=k/4: lanes (4j+0..4j+3) = B[4k0+0..4k0+3, j]
  → acols=16 → 4 B lines (k0=0..3)
"""
import numpy as np
from pathlib import Path

np.random.seed(0xCAFE)
IC, OC, K = 16, 16, 3
H_OUT, W_OUT = 4, 4

inp_f32 = np.random.randn(IC, H_OUT + K - 1, W_OUT + K - 1).astype(np.float32)
weight_f32 = np.random.randn(OC, IC, K, K).astype(np.float32)

# Symmetric quantization
w_scale = np.float32(np.abs(weight_f32).max() / 127.0)
a_scale = np.float32(np.abs(inp_f32).max() / 127.0)
weight_int8 = np.clip(np.round(weight_f32 / w_scale), -128, 127).astype(np.int8)
inp_int8 = np.clip(np.round(inp_f32 / a_scale), -128, 127).astype(np.int8)
print(f"w_scale={w_scale:.6f}  a_scale={a_scale:.6f}")
print(f"weight_int8 range [{weight_int8.min()}, {weight_int8.max()}]")
print(f"inp_int8     range [{inp_int8.min()}, {inp_int8.max()}]")

# Reference: int32 accumulation of int8*int8 products. Match TFMA's order
# (ky,kx outer, ic inner), per-pair to mirror the inner k+=4 loop.
out_ref_int32 = np.zeros((OC, H_OUT, W_OUT), dtype=np.int32)
for oc in range(OC):
    for oy in range(H_OUT):
        for ox in range(W_OUT):
            acc = np.int32(0)
            for ky in range(K):
                for kx in range(K):
                    for ic in range(IC):
                        a = np.int32(weight_int8[oc, ic, ky, kx])
                        b = np.int32(inp_int8[ic, oy + ky, ox + kx])
                        acc = np.int32(acc + a * b)
            out_ref_int32[oc, oy, ox] = acc

# Equivalent fp32 reference (int32 * scales)
out_ref_fp32 = (out_ref_int32.astype(np.float32) *
                (w_scale * a_scale)).astype(np.float32)

# Pack A: 9 chunks × 16 lines × 64 int8 lanes (16 used per row).
wpack = np.zeros((K * K, OC, 64), dtype=np.int8)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        for oc in range(OC):
            wpack[k, oc, :IC] = weight_int8[oc, :, ky, kx]

# Pack B: 9 chunks × 4 lines (k0=0..3) × 64 int8 lanes.
def im2col_get_int8(ky, kx, ic, j):
    oy = j // W_OUT
    ox = j % W_OUT
    return inp_int8[ic, oy + ky, ox + kx]

bpack = np.zeros((K * K, 4, 64), dtype=np.int8)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        for k0 in range(4):
            for j in range(16):
                for d in range(4):
                    bpack[k, k0, 4 * j + d] = im2col_get_int8(ky, kx, 4 * k0 + d, j)

OUT = Path("/tmp/tfma_probe")
wpack.tofile(OUT / "weight_pack_int8.bin")
bpack.tofile(OUT / "im2col_pack_int8.bin")

# Save kernel side reference (int32) and host fp32 equivalent + scales
out_ref_int32.tofile(OUT / "conv_ref_int32.bin")
out_ref_fp32.tofile(OUT / "conv_ref_fp32_from_int8.bin")
np.array([w_scale, a_scale], dtype=np.float32).tofile(OUT / "int8_scales.bin")

# Also keep an FP32 reference (computed without quantization) for context
inp_fp16 = inp_f32  # keep as reference fp32
weight_fp16 = weight_f32
out_ref_truefp32 = np.zeros((OC, H_OUT, W_OUT), dtype=np.float32)
for oc in range(OC):
    for oy in range(H_OUT):
        for ox in range(W_OUT):
            acc = np.float32(0.0)
            for ky in range(K):
                for kx in range(K):
                    for ic in range(IC):
                        acc = np.float32(acc + np.float32(weight_f32[oc, ic, ky, kx])
                                                 * np.float32(inp_f32[ic, oy + ky, ox + kx]))
            out_ref_truefp32[oc, oy, ox] = acc
out_ref_truefp32.tofile(OUT / "conv_ref_truefp32.bin")

quant_err = np.abs(out_ref_fp32 - out_ref_truefp32).max()
print(f"int8-vs-fp32 quantization error (rms): {np.abs(out_ref_fp32 - out_ref_truefp32).mean():.4f}")
print(f"int8-vs-fp32 quantization error (max): {quant_err:.4f}")
print(f"out_ref_int32 range: [{out_ref_int32.min()}, {out_ref_int32.max()}]")
print(f"out_ref_int32[0,:,:] = \n{out_ref_int32[0]}")
