#!/usr/bin/env python3
"""FP16 variant of the TFMA 3x3 conv test.

Same dims (IC=OC=16, K=3, output 4x4 → 16 pixels), but operands are FP16.
TFMA fp16->fp32 path: AROWS=16, ACOLS=16 (field=7 → (7+1)*2=16),
BCOLS=16 (field=3 → (3+1)*4=16). Each TFMA dispatches 16*16*16 = 4096 MACs;
9 dispatches accumulating gives the full 3x3 conv.

FP16 scratchpad layout (per sysemu tensor_fma16a32):
  A line (16 OC × 16 IC fp16):  lanes 0..15 hold A[oc, 0..15], lanes 16..31 unused
  B line (per k-pair k0=k/2):    interleaved
                                 lane (2j+0) = B[2k0+0, j]  (fp16)
                                 lane (2j+1) = B[2k0+1, j]  (fp16)
                                 → 8 lines for acols=16 (k0=0..7)

Pre-pack 9 chunks (one per ky,kx); each chunk has 16+8=24 cache lines.
"""
import numpy as np
from pathlib import Path

np.random.seed(0xCAFE)
IC, OC, K = 16, 16, 3
H_OUT, W_OUT = 4, 4

inp_f32 = np.random.randn(IC, H_OUT + K - 1, W_OUT + K - 1).astype(np.float32)
weight_f32 = np.random.randn(OC, IC, K, K).astype(np.float32)

# Quantize: round to fp16 then back to fp32 for "what TFMA actually sees"
inp_fp16 = inp_f32.astype(np.float16).astype(np.float32)
weight_fp16 = weight_f32.astype(np.float16).astype(np.float32)

# FP16 reference: TFMA-order accumulation with fp16 operands, fp32 accumulator.
out_ref = np.zeros((OC, H_OUT, W_OUT), dtype=np.float32)
for oc in range(OC):
    for oy in range(H_OUT):
        for ox in range(W_OUT):
            acc = np.float32(0.0)
            for ky in range(K):
                for kx in range(K):
                    for ic in range(IC):
                        # Inputs round to fp16; product still in fp32.
                        a = np.float16(weight_f32[oc, ic, ky, kx])
                        b = np.float16(inp_f32[ic, oy + ky, ox + kx])
                        acc = np.float32(acc + np.float32(a) * np.float32(b))
            out_ref[oc, oy, ox] = acc

# A pack: weight as fp16, padded.  9 chunks × 16 lines × 32 fp16 lanes.
wpack = np.zeros((K * K, OC, 32), dtype=np.float16)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        for oc in range(OC):
            wpack[k, oc, :IC] = weight_f32[oc, :, ky, kx].astype(np.float16)

# B pack: interleaved im2col.  9 chunks × 8 lines × 32 fp16 lanes.
# For chunk k=ky*3+kx and line k0 (acols/2 line index):
#   lanes (2j+0,2j+1) = (Im2col[2k0+0, j], Im2col[2k0+1, j]) for j=0..15
def im2col_get(ky, kx, ic, j):
    """Im2col activation at chunk (ky,kx), input channel ic, spatial pos j (0..15)."""
    oy = j // W_OUT
    ox = j % W_OUT
    return inp_f32[ic, oy + ky, ox + kx]

bpack = np.zeros((K * K, 8, 32), dtype=np.float16)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        for k0 in range(8):
            for j in range(16):
                bpack[k, k0, 2 * j + 0] = np.float16(im2col_get(ky, kx, 2 * k0 + 0, j))
                bpack[k, k0, 2 * j + 1] = np.float16(im2col_get(ky, kx, 2 * k0 + 1, j))

OUT = Path("/tmp/tfma_probe")
wpack.astype(np.float16).tofile(OUT / "weight_pack_fp16.bin")
bpack.astype(np.float16).tofile(OUT / "im2col_pack_fp16.bin")
out_ref.astype(np.float32).tofile(OUT / "conv_ref_fp16.bin")

w_bytes = wpack.size * 2
b_bytes = bpack.size * 2
print(f"weight_pack_fp16: {wpack.shape} = {w_bytes} B "
      f"({w_bytes // 64} cache lines, expect 9*16=144)")
print(f"im2col_pack_fp16: {bpack.shape} = {b_bytes} B "
      f"({b_bytes // 64} cache lines, expect 9*8=72)")
print(f"out_ref range: [{out_ref.min():.3f}, {out_ref.max():.3f}]")
print(f"out_ref[0,:,:] = \n{out_ref[0]}")
