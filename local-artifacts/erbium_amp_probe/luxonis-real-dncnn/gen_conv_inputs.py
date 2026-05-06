#!/usr/bin/env python3
"""Generate inputs for a small TFMA 3x3 conv test.

Layer: input [IC=16, 6, 6] (4x4 output + 2-pixel halo padded), weights
[OC=16, IC=16, 3, 3], output [OC=16, 4, 4].

Pre-im2col the input into 9 chunks, each [IC=16, 16] (row-major 4x4 spatial,
shifted by (ky, kx)). Pre-pack weights into 9 chunks, each [OC=16, IC=16].

Each chunk is exactly 16 cache lines × 16 fp32 lanes — a TFMA-ready operand.
"""
import numpy as np
from pathlib import Path

np.random.seed(0xCAFE)
IC, OC, K = 16, 16, 3
H_OUT, W_OUT = 4, 4

# Random fp32 inputs
inp = np.random.randn(IC, H_OUT + K - 1, W_OUT + K - 1).astype(np.float32)
weight = np.random.randn(OC, IC, K, K).astype(np.float32)
bias = np.zeros(OC, dtype=np.float32)  # zero bias for simplicity

# Reference: match TFMA's accumulation order — sum over (ky,kx) OUTER, ic INNER.
# (TFMA loops k=0..acols-1 inside each chunk, then we accumulate across 9 chunks.)
out_ref = np.zeros((OC, H_OUT, W_OUT), dtype=np.float32)
for oc in range(OC):
    for oy in range(H_OUT):
        for ox in range(W_OUT):
            acc = np.float32(0.0)
            for ky in range(K):
                for kx in range(K):
                    for ic in range(IC):
                        acc = np.float32(acc + np.float32(weight[oc, ic, ky, kx])
                                                 * np.float32(inp[ic, oy + ky, ox + kx]))
            out_ref[oc, oy, ox] = acc

# im2col: 9 chunks (one per ky,kx). Each chunk is [IC, 16] = 16 cache lines.
# For chunk k = ky*3 + kx, ic-th line holds inp[ic, ky:ky+H_OUT, kx:kx+W_OUT].flatten()
im2col = np.zeros((K * K, IC, H_OUT * W_OUT), dtype=np.float32)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        for ic in range(IC):
            patch = inp[ic, ky:ky + H_OUT, kx:kx + W_OUT]   # [4, 4]
            im2col[k, ic, :] = patch.flatten()

# Weight pack: 9 chunks, each [OC, IC] = 16 cache lines.
wpack = np.zeros((K * K, OC, IC), dtype=np.float32)
for ky in range(K):
    for kx in range(K):
        k = ky * K + kx
        wpack[k, :, :] = weight[:, :, ky, kx]

OUT = Path("/tmp/tfma_probe")
OUT.mkdir(parents=True, exist_ok=True)
im2col.astype(np.float32).tofile(OUT / "im2col_pack.bin")
wpack.astype(np.float32).tofile(OUT / "weight_pack.bin")
out_ref.astype(np.float32).tofile(OUT / "conv_ref.bin")

# Check sizes (each line = 64 B = 16 fp32)
print(f"im2col shape {im2col.shape} = {im2col.size * 4} B "
      f"({im2col.size * 4 // 64} cache lines, expect 9*16=144)")
print(f"wpack  shape {wpack.shape} = {wpack.size * 4} B "
      f"({wpack.size * 4 // 64} cache lines, expect 9*16=144)")
print(f"ref out shape {out_ref.shape}")
print(f"ref out range: [{out_ref.min():.3f}, {out_ref.max():.3f}]")
print(f"ref out[0,:,:] = \n{out_ref[0]}")
