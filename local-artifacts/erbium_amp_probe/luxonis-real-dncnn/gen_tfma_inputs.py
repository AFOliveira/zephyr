#!/usr/bin/env python3
"""Generate non-trivial 16x16 fp32 A and B matrices and the expected
A @ B reference. Layout matches what the kernel will tensor_load:
  - A: 16 cache-lines × 16 fp32 each (row-major, A[i,k] in line i, lane k)
  - B: 16 cache-lines × 16 fp32 each (row-major, B[k,j] in line k, lane j)
"""
import numpy as np

np.random.seed(0xC0FFEE)
A = np.random.randn(16, 16).astype(np.float32)
B = np.random.randn(16, 16).astype(np.float32)
C_ref = (A @ B).astype(np.float32)

# Pack 16 lines × 16 fp32 = 1024 bytes per matrix.
A.tofile("/tmp/tfma_probe/A.bin")
B.tofile("/tmp/tfma_probe/B.bin")
C_ref.tofile("/tmp/tfma_probe/C_ref.bin")

print(f"A range: [{A.min():.3f}, {A.max():.3f}]")
print(f"B range: [{B.min():.3f}, {B.max():.3f}]")
print(f"C_ref range: [{C_ref.min():.3f}, {C_ref.max():.3f}]")
print(f"C_ref[0,:4] = {C_ref[0,:4]}")
print(f"C_ref[15,12:] = {C_ref[15,12:]}")
