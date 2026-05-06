#!/usr/bin/env python3
"""Repack DnCNN3 hidden-layer weights for TFMA dispatch.

Existing layout (`dncnn3_luxonis_240x320_weights_packed_f32.bin`):
   W0[oc=64, ky*kx*ic=9]            (first conv: IC=1, K=3)
   B0[oc=64]
   WH[layer=18, oc=64, ky=3, kx=3, ic=64]
   BH[layer=18, oc=64]
   WF[oc=1, ic=64, ky=3, kx=3]
   BF[1]

TFMA dispatch (FP32) consumes operands in the role:
  C[p, oc] = A[p, ic] @ B[ic, oc]
where:
  - A is per-tile activation (NHWC: rows = 16 spatial pixels, cols = 16 IC)
  - B is per-tile weight (rows = 16 IC, cols = 16 OC), pre-transposed
The kernel iterates 9 (ky,kx) × 4 ic-chunks × 4 oc-chunks per spatial tile.

This script emits TFMA-ready weight blob:
   WH_tfma[L=18, k=9 (ky*3+kx), oc_chunk=4, ic_chunk=4, ic_in_tile=16, oc_in_tile=16]
   total: 18 * 9 * 4 * 4 * 16 * 16 * 4 = 2,654,208 bytes (= 2.53 MB)

Each (L, k, oc_c, ic_c) slice is exactly 16 cache lines × 16 fp32 = 1 KB,
loadable via a single tensor_load(num_lines=15, stride=64).

The first/last conv weights stay in their original layout — those layers
keep VPU code (negligible compute, 0.04 % of total).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")

W = np.fromfile(DIR / "dncnn3_luxonis_240x320_weights_packed_f32.bin",
                dtype=np.float32)

# Match the offsets in dncnn3_luxonis_real_vpu.c:50-63
W0_FLOATS = 64 * 9                  # 576
B0_FLOATS = 64
WH_FLOATS = 18 * 64 * 3 * 3 * 64    # 663552
BH_FLOATS = 18 * 64                 # 1152
WF_FLOATS = 9 * 64                  # 576
BF_FLOATS = 1

assert W.size == W0_FLOATS + B0_FLOATS + WH_FLOATS + BH_FLOATS + WF_FLOATS + BF_FLOATS, W.size

off = 0
W0 = W[off:off + W0_FLOATS].reshape(64, 9);                off += W0_FLOATS
B0 = W[off:off + B0_FLOATS].copy();                        off += B0_FLOATS
WH = W[off:off + WH_FLOATS].reshape(18, 64, 3, 3, 64);     off += WH_FLOATS
BH = W[off:off + BH_FLOATS].reshape(18, 64).copy();        off += BH_FLOATS
WF = W[off:off + WF_FLOATS].reshape(1, 64, 3, 3);          off += WF_FLOATS
BF = W[off:off + BF_FLOATS].copy();                        off += BF_FLOATS
assert off == W.size

# Repack hidden into TFMA layout: [L=18, k=9, oc_c=4, ic_c=4, ic=16, oc=16]
WH_tfma = np.zeros((18, 9, 4, 4, 16, 16), dtype=np.float32)
for L in range(18):
    for ky in range(3):
        for kx in range(3):
            k = ky * 3 + kx
            for oc_c in range(4):
                for ic_c in range(4):
                    g_oc = slice(oc_c * 16, oc_c * 16 + 16)   # 16 OCs
                    g_ic = slice(ic_c * 16, ic_c * 16 + 16)   # 16 ICs
                    # WH[L, oc, ky, kx, ic] -> WH_tfma[L, k, oc_c, ic_c, ic, oc]
                    tile = WH[L, g_oc, ky, kx, g_ic]   # shape (16 OC, 16 IC)
                    WH_tfma[L, k, oc_c, ic_c, :, :] = tile.T  # transpose to (16 IC, 16 OC)

WH_tfma.astype(np.float32).tofile(DIR / "dncnn3_tfma_hidden_weights_fp32.bin")

# Hidden biases stay simple [L, OC]
BH.astype(np.float32).tofile(DIR / "dncnn3_tfma_hidden_biases_fp32.bin")

# Original WH layout [L, OC, ky, kx, IC] flat — for scalar conv_hidden reference
WH.astype(np.float32).tofile(DIR / "dncnn3_tfma_hidden_orig_weights_fp32.bin")

# First conv weights (IC=1 → OC=64). Keep separate for VPU first-conv path.
# Layout: [oc=64, ky=3, kx=3] flat to [oc, k]
W0.astype(np.float32).tofile(DIR / "dncnn3_tfma_first_weights_fp32.bin")
B0.astype(np.float32).tofile(DIR / "dncnn3_tfma_first_biases_fp32.bin")

# Final conv (IC=64 → OC=1). Keep VPU path.
WF.astype(np.float32).tofile(DIR / "dncnn3_tfma_final_weights_fp32.bin")
BF.astype(np.float32).tofile(DIR / "dncnn3_tfma_final_biases_fp32.bin")

print(f"WH_tfma  {WH_tfma.shape} = {WH_tfma.nbytes:,} B")
print(f"BH       {BH.shape}        = {BH.nbytes:,} B")
print(f"W0       {W0.shape}        = {W0.nbytes:,} B")
print(f"B0       {B0.shape}        = {B0.nbytes:,} B")
print(f"WF       {WF.shape}        = {WF.nbytes:,} B")
print(f"BF       {BF.shape}        = {BF.nbytes:,} B")

# Sanity: per (L, k, oc_c, ic_c) tile is 1024 bytes (16 cache lines of 16 fp32)
tile_bytes = 16 * 16 * 4
n_tiles = 18 * 9 * 4 * 4
print(f"per-tile bytes: {tile_bytes}  total tiles: {n_tiles}  total: {n_tiles * tile_bytes:,}")
