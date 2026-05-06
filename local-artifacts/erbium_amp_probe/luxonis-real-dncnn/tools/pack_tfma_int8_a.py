#!/usr/bin/env python3
"""Repack INT8 hidden weights into A-side TFMA layout.

Source blob: dncnn3_int8_hidden_weights_int8.bin
  shape [L=18, OC=64, ky=3, kx=3, IC=64] int8

Target blob: dncnn3_tfma_int8_a_pack.bin
  shape [L=18, oc_tile=4, tap=9, oc_in_tile=16, ic=64] int8 = 663,552 B

Per (L, oc_tile, tap), the blob holds 16 SCP lines (one per OC in the tile),
each line = 64 int8 IC values. Loaded as A operand by tensor_fma type=3:
  A[oc_in_tile, ic] = src[L, 16*oc_tile + oc_in_tile, ky, kx, ic]
"""
from pathlib import Path
import numpy as np

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")
SRC = DIR / "dncnn3_int8_hidden_weights_int8.bin"
DST = DIR / "dncnn3_tfma_int8_a_pack.bin"

L, OC, K, IC = 18, 64, 3, 64
OC_TILES = 4
OC_PER_TILE = 16
TAPS = K * K

w = np.fromfile(SRC, dtype=np.int8).reshape(L, OC, K, K, IC)
out = np.zeros((L, OC_TILES, TAPS, OC_PER_TILE, IC), dtype=np.int8)

for layer in range(L):
    for oct in range(OC_TILES):
        for ky in range(K):
            for kx in range(K):
                tap = ky * K + kx
                for o in range(OC_PER_TILE):
                    oc = oct * OC_PER_TILE + o
                    out[layer, oct, tap, o, :] = w[layer, oc, ky, kx, :]

out.tofile(DST)
print(f"Wrote {DST}: shape {out.shape}, {out.nbytes:,} B")
print(f"  per-(layer, oc_tile, tap): {OC_PER_TILE * IC} = 1024 B")
print(f"  bytes per layer: {OC_TILES * TAPS * OC_PER_TILE * IC:,}")
