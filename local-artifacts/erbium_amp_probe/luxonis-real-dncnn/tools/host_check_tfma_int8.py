#!/usr/bin/env python3
"""Host-side simulation of the TFMA INT8 hidden-conv kernel for ONE output tile.

Verifies:
  1. A-pack layout matches what tensor_fma reads (permutation correctness).
  2. B-pack layout matches what tensor_fma reads.
  3. Marshalling math (dequant + bias + ReLU + requant) matches the v1 scalar path.

Compares against direct int32 conv on the same int8 inputs/weights.
"""
import numpy as np
from pathlib import Path

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")

# Load blobs (use layer 0 as the test layer)
W_orig = np.fromfile(DIR / "dncnn3_int8_hidden_weights_int8.bin", dtype=np.int8)
W_orig = W_orig.reshape(18, 64, 3, 3, 64)   # [L, OC, ky, kx, IC]

W_pack = np.fromfile(DIR / "dncnn3_tfma_int8_a_pack.bin", dtype=np.int8)
W_pack = W_pack.reshape(18, 4, 9, 16, 64)   # [L, oc_tile, tap, oc_in_tile, ic]

w_scales = np.fromfile(DIR / "dncnn3_int8_hidden_w_scales_fp32.bin", dtype=np.float32).reshape(18, 64)
a_scales = np.fromfile(DIR / "dncnn3_int8_act_scales_fp32.bin", dtype=np.float32)
inv_a    = np.fromfile(DIR / "dncnn3_int8_inv_act_scales_fp32.bin", dtype=np.float32)
b_hidden = np.fromfile(DIR / "dncnn3_tfma_hidden_biases_fp32.bin", dtype=np.float32).reshape(18, 64)

# Random int8 activation 240x320x64 for layer 0 input (will only verify one tile)
np.random.seed(0)
ACT = np.random.randint(-127, 128, size=(240, 320, 64), dtype=np.int8)

L = 0
OY_BASE = 5    # arbitrary interior row
OX_BASE = 32   # arbitrary interior col, +16 wide tile
TILE_W = 16
OC_PER_TILE = 16
K = 3

# 1. Sanity: W_pack matches reordered W_orig
for oct in range(4):
    for tap in range(9):
        ky, kx = tap // 3, tap % 3
        for o in range(OC_PER_TILE):
            oc = oct * OC_PER_TILE + o
            assert np.array_equal(W_pack[L, oct, tap, o, :], W_orig[L, oc, ky, kx, :]), \
                f"W_pack mismatch at oct={oct} tap={tap} o={o}"
print("OK: A-pack layout matches W_orig reordering")

# 2. Build B-pack the same way the kernel does it
B = np.zeros((9, 16, 64), dtype=np.int8)   # [tap, k0, lane=j*4+d]
for ky in range(K):
    y_in = OY_BASE + ky - 1
    for kx in range(K):
        tap = ky * K + kx
        for j in range(TILE_W):
            x_in = OX_BASE + kx - 1 + j
            if 0 <= y_in < 240 and 0 <= x_in < 320:
                src_p = ACT[y_in, x_in, :]   # 64 ICs
                for k0 in range(16):
                    for d in range(4):
                        B[tap, k0, j*4+d] = src_p[4*k0 + d]
            # else: leave zeros

# 3. Compute reference output: int32 conv at this tile
ref_int32 = np.zeros((4, OC_PER_TILE, TILE_W), dtype=np.int32)   # [oc_tile, oc_in, j]
for oct in range(4):
    for o in range(OC_PER_TILE):
        oc = oct * OC_PER_TILE + o
        for j in range(TILE_W):
            x_out = OX_BASE + j
            acc = np.int32(0)
            for ky in range(K):
                y_in = OY_BASE + ky - 1
                if not (0 <= y_in < 240): continue
                for kx in range(K):
                    x_in = x_out + kx - 1
                    if not (0 <= x_in < 320): continue
                    for ic in range(64):
                        a = np.int32(ACT[y_in, x_in, ic])
                        w = np.int32(W_orig[L, oc, ky, kx, ic])
                        acc = np.int32(acc + a * w)
            ref_int32[oct, o, j] = acc

# 4. Compute via TFMA semantics:
#    For each output tile (oc_tile fixed):
#      Sum over taps:
#        For each oc_in (i=0..15) and j=0..15:
#          For each k=0..63 step 4:
#            k0 = k/4
#            a1..a4 from W_pack[L, oct, tap, i, k..k+3]
#            b1..b4 from B[tap, k0, j*4..j*4+3]
#            acc += a1*b1 + a2*b2 + a3*b3 + a4*b4
sim_int32 = np.zeros((4, OC_PER_TILE, TILE_W), dtype=np.int32)
for oct in range(4):
    tenc = np.zeros((OC_PER_TILE, TILE_W), dtype=np.int32)
    for tap in range(9):
        # First-pass = (tap==0) overwrites; else accumulates
        for i in range(OC_PER_TILE):
            for j in range(TILE_W):
                pair_sum = np.int32(0)
                for k in range(0, 64, 4):
                    a1 = np.int32(W_pack[L, oct, tap, i, k+0])
                    a2 = np.int32(W_pack[L, oct, tap, i, k+1])
                    a3 = np.int32(W_pack[L, oct, tap, i, k+2])
                    a4 = np.int32(W_pack[L, oct, tap, i, k+3])
                    b1 = np.int32(B[tap, k//4, j*4+0])
                    b2 = np.int32(B[tap, k//4, j*4+1])
                    b3 = np.int32(B[tap, k//4, j*4+2])
                    b4 = np.int32(B[tap, k//4, j*4+3])
                    pair_sum = np.int32(pair_sum + a1*b1 + a2*b2 + a3*b3 + a4*b4)
                if tap == 0:
                    tenc[i, j] = pair_sum
                else:
                    tenc[i, j] = np.int32(tenc[i, j] + pair_sum)
    sim_int32[oct] = tenc

err = np.abs(ref_int32 - sim_int32).max()
print(f"int32 max diff (TFMA-pack vs direct conv): {err}")
assert err == 0, f"FAILURE: mismatch of {err}"
print("OK: TFMA int32 output bit-exact vs reference at this tile")

# 5. Verify marshalling: dequant + bias + ReLU + requant
a_in = a_scales[L]
inv_out = inv_a[L + 1]
ws = w_scales[L]
bs = b_hidden[L]

def sat_int8(x):
    v = int(np.round(x))
    return max(-128, min(127, v))

# Reference scalar v1 path
ref_int8 = np.zeros((4, OC_PER_TILE, TILE_W), dtype=np.int8)
for oct in range(4):
    for o in range(OC_PER_TILE):
        oc = oct * OC_PER_TILE + o
        for j in range(TILE_W):
            v = ref_int32[oct, o, j]
            dq = float(v) * float(ws[oc]) * float(a_in)
            r = max(0.0, dq + float(bs[oc]))
            ref_int8[oct, o, j] = sat_int8(r * float(inv_out))

# Sim (matches kernel marshalling)
sim_int8 = np.zeros((4, OC_PER_TILE, TILE_W), dtype=np.int8)
for oct in range(4):
    for r in range(OC_PER_TILE):
        oc = oct * OC_PER_TILE + r
        scale_r = float(ws[oc]) * float(a_in)
        bias_r = float(bs[oc])
        for j in range(TILE_W):
            v = sim_int32[oct, r, j]
            dq = float(v) * scale_r + bias_r
            rl = max(0.0, dq)
            sim_int8[oct, r, j] = sat_int8(rl * float(inv_out))

m = (ref_int8 != sim_int8).sum()
print(f"int8 mismatch count after marshalling: {m}")
assert m == 0, f"marshalling mismatch: {m} elements"
print("OK: marshalling matches v1 reference")

print("\nALL CHECKS PASS — kernel layout + math verified on host")
