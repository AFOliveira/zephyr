#!/usr/bin/env python3
"""Host reference for first conv + N hidden layers. Vectorized via scipy.

Output: NHWC padded buffer matching kernel's static_act0/act1 layout
(shape (242, 322, 64), with 1-pixel zero halo on each side)."""
import argparse, numpy as np
from pathlib import Path
from scipy.signal import correlate2d

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")

ap = argparse.ArgumentParser()
ap.add_argument('--hidden', type=int, default=1)
ap.add_argument('--out', type=str, default='/tmp/dncnn_tfma/host_act_ref.npy')
args = ap.parse_args()

W = np.fromfile(DIR / "dncnn3_luxonis_240x320_weights_packed_f32.bin", dtype=np.float32)
off = 0
W0 = W[off:off+576].reshape(64, 3, 3); off += 576
B0 = W[off:off+64].copy(); off += 64
WH = W[off:off+18*64*9*64].reshape(18, 64, 3, 3, 64); off += 18*64*9*64
BH = W[off:off+18*64].reshape(18, 64).copy(); off += 18*64

img = np.fromfile(DIR / "dncnn3_luxonis_240x320_input_f32.bin", dtype=np.float32).reshape(240, 320)
H, W_ = 240, 320
IC, OC, K = 64, 64, 3

def relu(x): return np.maximum(np.float32(0.0), x)

# Conv first: 1 channel -> 64 channels. Use scipy correlate2d (no kernel flip = correlation).
# ONNX conv = cross-correlation (no flip).
act = np.zeros((H, W_, OC), dtype=np.float32)
for oc in range(OC):
    c = correlate2d(img, W0[oc], mode='same', boundary='fill', fillvalue=0).astype(np.float32)
    act[:, :, oc] = relu(c + B0[oc])
print(f'after conv_first: act range [{act.min():.4f}, {act.max():.4f}]')

# Hidden layers — use im2col-style einsum via convolve over channels
for L in range(args.hidden):
    new = np.zeros((H, W_, OC), dtype=np.float32)
    # zero-pad spatial only (channels untouched)
    act_pad = np.pad(act, ((1, 1), (1, 1), (0, 0)), mode='constant').astype(np.float32)
    # for each OC: sum over IC of correlate(act_in[:,:,ic], WH[L, oc, :, :, ic])
    for oc in range(OC):
        acc = np.zeros((H, W_), dtype=np.float32)
        for ic in range(IC):
            kernel = WH[L, oc, :, :, ic].astype(np.float32)  # 3x3
            # correlate uses "same" so input must be H x W, output H x W.
            c = correlate2d(act[:, :, ic], kernel, mode='same', boundary='fill', fillvalue=0).astype(np.float32)
            acc += c
        new[:, :, oc] = relu(acc + BH[L, oc])
    act = new
    print(f'after hidden {L}: act range [{act.min():.4f}, {act.max():.4f}]')

# Pad to (H+2, W+2, OC) NHWC like the kernel
out = np.zeros((H+2, W_+2, OC), dtype=np.float32)
out[1:H+1, 1:W_+1, :] = act
np.save(args.out, out)
print(f'saved {args.out} shape {out.shape}')
