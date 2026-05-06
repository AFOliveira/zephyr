#!/usr/bin/env python3
"""Render a DnCNN3 input/output pair side-by-side.

Usage:
  render_dncnn.py <input_bin> <dump.bin> [--ref <ort_ref_bin>] [--out <png>]

  input_bin   : 240×320×1 FP32 input image (the noisy image)
  dump.bin    : 16 MB device-memory dump (output is at offset 0x300000)
  --ref       : optional FP32 ORT reference for diff visualization
  --out       : output PNG path (default: /tmp/dncnn_view.png)

Saves a 2- or 4-panel PNG; prints per-panel stats.
"""
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt

H, W = 240, 320
OUTPUT_OFFSET = 0x300000

ap = argparse.ArgumentParser()
ap.add_argument("input_bin")
ap.add_argument("dump_bin")
ap.add_argument("--ref", default=None)
ap.add_argument("--out", default="/tmp/dncnn_view.png")
ap.add_argument("--title", default="")
args = ap.parse_args()

inp = np.fromfile(args.input_bin, dtype=np.float32).reshape(H, W)
out = np.frombuffer(open(args.dump_bin, "rb").read()[OUTPUT_OFFSET:OUTPUT_OFFSET + H*W*4],
                    dtype=np.float32).reshape(H, W)
ref = np.fromfile(args.ref, dtype=np.float32).reshape(H, W) if args.ref else None

n_panels = 4 if ref is not None else 2
fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 4.5))
panels = [
    (inp, "INPUT (noisy)", "gray", 0, 1),
    (out, "OUTPUT silicon\n(denoised)", "gray", 0, 1),
]
if ref is not None:
    diff = out - ref
    panels.append((ref,  "ORT FP32 reference", "gray", 0, 1))
    panels.append((diff, f"silicon - ORT\nmax_abs={np.abs(diff).max():.3e}",
                  "RdBu_r", -0.05, 0.05))

for ax, (img, title, cmap, vmin, vmax) in zip(axes, panels):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if args.title:
    plt.suptitle(args.title)
plt.tight_layout()
plt.savefig(args.out, dpi=110, bbox_inches='tight')
print(f"Saved: {args.out}")
print()
print(f"INPUT  range [{inp.min():.4f}, {inp.max():.4f}]   mean {inp.mean():.4f}   std {inp.std():.4f}")
print(f"OUTPUT range [{out.min():.4f}, {out.max():.4f}]   mean {out.mean():.4f}   std {out.std():.4f}")
print(f"Predicted noise removed (input - output): mean_abs={np.abs(inp-out).mean():.4f}, std={np.abs(inp-out).std():.4f}")
if ref is not None:
    diff = out - ref
    print(f"Audit vs ORT: max_abs={np.abs(diff).max():.4e}  mean_abs={np.abs(diff).mean():.4e}")
