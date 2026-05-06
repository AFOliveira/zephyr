#!/usr/bin/env python3
"""Audit a tiled 240x320 DnCNN3 silicon inference.

For each of 20 tiles:
  1. Read the device dump (dump.bin), pull the kernel-side summary at 0x1000
     and confirm per-tile audit passed (max_abs vs linked tile-ORT-ref bin).
  2. Pull the per-tile output region from OUTPUT_OFFSET (0x300000) of the dump.
  3. Crop per audit_a1_geometry.json to get the kept output region.
  4. Place into the stitched 240x320 image at the tile's output coordinates.

Then compare stitched image vs full ORT 240x320 reference.
Reports per-tile + global audit numbers and timing summary.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

SUMMARY = struct.Struct("<16I")
DNCNN_MAGIC = 0xD3C11004
OUTPUT_OFFSET = 0x300000


def fnv1a(arr: np.ndarray) -> int:
    h = 2166136261
    for w in arr.astype("<f4", copy=False).reshape(-1).view(np.uint32):
        h ^= int(w); h = (h * 16777619) & 0xFFFFFFFF
    return h


def parse_summary(dump: Path) -> dict:
    with dump.open("rb") as f:
        f.seek(0x1000)
        s = SUMMARY.unpack(f.read(64))
    ops = s[11] | (s[12] << 32)
    return {
        "magic": s[0], "active_harts": s[1], "passes": s[2],
        "width": s[3], "height": s[4], "channels": s[5], "layers": s[6],
        "active_mask": s[7], "done_count": s[8],
        "output_hash": s[9], "reference_hash": s[10],
        "ops": ops, "max_abs": s[13]/1e9, "mean_abs": s[14]/1e9,
    }


def read_tile_output(dump: Path, h: int, w: int) -> np.ndarray:
    """Read input_h × input_w FP32 outputs from OUTPUT_OFFSET of the device dump."""
    with dump.open("rb") as f:
        f.seek(OUTPUT_OFFSET)
        data = f.read(h * w * 4)
    if len(data) != h * w * 4:
        raise ValueError(f"short read: got {len(data)} bytes, expected {h*w*4}")
    return np.frombuffer(data, dtype=np.float32).reshape(h, w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/tiled-fp32/audit_a1_geometry.json")
    ap.add_argument("--runs-dir", default="/tmp/tiled_240x320_runs")
    ap.add_argument("--full-ort", default="/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/dncnn3_luxonis_240x320_ort_output_f32.bin")
    ap.add_argument("--out", default="/tmp/tiled_240x320_stitched.bin")
    ap.add_argument("--report", default="/tmp/tiled_240x320_report.json")
    args = ap.parse_args()

    g = json.loads(Path(args.geometry).read_text())
    img_h, img_w = g["img_h"], g["img_w"]
    full_ort = np.fromfile(args.full_ort, dtype=np.float32).reshape(img_h, img_w)

    stitched = np.zeros((img_h, img_w), dtype=np.float32)
    per_tile = []

    runs_dir = Path(args.runs_dir)
    print(f"=== per-tile audit ===")
    print(f"{'tile':>4s} {'in_h x in_w':>12s} {'wait_s':>8s} {'GOPS':>7s} {'max_abs':>10s} {'audit':>6s} {'crop_match':>11s}")
    print("-" * 80)
    n_pass = 0
    total_wait_s = 0.0
    total_ops = 0
    for t in g["tiles"]:
        tid = t["tile_id"]
        run_dir = runs_dir / f"tile_{tid:02d}"
        # Find the run dir if naming differs (we have several layouts)
        if not run_dir.exists():
            for cand in runs_dir.iterdir():
                if cand.is_dir() and (f"tile_{tid:02d}" in cand.name or f"tile{tid:02d}" in cand.name):
                    run_dir = cand; break
        manifest = json.loads((run_dir / "manifest.json").read_text())
        wait_s = manifest.get("kernel_wait_s") or 0
        dump = run_dir / "dump.bin"
        s = parse_summary(dump)
        magic_ok = s["magic"] == DNCNN_MAGIC
        all_done = s["done_count"] == s["active_harts"]
        per_tile_pass = magic_ok and all_done and s["max_abs"] <= 1e-5
        if per_tile_pass:
            n_pass += 1
        total_wait_s += wait_s
        total_ops += s["ops"]
        gops = s["ops"] / wait_s / 1e9 if wait_s else 0.0

        # Pull tile output, crop, place into stitched
        out = read_tile_output(dump, t["input_h"], t["input_w"])
        cropped = out[t["crop_y0"]:t["crop_y1"], t["crop_x0"]:t["crop_x1"]]
        stitched[t["output_y0"]:t["output_y1"], t["output_x0"]:t["output_x1"]] = cropped

        # Compare cropped vs full-ORT region
        full_region = full_ort[t["output_y0"]:t["output_y1"], t["output_x0"]:t["output_x1"]]
        crop_max_abs = float(np.abs(cropped - full_region).max())
        per_tile.append({
            "tile_id": tid, "input_h": t["input_h"], "input_w": t["input_w"],
            "wait_s": wait_s, "gops": gops, "max_abs_per_tile": s["max_abs"],
            "crop_max_abs_vs_full_ort": crop_max_abs,
            "per_tile_audit_pass": per_tile_pass,
        })
        print(f"{tid:>4d} {t['input_h']:>5d} x{t['input_w']:>4d}   {wait_s:>8.4f} {gops:>7.3f} "
              f"{s['max_abs']:>10.2e} {'OK' if per_tile_pass else 'FAIL':>6s} {crop_max_abs:>11.2e}")

    # Stitched image vs full ORT
    diff = np.abs(stitched - full_ort)
    stitched_max_abs = float(diff.max())
    stitched_mean_abs = float(diff.mean())
    bit_exact = bool(stitched.astype("<f4").tobytes() == full_ort.astype("<f4").tobytes())
    allclose_1e5 = stitched_max_abs <= 1e-5

    full_image_gops = total_ops / total_wait_s / 1e9 if total_wait_s else 0
    print()
    print(f"=== stitched 240x320 vs full ORT ===")
    print(f"  bit_exact            = {bit_exact}")
    print(f"  allclose @ 1e-5      = {allclose_1e5}")
    print(f"  max_abs              = {stitched_max_abs:.3e}")
    print(f"  mean_abs             = {stitched_mean_abs:.3e}")
    print(f"  stitched_hash        = 0x{fnv1a(stitched):08x}")
    print(f"  full_ort_hash        = 0x{fnv1a(full_ort):08x}")
    print()
    print(f"=== timing aggregate ===")
    print(f"  per-tile audits pass = {n_pass}/{len(g['tiles'])}")
    print(f"  total kernel wait_s  = {total_wait_s:.4f}")
    print(f"  total ops            = {total_ops:,}")
    print(f"  full-image GOPS      = {full_image_gops:.3f}")
    print(f"  full-image latency   = {total_wait_s:.2f} s/inference")
    print()

    stitched.tofile(args.out)
    print(f"wrote stitched output to {args.out}")

    report = {
        "tiles": per_tile,
        "stitched": {
            "max_abs": stitched_max_abs,
            "mean_abs": stitched_mean_abs,
            "bit_exact": bit_exact,
            "allclose_1e5": allclose_1e5,
            "stitched_hash": f"0x{fnv1a(stitched):08x}",
            "full_ort_hash": f"0x{fnv1a(full_ort):08x}",
        },
        "timing": {
            "n_tiles": len(g["tiles"]),
            "per_tile_audit_pass": n_pass,
            "total_kernel_wait_s": total_wait_s,
            "total_ops": int(total_ops),
            "full_image_gops": full_image_gops,
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"wrote report to {args.report}")
    return 0 if (allclose_1e5 and n_pass == len(g["tiles"])) else 1


if __name__ == "__main__":
    sys.exit(main())
