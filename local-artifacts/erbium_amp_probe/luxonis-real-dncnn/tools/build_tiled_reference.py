#!/usr/bin/env python3
"""Track A Phase A1.1: tiled DnCNN3 reference.

Run the ONNX model on each (input_y0..input_y1, input_x0..input_x1) tile and
crop the relevant output region. Stitch outputs into the full image. Verify the
stitched result matches a single full-resolution ORT run within tolerance for
the chosen halo size.

Outputs:
  - audit_a1_geometry.json: per-tile coordinates (input region, output region, halo)
  - audit_a1_report.json: max_abs(stitched, full_ort) for each halo we tried
  - tiles/<i>_<j>/{input.bin, ort_output.bin}: per-tile bins for staging into ELFs

Halo math for DnCNN3:
  20 conv layers × 3x3 each adds 1 pixel of receptive field per side per layer.
  Total RF = 41x41 (offset of 20 from center each side). Halo of 20 should be
  exactly correct; we test halos {16, 20, 24, 32} to find the smallest that works.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

DEFAULT_ONNX = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/"
                    "luxonis-models/downloads/onnx/extracted/"
                    "dncnn3-240x320.onnx/dncnn3-240x320.onnx")


def patch_model_shape(src_onnx: Path, h: int, w: int, dst: Path) -> None:
    model = onnx.load(src_onnx)
    for vi in (model.graph.input[0], model.graph.output[0]):
        shape = vi.type.tensor_type.shape
        for dim, val in zip(shape.dim, [1, 1, h, w]):
            dim.ClearField("dim_param")
            dim.dim_value = int(val)
    onnx.save(model, dst)


def run_ort(onnx_path: Path, input_array: np.ndarray) -> np.ndarray:
    h, w = input_array.shape[-2:]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patch_model_shape(onnx_path, h, w, Path(tmp.name))
        sess = ort.InferenceSession(tmp.name, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        return sess.run(None, {"image": input_array.astype(np.float32)})[0]


def make_tile_geometry(img_h: int, img_w: int, tile_h: int, tile_w: int,
                       halo: int) -> list[dict]:
    """Compute per-tile (input region, output region) for a uniform tile grid.

    Output tiles are non-overlapping. Input tiles include `halo` pixels of
    surrounding context, clamped to image bounds. Boundary tiles whose halo
    falls outside the image just get clamped — the conv kernel's interior_y/x
    check handles boundary semantics natively.
    """
    tiles = []
    for oy0 in range(0, img_h, tile_h):
        for ox0 in range(0, img_w, tile_w):
            oy1 = min(oy0 + tile_h, img_h)
            ox1 = min(ox0 + tile_w, img_w)
            iy0 = max(oy0 - halo, 0)
            iy1 = min(oy1 + halo, img_h)
            ix0 = max(ox0 - halo, 0)
            ix1 = min(ox1 + halo, img_w)
            tiles.append({
                "tile_id": len(tiles),
                "output_y0": oy0, "output_y1": oy1,
                "output_x0": ox0, "output_x1": ox1,
                "input_y0":  iy0, "input_y1":  iy1,
                "input_x0":  ix0, "input_x1":  ix1,
                "input_h":   iy1 - iy0,
                "input_w":   ix1 - ix0,
                "output_h":  oy1 - oy0,
                "output_w":  ox1 - ox0,
                # Where in the cropped output to find the region we keep:
                "crop_y0":   oy0 - iy0,
                "crop_y1":   oy0 - iy0 + (oy1 - oy0),
                "crop_x0":   ox0 - ix0,
                "crop_x1":   ox0 - ix0 + (ox1 - ox0),
            })
    return tiles


def tiled_inference(onnx_path: Path, full_input: np.ndarray, tiles: list[dict]
                    ) -> np.ndarray:
    img_h, img_w = full_input.shape[-2:]
    out = np.zeros_like(full_input)
    for t in tiles:
        sub_in = full_input[..., t["input_y0"]:t["input_y1"],
                                 t["input_x0"]:t["input_x1"]]
        sub_out = run_ort(onnx_path, sub_in)
        cropped = sub_out[..., t["crop_y0"]:t["crop_y1"],
                              t["crop_x0"]:t["crop_x1"]]
        out[..., t["output_y0"]:t["output_y1"],
                t["output_x0"]:t["output_x1"]] = cropped
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--input", type=Path,
                    default=Path(__file__).parent.parent
                    / "dncnn3_luxonis_240x320_input_f32.bin")
    ap.add_argument("--ort_full", type=Path,
                    default=Path(__file__).parent.parent
                    / "dncnn3_luxonis_240x320_ort_output_f32.bin")
    ap.add_argument("--tile_h", type=int, default=64)
    ap.add_argument("--tile_w", type=int, default=64)
    ap.add_argument("--halos", type=int, nargs="+", default=[16, 20, 24, 32])
    ap.add_argument("--img_h", type=int, default=240)
    ap.add_argument("--img_w", type=int, default=320)
    ap.add_argument("--out_dir", type=Path,
                    default=Path(__file__).parent.parent / "tiled-fp32")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading input from {args.input}")
    full_input = np.fromfile(args.input, dtype=np.float32).reshape(
        1, 1, args.img_h, args.img_w)
    print(f"loading full ORT output from {args.ort_full}")
    full_ort = np.fromfile(args.ort_full, dtype=np.float32).reshape(
        1, 1, args.img_h, args.img_w)

    print(f"sanity: re-running full ORT on the input ...")
    fresh_ort = run_ort(args.onnx, full_input)
    fresh_diff = float(np.abs(fresh_ort - full_ort).max())
    print(f"  fresh-ORT vs saved-bin max_abs = {fresh_diff:.3e}")

    halo_results = []
    for halo in args.halos:
        print(f"\n=== halo = {halo} ===")
        tiles = make_tile_geometry(args.img_h, args.img_w,
                                   args.tile_h, args.tile_w, halo)
        print(f"  {len(tiles)} tiles")
        print(f"  input tile sizes: {sorted(set((t['input_h'], t['input_w']) for t in tiles))}")
        print(f"  output tile sizes: {sorted(set((t['output_h'], t['output_w']) for t in tiles))}")
        stitched = tiled_inference(args.onnx, full_input, tiles)
        diff = np.abs(stitched - full_ort)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        bit_exact = bool(stitched.astype("<f4").tobytes() ==
                         full_ort.astype("<f4").tobytes())
        result = {
            "halo": halo, "n_tiles": len(tiles),
            "max_abs": max_abs, "mean_abs": mean_abs,
            "bit_exact": bit_exact,
            "allclose_1e5": max_abs <= 1e-5,
            "input_tile_sizes": sorted(set((t['input_h'], t['input_w']) for t in tiles)),
        }
        print(f"  max_abs = {max_abs:.3e}  mean_abs = {mean_abs:.3e}  "
              f"bit_exact = {bit_exact}  allclose@1e-5 = {max_abs <= 1e-5}")
        halo_results.append(result)

    chosen = next((r for r in halo_results if r["allclose_1e5"]), None)
    if not chosen:
        print(f"\n!! NO HALO WORKED — try larger values")
        report = {"halos_tested": halo_results, "chosen_halo": None}
    else:
        print(f"\nchosen halo = {chosen['halo']}  "
              f"(smallest with max_abs ≤ 1e-5)")
        # Save the chosen geometry + per-tile bins
        tiles = make_tile_geometry(args.img_h, args.img_w,
                                   args.tile_h, args.tile_w, chosen["halo"])
        tiles_dir = args.out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for t in tiles:
            sub_in = full_input[..., t["input_y0"]:t["input_y1"],
                                     t["input_x0"]:t["input_x1"]]
            sub_out_full = run_ort(args.onnx, sub_in)
            sub_out_crop = sub_out_full[..., t["crop_y0"]:t["crop_y1"],
                                            t["crop_x0"]:t["crop_x1"]]
            tdir = tiles_dir / f"t{t['tile_id']:02d}"
            tdir.mkdir(exist_ok=True)
            sub_in.astype(np.float32).tofile(tdir / "input.bin")
            sub_out_full.astype(np.float32).tofile(tdir / "ort_output_full.bin")
            sub_out_crop.astype(np.float32).tofile(tdir / "ort_output_crop.bin")
        report = {
            "img_h": args.img_h, "img_w": args.img_w,
            "tile_h": args.tile_h, "tile_w": args.tile_w,
            "chosen_halo": chosen["halo"], "n_tiles": chosen["n_tiles"],
            "halos_tested": halo_results,
            "tiles": tiles,
        }
    out_json = args.out_dir / "audit_a1_geometry.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_json}")
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())
