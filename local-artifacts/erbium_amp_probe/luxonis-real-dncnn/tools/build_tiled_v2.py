#!/usr/bin/env python3
"""Build a tiled 240x320 inference with configurable halo, output stride,
and per-tile compile flags. Generates the geometry, the per-tile bins (via ORT),
and the per-tile ELFs in one go.
"""
from __future__ import annotations
import argparse, json, subprocess, tempfile, copy, shutil, os
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")
ROOT = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe")
INC = Path("/home/afonso/et-platform-vidas")
GCC = "/home/afonso/et/bin/riscv64-unknown-elf-gcc"
OBJCOPY = "/home/afonso/et/bin/riscv64-unknown-elf-objcopy"
ONNX_PATH = (Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/")
             / "luxonis-models/downloads/onnx/extracted/"
             / "dncnn3-240x320.onnx/dncnn3-240x320.onnx")

COMMON = [
    "-O3", "-funroll-loops",
    "-fno-tree-loop-distribute-patterns", "-fno-tree-loop-vectorize",
    "-march=rv64imfc", "-mabi=lp64f", "-mcmodel=medany", "-nostdlib",
    "-fno-zero-initialized-in-bss", "-ffunction-sections", "-fdata-sections",
    f"-I{INC}/et-common-libs/build-headers/erbium-soc1sim-staged-include",
    f"-I{INC}/hal/platform/erbium/include",
    f"-I{INC}/hal/platform/etsoc/include",
    f"-I{INC}/et-common-libs/include",
    "-Wl,--gc-sections", "-Wl,--no-warn-rwx-segments", "-Wl,--emit-relocs",
    "-T", f"{INC}/et-common-libs/share/erbium-soc1sim/erbium.ld",
    "-DDNCNN_STATIC_BLOBS", "-DDNCNN_TILE_BLOBS",
    "-DACTIVE_HARTS=16u", "-DNUM_HARTS=16",
    "-DDNCNN_VPU_FUSED9TAP=1", "-DDNCNN_VPU_PREFETCH_READ_WINDOW=1",
]


def patch_onnx_shape(src: Path, h: int, w: int, dst: Path):
    m = onnx.load(src)
    for vi in (m.graph.input[0], m.graph.output[0]):
        sh = vi.type.tensor_type.shape
        for d, v in zip(sh.dim, [1, 1, h, w]):
            d.ClearField("dim_param"); d.dim_value = int(v)
    onnx.save(m, dst)


def run_ort(input_arr: np.ndarray) -> np.ndarray:
    h, w = input_arr.shape[-2:]
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patch_onnx_shape(ONNX_PATH, h, w, Path(tmp.name))
        s = ort.InferenceSession(tmp.name, sess_options=opts, providers=["CPUExecutionProvider"])
        return s.run(None, {"image": input_arr.astype(np.float32)})[0]


def make_geometry(img_h: int, img_w: int, tile_h: int, tile_w: int, halo: int) -> list[dict]:
    tiles = []
    for oy0 in range(0, img_h, tile_h):
        for ox0 in range(0, img_w, tile_w):
            oy1 = min(oy0 + tile_h, img_h); ox1 = min(ox0 + tile_w, img_w)
            iy0 = max(oy0 - halo, 0); iy1 = min(oy1 + halo, img_h)
            ix0 = max(ox0 - halo, 0); ix1 = min(ox1 + halo, img_w)
            tiles.append({
                "tile_id": len(tiles),
                "output_y0": oy0, "output_y1": oy1, "output_x0": ox0, "output_x1": ox1,
                "input_y0": iy0, "input_y1": iy1, "input_x0": ix0, "input_x1": ix1,
                "input_h": iy1 - iy0, "input_w": ix1 - ix0,
                "output_h": oy1 - oy0, "output_w": ox1 - ox0,
                "crop_y0": oy0 - iy0, "crop_y1": oy0 - iy0 + (oy1 - oy0),
                "crop_x0": ox0 - ix0, "crop_x1": ox0 - ix0 + (ox1 - ox0),
            })
    return tiles


def build_tile_elfs(tiles: list[dict], out_dir: Path, full_input: np.ndarray,
                    extra_flags: list[str] = (), passes: int = 1, label: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"tiles": [], "label": label}
    for t in tiles:
        sub_in = full_input[..., t["input_y0"]:t["input_y1"],
                                 t["input_x0"]:t["input_x1"]]
        sub_out = run_ort(sub_in)
        work = out_dir / f"tile_{t['tile_id']:02d}"
        work.mkdir(exist_ok=True)
        sub_in.astype(np.float32).tofile(work / "dncnn_tile_input.bin")
        sub_out.astype(np.float32).tofile(work / "dncnn_tile_ref.bin")
        # objcopy from inside the work dir for clean symbols
        for fn in ("dncnn_tile_input", "dncnn_tile_ref"):
            subprocess.run([OBJCOPY, "-I", "binary", "-O", "elf64-littleriscv",
                "-B", "riscv", "--rename-section",
                ".data=.rodata,alloc,load,readonly,data,contents",
                f"{fn}.bin", f"{fn}.o"], cwd=work, check=True)
        elf = out_dir / f"tile_{t['tile_id']:02d}.elf"
        cmd = [GCC, *COMMON, *list(extra_flags),
               f"-DIMG_W={t['input_w']}u", f"-DIMG_H={t['input_h']}u",
               f"-DDNCNN_PASSES={passes}u",
               "-o", str(elf),
               str(DIR / "dncnn3_luxonis_real_vpu.c"),
               str(work / "dncnn_tile_input.o"),
               str(work / "dncnn_tile_ref.o"),
               str(DIR / "dncnn3_luxonis_240x320_weights_packed_f32.o"),
               str(ROOT / "hart-report" / "hart_report_crt.S"),
               str(INC / "erbium-examples" / "runtime" / "erbium-soc1sim" / "layout.c")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  BUILD FAIL tile_{t['tile_id']:02d}: {res.stderr[-200:]}")
            continue
        info["tiles"].append({**t, "elf": str(elf)})
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="experiment label, e.g. tg_h20_t64")
    ap.add_argument("--tile-h", type=int, default=64)
    ap.add_argument("--tile-w", type=int, default=64)
    ap.add_argument("--halo", type=int, default=20)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--flag", action="append", default=[], help="extra gcc flag")
    ap.add_argument("--macro", action="append", default=[], help="extra -D macro")
    ap.add_argument("--out-base", default="/tmp/tiled_v2")
    ap.add_argument("--input-bin", default=str(DIR / "dncnn3_luxonis_240x320_input_f32.bin"))
    args = ap.parse_args()

    tiles = make_geometry(240, 320, args.tile_h, args.tile_w, args.halo)
    print(f"label={args.label} tile={args.tile_h}x{args.tile_w} halo={args.halo} passes={args.passes}")
    print(f"  {len(tiles)} tiles, unique input shapes: "
          f"{sorted(set((t['input_h'],t['input_w']) for t in tiles))}")

    full_input = np.fromfile(args.input_bin, dtype=np.float32).reshape(1,1,240,320)
    out_dir = Path(args.out_base) / args.label
    extra = list(args.flag) + [f"-D{m}" for m in args.macro]
    info = build_tile_elfs(tiles, out_dir, full_input,
                           extra_flags=extra, passes=args.passes, label=args.label)
    info["tile_h"] = args.tile_h
    info["tile_w"] = args.tile_w
    info["halo"] = args.halo
    info["passes"] = args.passes
    info["extra"] = extra
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  built {len(info['tiles'])}/{len(tiles)} ELFs in {out_dir}")
    return 0 if len(info["tiles"]) == len(tiles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
