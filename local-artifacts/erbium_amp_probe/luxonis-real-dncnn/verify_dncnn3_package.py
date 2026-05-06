#!/usr/bin/env python3
"""Verify the Erbium DnCNN3 package against the Luxonis ONNX model.

This checker treats ONNXRuntime as the reference execution engine and verifies:

1. The packed Erbium weight blob was generated from the ONNX initializers.
2. The saved host reference outputs match ONNXRuntime for each generated tile.
3. Silicon dump summaries report output/reference agreement within tolerance.

The Erbium implementation uses a different FP32 accumulation order from
ONNXRuntime once VPU dot products are involved, so bit-exact output is not a
portable default.  Use --require-bit-exact when that is explicitly required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
DEFAULT_ONNX = (
    ARTIFACT_ROOT
    / "luxonis-models/downloads/onnx/extracted/dncnn3-240x320.onnx/"
    / "dncnn3-240x320.onnx"
)

DNCNN_MAGIC = 0xD3C11004
SUMMARY_STRUCT = struct.Struct("<16I")


def set_tensor_shape(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value_info.type.tensor_type.shape
    for dim, value in zip(shape.dim, dims):
        dim.ClearField("dim_param")
        dim.dim_value = int(value)


def patch_model_shape(model_path: Path, height: int, width: int, output: Path) -> None:
    model = onnx.load(model_path)
    set_tensor_shape(model.graph.input[0], [1, 1, height, width])
    set_tensor_shape(model.graph.output[0], [1, 1, height, width])
    onnx.save(model, output)


def run_onnxruntime(model_path: Path, input_nchw: np.ndarray) -> np.ndarray:
    height, width = input_nchw.shape[-2:]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patch_model_shape(model_path, height, width, Path(tmp.name))
        session = ort.InferenceSession(
            tmp.name,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return session.run(None, {"image": input_nchw.astype(np.float32)})[0]


def load_initializers(model_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(model_path)
    return {
        tensor.name: numpy_helper.to_array(tensor).astype(np.float32)
        for tensor in model.graph.initializer
    }


def pack_dncnn3_weights(initializers: dict[str, np.ndarray]) -> np.ndarray:
    chunks: list[np.ndarray] = []

    chunks.append(initializers["model.0.weight"].reshape(64, 9))
    chunks.append(initializers["model.0.bias"].reshape(64))

    hidden_pack: list[np.ndarray] = []
    hidden_bias: list[np.ndarray] = []
    for layer in range(2, 38, 2):
        weight = initializers[f"model.{layer}.weight"]
        packed = np.empty((64, 9, 64), dtype=np.float32)
        for oc in range(64):
            for ky in range(3):
                for kx in range(3):
                    packed[oc, ky * 3 + kx, :] = weight[oc, :, ky, kx]
        hidden_pack.append(packed.reshape(-1))
        hidden_bias.append(initializers[f"model.{layer}.bias"].reshape(64))

    chunks.append(np.concatenate(hidden_pack).astype(np.float32))
    chunks.append(np.concatenate(hidden_bias).astype(np.float32))

    final_weight = initializers["model.38.weight"]
    final_pack = np.empty((9, 64), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            final_pack[ky * 3 + kx, :] = final_weight[0, :, ky, kx]

    chunks.append(final_pack.reshape(-1))
    chunks.append(initializers["model.38.bias"].reshape(1))
    return np.concatenate([chunk.reshape(-1).astype(np.float32) for chunk in chunks])


def fnv1a_float_hash(values: np.ndarray) -> int:
    h = 2166136261
    words = values.astype("<f4", copy=False).reshape(-1).view(np.uint32)
    for word in words:
        h ^= int(word)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def tile_inputs(root: Path) -> list[tuple[int, int, Path, Path]]:
    tiles: list[tuple[int, int, Path, Path]] = []
    pattern = re.compile(r"dncnn3_luxonis_(\d+)x(\d+)_input_f32\.bin$")
    for input_path in sorted(root.glob("dncnn3_luxonis_*x*_input_f32.bin")):
        match = pattern.match(input_path.name)
        if not match:
            continue
        height = int(match.group(1))
        width = int(match.group(2))
        ref_path = root / f"dncnn3_luxonis_{height}x{width}_ort_output_f32.bin"
        if ref_path.exists():
            tiles.append((height, width, input_path, ref_path))
    return tiles


def latest_dump(root: Path, height: int, width: int) -> Path | None:
    remote = root / "remote-logs"
    aliases = {
        (1, 1): "tile1_static_1h",
        (4, 4): "tile4_static_1h",
        (16, 16): "tile16_static_4h",
        (64, 64): "tile64_static_16h",
        (240, 320): "static_16h",
    }
    alias = aliases.get((height, width))
    if not alias:
        return None
    dumps = sorted(remote.glob(f"dump_dncnn3_luxonis_real_{alias}_*.bin"))
    return dumps[-1] if dumps else None


def parse_summary(dump_path: Path) -> dict[str, int | float | str]:
    with dump_path.open("rb") as f:
        data = f.read(0x1000 + SUMMARY_STRUCT.size)
    fields = SUMMARY_STRUCT.unpack_from(data, 0x1000)
    ops = fields[11] | (fields[12] << 32)
    return {
        "dump": str(dump_path),
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "width": fields[3],
        "height": fields[4],
        "channels": fields[5],
        "layers": fields[6],
        "active_mask": fields[7],
        "done_count": fields[8],
        "output_hash": fields[9],
        "reference_hash": fields[10],
        "ops": ops,
        "max_abs": fields[13] / 1_000_000_000.0,
        "mean_abs": fields[14] / 1_000_000_000.0,
        "slot_checksum_sum": fields[15],
    }


def check_close(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "bit_exact": bool(actual.astype("<f4").tobytes() == expected.astype("<f4").tobytes()),
        "allclose": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "actual_hash": f"0x{fnv1a_float_hash(actual):08x}",
        "expected_hash": f"0x{fnv1a_float_hash(expected):08x}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--require-bit-exact", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    model_path = args.onnx.resolve()
    if not model_path.exists():
        raise SystemExit(f"ONNX model not found: {model_path}")

    report: dict[str, object] = {
        "onnx": str(model_path),
        "root": str(root),
        "atol": args.atol,
        "rtol": args.rtol,
    }

    initializers = load_initializers(model_path)
    expected_weights = pack_dncnn3_weights(initializers)
    weight_path = root / "dncnn3_luxonis_240x320_weights_packed_f32.bin"
    actual_weights = np.fromfile(weight_path, dtype=np.float32)
    weight_ok = (
        actual_weights.shape == expected_weights.shape
        and actual_weights.astype("<f4").tobytes() == expected_weights.astype("<f4").tobytes()
    )
    report["weights"] = {
        "path": str(weight_path),
        "floats": int(actual_weights.size),
        "expected_floats": int(expected_weights.size),
        "bit_exact": bool(weight_ok),
    }

    tile_reports = []
    overall_ok = bool(weight_ok)
    for height, width, input_path, ref_path in tile_inputs(root):
        input_values = np.fromfile(input_path, dtype=np.float32).reshape(1, 1, height, width)
        saved_ref = np.fromfile(ref_path, dtype=np.float32).reshape(1, 1, height, width)
        ort_ref = run_onnxruntime(model_path, input_values)
        host_check = check_close(saved_ref, ort_ref, args.atol, args.rtol)

        tile_report: dict[str, object] = {
            "tile": f"{height}x{width}",
            "input": str(input_path),
            "saved_reference": str(ref_path),
            "host_reference_vs_onnxruntime": host_check,
        }

        tile_ok = host_check["bit_exact"] if args.require_bit_exact else host_check["allclose"]
        dump_path = latest_dump(root, height, width)
        if dump_path:
            summary = parse_summary(dump_path)
            silicon_bit_exact = summary["output_hash"] == summary["reference_hash"]
            silicon_allclose = (
                summary["magic"] == DNCNN_MAGIC
                and summary["done_count"] == summary["active_harts"]
                and summary["height"] == height
                and summary["width"] == width
                and summary["max_abs"] <= args.atol
            )
            summary["output_hash"] = f"0x{summary['output_hash']:08x}"
            summary["reference_hash"] = f"0x{summary['reference_hash']:08x}"
            summary["active_mask"] = f"0x{summary['active_mask']:x}"
            summary["slot_checksum_sum"] = f"0x{summary['slot_checksum_sum']:08x}"
            summary["bit_exact"] = bool(silicon_bit_exact)
            summary["allclose"] = bool(silicon_allclose)
            tile_report["silicon_summary"] = summary
            tile_ok = tile_ok and (
                silicon_bit_exact if args.require_bit_exact else silicon_allclose
            )

        tile_report["ok"] = bool(tile_ok)
        overall_ok = overall_ok and bool(tile_ok)
        tile_reports.append(tile_report)

    report["tiles"] = tile_reports
    report["ok"] = bool(overall_ok)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write_report:
        out = root / "dncnn3_package_verify_report.json"
        out.write_text(text + "\n")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
