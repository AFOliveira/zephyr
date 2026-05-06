#!/usr/bin/env python3
"""Track B B1: static INT8 quantization and ORT audit for Luxonis DnCNN3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

HERE = Path(__file__).resolve().parent
DN_ROOT = HERE.parent
MODELS_ROOT = DN_ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ONNX = MODELS_ROOT / "dncnn3-240x320.onnx" / "dncnn3-240x320.onnx"


class AugmentedImageReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        image_hw: np.ndarray,
        samples: int,
        seed: int,
        noise_std: float,
        clip_min: float,
        clip_max: float,
    ) -> None:
        self.input_name = input_name
        self.image_hw = image_hw.astype(np.float32, copy=False)
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= self.samples:
            return None
        if self.index == 0:
            sample = self.image_hw
        else:
            noise = self.rng.normal(0.0, self.noise_std, self.image_hw.shape).astype(np.float32)
            sample = np.clip(self.image_hw + noise, self.clip_min, self.clip_max)
        self.index += 1
        return {self.input_name: sample.reshape(1, 1, *sample.shape).astype(np.float32)}


def run_ort(model_path: Path, input_name: str, output_name: str, image_hw: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    got_input = sess.get_inputs()[0].name
    got_output = sess.get_outputs()[0].name
    if got_input != input_name or got_output != output_name:
        raise RuntimeError(
            f"model IO changed: expected {input_name}->{output_name}, got {got_input}->{got_output}")
    out = sess.run([output_name], {input_name: image_hw.reshape(1, 1, *image_hw.shape).astype(np.float32)})[0]
    return out.reshape(image_hw.shape).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(peak / math.sqrt(mse))


def tensor_to_json(arr: np.ndarray) -> Any:
    flat = arr.reshape(-1)
    obj: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "count": int(flat.size),
    }
    if flat.size <= 2048:
        obj["values"] = flat.tolist()
    else:
        obj.update({
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
        })
    return obj


def extract_quant_params(model_path: Path, out_json: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    params: dict[str, Any] = {"model": str(model_path), "tensors": {}}
    for init in model.graph.initializer:
        if not (init.name.endswith("_scale") or init.name.endswith("_zero_point")):
            continue
        arr = numpy_helper.to_array(init)
        params["tensors"][init.name] = tensor_to_json(arr)
    out_json.write_text(json.dumps(params, indent=2) + "\n")
    return params


def crop_metrics(a: np.ndarray, b: np.ndarray, geometry_path: Path) -> list[dict[str, Any]]:
    if not geometry_path.exists():
        return []
    geom = json.loads(geometry_path.read_text())
    rows = []
    for tile in geom["tiles"]:
        y0 = int(tile["output_y0"])
        y1 = int(tile["output_y1"])
        x0 = int(tile["output_x0"])
        x1 = int(tile["output_x1"])
        qa = a[y0:y1, x0:x1]
        fb = b[y0:y1, x0:x1]
        diff = np.abs(qa - fb)
        rows.append({
            "tile_id": int(tile["tile_id"]),
            "output_h": int(tile["output_h"]),
            "output_w": int(tile["output_w"]),
            "psnr_peak1": psnr(qa, fb, 1.0),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        })
    return rows


def append_master(path: Path, row: dict[str, Any]) -> None:
    cols = [
        "timestamp", "variant", "quantized_onnx", "int8_ort_output",
        "samples", "seed", "noise_std", "quant_scheme",
        "dequant_placement", "per_channel", "activation_type",
        "weight_type", "psnr_vs_fp32", "psnr_dynamic_peak",
        "max_abs", "mean_abs", "rmse", "max_abs_lsb",
        "worst_tile_psnr", "worst_tile_max_abs", "passes_psnr_gate",
        "status",
    ]
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--input", type=Path,
                    default=DN_ROOT / "dncnn3_luxonis_240x320_input_f32.bin")
    ap.add_argument("--fp32-ort-output", type=Path,
                    default=DN_ROOT / "dncnn3_luxonis_240x320_ort_output_f32.bin")
    ap.add_argument("--geometry", type=Path,
                    default=DN_ROOT / "tiled-fp32" / "audit_a1_geometry.json")
    ap.add_argument("--out-dir", type=Path, default=DN_ROOT / "dncnn3-int8")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--noise-std", type=float, default=0.03)
    ap.add_argument("--psnr-gate", type=float, default=35.0)
    ap.add_argument("--format", choices=["qdq", "qoperator"], default="qdq")
    ap.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--activation-type", choices=["quint8", "qint8"], default="quint8")
    ap.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    ap.add_argument("--master", type=Path, default=DN_ROOT / "audit_master_int8.tsv")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    output_meta = sess.get_outputs()[0]
    if input_meta.shape != [1, 1, 240, 320]:
        raise SystemExit(f"unexpected input shape: {input_meta.shape}")
    input_name = input_meta.name
    output_name = output_meta.name

    image = np.fromfile(args.input, dtype="<f4").reshape(240, 320)
    fp32_ref = np.fromfile(args.fp32_ort_output, dtype="<f4").reshape(240, 320)
    clip_min = float(np.min(image))
    clip_max = float(np.max(image))
    reader = AugmentedImageReader(
        input_name=input_name,
        image_hw=image,
        samples=args.samples,
        seed=args.seed,
        noise_std=args.noise_std,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    quant_model = args.out_dir / "dncnn3-240x320-int8.onnx"
    quant_format = QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator
    activation_type = QuantType.QUInt8 if args.activation_type == "quint8" else QuantType.QInt8
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    if args.reuse_existing and quant_model.exists():
        print(f"reusing {quant_model}")
    else:
        quantize_static(
            model_input=args.onnx,
            model_output=quant_model,
            calibration_data_reader=reader,
            quant_format=quant_format,
            op_types_to_quantize=["Conv"],
            per_channel=args.per_channel,
            activation_type=activation_type,
            weight_type=weight_type,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                "ActivationSymmetric": args.activation_type == "qint8",
                "WeightSymmetric": args.weight_type == "qint8",
            },
        )

    int8_output = run_ort(quant_model, input_name, output_name, image)
    int8_output_path = args.out_dir / "dncnn3_luxonis_240x320_int8_ort_output_f32.bin"
    int8_output.astype("<f4").tofile(int8_output_path)

    diff = np.abs(int8_output - fp32_ref)
    dynamic_peak = float(np.max(fp32_ref) - np.min(fp32_ref)) or 1.0
    metrics = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
        "psnr_peak1": psnr(int8_output, fp32_ref, 1.0),
        "psnr_dynamic_peak": psnr(int8_output, fp32_ref, dynamic_peak),
        "psnr_gate": args.psnr_gate,
        "passes_psnr_gate_peak1": psnr(int8_output, fp32_ref, 1.0) >= args.psnr_gate,
    }

    scales_path = args.out_dir / "int8_scales.json"
    quant_params = extract_quant_params(quant_model, scales_path)
    tile_rows = crop_metrics(int8_output, fp32_ref, args.geometry)
    report = {
        "onnx": str(args.onnx),
        "quantized_onnx": str(quant_model),
        "input": str(args.input),
        "fp32_reference": str(args.fp32_ort_output),
        "int8_ort_output": str(int8_output_path),
        "int8_scales": str(scales_path),
        "input_name": input_name,
        "output_name": output_name,
        "calibration": {
            "samples": args.samples,
            "seed": args.seed,
            "noise_std": args.noise_std,
            "clip_min": clip_min,
            "clip_max": clip_max,
        },
        "quantization": {
            "format": args.format,
            "op_types": ["Conv"],
            "per_channel": args.per_channel,
            "activation_type": args.activation_type,
            "weight_type": args.weight_type,
            "calibrate_method": "MinMax",
            "n_quant_param_tensors": len(quant_params["tensors"]),
        },
        "metrics_vs_fp32_ort": metrics,
        "tile_crop_metrics_vs_fp32_ort": tile_rows,
    }
    report_path = args.out_dir / "quantize_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    worst_tile_psnr = min((t["psnr_peak1"] for t in tile_rows), default="")
    worst_tile_max_abs = max((t["max_abs"] for t in tile_rows), default="")
    append_master(args.master, {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": args.out_dir.name,
        "quantized_onnx": str(quant_model),
        "int8_ort_output": str(int8_output_path),
        "samples": args.samples,
        "seed": args.seed,
        "noise_std": args.noise_std,
        "quant_scheme": f"{args.format}:conv:{args.activation_type}/{args.weight_type}",
        "dequant_placement": "QDQ graph output fp32",
        "per_channel": int(args.per_channel),
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "psnr_vs_fp32": f"{metrics['psnr_peak1']:.6f}",
        "psnr_dynamic_peak": f"{metrics['psnr_dynamic_peak']:.6f}",
        "max_abs": f"{metrics['max_abs']:.9g}",
        "mean_abs": f"{metrics['mean_abs']:.9g}",
        "rmse": f"{metrics['rmse']:.9g}",
        "max_abs_lsb": "",
        "worst_tile_psnr": f"{worst_tile_psnr:.6f}" if worst_tile_psnr != "" else "",
        "worst_tile_max_abs": f"{worst_tile_max_abs:.9g}" if worst_tile_max_abs != "" else "",
        "passes_psnr_gate": int(metrics["passes_psnr_gate_peak1"]),
        "status": "ok" if metrics["passes_psnr_gate_peak1"] else "fail",
    })

    print(json.dumps(report["metrics_vs_fp32_ort"], indent=2))
    print(f"wrote {quant_model}")
    print(f"wrote {int8_output_path}")
    print(f"wrote {scales_path}")
    print(f"wrote {report_path}")
    print(f"appended {args.master}")
    return 0 if metrics["passes_psnr_gate_peak1"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
