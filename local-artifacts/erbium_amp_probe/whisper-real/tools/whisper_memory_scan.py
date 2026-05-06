#!/usr/bin/env python3
"""Phase 0.1 memory feasibility scan for Luxonis Whisper Tiny En ONNX files."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ENCODER = MODELS / "whisper_tiny_en_encoder_qaihub.onnx" / "whisper_tiny_en_encoder_qaihub.onnx"
DEFAULT_DECODER = MODELS / "whisper_tiny_en_decoder_qaihub.onnx" / "whisper_tiny_en_decoder_qaihub.onnx"
DEFAULT_ENCODER_CONFIG = DEFAULT_ENCODER.parent / "config.json"
DEFAULT_DECODER_CONFIG = DEFAULT_DECODER.parent / "config.json"


DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.INT64: 8,
    TensorProto.INT32: 4,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.BOOL: 1,
    TensorProto.DOUBLE: 8,
}


def prod(vals: list[int]) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def mib(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0)


def precision_bytes(elems: int) -> dict[str, int]:
    return {
        "fp32": elems * 4,
        "fp16": elems * 2,
        "int8": elems,
        "int4": (elems + 1) // 2,
    }


def scan_io_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    model = cfg["model"]
    rows = []
    totals = {"actual": 0, "fp32": 0, "fp16": 0, "int8": 0, "int4": 0}
    for kind in ("inputs", "outputs"):
        for item in model.get(kind, []):
            dtype = item["dtype"]
            shape = [int(x) for x in item["shape"]]
            dtype_size = {"float32": 4, "int32": 4, "float16": 2, "int8": 1}.get(dtype, 4)
            elems = prod(shape)
            size = prod(shape) * dtype_size
            totals["actual"] += size
            if dtype.startswith("float"):
                pb = precision_bytes(elems)
                for key in ("fp32", "fp16", "int8", "int4"):
                    totals[key] += pb[key]
            else:
                for key in ("fp32", "fp16", "int8", "int4"):
                    totals[key] += size
            rows.append({
                "kind": kind[:-1],
                "name": item["name"],
                "dtype": dtype,
                "shape": shape,
                "bytes": size,
                "mib": mib(size),
            })
    return {
        "rows": rows,
        "total_bytes": totals["actual"],
        "total_mib": mib(totals["actual"]),
        "precision_mib": {k: mib(v) for k, v in totals.items()},
    }


def scan_model(name: str, onnx_path: Path, config_path: Path) -> dict[str, Any]:
    model = onnx.load(onnx_path, load_external_data=False)
    init_rows = []
    total_elems_float = 0
    total_bytes_actual = 0
    init_by_name = {}
    for init in model.graph.initializer:
        dims = [int(d) for d in init.dims]
        elems = prod(dims)
        dtype_size = DTYPE_BYTES.get(init.data_type, 4)
        actual = elems * dtype_size
        total_bytes_actual += actual
        if init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
            total_elems_float += elems
        row = {
            "name": init.name,
            "dims": dims,
            "data_type": TensorProto.DataType.Name(init.data_type),
            "elements": elems,
            "actual_bytes": actual,
            "actual_mib": mib(actual),
            **{f"{k}_bytes": v for k, v in precision_bytes(elems).items()},
        }
        init_rows.append(row)
        init_by_name[init.name] = row

    op_counts = Counter(node.op_type for node in model.graph.node)
    matmul_weights = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for idx, inp in enumerate(node.input):
            if inp in init_by_name:
                matmul_weights.append({
                    "node": node.name,
                    "input_index": idx,
                    "initializer": inp,
                    "dims": init_by_name[inp]["dims"],
                    "elements": init_by_name[inp]["elements"],
                    "fp32_mib": mib(init_by_name[inp]["fp32_bytes"]),
                    "int8_mib": mib(init_by_name[inp]["int8_bytes"]),
                    "int4_mib": mib(init_by_name[inp]["int4_bytes"]),
                })

    totals = {
        "actual_mib": mib(total_bytes_actual),
        "float_weight_fp32_mib": mib(total_elems_float * 4),
        "float_weight_fp16_mib": mib(total_elems_float * 2),
        "float_weight_int8_mib": mib(total_elems_float),
        "float_weight_int4_mib": mib(math.ceil(total_elems_float / 2)),
    }
    return {
        "model": name,
        "onnx": str(onnx_path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_counts": dict(op_counts.most_common()),
        "totals": totals,
        "io": scan_io_config(config_path),
        "largest_initializers": sorted(init_rows, key=lambda r: r["actual_bytes"], reverse=True)[:25],
        "matmul_weight_initializers": sorted(matmul_weights, key=lambda r: r["elements"], reverse=True),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Whisper Tiny En Phase 0.1 Memory Scan", ""]
    lines.append("All sizes below are for static ONNX initializers unless otherwise noted.")
    lines.append("")
    lines.append("| Model | Nodes | Initializers | FP32 weights MiB | FP16 | INT8 | INT4 | I/O FP32 MiB | I/O FP16 | I/O INT8 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in report["models"]:
        t = m["totals"]
        io = m["io"]["precision_mib"]
        lines.append(
            f"| {m['model']} | {m['nodes']} | {m['initializers']} | "
            f"{t['float_weight_fp32_mib']:.3f} | {t['float_weight_fp16_mib']:.3f} | "
            f"{t['float_weight_int8_mib']:.3f} | {t['float_weight_int4_mib']:.3f} | "
            f"{io['fp32']:.3f} | {io['fp16']:.3f} | {io['int8']:.3f} |"
        )
    lines.append("")
    lines.append("## Decision Notes")
    lines.append("")
    lines.append("- Encoder static weights fit inside 16 MiB at INT8 and INT4, but not FP16/FP32.")
    lines.append("- Encoder FP32 I/O does not fit inside 16 MiB because the cross-attention K/V cache outputs are about 17.6 MiB; FP16 or INT8 cache storage is needed.")
    lines.append("- Decoder static weights do not fit inside 16 MiB even at INT4 in this exported graph.")
    lines.append("- Decoder runtime I/O is also dominated by cross-attention and self-attention KV caches, so streaming, cache quantization, or host-assisted paging is required for a full decoder.")
    lines.append("- The next silicon-relevant step is not full-model C yet; it is real-shape MatMul/streaming probes for the largest encoder and decoder weight blocks.")
    lines.append("")
    for m in report["models"]:
        lines.append(f"## {m['model']} Op Counts")
        lines.append("")
        lines.append("| Op | Count |")
        lines.append("|---|---:|")
        for op, count in m["op_counts"].items():
            lines.append(f"| {op} | {count} |")
        lines.append("")
        lines.append(f"## {m['model']} Largest Initializers")
        lines.append("")
        lines.append("| Initializer | Shape | Type | MiB | INT8 MiB | INT4 MiB |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for row in m["largest_initializers"][:15]:
            lines.append(
                f"| `{row['name']}` | `{row['dims']}` | {row['data_type']} | "
                f"{row['actual_mib']:.3f} | {mib(row['int8_bytes']):.3f} | {mib(row['int4_bytes']):.3f} |"
            )
        lines.append("")
        lines.append(f"## {m['model']} Largest MatMul Weight Initializers")
        lines.append("")
        lines.append("| Initializer | Shape | FP32 MiB | INT8 MiB | INT4 MiB |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in m["matmul_weight_initializers"][:15]:
            lines.append(
                f"| `{row['initializer']}` | `{row['dims']}` | "
                f"{row['fp32_mib']:.3f} | {row['int8_mib']:.3f} | {row['int4_mib']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--encoder-config", type=Path, default=DEFAULT_ENCODER_CONFIG)
    ap.add_argument("--decoder-config", type=Path, default=DEFAULT_DECODER_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "phase0")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "models": [
            scan_model("encoder", args.encoder, args.encoder_config),
            scan_model("decoder", args.decoder, args.decoder_config),
        ]
    }
    json_path = args.out_dir / "whisper_memory_scan.json"
    md_path = args.out_dir / "whisper_memory_scan.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "encoder": report["models"][0]["totals"],
        "decoder": report["models"][1]["totals"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
