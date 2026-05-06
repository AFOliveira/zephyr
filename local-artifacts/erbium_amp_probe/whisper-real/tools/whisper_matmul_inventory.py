#!/usr/bin/env python3
"""Infer Whisper Tiny En MatMul shapes and rank silicon benchmark targets."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ENCODER = MODELS / "whisper_tiny_en_encoder_qaihub.onnx" / "whisper_tiny_en_encoder_qaihub.onnx"
DEFAULT_DECODER = MODELS / "whisper_tiny_en_decoder_qaihub.onnx" / "whisper_tiny_en_decoder_qaihub.onnx"


def dims_from_vi(vi: onnx.ValueInfoProto) -> list[int | str] | None:
    tensor = vi.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dims: list[int | str] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def prod(vals: list[int]) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def matmul_ops(a: list[int], b: list[int]) -> int | None:
    if len(a) < 2 or len(b) < 2:
        return None
    if not all(isinstance(v, int) for v in a + b):
        return None
    m = int(a[-2])
    k = int(a[-1])
    n = int(b[-1])
    if int(b[-2]) != k:
        return None
    batch_a = [int(v) for v in a[:-2]]
    batch_b = [int(v) for v in b[:-2]]
    max_rank = max(len(batch_a), len(batch_b))
    batch_a = [1] * (max_rank - len(batch_a)) + batch_a
    batch_b = [1] * (max_rank - len(batch_b)) + batch_b
    batch = []
    for x, y in zip(batch_a, batch_b):
        if x == y:
            batch.append(x)
        elif x == 1:
            batch.append(y)
        elif y == 1:
            batch.append(x)
        else:
            return None
    return prod(batch) * m * n * k * 2


def scan_model(name: str, path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes: dict[str, list[int | str] | None] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[vi.name] = dims_from_vi(vi)
    initializer_names = set()
    for init in inferred.graph.initializer:
        shapes[init.name] = [int(d) for d in init.dims]
        initializer_names.add(init.name)

    rows = []
    grouped: dict[tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]], dict[str, Any]] = {}
    for node in inferred.graph.node:
        if node.op_type != "MatMul":
            continue
        a = shapes.get(node.input[0])
        b = shapes.get(node.input[1])
        out = shapes.get(node.output[0])
        ops = matmul_ops(a or [], b or [])
        row = {
            "node": node.name,
            "a": a,
            "b": b,
            "out": out,
            "ops": ops,
            "weight_input": node.input[0] if node.input[0] in initializer_names else node.input[1] if node.input[1] in initializer_names else None,
        }
        rows.append(row)
        key = (tuple(a or []), tuple(b or []), tuple(out or []))
        if key not in grouped:
            grouped[key] = {
                "a": a,
                "b": b,
                "out": out,
                "count": 0,
                "ops_each": ops,
                "ops_total": 0,
                "examples": [],
            }
        grouped[key]["count"] += 1
        grouped[key]["ops_total"] += ops or 0
        if len(grouped[key]["examples"]) < 5:
            grouped[key]["examples"].append(node.name)

    return {
        "model": name,
        "matmul_count": len(rows),
        "groups": sorted(grouped.values(), key=lambda r: r["ops_total"], reverse=True),
        "total_ops": sum((r["ops"] or 0) for r in rows),
    }


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = ["# Whisper Tiny En MatMul Inventory", ""]
    lines.append("Shapes are from ONNX shape inference. Ops are one forward pass for the exported encoder, or one decoder token step for the exported decoder.")
    lines.append("")
    for model in report["models"]:
        lines.append(f"## {model['model']}")
        lines.append("")
        lines.append(f"- MatMul nodes: {model['matmul_count']}")
        lines.append(f"- Total MatMul ops: {model['total_ops']:,}")
        lines.append("")
        lines.append("| Count | A shape | B shape | Output shape | Ops each | Ops total | Example nodes |")
        lines.append("|---:|---|---|---|---:|---:|---|")
        for row in model["groups"]:
            lines.append(
                f"| {row['count']} | `{row['a']}` | `{row['b']}` | `{row['out']}` | "
                f"{row['ops_each'] or 0:,} | {row['ops_total']:,} | `{', '.join(row['examples'])}` |"
            )
        lines.append("")
    lines.append("## Silicon Targets")
    lines.append("")
    lines.append("- Encoder first target: row-tiled MLP MatMul with `M=128, K=384, N=1536`; this mirrors the largest encoder MLP weight shape while staying under the 16 MiB arg buffer.")
    lines.append("- Encoder attention target: row-tiled score MatMul with `batch=6, M=128, K=64, N=1500`; this tests the long-context attention access pattern.")
    lines.append("- Decoder first target: vocab projection with `M=1, K=384, N=51864`; the full INT8 weight is still about 19 MiB, so it must be column-tiled or streamed.")
    lines.append("- Existing `whisper-bench` proxy covers only `TOK=256, DIM=64, HIDDEN=256`; keep it as a tuning baseline, not a model-shape proof.")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "phase0")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "models": [
            scan_model("encoder", args.encoder),
            scan_model("decoder", args.decoder),
        ]
    }
    json_path = args.out_dir / "whisper_matmul_inventory.json"
    md_path = args.out_dir / "whisper_matmul_inventory.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_md(report, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "encoder_total_ops": report["models"][0]["total_ops"],
        "decoder_total_ops_per_token": report["models"][1]["total_ops"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
