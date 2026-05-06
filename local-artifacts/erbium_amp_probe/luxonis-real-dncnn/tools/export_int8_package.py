#!/usr/bin/env python3
"""Export a DnCNN3 QDQ ONNX model into explicit INT8 layer parameters."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
DN_ROOT = HERE.parent
DEFAULT_MODEL = DN_ROOT / "dncnn3-int8" / "dncnn3-240x320-int8.onnx"


def clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def scalar_or_list(arr: np.ndarray) -> Any:
    if arr.size == 1:
        v = arr.reshape(-1)[0]
        if np.issubdtype(arr.dtype, np.integer):
            return int(v)
        return float(v)
    return arr.reshape(-1).tolist()


def find_single_consumer(nodes: list[onnx.NodeProto], tensor: str, op_type: str) -> onnx.NodeProto:
    matches = [n for n in nodes if n.op_type == op_type and tensor in n.input]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {op_type} consumer for {tensor}, found {len(matches)}")
    return matches[0]


def qparams_from_dq(dq_node: onnx.NodeProto, init: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "quantized_tensor": dq_node.input[0],
        "scale_tensor": dq_node.input[1],
        "zero_point_tensor": dq_node.input[2],
        "scale": scalar_or_list(init[dq_node.input[1]]),
        "zero_point": scalar_or_list(init[dq_node.input[2]]),
    }


def qparams_from_q(q_node: onnx.NodeProto, init: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "scale_tensor": q_node.input[1],
        "zero_point_tensor": q_node.input[2],
        "scale": scalar_or_list(init[q_node.input[1]]),
        "zero_point": scalar_or_list(init[q_node.input[2]]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, default=DN_ROOT / "dncnn3-int8-package")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.model))
    nodes = list(model.graph.node)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    producer = {out: n for n in nodes for out in n.output}

    conv_nodes = [n for n in nodes if n.op_type == "Conv"]
    arrays: dict[str, np.ndarray] = {}
    layers: list[dict[str, Any]] = []

    for idx, conv in enumerate(conv_nodes):
        lname = clean(conv.name or f"conv_{idx}")
        input_dq = producer[conv.input[0]]
        weight_dq = producer[conv.input[1]]
        bias_dq = producer[conv.input[2]]
        if input_dq.op_type != "DequantizeLinear":
            raise RuntimeError(f"{conv.name}: input is not QDQ dequantized")
        if weight_dq.op_type != "DequantizeLinear":
            raise RuntimeError(f"{conv.name}: weight is not QDQ dequantized")
        if bias_dq.op_type != "DequantizeLinear":
            raise RuntimeError(f"{conv.name}: bias is not QDQ dequantized")

        conv_q = find_single_consumer(nodes, conv.output[0], "QuantizeLinear")
        conv_dq = find_single_consumer(nodes, conv_q.output[0], "DequantizeLinear")

        relu_qparams = None
        relu_node = None
        relu_consumers = [n for n in nodes if n.op_type == "Relu" and conv_dq.output[0] in n.input]
        if relu_consumers:
            if len(relu_consumers) != 1:
                raise RuntimeError(f"{conv.name}: ambiguous Relu consumers")
            relu_node = relu_consumers[0]
            relu_q = find_single_consumer(nodes, relu_node.output[0], "QuantizeLinear")
            relu_qparams = qparams_from_q(relu_q, init)

        wq_name = weight_dq.input[0]
        bq_name = bias_dq.input[0]
        wq = init[wq_name]
        bq = init[bq_name]
        arrays[f"{lname}_weight_q"] = wq
        arrays[f"{lname}_bias_q"] = bq

        input_q = qparams_from_dq(input_dq, init)
        weight_q = qparams_from_dq(weight_dq, init)
        bias_q = qparams_from_dq(bias_dq, init)
        conv_out_q = qparams_from_q(conv_q, init)

        in_scale = np.asarray(init[input_dq.input[1]], dtype=np.float64).reshape(-1)
        w_scale = np.asarray(init[weight_dq.input[1]], dtype=np.float64).reshape(-1)
        b_scale = np.asarray(init[bias_dq.input[1]], dtype=np.float64).reshape(-1)
        expected_b_scale = in_scale[0] * w_scale
        bias_scale_max_abs_error = float(np.max(np.abs(b_scale - expected_b_scale)))

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
        layers.append({
            "index": idx,
            "name": conv.name,
            "key": lname,
            "has_relu": relu_node is not None,
            "input": conv.input[0],
            "conv_output": conv.output[0],
            "weight_shape": list(wq.shape),
            "weight_dtype": str(wq.dtype),
            "bias_shape": list(bq.shape),
            "bias_dtype": str(bq.dtype),
            "kernel_shape": list(attrs.get("kernel_shape", [])),
            "pads": list(attrs.get("pads", [])),
            "strides": list(attrs.get("strides", [])),
            "input_q": input_q,
            "weight_q": weight_q,
            "bias_q": bias_q,
            "conv_output_q": conv_out_q,
            "relu_output_q": relu_qparams,
            "bias_scale_max_abs_error": bias_scale_max_abs_error,
        })

    npz_path = args.out_dir / "dncnn3_int8_package.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "source_model": str(args.model),
        "package_npz": str(npz_path),
        "n_layers": len(layers),
        "layers": layers,
        "notes": {
            "conv_semantics": "NCHW Conv, pads/strides from ONNX, per-channel weight scales on output channels",
            "kernel_target": "Tensor IMA8A32 can consume signed int8 weights and quantized activations with int32 accumulation",
            "bias_scale_check": "bias_scale should equal input_scale * weight_scale per output channel",
        },
    }
    manifest_path = args.out_dir / "dncnn3_int8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    worst_bias_scale_error = max(l["bias_scale_max_abs_error"] for l in layers)
    print(json.dumps({
        "n_layers": len(layers),
        "package_npz": str(npz_path),
        "manifest": str(manifest_path),
        "worst_bias_scale_max_abs_error": worst_bias_scale_error,
        "weight_dtypes": sorted({l["weight_dtype"] for l in layers}),
        "bias_dtypes": sorted({l["bias_dtype"] for l in layers}),
    }, indent=2))
    return 0 if worst_bias_scale_error <= 1e-9 else 2


if __name__ == "__main__":
    raise SystemExit(main())
