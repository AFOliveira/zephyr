#!/usr/bin/env python3
"""Export Whisper encoder Conv1d nodes as padded MatMul audit tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from run_whisper_e2e_audit import (
    DEFAULT_ENCODER,
    PHASE0,
    add_outputs,
    prepare_features,
    sha256_array,
    sha256_path,
)


DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "audio" / "openai_whisper_jfk_16k.wav"


def align_up(value: int, align: int = 16) -> int:
    return (value + align - 1) & ~(align - 1)


def node_attr_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return [int(v) for v in attr.ints]
    return default


def node_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def make_session_with_outputs(model_path: Path, outputs: list[str]):
    import tempfile
    import onnxruntime as ort

    model = onnx.load(model_path)
    patched = add_outputs(model, outputs)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(patched, tmp.name)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(tmp.name, sess_options=opts, providers=["CPUExecutionProvider"])


def initializer_map(model_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(model_path)
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def conv_nodes(model_path: Path) -> dict[str, onnx.NodeProto]:
    model = onnx.load(model_path)
    return {node.name: node for node in model.graph.node if node.op_type == "Conv"}


def export_conv(args: argparse.Namespace) -> dict[str, Any]:
    nodes = conv_nodes(args.encoder)
    if args.node_name not in nodes:
        raise SystemExit(f"Conv node not found: {args.node_name}")
    node = nodes[args.node_name]
    if len(node.input) != 3:
        raise SystemExit(f"{node.name}: expected data, weight, bias inputs")

    initializers = initializer_map(args.encoder)
    weight = np.asarray(initializers[node.input[1]], dtype=np.float32)
    bias = np.asarray(initializers[node.input[2]], dtype=np.float32)
    if weight.ndim != 3:
        raise SystemExit(f"{node.name}: expected 1D Conv weight, got {weight.shape}")

    kernel = node_attr_ints(node, "kernel_shape", [weight.shape[2]])[0]
    pads = node_attr_ints(node, "pads", [0, 0])
    strides = node_attr_ints(node, "strides", [1])
    dilations = node_attr_ints(node, "dilations", [1])
    groups = node_attr_int(node, "group", 1)
    if groups != 1 or dilations != [1]:
        raise SystemExit(f"{node.name}: unsupported Conv attrs groups={groups} dilations={dilations}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_names = [node.input[0], node.output[0]]
    session = make_session_with_outputs(args.encoder, output_names)
    audio, audio_meta = prepare_features(args.audio, args.out_dir)
    data, ref = session.run(output_names, {"audio": audio})
    data = np.asarray(data, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 1:
        raise SystemExit(f"{node.name}: unsupported input shape {data.shape}")
    if ref.ndim != 3 or ref.shape[0] != 1:
        raise SystemExit(f"{node.name}: unsupported output shape {ref.shape}")

    out_channels, in_channels, kernel_check = weight.shape
    if kernel_check != kernel or bias.shape != (out_channels,):
        raise SystemExit(f"{node.name}: bad weight/bias shapes {weight.shape} {bias.shape}")
    if data.shape[1] != in_channels or ref.shape[1] != out_channels:
        raise SystemExit(f"{node.name}: channel mismatch input={data.shape} output={ref.shape}")

    out_len = ref.shape[2]
    raw_k = in_channels * kernel + 1
    k_dim = align_up(raw_k, 16)
    act = np.zeros((out_len, k_dim), dtype=np.float32)
    weight_2d = np.zeros((k_dim, out_channels), dtype=np.float32)
    pad_left = pads[0] if pads else 0
    stride = strides[0] if strides else 1
    source = data[0]

    for out_x in range(out_len):
        base_x = out_x * stride - pad_left
        for ci in range(in_channels):
            for kw in range(kernel):
                in_x = base_x + kw
                if 0 <= in_x < source.shape[1]:
                    act[out_x, ci * kernel + kw] = source[ci, in_x]
    act[:, in_channels * kernel] = 1.0

    for co in range(out_channels):
        for ci in range(in_channels):
            for kw in range(kernel):
                weight_2d[ci * kernel + kw, co] = weight[co, ci, kw]
        weight_2d[in_channels * kernel, co] = bias[co]

    ref_2d = np.ascontiguousarray(ref[0].T, dtype=np.float32)
    numpy_ref = act @ weight_2d
    max_abs_numpy_vs_ort = float(np.max(np.abs(numpy_ref - ref_2d)))

    act.astype("<f4").tofile(args.out_dir / "act_full.bin")
    weight_2d.astype("<f4").tofile(args.out_dir / "weight.bin")
    weight_2d.T.copy().astype("<f4").tofile(args.out_dir / "weight_t.bin")
    ref_2d.astype("<f4").tofile(args.out_dir / "ref_full.bin")
    audio.astype("<f4").tofile(args.out_dir / "audio_input.bin")

    manifest = {
        "kind": "whisper_encoder_conv_as_matmul",
        "node_name": node.name,
        "input_name": node.input[0],
        "output_name": node.output[0],
        "encoder": str(args.encoder),
        "encoder_sha256": sha256_path(args.encoder),
        "audio": str(args.audio),
        "audio_meta": audio_meta,
        "audio_sha256": sha256_array(audio),
        "input_shape": list(data.shape),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "output_shape": list(ref.shape),
        "rows": int(out_len),
        "k_dim": int(k_dim),
        "raw_k_dim": int(raw_k),
        "n_cols": int(out_channels),
        "ops": int(out_len * k_dim * out_channels * 2),
        "model_ops_unpadded": int(out_len * raw_k * out_channels * 2),
        "kernel_shape": [int(kernel)],
        "pads": [int(v) for v in pads],
        "strides": [int(v) for v in strides],
        "max_abs_numpy_vs_ort": max_abs_numpy_vs_ort,
        "files": {
            "audio": "audio_input.bin",
            "act_full": "act_full.bin",
            "weight": "weight.bin",
            "weight_t": "weight_t.bin",
            "ref_full": "ref_full.bin",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--node-name", required=True)
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "encoder_conv_nodes" / "conv")
    args = ap.parse_args()
    export_conv(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
