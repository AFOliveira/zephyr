#!/usr/bin/env python3
"""Export real ORT tensors for one Whisper decoder MatMul node."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ENCODER = MODELS / "whisper_tiny_en_encoder_qaihub.onnx" / "whisper_tiny_en_encoder_qaihub.onnx"
DEFAULT_DECODER = MODELS / "whisper_tiny_en_decoder_qaihub.onnx" / "whisper_tiny_en_decoder_qaihub.onnx"


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {
        vi.name: vi
        for vi in list(inferred.graph.value_info)
        + list(inferred.graph.output)
        + list(inferred.graph.input)
    }
    existing = {o.name for o in model.graph.output}
    for name in names:
        if name in existing:
            continue
        vi = known.get(name)
        if vi is None:
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        model.graph.output.append(vi)
        existing.add(name)
    return model


def run_session(model_path: str, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return sess.run(outputs, inputs)


def find_node(model: onnx.ModelProto, node_name: str) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if node.name == node_name]
    if not matches:
        raise SystemExit(f"decoder node not found: {node_name}")
    if len(matches) > 1:
        raise SystemExit(f"decoder node name is ambiguous: {node_name}")
    node = matches[0]
    if node.op_type != "MatMul":
        raise SystemExit(f"node is {node.op_type}, expected MatMul: {node_name}")
    return node


def make_decoder_inputs(
    encoder: Path,
    seed: int,
    audio_scale: float,
    token_id: int,
    index: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal((1, 80, 3000), dtype=np.float32) * np.float32(audio_scale)
    k_cross, v_cross = run_session(
        str(encoder), ["k_cache_cross", "v_cache_cross"], {"audio": audio}
    )
    return {
        "x": np.asarray([[token_id]], dtype=np.int32),
        "index": np.asarray([[index]], dtype=np.int32),
        "k_cache_cross": np.asarray(k_cross, dtype=np.float32),
        "v_cache_cross": np.asarray(v_cross, dtype=np.float32),
        "k_cache_self": np.zeros((4, 6, 64, 224), dtype=np.float32),
        "v_cache_self": np.zeros((4, 6, 224, 64), dtype=np.float32),
    }


def to_single_row_matmul(
    a: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    pad_k: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)

    if a.shape[-1] != b.shape[-2]:
        raise SystemExit(f"MatMul K mismatch: A {a.shape}, B {b.shape}")

    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-2], b.shape[-1])
    out2 = out.reshape(-1, out.shape[-1])
    if a2.shape[0] != 1 or b2.shape[0] != 1 or out2.shape[0] != 1:
        raise SystemExit(
            "this exporter only handles single-row decoder MatMuls; "
            f"got A {a.shape}, B {b.shape}, OUT {out.shape}"
        )

    k_dim = a2.shape[1]
    n_cols = b2.shape[2]
    k_padded = ((k_dim + 15) // 16) * 16
    if k_padded != k_dim and not pad_k:
        raise SystemExit(f"K={k_dim} is not divisible by 16; pass --pad-k")
    if n_cols % 2 != 0:
        raise SystemExit(f"N={n_cols} must be even for the VPU x2 kernel")

    act = np.zeros((1, k_padded), dtype=np.float32)
    weight = np.zeros((k_padded, n_cols), dtype=np.float32)
    act[:, :k_dim] = a2
    weight[:k_dim, :] = b2[0]
    ref = out2[0].copy()

    numpy_ref = (act @ weight)[0]
    max_abs_numpy_vs_ort = float(np.abs(numpy_ref - ref).max())
    mean_abs_numpy_vs_ort = float(np.abs(numpy_ref - ref).mean())
    meta = {
        "act_original_shape": list(a.shape),
        "weight_original_shape": list(b.shape),
        "out_original_shape": list(out.shape),
        "k_dim_unpadded": int(k_dim),
        "k_dim": int(k_padded),
        "n_cols": int(n_cols),
        "k_was_padded": bool(k_padded != k_dim),
        "max_abs_numpy_vs_ort": max_abs_numpy_vs_ort,
        "mean_abs_numpy_vs_ort": mean_abs_numpy_vs_ort,
    }
    return act[0], weight.T.copy(), ref, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--node-name", required=True)
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--audio-scale", type=float, default=0.25)
    ap.add_argument("--token-id", type=int, default=50258)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--pad-k", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decoder = onnx.load(args.decoder)
    node = find_node(decoder, args.node_name)
    inputs = make_decoder_inputs(
        args.encoder, args.seed, args.audio_scale, args.token_id, args.index
    )

    output_names = []
    for name in [node.input[0], node.input[1], node.output[0], "logits"]:
        if name not in output_names:
            output_names.append(name)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patched = add_outputs(decoder, output_names)
        onnx.save(patched, tmp.name)
        raw_values = run_session(str(tmp.name), output_names, inputs)
        values_by_name = dict(zip(output_names, raw_values))

    act, weight_t, ref, meta = to_single_row_matmul(
        values_by_name[node.input[0]],
        values_by_name[node.input[1]],
        values_by_name[node.output[0]],
        args.pad_k,
    )
    logits = np.asarray(values_by_name["logits"], dtype=np.float32)
    init = {i.name: numpy_helper.to_array(i) for i in decoder.graph.initializer}
    weight_is_initializer = node.input[1] in init

    act.astype("<f4").tofile(args.out_dir / "act_full.bin")
    weight_t.astype("<f4").tofile(args.out_dir / "weight_t.bin")
    ref.astype("<f4").tofile(args.out_dir / "ref_full.bin")
    logits.reshape(-1).astype("<f4").tofile(args.out_dir / "decoder_logits_full.bin")

    manifest: dict[str, Any] = {
        "encoder": str(args.encoder),
        "decoder": str(args.decoder),
        "node_name": args.node_name,
        "act_name": node.input[0],
        "weight_name": node.input[1],
        "out_name": node.output[0],
        "weight_is_initializer": bool(weight_is_initializer),
        "seed": args.seed,
        "audio_scale": args.audio_scale,
        "token_id": args.token_id,
        "index": args.index,
        "rows": 1,
        "k_dim_unpadded": meta["k_dim_unpadded"],
        "k_dim": meta["k_dim"],
        "n_cols": meta["n_cols"],
        "ops": int(meta["k_dim_unpadded"] * meta["n_cols"] * 2),
        "padded_ops_executed": int(meta["k_dim"] * meta["n_cols"] * 2),
        **meta,
        "files": {
            "act_full": "act_full.bin",
            "weight_t": "weight_t.bin",
            "ref_full": "ref_full.bin",
            "decoder_logits_full": "decoder_logits_full.bin",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
