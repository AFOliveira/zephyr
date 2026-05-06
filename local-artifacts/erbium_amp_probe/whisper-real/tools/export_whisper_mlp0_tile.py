#!/usr/bin/env python3
"""Export real ORT tensors for the first Whisper encoder MLP MatMul tile."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ENCODER = MODELS / "whisper_tiny_en_encoder_qaihub.onnx" / "whisper_tiny_en_encoder_qaihub.onnx"

ACT_NAME = "/encoder/blocks.0/mlp_ln/LayerNormalization_output_0"
WEIGHT_NAME = "onnx::MatMul_780"
OUT_NAME = "/encoder/blocks.0/mlp/0/MatMul_output_0"


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {vi.name: vi for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "phase0" / "mlp0_tile")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260503)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.encoder)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    weight = init[WEIGHT_NAME].astype(np.float32)
    if weight.shape != (384, 1536):
        raise SystemExit(f"unexpected weight shape {weight.shape}")

    rng = np.random.default_rng(args.seed)
    audio = rng.standard_normal((1, 80, 3000), dtype=np.float32) * np.float32(0.25)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patched = add_outputs(model, [ACT_NAME, OUT_NAME])
        onnx.save(patched, tmp.name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(tmp.name, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        outputs = sess.run([ACT_NAME, OUT_NAME], {"audio": audio})

    act_full = outputs[0].astype(np.float32)
    out_full = outputs[1].astype(np.float32)
    if act_full.shape != (1, 1500, 384):
        raise SystemExit(f"unexpected activation shape {act_full.shape}")
    if out_full.shape != (1, 1500, 1536):
        raise SystemExit(f"unexpected output shape {out_full.shape}")

    act = act_full[0, :args.rows, :].copy()
    ref = out_full[0, :args.rows, :].copy()
    weight_t = weight.T.copy()
    numpy_ref = act @ weight
    max_abs_numpy_vs_ort = float(np.abs(numpy_ref - ref).max())

    audio.astype("<f4").tofile(args.out_dir / "audio_input.bin")
    act.astype("<f4").tofile(args.out_dir / "mlp0_act_128x384.bin")
    weight.astype("<f4").tofile(args.out_dir / "mlp0_weight_384x1536.bin")
    weight_t.astype("<f4").tofile(args.out_dir / "mlp0_weight_t_1536x384.bin")
    ref.astype("<f4").tofile(args.out_dir / "mlp0_ref_128x1536.bin")
    act_full.astype("<f4").tofile(args.out_dir / "mlp0_act_full_1500x384.bin")
    out_full.astype("<f4").tofile(args.out_dir / "mlp0_ref_full_1500x1536.bin")

    manifest = {
        "encoder": str(args.encoder),
        "act_name": ACT_NAME,
        "weight_name": WEIGHT_NAME,
        "out_name": OUT_NAME,
        "rows": args.rows,
        "seed": args.seed,
        "audio_shape": list(audio.shape),
        "act_tile_shape": list(act.shape),
        "weight_shape": list(weight.shape),
        "weight_t_shape": list(weight_t.shape),
        "ref_tile_shape": list(ref.shape),
        "max_abs_numpy_vs_ort": max_abs_numpy_vs_ort,
        "files": {
            "audio": "audio_input.bin",
            "act": "mlp0_act_128x384.bin",
            "weight": "mlp0_weight_384x1536.bin",
            "weight_t": "mlp0_weight_t_1536x384.bin",
            "ref": "mlp0_ref_128x1536.bin",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
