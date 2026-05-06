#!/usr/bin/env python3
"""Export real ORT tensors for one Whisper encoder MatMul node."""
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


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {
        vi.name: vi
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--act-name", required=True)
    ap.add_argument("--weight-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--scale", type=float, default=0.25)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.encoder)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    if args.weight_name not in init:
        raise SystemExit(f"initializer not found: {args.weight_name}")
    weight = init[args.weight_name].astype(np.float32)
    if weight.ndim != 2:
        raise SystemExit(f"weight must be rank-2, got {weight.shape}")

    rng = np.random.default_rng(args.seed)
    audio = rng.standard_normal((1, 80, 3000), dtype=np.float32) * np.float32(args.scale)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patched = add_outputs(model, [args.act_name, args.out_name])
        onnx.save(patched, tmp.name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            tmp.name, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        act_full, out_full = sess.run(
            [args.act_name, args.out_name], {"audio": audio}
        )

    act_full = np.asarray(act_full, dtype=np.float32)
    out_full = np.asarray(out_full, dtype=np.float32)
    if act_full.ndim == 3 and act_full.shape[0] == 1:
        act_2d = act_full[0]
    elif act_full.ndim == 2:
        act_2d = act_full
    else:
        raise SystemExit(f"unsupported activation shape {act_full.shape}")
    if out_full.ndim == 3 and out_full.shape[0] == 1:
        out_2d = out_full[0]
    elif out_full.ndim == 2:
        out_2d = out_full
    else:
        raise SystemExit(f"unsupported output shape {out_full.shape}")

    rows, k_dim = act_2d.shape
    w_k, n_cols = weight.shape
    if k_dim != w_k:
        raise SystemExit(
            f"shape mismatch: activation K={k_dim}, weight shape={weight.shape}"
        )
    if out_2d.shape != (rows, n_cols):
        raise SystemExit(
            f"shape mismatch: output {out_2d.shape}, expected {(rows, n_cols)}"
        )
    if k_dim % 16 != 0:
        raise SystemExit(f"K dimension must be divisible by 16 for this kernel: {k_dim}")
    if n_cols % 2 != 0:
        raise SystemExit(f"N dimension must be even for this kernel: {n_cols}")

    numpy_ref = act_2d @ weight
    max_abs_numpy_vs_ort = float(np.abs(numpy_ref - out_2d).max())
    weight_t = weight.T.copy()

    audio.astype("<f4").tofile(args.out_dir / "audio_input.bin")
    act_2d.astype("<f4").tofile(args.out_dir / "act_full.bin")
    weight.astype("<f4").tofile(args.out_dir / "weight.bin")
    weight_t.astype("<f4").tofile(args.out_dir / "weight_t.bin")
    out_2d.astype("<f4").tofile(args.out_dir / "ref_full.bin")

    manifest = {
        "encoder": str(args.encoder),
        "act_name": args.act_name,
        "weight_name": args.weight_name,
        "out_name": args.out_name,
        "seed": args.seed,
        "audio_shape": list(audio.shape),
        "act_shape": list(act_2d.shape),
        "weight_shape": list(weight.shape),
        "weight_t_shape": list(weight_t.shape),
        "ref_shape": list(out_2d.shape),
        "rows": rows,
        "k_dim": k_dim,
        "n_cols": n_cols,
        "ops": int(rows * k_dim * n_cols * 2),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
