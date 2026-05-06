#!/usr/bin/env python3
"""Export per-head tensors for a Whisper batched MatMul node."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


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


def pad_k(a: np.ndarray, b: np.ndarray, multiple: int) -> tuple[np.ndarray, np.ndarray]:
    if multiple <= 1:
        return a, b
    k = a.shape[1]
    padded = ((k + multiple - 1) // multiple) * multiple
    if padded == k:
        return a, b
    a2 = np.zeros((a.shape[0], padded), dtype=np.float32)
    b2 = np.zeros((padded, b.shape[1]), dtype=np.float32)
    a2[:, :k] = a
    b2[:k, :] = b
    return a2, b2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--a-name", required=True)
    ap.add_argument("--b-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--pad-k-multiple", type=int, default=16)
    ap.add_argument("--heads", nargs="*", type=int)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.encoder)

    rng = np.random.default_rng(args.seed)
    audio = rng.standard_normal((1, 80, 3000), dtype=np.float32) * np.float32(args.scale)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        patched = add_outputs(model, [args.a_name, args.b_name, args.out_name])
        onnx.save(patched, tmp.name)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            tmp.name, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        a_full, b_full, out_full = sess.run(
            [args.a_name, args.b_name, args.out_name], {"audio": audio}
        )

    a_full = np.asarray(a_full, dtype=np.float32)
    b_full = np.asarray(b_full, dtype=np.float32)
    out_full = np.asarray(out_full, dtype=np.float32)
    if a_full.ndim != 4 or b_full.ndim != 4 or out_full.ndim != 4:
        raise SystemExit(
            f"expected rank-4 tensors, got A={a_full.shape}, B={b_full.shape}, O={out_full.shape}"
        )
    if a_full.shape[0] != 1 or b_full.shape[0] != 1 or out_full.shape[0] != 1:
        raise SystemExit("batch dimension must be one")
    heads, rows, k_dim = a_full.shape[1], a_full.shape[2], a_full.shape[3]
    if b_full.shape[:3] != (1, heads, k_dim):
        raise SystemExit(f"B shape mismatch: A={a_full.shape}, B={b_full.shape}")
    n_cols = b_full.shape[3]
    if out_full.shape != (1, heads, rows, n_cols):
        raise SystemExit(
            f"output shape mismatch: got {out_full.shape}, expected {(1, heads, rows, n_cols)}"
        )

    selected = args.heads if args.heads else list(range(heads))
    manifests = []
    audio.astype("<f4").tofile(args.out_root / "audio_input.bin")
    for head in selected:
        if head < 0 or head >= heads:
            raise SystemExit(f"invalid head {head}; valid range 0..{heads - 1}")
        out_dir = args.out_root / f"{args.prefix}_h{head}"
        out_dir.mkdir(parents=True, exist_ok=True)
        a = a_full[0, head].copy()
        b = b_full[0, head].copy()
        ref = out_full[0, head].copy()
        a_padded, b_padded = pad_k(a, b, args.pad_k_multiple)
        weight_t = b_padded.T.copy()
        numpy_ref = a_padded @ b_padded
        max_abs_numpy_vs_ort = float(np.abs(numpy_ref - ref).max())

        a_padded.astype("<f4").tofile(out_dir / "act_full.bin")
        b_padded.astype("<f4").tofile(out_dir / "weight.bin")
        weight_t.astype("<f4").tofile(out_dir / "weight_t.bin")
        ref.astype("<f4").tofile(out_dir / "ref_full.bin")
        manifest = {
            "encoder": str(args.encoder),
            "a_name": args.a_name,
            "b_name": args.b_name,
            "act_name": args.a_name,
            "weight_name": args.b_name,
            "out_name": args.out_name,
            "head": head,
            "seed": args.seed,
            "audio_shape": list(audio.shape),
            "source_act_shape": [rows, k_dim],
            "source_weight_shape": [k_dim, n_cols],
            "source_ref_shape": list(ref.shape),
            "act_shape": list(a_padded.shape),
            "weight_shape": list(b_padded.shape),
            "weight_t_shape": list(weight_t.shape),
            "ref_shape": list(ref.shape),
            "rows": rows,
            "k_dim": int(a_padded.shape[1]),
            "source_k_dim": k_dim,
            "n_cols": n_cols,
            "ops": int(rows * int(a_padded.shape[1]) * n_cols * 2),
            "useful_ops": int(rows * k_dim * n_cols * 2),
            "max_abs_numpy_vs_ort": max_abs_numpy_vs_ort,
            "files": {
                "act_full": "act_full.bin",
                "weight": "weight.bin",
                "weight_t": "weight_t.bin",
                "ref_full": "ref_full.bin",
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(manifest)

    index = {
        "encoder": str(args.encoder),
        "a_name": args.a_name,
        "b_name": args.b_name,
        "out_name": args.out_name,
        "prefix": args.prefix,
        "heads": selected,
        "tensor_shapes": {
            "a": list(a_full.shape),
            "b": list(b_full.shape),
            "out": list(out_full.shape),
        },
        "manifests": [f"{args.prefix}_h{m['head']}/manifest.json" for m in manifests],
    }
    (args.out_root / f"{args.prefix}_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
