#!/usr/bin/env python3
"""Build an auditable Whisper native-v1 bundle.

The bundle is the handoff format for the ET-SoC1 native executor work.  It
freezes the exact ONNX models, audio-derived log-mel input, FP32 reference token
sequence, dynamically quantized INT8 ONNX references, and raw INT8 weight files
that a device-side executor can page/tile.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE1 = ROOT / "phase1"
sys.path.insert(0, str(HERE))

from run_whisper_e2e_audit import (  # noqa: E402
    DEFAULT_AUDIO,
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    decode_loop,
    make_session,
    prepare_features,
    sha256_array,
    sha256_path,
)


def sanitize(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return out or "tensor"


def model_inventory(model_path: Path) -> dict[str, Any]:
    model = onnx.load(model_path)
    op_counts = Counter(node.op_type for node in model.graph.node)
    initializers = []
    total_fp32_bytes = 0
    total_int8_bytes = 0
    total_int4_bytes = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        bytes_fp32 = int(arr.size * 4) if np.issubdtype(arr.dtype, np.floating) else int(arr.nbytes)
        bytes_int8 = int(arr.size) if np.issubdtype(arr.dtype, np.floating) else int(arr.nbytes)
        bytes_int4 = int((arr.size + 1) // 2) if np.issubdtype(arr.dtype, np.floating) else int(arr.nbytes)
        total_fp32_bytes += bytes_fp32
        total_int8_bytes += bytes_int8
        total_int4_bytes += bytes_int4
        initializers.append({
            "name": init.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "elements": int(arr.size),
            "bytes": int(arr.nbytes),
            "fp32_equiv_bytes": bytes_fp32,
            "int8_equiv_bytes": bytes_int8,
            "int4_equiv_bytes": bytes_int4,
        })
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
        "weight_bytes": {
            "fp32_equiv": total_fp32_bytes,
            "int8_equiv": total_int8_bytes,
            "int4_equiv": total_int4_bytes,
        },
        "largest_initializers": sorted(
            initializers, key=lambda x: x["fp32_equiv_bytes"], reverse=True
        )[:24],
    }


def quantize_weight_bins(model_name: str, model_path: Path, out_dir: Path) -> dict[str, Any]:
    model = onnx.load(model_path)
    qdir = out_dir / model_name
    qdir.mkdir(parents=True, exist_ok=True)
    entries = []
    total_qbytes = 0
    total_fp32_bytes = 0
    worst_max_abs = 0.0
    worst_name = ""
    seen: dict[str, int] = {}

    for idx, init in enumerate(model.graph.initializer):
        arr = numpy_helper.to_array(init)
        if not np.issubdtype(arr.dtype, np.floating):
            entries.append({
                "name": init.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "quantized": False,
                "reason": "non-floating initializer",
            })
            continue

        src = np.asarray(arr, dtype=np.float32)
        max_abs = float(np.max(np.abs(src))) if src.size else 0.0
        scale = max_abs / 127.0 if max_abs else 1.0
        q = np.clip(np.rint(src / scale), -127, 127).astype(np.int8)
        recon = q.astype(np.float32) * np.float32(scale)
        diff = np.abs(src - recon)
        max_err = float(diff.max()) if diff.size else 0.0
        mean_err = float(diff.mean()) if diff.size else 0.0
        if max_err > worst_max_abs:
            worst_max_abs = max_err
            worst_name = init.name

        base = sanitize(init.name)
        suffix = seen.get(base, 0)
        seen[base] = suffix + 1
        file_name = f"{idx:03d}_{base}{'_' + str(suffix) if suffix else ''}.i8"
        out_path = qdir / file_name
        q.tofile(out_path)
        total_qbytes += int(q.nbytes)
        total_fp32_bytes += int(src.size * 4)
        entries.append({
            "name": init.name,
            "shape": list(src.shape),
            "dtype": "int8_symmetric_per_tensor",
            "source_dtype": str(arr.dtype),
            "scale": scale,
            "zero_point": 0,
            "elements": int(src.size),
            "fp32_bytes": int(src.size * 4),
            "int8_bytes": int(q.nbytes),
            "file": str(out_path.relative_to(out_dir)),
            "sha256": sha256_path(out_path),
            "source_sha256": sha256_array(src),
            "max_abs_source": max_abs,
            "max_abs_reconstruction_error": max_err,
            "mean_abs_reconstruction_error": mean_err,
            "quantized": True,
        })

    return {
        "model": model_name,
        "source": str(model_path),
        "source_sha256": sha256_path(model_path),
        "weights": entries,
        "totals": {
            "fp32_bytes": total_fp32_bytes,
            "int8_bytes": total_qbytes,
            "compression_ratio": (
                float(total_fp32_bytes) / float(total_qbytes)
                if total_qbytes else None
            ),
        },
        "worst_reconstruction": {
            "name": worst_name,
            "max_abs": worst_max_abs,
        },
    }


def run_host_reference(
    encoder_path: Path,
    decoder_path: Path,
    features: np.ndarray,
    run_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    encoder = make_session(encoder_path)
    decoder = make_session(decoder_path, ["/3/Add_2_output_0", "/ln/LayerNormalization_output_0"])
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    result = decode_loop(
        "host-only",
        run_dir,
        decoder,
        tokenizer,
        k_cross,
        v_cross,
        max_new_tokens,
        None,
    )
    return {
        "result": result,
        "k_cache_cross_shape": list(k_cross.shape),
        "v_cache_cross_shape": list(v_cross.shape),
        "k_cache_cross_sha256": sha256_array(k_cross),
        "v_cache_cross_sha256": sha256_array(v_cross),
    }


def compare_tokens(fp32: dict[str, Any], int8: dict[str, Any]) -> dict[str, Any]:
    a = fp32["result"]["tokens"]
    b = int8["result"]["tokens"]
    limit = min(len(a), len(b))
    mismatches = [
        {"index": i, "fp32": int(a[i]), "int8": int(b[i])}
        for i in range(limit)
        if int(a[i]) != int(b[i])
    ]
    if len(a) != len(b):
        mismatches.append({"index": limit, "fp32_tail": a[limit:], "int8_tail": b[limit:]})
    return {
        "token_sequence_match": not mismatches,
        "text_match": fp32["result"]["text"] == int8["result"]["text"],
        "mismatches": mismatches,
        "fp32_generated_nonprompt_tokens": fp32["result"]["generated_nonprompt_tokens"],
        "int8_generated_nonprompt_tokens": int8["result"]["generated_nonprompt_tokens"],
        "fp32_text": fp32["result"]["text"],
        "int8_text": int8["result"]["text"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=ROOT / "audio" / "openai_whisper_jfk_16k.wav")
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--out-dir", type=Path, default=PHASE1 / "native_v1_bundles")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--skip-dynamic-onnx", action="store_true")
    args = ap.parse_args()

    if not args.audio.exists():
        args.audio = DEFAULT_AUDIO
    if not args.audio.exists():
        raise SystemExit(f"audio file not found: {args.audio}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bundle_dir = args.out_dir / f"bundle_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    features, input_meta = prepare_features(args.audio, bundle_dir)
    shutil.copy2(args.encoder, bundle_dir / "encoder_fp32.onnx")
    shutil.copy2(args.decoder, bundle_dir / "decoder_fp32.onnx")

    q_encoder = bundle_dir / "encoder_dynamic_int8.onnx"
    q_decoder = bundle_dir / "decoder_dynamic_int8.onnx"
    if not args.skip_dynamic_onnx:
        quantize_dynamic(
            str(args.encoder),
            str(q_encoder),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        quantize_dynamic(
            str(args.decoder),
            str(q_decoder),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )

    fp32_ref = run_host_reference(args.encoder, args.decoder, features, bundle_dir / "host_fp32", args.max_new_tokens)
    int8_ref = None
    int8_cmp = {"available": False}
    if q_encoder.exists() and q_decoder.exists():
        int8_ref = run_host_reference(q_encoder, q_decoder, features, bundle_dir / "host_dynamic_int8", args.max_new_tokens)
        int8_cmp = {"available": True, **compare_tokens(fp32_ref, int8_ref)}

    weight_pkg_dir = bundle_dir / "weights_int8_symmetric"
    encoder_weights = quantize_weight_bins("encoder", args.encoder, weight_pkg_dir)
    decoder_weights = quantize_weight_bins("decoder", args.decoder, weight_pkg_dir)

    manifest = {
        "schema": "whisper-native-v1-bundle",
        "version": 1,
        "timestamp": stamp,
        "scope": "Host-generated log-mel input and ONNX references for ET-SoC1 log-mel-to-token-ID execution.",
        "input": input_meta,
        "models": {
            "encoder_fp32": {
                "path": str(args.encoder),
                "bundle_file": "encoder_fp32.onnx",
                "sha256": sha256_path(args.encoder),
                "inventory": model_inventory(args.encoder),
            },
            "decoder_fp32": {
                "path": str(args.decoder),
                "bundle_file": "decoder_fp32.onnx",
                "sha256": sha256_path(args.decoder),
                "inventory": model_inventory(args.decoder),
            },
            "encoder_dynamic_int8": {
                "bundle_file": q_encoder.name if q_encoder.exists() else None,
                "sha256": sha256_path(q_encoder) if q_encoder.exists() else None,
            },
            "decoder_dynamic_int8": {
                "bundle_file": q_decoder.name if q_decoder.exists() else None,
                "sha256": sha256_path(q_decoder) if q_decoder.exists() else None,
            },
        },
        "host_references": {
            "fp32": fp32_ref,
            "dynamic_int8": int8_ref,
            "comparison": int8_cmp,
        },
        "device_weight_package": {
            "format": "int8_symmetric_per_tensor_raw_bins",
            "encoder": encoder_weights,
            "decoder": decoder_weights,
        },
        "native_executor_contract": {
            "input_tensor": "input_features_1x80x3000.bin",
            "output": "token_ids",
            "first_correctness_gate": "match fp32 ONNXRuntime token sequence for this bundle",
            "quantized_correctness_gate": "match dynamic INT8 ONNXRuntime token sequence before optimization",
        },
    }

    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle_dir / "reference_tokens.json").write_text(json.dumps({
        "fp32_tokens": fp32_ref["result"]["tokens"],
        "fp32_text": fp32_ref["result"]["text"],
        "dynamic_int8_tokens": int8_ref["result"]["tokens"] if int8_ref else None,
        "dynamic_int8_text": int8_ref["result"]["text"] if int8_ref else None,
        "comparison": int8_cmp,
    }, indent=2) + "\n")
    (bundle_dir / "README.md").write_text(
        "# Whisper native-v1 bundle\n\n"
        "This bundle freezes the log-mel input, FP32 ONNX reference, dynamic "
        "INT8 ONNX reference, and raw INT8 weight files for the ET-SoC1 "
        "log-mel-to-token-ID executor.\n\n"
        f"- FP32 text: `{fp32_ref['result']['text']}`\n"
        f"- Dynamic INT8 text: `{int8_ref['result']['text'] if int8_ref else 'not generated'}`\n"
        f"- INT8 token match: `{int8_cmp.get('token_sequence_match')}`\n"
        f"- Manifest: `bundle_manifest.json`\n",
    )

    print(json.dumps({
        "bundle_dir": str(bundle_dir),
        "fp32_text": fp32_ref["result"]["text"],
        "dynamic_int8_text": int8_ref["result"]["text"] if int8_ref else None,
        "dynamic_int8_token_match": int8_cmp.get("token_sequence_match"),
        "manifest": str(bundle_dir / "bundle_manifest.json"),
    }, indent=2))
    return 0 if (not int8_cmp["available"] or int8_cmp["token_sequence_match"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
