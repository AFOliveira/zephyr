#!/usr/bin/env python3
"""Run Whisper from the packed raw INT8 resident weight manifest on host.

This is the host-first correctness gate for the resident device plan.  It
consumes the exact `resident_weight_manifest.json` that the ET-SoC1 executor
will consume, reconstructs FP32 initializers from the raw symmetric INT8 blobs,
and runs the complete log-mel -> token-ID Whisper loop through ONNXRuntime.

This is not a performance model for the eventual device executor.  It validates
the weight package and quantization quality for the first practical mixed path:
INT8 weights with FP32 activations/nonlinear ops.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from transformers import WhisperTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLES = PHASE1 / "native_v1_bundles"
DEFAULT_LAYOUT_ROOT = PHASE1 / "resident_layout_host"
DEFAULT_OUT = PHASE1 / "raw_int8_host_executor"
sys.path.insert(0, str(HERE))

from run_whisper_e2e_audit import (  # noqa: E402
    decode_loop,
    make_session,
    sha256_array,
    sha256_path,
)


def latest_dir(root: Path, prefix: str) -> Path:
    dirs = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no {prefix} directories found under {root}")
    return dirs[-1]


def compare_tokens(a: list[int], b: list[int]) -> dict[str, Any]:
    limit = min(len(a), len(b))
    mismatches = [
        {"index": i, "a": int(a[i]), "b": int(b[i])}
        for i in range(limit)
        if int(a[i]) != int(b[i])
    ]
    if len(a) != len(b):
        mismatches.append({"index": limit, "a_tail": a[limit:], "b_tail": b[limit:]})
    return {
        "match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:64],
    }


def load_resident_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema") != "whisper-resident-weight-region":
        raise SystemExit(f"not a resident weight manifest: {path}")
    return data


def find_bundle(layout_manifest: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_BUNDLES / layout_manifest["source_bundle"]


def entry_index(resident_manifest: dict[str, Any], part: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in resident_manifest["entries"]:
        if entry["model_part"] != part:
            continue
        name = entry["name"]
        if name in out:
            raise SystemExit(f"duplicate resident entry name for {part}: {name}")
        out[name] = entry
    return out


def patch_model_from_raw_int8(
    *,
    part: str,
    source_model: Path,
    bundle_dir: Path,
    resident_manifest: dict[str, Any],
    out_model: Path,
) -> dict[str, Any]:
    model = onnx.load(source_model)
    index = entry_index(resident_manifest, part)
    replaced = 0
    skipped_nonfloat = 0
    missing_float = []
    total_int8_bytes = 0
    total_fp32_bytes = 0
    max_recon_abs = 0.0
    largest = []

    for init in model.graph.initializer:
        src = numpy_helper.to_array(init)
        if not np.issubdtype(src.dtype, np.floating):
            skipped_nonfloat += 1
            continue
        entry = index.get(init.name)
        if entry is None:
            missing_float.append(init.name)
            continue

        raw_path = bundle_dir / "weights_int8_symmetric" / entry["source_file"]
        q = np.fromfile(raw_path, dtype=np.int8)
        expected_elems = int(np.prod(entry["shape"], dtype=np.int64)) if entry["shape"] else 1
        if q.size != expected_elems:
            raise SystemExit(f"element count mismatch for {raw_path}: {q.size} != {expected_elems}")
        q = q.reshape(entry["shape"] if entry["shape"] else ())
        deq = q.astype(np.float32) * np.float32(entry["scale"])
        if list(deq.shape) != list(src.shape):
            raise SystemExit(f"shape mismatch for {init.name}: {deq.shape} != {src.shape}")
        err = np.abs(np.asarray(src, dtype=np.float32) - deq)
        local_max = float(err.max()) if err.size else 0.0
        max_recon_abs = max(max_recon_abs, local_max)
        total_int8_bytes += int(q.nbytes)
        total_fp32_bytes += int(deq.size * 4)
        largest.append({
            "name": init.name,
            "shape": list(deq.shape),
            "int8_bytes": int(q.nbytes),
            "scale": float(entry["scale"]),
            "max_abs_reconstruction_error": local_max,
        })
        init.CopyFrom(numpy_helper.from_array(deq.astype(np.float32), name=init.name))
        replaced += 1

    if missing_float:
        raise SystemExit(
            f"{part}: missing raw INT8 entries for {len(missing_float)} float initializers: "
            + ", ".join(missing_float[:16])
        )

    onnx.save(model, out_model)
    return {
        "part": part,
        "source_model": str(source_model),
        "output_model": str(out_model),
        "source_sha256": sha256_path(source_model),
        "output_sha256": sha256_path(out_model),
        "replaced_float_initializers": replaced,
        "skipped_nonfloat_initializers": skipped_nonfloat,
        "total_int8_bytes": total_int8_bytes,
        "total_fp32_materialized_bytes": total_fp32_bytes,
        "compression_ratio": (total_fp32_bytes / total_int8_bytes) if total_int8_bytes else None,
        "max_abs_reconstruction_error": max_recon_abs,
        "largest_replaced": sorted(largest, key=lambda x: x["int8_bytes"], reverse=True)[:16],
    }


def load_features(bundle_dir: Path, manifest: dict[str, Any]) -> np.ndarray:
    shape = manifest["input"]["feature_shape"]
    path = bundle_dir / manifest["native_executor_contract"]["input_tensor"]
    arr = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape, dtype=np.int64))
    if arr.size != expected:
        raise SystemExit(f"feature element count mismatch: {arr.size} != {expected}")
    return np.ascontiguousarray(arr.reshape(shape), dtype=np.float32)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Whisper Raw INT8 Host Executor Report",
        "",
        f"- Result: `{'PASS' if report['ok'] else 'FAIL'}`",
        f"- Bundle: `{report['bundle_dir']}`",
        f"- Resident manifest: `{report['resident_manifest']}`",
        f"- Raw INT8 text: `{report['raw_int8_result']['text']}`",
        f"- FP32 text: `{report['references']['fp32_text']}`",
        f"- Dynamic INT8 text: `{report['references']['dynamic_int8_text']}`",
        f"- Token match vs FP32: `{report['comparisons']['vs_fp32']['match']}`",
        f"- Token match vs dynamic INT8: `{report['comparisons']['vs_dynamic_int8']['match']}`",
        f"- Decode wall seconds: `{report['raw_int8_result']['decode_wall_s']:.6f}`",
        f"- Tokens/s wall: `{report['raw_int8_result']['tokens_per_s_wall']:.6f}`",
        "",
        "## Model Patching",
        "",
        "| Part | Replaced float initializers | INT8 bytes | Materialized FP32 bytes | Max reconstruction error |",
        "|---|---:|---:|---:|---:|",
    ]
    for part in ("encoder", "decoder"):
        p = report["patched_models"][part]
        lines.append(
            f"| {part} | {p['replaced_float_initializers']} | {p['total_int8_bytes']} | "
            f"{p['total_fp32_materialized_bytes']} | {p['max_abs_reconstruction_error']:.9g} |"
        )
    if not report["ok"]:
        lines.extend([
            "",
            "## Mismatches",
            "",
            "```json",
            json.dumps(report["comparisons"], indent=2),
            "```",
        ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-manifest", type=Path, default=None)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    resident_manifest_path = args.resident_manifest
    if resident_manifest_path is None:
        resident_manifest_path = latest_dir(DEFAULT_LAYOUT_ROOT, "run_") / "resident_weight_manifest.json"
    resident_manifest = load_resident_manifest(resident_manifest_path)
    bundle_dir = find_bundle(resident_manifest, args.bundle_dir)
    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder_model = run_dir / "encoder_raw_int8_dequant.onnx"
    decoder_model = run_dir / "decoder_raw_int8_dequant.onnx"
    patched_encoder = patch_model_from_raw_int8(
        part="encoder",
        source_model=bundle_dir / "encoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=encoder_model,
    )
    patched_decoder = patch_model_from_raw_int8(
        part="decoder",
        source_model=bundle_dir / "decoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=decoder_model,
    )

    features = load_features(bundle_dir, bundle_manifest)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    encoder = make_session(encoder_model)
    decoder = make_session(decoder_model, ["/3/Add_2_output_0", "/ln/LayerNormalization_output_0"])

    t_enc0 = time.perf_counter()
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    encoder_wall_s = time.perf_counter() - t_enc0
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    k_cross.astype("<f4").tofile(run_dir / "raw_int8_k_cache_cross.bin")
    v_cross.astype("<f4").tofile(run_dir / "raw_int8_v_cache_cross.bin")

    ref_fp32 = bundle_manifest["host_references"]["fp32"]["result"]
    ref_dyn = bundle_manifest["host_references"]["dynamic_int8"]["result"]
    max_new = args.max_new_tokens
    if max_new is None:
        max_new = int(bundle_manifest["host_references"]["fp32"]["result"]["generated_nonprompt_tokens"])

    result = decode_loop(
        "host-only",
        run_dir,
        decoder,
        tokenizer,
        k_cross,
        v_cross,
        max_new,
        None,
    )

    cmp_fp32 = compare_tokens(result["tokens"], ref_fp32["tokens"])
    cmp_dyn = compare_tokens(result["tokens"], ref_dyn["tokens"])
    text_match_fp32 = result["text"] == ref_fp32["text"]
    text_match_dyn = result["text"] == ref_dyn["text"]
    ok = bool(cmp_fp32["match"] and cmp_dyn["match"] and text_match_fp32 and text_match_dyn)

    report = {
        "schema": "whisper-raw-int8-host-executor-report",
        "version": 1,
        "timestamp": stamp,
        "ok": ok,
        "bundle_dir": str(bundle_dir),
        "resident_manifest": str(resident_manifest_path),
        "patched_models": {
            "encoder": patched_encoder,
            "decoder": patched_decoder,
        },
        "encoder": {
            "wall_s": encoder_wall_s,
            "k_cache_cross_shape": list(k_cross.shape),
            "v_cache_cross_shape": list(v_cross.shape),
            "k_cache_cross_sha256": sha256_array(k_cross),
            "v_cache_cross_sha256": sha256_array(v_cross),
        },
        "raw_int8_result": result,
        "references": {
            "fp32_tokens": ref_fp32["tokens"],
            "fp32_text": ref_fp32["text"],
            "dynamic_int8_tokens": ref_dyn["tokens"],
            "dynamic_int8_text": ref_dyn["text"],
        },
        "comparisons": {
            "vs_fp32": {
                **cmp_fp32,
                "text_match": text_match_fp32,
            },
            "vs_dynamic_int8": {
                **cmp_dyn,
                "text_match": text_match_dyn,
            },
        },
    }

    report_path = run_dir / "raw_int8_host_executor_report.json"
    md_path = run_dir / "RAW_INT8_HOST_EXECUTOR_REPORT.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)

    print(json.dumps({
        "ok": ok,
        "report": str(report_path),
        "markdown": str(md_path),
        "text": result["text"],
        "match_fp32_tokens": cmp_fp32["match"],
        "match_dynamic_int8_tokens": cmp_dyn["match"],
        "tokens_per_s_wall": result["tokens_per_s_wall"],
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
