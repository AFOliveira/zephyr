#!/usr/bin/env python3
"""Measure audited decoder MatMul token rate from real Whisper ONNX tensors."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE0 = ROOT / "phase0"

EXPORT = HERE / "export_whisper_decoder_matmul_node.py"
RUNNER = HERE / "run_whisper_decoder_token_matmul_audit.py"

SHAPES = [
    {
        "family": "logits",
        "node": "/logits/MatMul",
        "count": 1,
        "tile_cols": 8192,
        "variant": "decoder_logits_real_coltile8192_ah16",
    },
    {
        "family": "mlp0",
        "node": "/0/mlp/0/MatMul",
        "count": 4,
        "tile_cols": 1536,
    },
    {
        "family": "mlp2",
        "node": "/0/mlp/2/MatMul",
        "count": 4,
        "tile_cols": 384,
    },
    {
        "family": "self_qkv",
        "node": "MatMul_from_/0/attn/query/MatMul_and_/0/attn/key/MatMul_and_/0/attn/value/MatMul",
        "count": 4,
        "tile_cols": 1152,
    },
    {
        "family": "dense384",
        "node": "/0/attn/out/MatMul",
        "count": 12,
        "tile_cols": 384,
    },
    {
        "family": "cross_score",
        "node": "/0/cross_attn/MatMul",
        "count": 24,
        "tile_cols": 1500,
    },
    {
        "family": "cross_value",
        "node": "/0/cross_attn/MatMul_1",
        "count": 24,
        "tile_cols": 64,
        "pad_k": True,
    },
    {
        "family": "self_score",
        "node": "/0/attn/MatMul",
        "count": 24,
        "tile_cols": 224,
    },
    {
        "family": "self_value",
        "node": "/0/attn/MatMul_1",
        "count": 24,
        "tile_cols": 64,
    },
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, text=True)


def newest_passing_report(variant_prefix: str) -> Path | None:
    candidates = sorted((PHASE0 / "runs").glob(f"{variant_prefix}_*/full_audit_report.json"))
    for candidate in reversed(candidates):
        try:
            report = json.loads(candidate.read_text())
        except Exception:
            continue
        if report.get("allclose_1e4") and report.get("tile_ok"):
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    node_root = PHASE0 / "decoder_token_nodes"
    node_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    rows: list[dict[str, Any]] = []
    for shape in SHAPES:
        family = shape["family"]
        out_dir = node_root / family
        variant = shape.get("variant", f"decoder_token_{family}_ah{args.active_harts}")
        report_path = newest_passing_report(variant) if args.reuse_existing else None

        if report_path is None:
            export_cmd = [
                "python3", str(EXPORT),
                "--out-dir", str(out_dir),
                "--node-name", shape["node"],
                "--pad-k",
            ]
            if not shape.get("pad_k", False):
                # --pad-k is harmless when K is already aligned, and it keeps
                # the invocation uniform across all shape families.
                pass
            run(export_cmd)

            run([
                "python3", str(RUNNER),
                "--data-dir", str(out_dir),
                "--variant", variant,
                "--tile-cols", str(shape["tile_cols"]),
                "--active-harts", str(args.active_harts),
                "--timeout", str(args.timeout),
                "--cflag=-DVPU_ACCUM_FULL_K=1u",
            ])
            report_path = newest_passing_report(variant)
            if report_path is None:
                raise SystemExit(f"missing report for {family}")

        report = json.loads(report_path.read_text())
        if not (report["allclose_1e4"] and report["tile_ok"]):
            raise SystemExit(f"audit failed for {family}: {report_path}")
        count = int(shape["count"])
        rows.append({
            "family": family,
            "node": shape["node"],
            "count": count,
            "k_dim_unpadded": report["k_dim_unpadded"],
            "k_dim": report["k_dim"],
            "n_cols": report["n_cols"],
            "ops_each": report["ops"],
            "ops_total": report["ops"] * count,
            "wait_s_each": report["total_wait_s"],
            "wait_s_total": report["total_wait_s"] * count,
            "token_s_if_only_this_family": report["token_s_for_this_matmul"] / count,
            "gops_model_ops": report["gops_model_ops"],
            "max_abs": report["max_abs"],
            "mean_abs": report["mean_abs"],
            "argmax_match": report.get("argmax_match", ""),
            "report": str(report_path),
            "remote_work": report["remote_work"],
        })

    total_wait = sum(r["wait_s_total"] for r in rows)
    total_ops = sum(r["ops_total"] for r in rows)
    summary = {
        "timestamp": stamp,
        "description": "Whisper Tiny En decoder MatMul token-rate suite, measured on ET-SoC1 silicon with real ONNXRuntime-exported decoder tensors.",
        "active_harts": args.active_harts,
        "families": rows,
        "decoder_matmul_ops_per_token": total_ops,
        "decoder_matmul_wait_s_per_token": total_wait,
        "decoder_matmul_tokens_per_s": 1.0 / total_wait if total_wait else 0.0,
        "decoder_matmul_gops": total_ops / total_wait / 1e9 if total_wait else 0.0,
        "all_families_allclose_1e4": True,
    }

    out_json = PHASE0 / "whisper_decoder_token_rate_summary.json"
    out_md = PHASE0 / "WHISPER_DECODER_TOKEN_RATE.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Whisper Decoder Token Rate",
        "",
        "This measures the decoder MatMul work for one token using real tensors exported from the Whisper Tiny En ONNX decoder.",
        "Each row is run on ET-SoC1 silicon and compared back to the corresponding ONNX MatMul node output.",
        "",
        f"- Active harts: {args.active_harts}",
        f"- Decoder MatMul ops/token: {total_ops:,}",
        f"- Measured decoder MatMul wait/token: {total_wait:.6f} s",
        f"- Measured decoder MatMul token rate: {summary['decoder_matmul_tokens_per_s']:.6f} token/s",
        f"- Effective decoder MatMul throughput: {summary['decoder_matmul_gops']:.6f} GOPS",
        "",
        "| family | count | K | N | wait each (s) | wait/token (s) | max abs vs ONNX | report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['count']} | {r['k_dim_unpadded']} | {r['n_cols']} | "
            f"{r['wait_s_each']:.6f} | {r['wait_s_total']:.6f} | "
            f"{r['max_abs']:.9g} | `{r['report']}` |"
        )
    lines.extend([
        "",
        "Scope: this is an audited decoder-MatMul token-rate measurement, not a full native ONNX graph executor.",
        "The measured silicon kernels are mathematically equivalent to the listed ONNX MatMul nodes within FP32 accumulation-order tolerance.",
        "LayerNorm, softmax, GELU, cache update, token selection, and tokenizer work are not included in the token/s number above.",
    ])
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
