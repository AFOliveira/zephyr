#!/usr/bin/env python3
"""Summarize audited Whisper native-v1 silicon coverage.

This is intentionally read-only with respect to the board.  It inspects the
local audit ledgers/reports and emits a compact status file so the next silicon
iteration can target the actual remaining gaps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE0 = ROOT / "phase0"
PHASE1 = ROOT / "phase1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def newest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def newest_glob(pattern: str) -> Path | None:
    return newest(list(ROOT.glob(pattern)))


def onnx_counts(path: Path) -> dict[str, int]:
    model = onnx.load(path)
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def report_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "pass": False}
    report = load_json(path)
    return {
        "present": True,
        "pass": bool(report.get("all_pass", False)),
        "path": str(path),
        "selected_nodes": report.get("selected_nodes"),
        "audited_nodes": report.get("audited_nodes"),
        "audited_tiles": report.get("audited_tiles"),
        "skipped_nodes": report.get("skipped_nodes"),
        "max_abs": report.get("max_abs"),
        "mean_wait_s": report.get("mean_wait_s"),
        "tile_elements": report.get("tile_elements"),
        "graph": report.get("graph"),
        "op_types": report.get("op_types"),
    }


def latest_encoder_scalar_summary() -> dict[str, Any]:
    full = newest_glob("phase0/encoder_scalar_ops_audit_runs/run_*/encoder_scalar_ops_audit_report.json")
    remote_full = newest_glob(
        "phase1/remote_native_encoder_scalar_runs/run_*/results_full_*/encoder_scalar_remote_native_report.json"
    )
    model = PHASE1 / "native_v1_bundles" / "bundle_20260505-205553" / "encoder_fp32.onnx"
    expected = None
    if model.exists():
        counts = onnx_counts(model)
        expected = sum(counts.get(op, 0) for op in ("Add", "Mul", "Div", "Erf", "Softmax"))
    if remote_full is not None:
        item = report_summary(remote_full)
        item["expected_selected_nodes"] = expected
        item["passing_unique_node_indexes"] = item.get("selected_nodes")
        item["pass"] = bool(item.get("pass")) and (
            expected is None or int(item.get("selected_nodes") or 0) >= expected
        )
        return item
    rows = tsv_rows(PHASE0 / "whisper_encoder_scalar_ops_audit.tsv")
    passed_indexes = {
        int(r["index"])
        for r in rows
        if r.get("audit_pass") == "True" and (r.get("index") or "").isdigit()
    }
    if full is not None:
        item = report_summary(full)
        item["expected_selected_nodes"] = expected
        item["passing_unique_node_indexes"] = len(passed_indexes)
        item["pass"] = bool(item.get("pass")) and (
            expected is None or len(passed_indexes) >= expected
        )
        return item
    partial = newest_glob("phase0/encoder_scalar_ops_audit_runs/run_*/partial_encoder_scalar_ops_summary.json")
    if partial is None:
        return {"present": False, "pass": False}
    report = load_json(partial)
    return {
        "present": True,
        "pass": False,
        "partial": True,
        "path": str(partial),
        "completed_tiles": report.get("completed_tiles"),
        "unique_node_indexes": report.get("unique_node_indexes"),
        "node_index_min": report.get("node_index_min"),
        "node_index_max": report.get("node_index_max"),
        "op_tile_counts": report.get("op_tile_counts"),
        "max_abs": report.get("max_abs_overall"),
        "op_summary": report.get("op_summary"),
        "expected_selected_nodes": expected,
        "passing_unique_node_indexes": len(passed_indexes),
    }


def latest_decoder_scalar_summary(bundle: Path) -> dict[str, Any]:
    full = newest(list(ROOT.glob("phase1/decoder_scalar_fullstep_runs/run_*/decoder_scalar_ops_audit_report.json")) +
                  list(ROOT.glob("phase0/decoder_scalar_ops_audit_runs/run_*/decoder_scalar_ops_audit_report.json")))
    decoder = bundle / "decoder_fp32.onnx"
    expected = None
    if decoder.exists():
        counts = onnx_counts(decoder)
        expected = sum(counts.get(op, 0) for op in ("Add", "Mul", "Div", "Erf", "Softmax"))
    if full is None:
        return {"present": False, "pass": False, "expected_selected_nodes": expected}
    item = report_summary(full)
    item["expected_selected_nodes"] = expected
    item["pass"] = bool(item.get("pass")) and (
        expected is None or int(item.get("selected_nodes") or 0) >= expected
    )
    return item


def tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def latest_e2e_summary() -> dict[str, Any]:
    ledgers = sorted(ROOT.glob("phase1/overnight_runs/overnight_*/ledger.tsv"))
    if not ledgers:
        ledgers = sorted(ROOT.glob("phase1/overnight_smoke_runs*/overnight_*/ledger.tsv"))
    rows: list[dict[str, Any]] = []
    for ledger in ledgers:
        for row in tsv_rows(ledger):
            if row.get("kind") != "e2e":
                continue
            try:
                hybrid = float(row.get("hybrid_tokens_per_s_wall") or "nan")
                silicon = float(row.get("silicon_tokens_per_s") or "nan")
            except ValueError:
                hybrid = math.nan
                silicon = math.nan
            rows.append({
                "ledger": str(ledger),
                "timestamp": row.get("timestamp"),
                "variant": row.get("variant"),
                "pass": row.get("pass") == "True",
                "token_sequence_match": row.get("token_sequence_match") == "True",
                "text_match": row.get("text_match") == "True",
                "silicon_argmax_match": row.get("silicon_argmax_match") == "True",
                "silicon_audit_pass": row.get("silicon_audit_pass") == "True",
                "generated_nonprompt_tokens": int(row.get("generated_nonprompt_tokens") or 0),
                "host_tokens_per_s_wall": float(row.get("host_tokens_per_s_wall") or "nan"),
                "hybrid_tokens_per_s_wall": hybrid,
                "silicon_tokens_per_s": silicon,
                "silicon_wait_s": float(row.get("silicon_wait_s") or "nan"),
                "report": row.get("report"),
                "log": row.get("log"),
            })
    passing = [r for r in rows if r["pass"]]
    best_wall = max(passing, key=lambda r: r["hybrid_tokens_per_s_wall"], default=None)
    best_silicon = max(passing, key=lambda r: r["silicon_tokens_per_s"], default=None)
    return {
        "rows": len(rows),
        "passing_rows": len(passing),
        "best_wall": best_wall,
        "best_silicon": best_silicon,
        "latest": rows[-1] if rows else None,
    }


def matmul_summary() -> dict[str, Any]:
    rows = tsv_rows(PHASE0 / "whisper_real_full_node_audit.tsv")
    passing = [
        r for r in rows
        if r.get("allclose_1e4") == "True" and r.get("tile_ok") == "True"
    ]
    by_node = {}
    for r in passing:
        node = r.get("onnx_node") or r.get("node_name") or ""
        if not node:
            continue
        by_node[node] = r
    decoder_rate = PHASE0 / "whisper_decoder_token_rate_summary.json"
    rate = load_json(decoder_rate) if decoder_rate.exists() else None
    return {
        "full_node_rows": len(rows),
        "full_node_passing_rows": len(passing),
        "full_node_unique_passing": len(by_node),
        "full_node_examples": sorted(by_node)[:12],
        "decoder_token_rate": rate,
    }


def build_summary(bundle: Path) -> dict[str, Any]:
    encoder = bundle / "encoder_fp32.onnx"
    decoder = bundle / "decoder_fp32.onnx"
    summary = {
        "bundle": str(bundle),
        "onnx_node_counts": {
            "encoder": onnx_counts(encoder) if encoder.exists() else {},
            "decoder": onnx_counts(decoder) if decoder.exists() else {},
        },
        "coverage": {
            "encoder_data_ops": report_summary(newest_glob("phase0/encoder_data_ops_audit_runs/run_*/encoder_data_ops_audit_report.json")),
            "encoder_layernorm": report_summary(newest_glob("phase0/encoder_layernorm_audit_runs/run_*/encoder_layernorm_rows_audit_report.json")),
            "encoder_scalar_ops": latest_encoder_scalar_summary(),
            "decoder_data_ops": report_summary(newest_glob("phase0/decoder_data_ops_audit_runs/run_*/decoder_data_ops_audit_report.json")),
            "decoder_layernorm": report_summary(newest_glob("phase0/decoder_layernorm_audit_runs/run_*/decoder_layernorm_audit_report.json")),
            "decoder_scalar_ops": latest_decoder_scalar_summary(bundle),
            "matmul_and_conv_kernels": matmul_summary(),
            "hybrid_e2e": latest_e2e_summary(),
        },
        "remaining_gaps": [],
    }
    gaps = summary["remaining_gaps"]
    cov = summary["coverage"]
    if not cov["encoder_scalar_ops"].get("pass"):
        gaps.append("encoder scalar/Softmax audit is partial; continue from the first unaudited selected node")
    if not cov["hybrid_e2e"].get("passing_rows"):
        gaps.append("hybrid E2E tail-offload run has no passing silicon rows")
    gaps.append("full resident graph executor still needs dynamic-memory paging/provenance validation")
    gaps.append("dynamic INT8 ONNX reference exists, but native INT8 kernels are not yet audited layer-by-layer")
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    cov = summary["coverage"]
    lines = [
        "# Whisper Native-v1 Coverage Summary",
        "",
        f"Bundle: `{summary['bundle']}`",
        "",
        "## ONNX Node Counts",
        "",
        "| graph | op | count |",
        "| --- | --- | ---: |",
    ]
    for graph, counts in summary["onnx_node_counts"].items():
        for op, count in counts.items():
            lines.append(f"| {graph} | {op} | {count} |")
    lines.extend([
        "",
        "## Audited Silicon Coverage",
        "",
        "| area | pass | selected | audited | skipped | max abs | report |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for name in [
        "encoder_data_ops",
        "encoder_layernorm",
        "encoder_scalar_ops",
        "decoder_data_ops",
        "decoder_layernorm",
        "decoder_scalar_ops",
    ]:
        item = cov[name]
        selected = item.get("selected_nodes") or item.get("unique_node_indexes") or ""
        audited = item.get("audited_nodes") or item.get("audited_tiles") or item.get("completed_tiles") or ""
        skipped = item.get("skipped_nodes") if item.get("skipped_nodes") is not None else ""
        max_abs = item.get("max_abs")
        max_abs_s = "" if max_abs is None else f"{float(max_abs):.6g}"
        report = item.get("path", "")
        lines.append(
            f"| {name} | {item.get('pass')} | {selected} | {audited} | "
            f"{skipped} | {max_abs_s} | `{report}` |"
        )
    e2e = cov["hybrid_e2e"]
    best_wall = e2e.get("best_wall") or {}
    best_silicon = e2e.get("best_silicon") or {}
    lines.extend([
        "",
        "## E2E Tail-Offload Status",
        "",
        f"- Passing rows: `{e2e.get('passing_rows')}` / `{e2e.get('rows')}`",
        f"- Best host-orchestrated wall rate: `{best_wall.get('hybrid_tokens_per_s_wall')}` token/s from `{best_wall.get('variant')}`",
        f"- Best raw silicon tail rate: `{best_silicon.get('silicon_tokens_per_s')}` token/s from `{best_silicon.get('variant')}`",
        "",
        "## Remaining Gaps",
        "",
    ])
    lines.extend(f"- {gap}" for gap in summary["remaining_gaps"])
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=PHASE1 / "native_v1_bundles" / "bundle_20260505-205553")
    ap.add_argument("--out-json", type=Path, default=PHASE1 / "whisper_native_v1_coverage_summary.json")
    ap.add_argument("--out-md", type=Path, default=PHASE1 / "WHISPER_NATIVE_V1_COVERAGE_SUMMARY.md")
    args = ap.parse_args()
    summary = build_summary(args.bundle)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
