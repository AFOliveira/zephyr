#!/usr/bin/env python3
"""Host-side proof for the Whisper resident memory contract.

This verifies the layout we want before moving it to ET-SoC1:

* 64 MiB read-only region: raw INT8 weights plus compact manifest/scales.
* 16 MiB runtime region: log-mel input, encoder/cross-attention state, KV cache,
  tiled activation buffers, and token output.

The script intentionally does not depend on ONNXRuntime for execution.  It uses
the already-frozen native-v1 bundle as the source of truth and proves that the
device-facing raw files can be addressed, read back, and budgeted inside the
target regions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLES = PHASE1 / "native_v1_bundles"
DEFAULT_OUT = PHASE1 / "resident_layout_host"
MIB = 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def prod(vals: list[int]) -> int:
    out = 1
    for v in vals:
        out *= int(v)
    return out


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_bundle() -> Path:
    bundles = sorted(DEFAULT_BUNDLES.glob("bundle_*"))
    if not bundles:
        raise SystemExit(f"no native-v1 bundles found under {DEFAULT_BUNDLES}")
    return bundles[-1]


@dataclass
class Allocation:
    name: str
    offset: int
    size: int
    alignment: int
    end: int
    source: str | None = None
    sha256: str | None = None
    notes: str | None = None


class RegionAllocator:
    def __init__(self, size: int, alignment: int) -> None:
        self.size = int(size)
        self.alignment = int(alignment)
        self.cursor = 0
        self.allocations: list[Allocation] = []

    def reserve(
        self,
        name: str,
        size: int,
        *,
        alignment: int | None = None,
        source: str | None = None,
        sha256: str | None = None,
        notes: str | None = None,
    ) -> Allocation:
        use_align = int(alignment or self.alignment)
        offset = align_up(self.cursor, use_align)
        end = offset + int(size)
        if end > self.size:
            raise MemoryError(
                f"{name} exceeds region: end={end} size={self.size} "
                f"need_extra={end - self.size}"
            )
        alloc = Allocation(name, offset, int(size), use_align, end, source, sha256, notes)
        self.allocations.append(alloc)
        self.cursor = end
        return alloc

    @property
    def used(self) -> int:
        return max((a.end for a in self.allocations), default=0)

    @property
    def free(self) -> int:
        return self.size - self.used

    def as_report(self) -> dict[str, Any]:
        return {
            "size_bytes": self.size,
            "size_mib": self.size / MIB,
            "used_bytes": self.used,
            "used_mib": self.used / MIB,
            "free_bytes": self.free,
            "free_mib": self.free / MIB,
            "allocations": [asdict(a) for a in self.allocations],
        }


def iter_weight_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pkg = manifest["device_weight_package"]
    for part in ("encoder", "decoder"):
        for entry in pkg[part]["weights"]:
            if not entry.get("quantized") or not entry.get("file"):
                continue
            item = dict(entry)
            item["model_part"] = part
            out.append(item)
    return out


def pack_weight_region(
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    region_bytes: int,
    alignment: int,
    write_region_image: bool,
    out_dir: Path,
) -> dict[str, Any]:
    alloc = RegionAllocator(region_bytes, alignment)
    weight_root = bundle_dir / "weights_int8_symmetric"
    region = bytearray(region_bytes)
    device_entries = []
    total_file_bytes = 0

    for entry in iter_weight_entries(manifest):
        rel = Path(entry["file"])
        src = weight_root / rel
        if not src.exists():
            raise FileNotFoundError(src)
        expected_size = int(entry["int8_bytes"])
        actual_size = src.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"size mismatch for {src}: {actual_size} != {expected_size}")
        expected_sha = entry["sha256"]
        actual_sha = sha256_path(src)
        if actual_sha != expected_sha:
            raise ValueError(f"sha mismatch for {src}: {actual_sha} != {expected_sha}")

        a = alloc.reserve(
            f"{entry['model_part']}:{entry['name']}",
            actual_size,
            source=str(rel),
            sha256=actual_sha,
            notes=f"shape={entry['shape']} scale={entry['scale']}",
        )
        data = src.read_bytes()
        region[a.offset:a.end] = data
        readback_sha = sha256_bytes(memoryview(region)[a.offset:a.end])
        if readback_sha != expected_sha:
            raise ValueError(f"packed readback mismatch for {src}")
        total_file_bytes += actual_size
        device_entries.append({
            "name": entry["name"],
            "model_part": entry["model_part"],
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "scale": entry["scale"],
            "zero_point": entry["zero_point"],
            "offset": a.offset,
            "nbytes": a.size,
            "sha256": actual_sha,
            "source_file": str(rel),
        })

    compact_manifest = {
        "schema": "whisper-resident-weight-region",
        "version": 1,
        "source_bundle": bundle_dir.name,
        "region_size_bytes": region_bytes,
        "alignment": alignment,
        "entries": device_entries,
    }
    manifest_bytes = (json.dumps(compact_manifest, separators=(",", ":")) + "\n").encode("utf-8")
    manifest_alloc = alloc.reserve(
        "resident_weight_manifest.json",
        len(manifest_bytes),
        source="generated",
        sha256=sha256_bytes(manifest_bytes),
        notes="compact device-facing offsets/scales manifest",
    )
    region[manifest_alloc.offset:manifest_alloc.end] = manifest_bytes
    compact_manifest["manifest_region_offset"] = manifest_alloc.offset
    compact_manifest["manifest_region_nbytes"] = manifest_alloc.size

    manifest_path = out_dir / "resident_weight_manifest.json"
    manifest_path.write_text(json.dumps(compact_manifest, indent=2) + "\n")

    image_path = None
    if write_region_image:
        image_path = out_dir / "weights_region_64m.bin"
        image_path.write_bytes(region)

    return {
        "ok": alloc.used <= region_bytes,
        "region": alloc.as_report(),
        "raw_weight_file_bytes": total_file_bytes,
        "raw_weight_file_mib": total_file_bytes / MIB,
        "packed_region_sha256": sha256_bytes(region),
        "compact_manifest_path": str(manifest_path),
        "compact_manifest_bytes": len(manifest_bytes),
        "region_image_path": str(image_path) if image_path else None,
    }


def runtime_shape_bytes(shape: list[int], dtype_bytes: int) -> int:
    return prod(shape) * dtype_bytes


def build_runtime_region(
    manifest: dict[str, Any],
    *,
    region_bytes: int,
    alignment: int,
    max_decode_tokens: int,
    cross_cache_dtype: str,
    source_tile: int,
) -> dict[str, Any]:
    alloc = RegionAllocator(region_bytes, alignment)
    runtime = bytearray(region_bytes)

    input_file_size = int(runtime_shape_bytes(manifest["input"]["feature_shape"], 4))
    cross_k_shape = manifest["host_references"]["fp32"]["k_cache_cross_shape"]
    cross_v_shape = manifest["host_references"]["fp32"]["v_cache_cross_shape"]
    cross_dtype_bytes = {"int8": 1, "fp16": 2, "fp32": 4}[cross_cache_dtype]

    # Whisper tiny.en dimensions from the frozen QAIHub graph.
    decoder_layers = int(cross_k_shape[0])
    heads = int(cross_k_shape[1])
    head_dim = int(cross_k_shape[2])
    hidden = heads * head_dim
    mlp_hidden = hidden * 4

    blocks = [
        ("control_block", 64 * 1024, "host-filled run ABI and status"),
        ("input_mel_fp32", input_file_size, "1x80x3000 log-mel input"),
        (
            "encoder_hidden_fp32_reusable",
            runtime_shape_bytes([1, 1500, hidden], 4),
            "used while building cross cache; reusable as scratch during decode",
        ),
        (
            f"cross_k_cache_{cross_cache_dtype}",
            runtime_shape_bytes(cross_k_shape, cross_dtype_bytes),
            "persistent decoder cross-attention K cache",
        ),
        (
            f"cross_v_cache_{cross_cache_dtype}",
            runtime_shape_bytes(cross_v_shape, cross_dtype_bytes),
            "persistent decoder cross-attention V cache",
        ),
        ("cross_cache_scales_fp32", decoder_layers * heads * 2 * 4, "per-layer/head K/V scales"),
        (
            "self_k_cache_fp32",
            runtime_shape_bytes([decoder_layers, heads, head_dim, max_decode_tokens], 4),
            "persistent decoder self-attention K cache",
        ),
        (
            "self_v_cache_fp32",
            runtime_shape_bytes([decoder_layers, heads, max_decode_tokens, head_dim], 4),
            "persistent decoder self-attention V cache",
        ),
        (
            "decoder_hidden_a_fp32",
            runtime_shape_bytes([max_decode_tokens, hidden], 4),
            "ping-pong hidden state buffer",
        ),
        (
            "decoder_hidden_b_fp32",
            runtime_shape_bytes([max_decode_tokens, hidden], 4),
            "ping-pong hidden state buffer",
        ),
        (
            "mlp_tile_fp32",
            runtime_shape_bytes([max_decode_tokens, mlp_hidden], 4),
            "tiled MLP expansion buffer",
        ),
        (
            "attention_scores_tile_fp32",
            runtime_shape_bytes([heads, max_decode_tokens, source_tile], 4),
            "cross/self attention source-window tile",
        ),
        (
            "matmul_accum_tile_fp32",
            1 * MIB,
            "generic tiled GEMM accumulation scratch",
        ),
        ("token_output_u32", max_decode_tokens * 4, "generated token IDs"),
        ("summary_status", 64 * 1024, "timing, PMCs, audit status"),
        ("runtime_guard_slack", 512 * 1024, "reserved guard for alignment and kernel-local metadata"),
    ]

    for name, size, notes in blocks:
        a = alloc.reserve(name, int(size), notes=notes)
        # Touch the allocation to catch off-by-one/overlap mistakes in the host
        # model.  Full pattern fill keeps the check deterministic.
        runtime[a.offset:a.end] = bytes([len(alloc.allocations) & 0xFF]) * a.size

    return {
        "ok": alloc.used <= region_bytes,
        "region": alloc.as_report(),
        "parameters": {
            "max_decode_tokens": max_decode_tokens,
            "cross_cache_dtype": cross_cache_dtype,
            "source_tile": source_tile,
            "decoder_layers": decoder_layers,
            "heads": heads,
            "head_dim": head_dim,
            "hidden": hidden,
            "mlp_hidden": mlp_hidden,
        },
        "runtime_region_sha256": sha256_bytes(runtime),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    def mib(n: int | float) -> str:
        return f"{float(n) / MIB:.3f}"

    lines = [
        "# Whisper Resident Layout Host Verification",
        "",
        f"- Bundle: `{report['bundle_dir']}`",
        f"- Result: `{'PASS' if report['ok'] else 'FAIL'}`",
        f"- Weight region: {mib(report['weight_region']['region']['used_bytes'])} / {mib(report['weight_region']['region']['size_bytes'])} MiB used",
        f"- Runtime region: {mib(report['runtime_region']['region']['used_bytes'])} / {mib(report['runtime_region']['region']['size_bytes'])} MiB used",
        f"- FP32 vs dynamic INT8 token match in bundle: `{report['host_reference']['fp32_dynamic_int8_token_match']}`",
        "",
        "## Contract",
        "",
        "This is the host-side proof for:",
        "",
        "```text",
        "16 MiB region + 64 MiB region is enough for:",
        "  INT8 raw weights",
        "  tiled/resident executor",
        "  log-mel -> token IDs",
        "  short/normal decode lengths",
        "```",
        "",
        "The verified layout does not embed ONNX files or FP32 weights on chip.",
        "It uses raw INT8 weight blobs and a compact offset/scale manifest.",
        "",
        "## Weight Region",
        "",
        "| Item | Bytes | MiB |",
        "|---|---:|---:|",
        f"| Raw INT8 weight files | {report['weight_region']['raw_weight_file_bytes']} | {mib(report['weight_region']['raw_weight_file_bytes'])} |",
        f"| Compact manifest | {report['weight_region']['compact_manifest_bytes']} | {mib(report['weight_region']['compact_manifest_bytes'])} |",
        f"| Packed high-water | {report['weight_region']['region']['used_bytes']} | {mib(report['weight_region']['region']['used_bytes'])} |",
        f"| Free | {report['weight_region']['region']['free_bytes']} | {mib(report['weight_region']['region']['free_bytes'])} |",
        "",
        "## Runtime Region",
        "",
        "| Allocation | Offset | Bytes | MiB | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for a in report["runtime_region"]["region"]["allocations"]:
        lines.append(
            f"| `{a['name']}` | {a['offset']} | {a['size']} | {mib(a['size'])} | {a.get('notes') or ''} |"
        )
    lines.extend([
        "",
        "## Host Checks",
        "",
        "- Every raw INT8 weight file was size-checked, SHA-checked, copied into a simulated 64 MiB region, and SHA-checked again from the packed region.",
        "- The 16 MiB runtime arena was allocated with the same block sizes the resident executor needs and fully touched with deterministic patterns.",
        "- The frozen bundle already shows FP32 ONNXRuntime and dynamic INT8 ONNXRuntime produce the same token sequence for the JFK sample.",
        "",
        "## Important Caveat",
        "",
        "FP32 cross-attention K/V does not fit in this 16 MiB runtime plan.  The passing plan uses compressed cross K/V (`"
        + report["runtime_region"]["parameters"]["cross_cache_dtype"]
        + "`) or an equivalent tiled/recomputed cross-attention path.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--weights-region-mib", type=int, default=64)
    ap.add_argument("--runtime-region-mib", type=int, default=16)
    ap.add_argument("--alignment", type=int, default=64)
    ap.add_argument("--max-decode-tokens", type=int, default=224)
    ap.add_argument("--cross-cache-dtype", choices=("int8", "fp16", "fp32"), default="int8")
    ap.add_argument("--source-tile", type=int, default=256)
    ap.add_argument("--write-region-image", action="store_true")
    args = ap.parse_args()

    bundle_dir = args.bundle_dir or latest_bundle()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"bundle manifest not found: {manifest_path}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    weight_report = pack_weight_region(
        bundle_dir,
        manifest,
        region_bytes=args.weights_region_mib * MIB,
        alignment=args.alignment,
        write_region_image=args.write_region_image,
        out_dir=out_dir,
    )
    runtime_report = build_runtime_region(
        manifest,
        region_bytes=args.runtime_region_mib * MIB,
        alignment=args.alignment,
        max_decode_tokens=args.max_decode_tokens,
        cross_cache_dtype=args.cross_cache_dtype,
        source_tile=args.source_tile,
    )

    cmp = manifest["host_references"].get("comparison", {})
    report = {
        "schema": "whisper-resident-layout-host-verification",
        "version": 1,
        "timestamp": stamp,
        "bundle_dir": str(bundle_dir),
        "ok": bool(weight_report["ok"] and runtime_report["ok"] and cmp.get("token_sequence_match")),
        "host_reference": {
            "fp32_dynamic_int8_token_match": cmp.get("token_sequence_match"),
            "fp32_dynamic_int8_text_match": cmp.get("text_match"),
            "text": cmp.get("fp32_text"),
            "generated_nonprompt_tokens": cmp.get("fp32_generated_nonprompt_tokens"),
        },
        "weight_region": weight_report,
        "runtime_region": runtime_report,
    }

    report_path = out_dir / "resident_layout_report.json"
    md_path = out_dir / "RESIDENT_LAYOUT_REPORT.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)

    print(json.dumps({
        "ok": report["ok"],
        "report": str(report_path),
        "markdown": str(md_path),
        "weight_used_mib": report["weight_region"]["region"]["used_mib"],
        "runtime_used_mib": report["runtime_region"]["region"]["used_mib"],
        "host_token_match": report["host_reference"]["fp32_dynamic_int8_token_match"],
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
