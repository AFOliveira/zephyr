#!/usr/bin/env python3
"""Audit real Whisper LayerNormalization tensors on ET-SoC1 in row tiles."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from transformers import WhisperTokenizer

from run_whisper_decoder_layernorm_audit import run_decoder_to_step, safe_name
from run_whisper_decoder_scalar_ops_audit import session_with_outputs
from run_whisper_e2e_audit import (
    BASE_FLAGS,
    CRT,
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    GCC,
    LAYOUT,
    PHASE0,
    REMOTE_BASE,
    REMOTE_HOST,
    REMOTE_PARENT,
    RSYNC_RSH,
    SSH_CMD,
    make_session,
    prepare_features,
    run,
    run_retry,
    sha256_array,
    sha256_path,
)


DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "audio" / "openai_whisper_jfk_16k.wav"
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "whisper_layernorm_rows_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/layernorm-rows-audit"

MAGIC = 0x574C4E52
LN_DIM = 384
OUT_OFFSET = 0x4000
ARENA_BYTES = 16 * 1024 * 1024
ALIGN = 0x1000
THRESHOLD = 1e-4


def align_up(value: int, align: int = ALIGN) -> int:
    return (value + align - 1) & ~(align - 1)


def layernorm_layout(rows: int, dim: int = LN_DIM) -> dict[str, int]:
    tensor_bytes = rows * dim * 4
    vector_bytes = dim * 4
    in_offset = align_up(OUT_OFFSET + tensor_bytes)
    weight_offset = align_up(in_offset + tensor_bytes)
    bias_offset = align_up(weight_offset + vector_bytes)
    ref_offset = align_up(bias_offset + vector_bytes)
    required = ref_offset + tensor_bytes
    if required > ARENA_BYTES:
        raise ValueError(f"requires {required} bytes but arena is {ARENA_BYTES}")
    return {
        "out_offset": OUT_OFFSET,
        "in_offset": in_offset,
        "weight_offset": weight_offset,
        "bias_offset": bias_offset,
        "ref_offset": ref_offset,
        "required_bytes": required,
    }


def layernorm_nodes(model_path: Path) -> list[onnx.NodeProto]:
    model = onnx.load(model_path)
    return [node for node in model.graph.node if node.op_type == "LayerNormalization"]


def selected_output_names(nodes: list[onnx.NodeProto]) -> list[str]:
    seen = set()
    names: list[str] = []
    for node in nodes:
        for name in (node.input[0], node.output[0]):
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def initializers(model_path: Path) -> dict[str, np.ndarray]:
    return {i.name: numpy_helper.to_array(i) for i in onnx.load(model_path).graph.initializer}


class LayerNormRowsRunner:
    def __init__(self, run_dir: Path, active_harts: int, timeout: int) -> None:
        self.run_dir = run_dir
        self.active_harts = active_harts
        self.timeout = timeout
        self.build_dir = run_dir / "build"
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.remote = f"{REMOTE_ROOT}/{run_dir.name}"
        self.elf_cache: dict[tuple[int, int, int, int, int, int], Path] = {}
        run_retry([*SSH_CMD, "true"], timeout=30)
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(self.remote)}"], timeout=30)

    def build_elf(self, rows: int, layout: dict[str, int]) -> Path:
        key = (
            rows,
            layout["out_offset"],
            layout["in_offset"],
            layout["weight_offset"],
            layout["bias_offset"],
            layout["ref_offset"],
        )
        if key in self.elf_cache:
            return self.elf_cache[key]
        elf = self.build_dir / (
            f"whisper_layernorm_rows_r{rows}_"
            f"i{layout['in_offset']:x}_w{layout['weight_offset']:x}_"
            f"b{layout['bias_offset']:x}_r{layout['ref_offset']:x}.elf"
        )
        cmd = [
            GCC,
            "-O3",
            "-funroll-loops",
            *BASE_FLAGS,
            "-DNUM_HARTS=16",
            f"-DACTIVE_HARTS={self.active_harts}u",
            f"-DROWS={rows}u",
            f"-DLN_DIM={LN_DIM}u",
            f"-DOUT_OFFSET=0x{layout['out_offset']:x}u",
            f"-DIN_OFFSET=0x{layout['in_offset']:x}u",
            f"-DWEIGHT_OFFSET=0x{layout['weight_offset']:x}u",
            f"-DBIAS_OFFSET=0x{layout['bias_offset']:x}u",
            f"-DREF_OFFSET=0x{layout['ref_offset']:x}u",
            "-o",
            str(elf),
            str(SRC),
            str(CRT),
            str(LAYOUT),
        ]
        run(cmd)
        self.elf_cache[key] = elf
        run_retry(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
                   str(elf), f"{REMOTE_HOST}:{self.remote}/"],
                  attempts=5, delay_s=10.0)
        return elf

    def run_tile(self, node_index: int, node_name: str, row0: int,
                 ln_in: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                 ref: np.ndarray) -> dict[str, Any]:
        rows = int(ln_in.shape[0])
        layout = layernorm_layout(rows)
        elf = self.build_elf(rows, layout)
        tile_label = f"rows{row0}_{row0 + rows}"
        node_safe = f"{node_index:03d}_{safe_name(node_name)}_{tile_label}"
        local_node = self.run_dir / node_safe
        local_node.mkdir(parents=True, exist_ok=True)
        remote_node = f"{self.remote}/{node_safe}"

        (local_node / "in.bin").write_bytes(ln_in.astype("<f4").tobytes())
        (local_node / "weight.bin").write_bytes(weight.astype("<f4").tobytes())
        (local_node / "bias.bin").write_bytes(bias.astype("<f4").tobytes())
        (local_node / "ref.bin").write_bytes(ref.astype("<f4").tobytes())
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(remote_node)}"], timeout=30)
        run_retry(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
                   str(local_node / "in.bin"), str(local_node / "weight.bin"),
                   str(local_node / "bias.bin"), str(local_node / "ref.bin"),
                   f"{REMOTE_HOST}:{remote_node}/"],
                  attempts=8, delay_s=15.0)

        script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_node}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero2m.bin
cd "$WORK"
ln -sf {shlex.quote(self.remote + '/' + elf.name)} {shlex.quote(elf.name)}
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"$LAUNCH" --elf-load ./{elf.name} --shire 0 --file_load 0x0,$ZERO \\
  --file_load 0x{layout['in_offset']:x},in.bin \\
  --file_load 0x{layout['weight_offset']:x},weight.bin \\
  --file_load 0x{layout['bias_offset']:x},bias.bin \\
  --file_load 0x{layout['ref_offset']:x},ref.bin \\
  --dump_after dump.bin --timeout {self.timeout} > run.log 2>&1
grep -oE 'Kernel wait seconds: [0-9.eE+-]+' run.log | tail -1 || true
python3 - <<'PY'
import json, re, struct
from pathlib import Path
data = Path("dump.bin").read_bytes()
summary = struct.unpack_from("<32I", data, 0x1000)
slot_struct = struct.Struct("<16I")
slots = []
for h in range({self.active_harts}):
    raw = slot_struct.unpack_from(data, h * 64)
    slots.append({{
        "magic": raw[0], "hart_id": raw[1], "minion_id": raw[2],
        "thread_id": raw[3], "row0": raw[4], "row1": raw[5],
        "checksum": raw[7], "done": raw[8],
    }})
Path("out.bin").write_bytes(data[0x{layout['out_offset']:x}:0x{layout['out_offset']:x} + {rows} * {LN_DIM} * 4])
text = Path("run.log").read_text(errors="ignore")
m = re.search(r"Kernel wait seconds:\\s*([0-9.eE+-]+)", text)
Path("summary.json").write_text(json.dumps({{
    "wait_s": float(m.group(1)) if m else None,
    "summary": list(summary),
    "slots": slots,
    "stream_error": "Stream error" in text,
    "kernel_launch_error": "Error on kernel launch" in text,
}}, indent=2) + "\\n")
PY
"""
        run([*SSH_CMD, "bash", "-s"], input=script)
        run_retry(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
                   f"{REMOTE_HOST}:{remote_node}/summary.json",
                   f"{REMOTE_HOST}:{remote_node}/out.bin",
                   f"{REMOTE_HOST}:{remote_node}/run.log",
                   str(local_node) + "/"],
                  attempts=12, delay_s=20.0)

        raw = json.loads((local_node / "summary.json").read_text())
        summary = raw["summary"]
        out = np.fromfile(local_node / "out.bin", dtype="<f4").reshape(rows, LN_DIM).astype(np.float32)
        diff = np.abs(out - ref)
        hpm = {
            "hpm3_cycles": int(summary[10]) | (int(summary[11]) << 32),
            "hpm4_inst0": int(summary[12]) | (int(summary[13]) << 32),
            "hpm5_inst1": int(summary[14]) | (int(summary[15]) << 32),
            "hpm6_l2_miss_req": int(summary[16]) | (int(summary[17]) << 32),
            "hpm7_icache_req": int(summary[18]) | (int(summary[19]) << 32),
            "hpm8_icache_etlink_req": int(summary[20]) | (int(summary[21]) << 32),
        }
        return {
            "case_label": tile_label,
            "row0": row0,
            "row1": row0 + rows,
            "rows": rows,
            "wait_s": raw["wait_s"],
            "required_bytes": layout["required_bytes"],
            "summary_magic": int(summary[0]),
            "active_mask": int(summary[4]),
            "done_count": int(summary[5]),
            "summary_max_abs": float(summary[8]) / 1_000_000.0,
            "summary_mean_abs": float(summary[9]) / 1_000_000.0,
            "host_max_abs": float(diff.max()) if diff.size else 0.0,
            "host_mean_abs": float(diff.mean()) if diff.size else 0.0,
            "audit_pass": int(summary[0]) == MAGIC and int(summary[5]) == self.active_harts and float(diff.max()) <= THRESHOLD,
            "stream_error": raw["stream_error"],
            "kernel_launch_error": raw["kernel_launch_error"],
            "hpm": hpm,
            "node_dir": str(local_node),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--graph", choices=("decoder", "encoder"), default="encoder")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name-filter", default="")
    ap.add_argument("--tile-rows", type=int, default=512)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "encoder_layernorm_audit_runs")
    args = ap.parse_args()

    model_path = args.encoder if args.graph == "encoder" else args.decoder
    nodes = layernorm_nodes(model_path)
    if args.name_filter:
        rx = re.compile(args.name_filter)
        nodes = [node for node in nodes if rx.search(node.name)]
    if args.start_index:
        nodes = nodes[args.start_index:]
    if args.limit:
        nodes = nodes[:args.limit]
    if not nodes:
        raise SystemExit("no LayerNormalization nodes selected")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_names = selected_output_names(nodes)
    init = initializers(model_path)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    features, input_meta = prepare_features(args.audio, run_dir)
    tokens: list[int] = []
    k_cross = np.asarray([], dtype=np.float32)
    v_cross = np.asarray([], dtype=np.float32)
    if args.graph == "encoder":
        session = session_with_outputs(args.encoder, output_names)
        values = session.run(output_names, {"audio": features})
        outputs = {name: np.asarray(value, dtype=np.float32) for name, value in zip(output_names, values)}
    else:
        decoder = session_with_outputs(args.decoder, output_names)
        encoder = make_session(args.encoder)
        k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
        k_cross = np.asarray(k_cross, dtype=np.float32)
        v_cross = np.asarray(v_cross, dtype=np.float32)
        outputs, tokens = run_decoder_to_step(decoder, tokenizer, k_cross, v_cross, args.step, output_names)

    runner = LayerNormRowsRunner(run_dir, args.active_harts, args.timeout)
    results: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        ln_in = np.asarray(outputs[node.input[0]], dtype=np.float32)
        ln_ref = np.asarray(outputs[node.output[0]], dtype=np.float32)
        if ln_in.shape != ln_ref.shape or ln_in.shape[-1] != LN_DIM:
            raise SystemExit(f"{node.name}: unsupported shape {ln_in.shape} -> {ln_ref.shape}")
        rows_total = int(np.prod(ln_in.shape[:-1], dtype=np.int64))
        in_rows = np.ascontiguousarray(ln_in.reshape(rows_total, LN_DIM), dtype=np.float32)
        ref_rows = np.ascontiguousarray(ln_ref.reshape(rows_total, LN_DIM), dtype=np.float32)
        weight = np.asarray(init[node.input[1]], dtype=np.float32).reshape(LN_DIM)
        bias = np.asarray(init[node.input[2]], dtype=np.float32).reshape(LN_DIM)
        stitched = np.zeros_like(ref_rows)
        tile_results = []
        for row0 in range(0, rows_total, args.tile_rows):
            row1 = min(rows_total, row0 + args.tile_rows)
            tile = runner.run_tile(
                index,
                node.name,
                row0,
                in_rows[row0:row1],
                weight,
                bias,
                ref_rows[row0:row1],
            )
            out = np.fromfile(Path(tile["node_dir"]) / "out.bin", dtype="<f4").reshape(row1 - row0, LN_DIM)
            stitched[row0:row1] = out
            tile_results.append(tile)
            print(
                f"{index:03d} LayerNorm {node.name}:{tile['case_label']}: "
                f"pass={tile['audit_pass']} max_abs={tile['host_max_abs']:.9g} wait={tile['wait_s']}",
                flush=True,
            )
        diff = np.abs(stitched - ref_rows)
        result = {
            "index": index,
            "node": node.name,
            "input": node.input[0],
            "output": node.output[0],
            "shape": list(ln_ref.shape),
            "rows": rows_total,
            "tile_rows": args.tile_rows,
            "tiles": len(tile_results),
            "wait_s": sum(float(t["wait_s"] or 0.0) for t in tile_results),
            "host_max_abs": float(diff.max()) if diff.size else 0.0,
            "host_mean_abs": float(diff.mean()) if diff.size else 0.0,
            "audit_pass": all(t["audit_pass"] for t in tile_results) and float(diff.max()) <= THRESHOLD,
            "tiles_detail": tile_results,
        }
        results.append(result)

    all_pass = all(r["audit_pass"] for r in results)
    report = {
        "timestamp": stamp,
        "scope": f"Real Whisper {args.graph} LayerNormalization row-tiled audit on ET-SoC1",
        "graph": args.graph,
        "audio": input_meta,
        "model": {
            "encoder": str(args.encoder),
            "encoder_sha256": sha256_path(args.encoder),
            "decoder": str(args.decoder),
            "decoder_sha256": sha256_path(args.decoder),
            "tokenizer": "openai/whisper-tiny.en",
        },
        "decoder_step": args.step if args.graph == "decoder" else None,
        "tokens_to_step": [int(t) for t in tokens],
        "token_text_to_step": [tokenizer.decode([int(t)]) for t in tokens],
        "active_harts": args.active_harts,
        "tile_rows": args.tile_rows,
        "selected_nodes": len(nodes),
        "audited_tiles": sum(r["tiles"] for r in results),
        "all_pass": all_pass,
        "max_abs": max(r["host_max_abs"] for r in results),
        "mean_wait_s_per_node": sum(r["wait_s"] for r in results) / len(results),
        "feature_sha256": sha256_array(features),
        "k_cache_cross_sha256": sha256_array(k_cross) if k_cross.size else None,
        "v_cache_cross_sha256": sha256_array(v_cross) if v_cross.size else None,
        "results": results,
    }
    report_path = run_dir / f"{args.graph}_layernorm_rows_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# Whisper {args.graph.title()} LayerNorm Rows Audit",
        "",
        f"- Graph: `{args.graph}`",
        f"- Selected nodes: `{len(nodes)}`",
        f"- Audited tiles: `{report['audited_tiles']}`",
        f"- Tile rows: `{args.tile_rows}`",
        f"- All pass: `{all_pass}`",
        f"- Max abs diff: `{report['max_abs']:.9g}`",
        f"- Mean wait per node: `{report['mean_wait_s_per_node']:.6f} s`",
        "",
        "| index | node | rows | tiles | max abs | wait s | pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | `{r['node']}` | {r['rows']} | {r['tiles']} | "
            f"{r['host_max_abs']:.9g} | {r['wait_s']:.6f} | {r['audit_pass']} |"
        )
    (run_dir / f"{args.graph.upper()}_LAYERNORM_ROWS_AUDIT.md").write_text("\n".join(lines) + "\n")

    tsv = PHASE0 / f"whisper_{args.graph}_layernorm_rows_audit.tsv"
    new = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "step", "index", "node", "rows", "tiles",
            "tile_rows", "wait_s", "host_max_abs", "host_mean_abs",
            "audit_pass", "report",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": stamp,
                "step": args.step if args.graph == "decoder" else "",
                "index": r["index"],
                "node": r["node"],
                "rows": r["rows"],
                "tiles": r["tiles"],
                "tile_rows": r["tile_rows"],
                "wait_s": r["wait_s"],
                "host_max_abs": r["host_max_abs"],
                "host_mean_abs": r["host_mean_abs"],
                "audit_pass": r["audit_pass"],
                "report": str(report_path),
            })

    print()
    print(f"report: {report_path}")
    print(f"all_pass: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
