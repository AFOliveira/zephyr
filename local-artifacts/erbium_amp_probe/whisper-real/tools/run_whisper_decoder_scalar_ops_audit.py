#!/usr/bin/env python3
"""Audit real Whisper scalar/Softmax nodes on ET-SoC1."""
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
    add_outputs,
    decoder_prompt,
    make_session,
    prepare_features,
    run,
    run_retry,
    sha256_array,
    sha256_path,
)


DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "audio" / "openai_whisper_jfk_16k.wav"
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "whisper_scalar_ops_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/scalar-op-audit"

MAGIC = 0x57534F50
OUT_OFFSET = 0x4000
A_OFFSET = 0x10000
B_OFFSET = 0x40000
REF_OFFSET = 0x70000
ARENA_BYTES = 16 * 1024 * 1024
ALIGN = 0x1000

OP_MODES = {
    "Add": 1,
    "Mul": 2,
    "Div": 3,
    "Erf": 4,
    "Softmax": 5,
}

DEFAULT_THRESHOLDS = {
    "Add": 2e-6,
    "Mul": 2e-6,
    "Div": 2e-6,
    "Erf": 3e-4,
    "Softmax": 5e-4,
}


def align_up(value: int, align: int = ALIGN) -> int:
    return (value + align - 1) & ~(align - 1)


def scalar_layout(elem_count: int, arena_bytes: int = ARENA_BYTES) -> dict[str, int]:
    out_bytes = elem_count * 4
    a_bytes = elem_count * 4
    b_bytes = elem_count * 4
    ref_bytes = elem_count * 4

    a_offset = align_up(OUT_OFFSET + out_bytes)
    b_offset = align_up(a_offset + a_bytes)
    ref_offset = align_up(b_offset + b_bytes)
    required = ref_offset + ref_bytes
    if required > arena_bytes:
        raise ValueError(
            f"requires {required} bytes but arena is {arena_bytes} "
            f"(elem_count={elem_count})"
        )
    return {
        "out_offset": OUT_OFFSET,
        "a_offset": a_offset,
        "b_offset": b_offset,
        "ref_offset": ref_offset,
        "required_bytes": required,
        "dump_size": OUT_OFFSET + out_bytes,
    }


def node_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def selected_nodes(decoder_path: Path, op_types: set[str]) -> list[onnx.NodeProto]:
    model = onnx.load(decoder_path)
    return [node for node in model.graph.node if node.op_type in op_types]


def model_inputs(model_path: Path) -> set[str]:
    model = onnx.load(model_path)
    return {i.name for i in model.graph.input}


def value_map_from_initializers(decoder_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(decoder_path)
    values = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                values[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return values


def constant_output_names(decoder_path: Path) -> set[str]:
    model = onnx.load(decoder_path)
    return {
        out
        for node in model.graph.node
        if node.op_type == "Constant"
        for out in node.output
    }


def output_names_for_nodes(nodes: list[onnx.NodeProto], initializers: dict[str, np.ndarray],
                           constants: set[str], graph_inputs: set[str]) -> list[str]:
    names: list[str] = []
    seen = set()
    for node in nodes:
        for name in [*node.input, *node.output]:
            if not name or name in initializers or name in graph_inputs:
                continue
            if name not in constants and name.startswith("onnx::"):
                continue
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def session_with_outputs(model_path: Path, outputs: list[str]):
    model = onnx.load(model_path)
    patched = add_outputs(model, outputs)
    import tempfile
    import onnxruntime as ort

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(patched, tmp.name)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(tmp.name, sess_options=opts, providers=["CPUExecutionProvider"])


def broadcast_input(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(np.broadcast_to(value, target_shape), dtype=np.float32)


def node_tensors(node: onnx.NodeProto, outputs: dict[str, np.ndarray],
                 initializers: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    def get_value(name: str) -> np.ndarray:
        if name in outputs:
            return np.asarray(outputs[name], dtype=np.float32)
        if name in initializers:
            return np.asarray(initializers[name], dtype=np.float32)
        raise KeyError(f"{node.name}: missing tensor {name}")

    ref = np.asarray(outputs[node.output[0]], dtype=np.float32)
    shape = tuple(ref.shape)
    a = broadcast_input(get_value(node.input[0]), shape)
    if node.op_type in {"Add", "Mul", "Div"}:
        b = broadcast_input(get_value(node.input[1]), shape)
    else:
        b = np.zeros(shape, dtype=np.float32)

    if node.op_type == "Softmax":
        axis = node_attr_int(node, "axis", -1)
        if axis < 0:
            axis += len(shape)
        if axis != len(shape) - 1:
            raise SystemExit(f"{node.name}: only last-axis Softmax is supported, got axis {axis}")
        cols = shape[-1]
        rows = int(np.prod(shape[:-1], dtype=np.int64))
    else:
        cols = int(np.prod(shape, dtype=np.int64))
        rows = 1

    return (
        np.ascontiguousarray(a.reshape(-1), dtype=np.float32),
        np.ascontiguousarray(b.reshape(-1), dtype=np.float32),
        np.ascontiguousarray(ref.reshape(-1), dtype=np.float32),
        rows,
        cols,
    )


def scalar_tile_cases(node: onnx.NodeProto, a: np.ndarray, b: np.ndarray,
                      ref: np.ndarray, rows: int, cols: int,
                      tile_elements: int, arena_bytes: int
                      ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, int]]:
    if tile_elements <= 0:
        scalar_layout(int(ref.size), arena_bytes)
        return [("", a, b, ref, rows, cols, 0)]

    try:
        scalar_layout(int(ref.size), arena_bytes)
        return [("", a, b, ref, rows, cols, 0)]
    except ValueError:
        pass

    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, int]] = []
    if node.op_type == "Softmax":
        if cols <= 0:
            raise ValueError(f"{node.name}: invalid Softmax cols {cols}")
        rows_per_tile = max(1, tile_elements // cols)
        for row0 in range(0, rows, rows_per_tile):
            row1 = min(rows, row0 + rows_per_tile)
            elem0 = row0 * cols
            elem1 = row1 * cols
            cases.append((
                f"rows{row0}_{row1}",
                np.ascontiguousarray(a[elem0:elem1], dtype=np.float32),
                np.ascontiguousarray(b[elem0:elem1], dtype=np.float32),
                np.ascontiguousarray(ref[elem0:elem1], dtype=np.float32),
                row1 - row0,
                cols,
                elem0,
            ))
    else:
        for elem0 in range(0, int(ref.size), tile_elements):
            elem1 = min(int(ref.size), elem0 + tile_elements)
            cases.append((
                f"elems{elem0}_{elem1}",
                np.ascontiguousarray(a[elem0:elem1], dtype=np.float32),
                np.ascontiguousarray(b[elem0:elem1], dtype=np.float32),
                np.ascontiguousarray(ref[elem0:elem1], dtype=np.float32),
                1,
                elem1 - elem0,
                elem0,
            ))

    for label, ta, tb, tref, trows, tcols, _ in cases:
        try:
            scalar_layout(int(tref.size), arena_bytes)
        except ValueError as exc:
            raise ValueError(f"{node.name}:{label}: tile still too large: {exc}") from exc
        if node.op_type == "Softmax" and int(tref.size) != trows * tcols:
            raise ValueError(f"{node.name}:{label}: bad Softmax tile shape")
        if ta.size != tref.size or tb.size != tref.size:
            raise ValueError(f"{node.name}:{label}: tile input/ref size mismatch")
    return cases


class ScalarOpRunner:
    def __init__(self, run_dir: Path, active_harts: int, timeout: int,
                 mem_size: int) -> None:
        self.run_dir = run_dir
        self.active_harts = active_harts
        self.timeout = timeout
        self.mem_size = mem_size
        self.use_dynamic_launcher = mem_size > ARENA_BYTES
        self.local_static = run_dir / "build"
        self.local_static.mkdir(parents=True, exist_ok=True)
        self.remote = f"{REMOTE_ROOT}/{run_dir.name}"
        run_retry([*SSH_CMD, "true"], timeout=30)
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(self.remote)}"], timeout=30)
        self.elf_cache: dict[tuple[str, int, int, int, int, int, int, int], Path] = {}

    def build_elf(self, op_type: str, elem_count: int, rows: int, cols: int,
                  layout: dict[str, int]) -> Path:
        key = (
            op_type, elem_count, rows, cols,
            layout["out_offset"], layout["a_offset"],
            layout["b_offset"], layout["ref_offset"],
        )
        if key in self.elf_cache:
            return self.elf_cache[key]

        safe = f"{op_type.lower()}_n{elem_count}_r{rows}_c{cols}"
        elf = self.local_static / f"whisper_scalar_{safe}.elf"
        cmd = [
            GCC,
            "-O3",
            "-funroll-loops",
            *BASE_FLAGS,
            "-DNUM_HARTS=16",
            f"-DACTIVE_HARTS={self.active_harts}u",
            f"-DOP_MODE={OP_MODES[op_type]}u",
            f"-DELEM_COUNT={elem_count}u",
            f"-DROWS={rows}u",
            f"-DCOLS={cols}u",
            f"-DOUT_OFFSET=0x{layout['out_offset']:x}u",
            f"-DA_OFFSET=0x{layout['a_offset']:x}u",
            f"-DB_OFFSET=0x{layout['b_offset']:x}u",
            f"-DREF_OFFSET=0x{layout['ref_offset']:x}u",
            "-o",
            str(elf),
            str(SRC),
            str(CRT),
            str(LAYOUT),
        ]
        run(cmd)
        self.elf_cache[key] = elf
        run_retry(["rsync", "-e", RSYNC_RSH, "-azq", "--timeout=60",
                   "--partial", "--inplace",
                   str(elf), f"{REMOTE_HOST}:{self.remote}/"],
                  attempts=5, delay_s=10.0)
        return elf

    def run_node(self, index: int, node: onnx.NodeProto, a: np.ndarray, b: np.ndarray,
                 ref: np.ndarray, rows: int, cols: int,
                 case_label: str = "", full_elem_count: int | None = None,
                 tile_offset: int = 0) -> dict[str, Any]:
        elem_count = int(ref.size)
        layout = scalar_layout(elem_count, self.mem_size)
        elf = self.build_elf(node.op_type, elem_count, rows, cols, layout)
        label = f"_{safe_name(case_label)}" if case_label else ""
        node_safe = f"{index:03d}_{safe_name(node.name)}{label}"
        local_node = self.run_dir / node_safe
        local_node.mkdir(parents=True, exist_ok=True)
        remote_node = f"{self.remote}/{node_safe}"

        (local_node / "a.bin").write_bytes(a.astype("<f4").tobytes())
        (local_node / "b.bin").write_bytes(b.astype("<f4").tobytes())
        (local_node / "ref.bin").write_bytes(ref.astype("<f4").tobytes())
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(remote_node)}"], timeout=30)
        run_retry(["rsync", "-e", RSYNC_RSH, "-azq", "--timeout=60",
                   "--partial", "--inplace",
                   str(local_node / "a.bin"), str(local_node / "b.bin"),
                   str(local_node / "ref.bin"), f"{REMOTE_HOST}:{remote_node}/"],
                  attempts=8, delay_s=15.0)

        script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_node}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
if [ {1 if self.use_dynamic_launcher else 0} -eq 1 ]; then
  LAUNCH=$PARENT/erbium_soc1sim_argbuf_dynmem.remote.board
fi
ZERO=$PARENT/zero2m.bin
cd "$WORK"
ln -sf {shlex.quote(self.remote + '/' + elf.name)} {shlex.quote(elf.name)}
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
if [ {1 if self.use_dynamic_launcher else 0} -eq 1 ]; then
  "$LAUNCH" --elf-load ./{elf.name} --shire 0 --mem_size {self.mem_size} \\
    --dump_size {layout['dump_size']} --file_load 0x0,$ZERO \\
    --file_load 0x{layout['a_offset']:x},a.bin \\
    --file_load 0x{layout['b_offset']:x},b.bin \\
    --file_load 0x{layout['ref_offset']:x},ref.bin \\
    --dump_after dump.bin --timeout {self.timeout} > run.log 2>&1
else
  "$LAUNCH" --elf-load ./{elf.name} --shire 0 --file_load 0x0,$ZERO \\
    --file_load 0x{layout['a_offset']:x},a.bin \\
    --file_load 0x{layout['b_offset']:x},b.bin \\
    --file_load 0x{layout['ref_offset']:x},ref.bin \\
    --dump_after dump.bin --timeout {self.timeout} > run.log 2>&1
fi
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
        "thread_id": raw[3], "op_mode": raw[4], "i0": raw[5],
        "i1": raw[6], "checksum": raw[7], "done": raw[8],
    }})
Path("out.bin").write_bytes(data[0x{layout['out_offset']:x}:0x{layout['out_offset']:x} + {elem_count} * 4])
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
        run_retry(["rsync", "-e", RSYNC_RSH, "-azq", "--timeout=60",
                   "--partial", "--inplace",
                   f"{REMOTE_HOST}:{remote_node}/summary.json",
                   f"{REMOTE_HOST}:{remote_node}/out.bin",
                   f"{REMOTE_HOST}:{remote_node}/run.log",
                   str(local_node) + "/"],
                  attempts=12, delay_s=20.0)

        raw = json.loads((local_node / "summary.json").read_text())
        summary = raw["summary"]
        out = np.fromfile(local_node / "out.bin", dtype="<f4").astype(np.float32)
        host_max_abs = float(np.max(np.abs(out - ref))) if ref.size else 0.0
        host_mean_abs = float(np.mean(np.abs(out - ref))) if ref.size else 0.0
        threshold = DEFAULT_THRESHOLDS[node.op_type]
        hpm = {
            "hpm3_cycles": int(summary[12]) | (int(summary[13]) << 32),
            "hpm4_inst0": int(summary[14]) | (int(summary[15]) << 32),
            "hpm5_inst1": int(summary[16]) | (int(summary[17]) << 32),
            "hpm6_l2_miss_req": int(summary[18]) | (int(summary[19]) << 32),
            "hpm7_icache_req": int(summary[20]) | (int(summary[21]) << 32),
            "hpm8_icache_etlink_req": int(summary[22]) | (int(summary[23]) << 32),
        }
        result = {
            "index": index,
            "node": node.name,
            "case_label": case_label,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "output": node.output[0],
            "shape": list(ref.shape),
            "elem_count": elem_count,
            "full_elem_count": int(full_elem_count if full_elem_count is not None else elem_count),
            "tile_offset": int(tile_offset),
            "rows": rows,
            "cols": cols,
            "required_bytes": layout["required_bytes"],
            "mem_size": self.mem_size,
            "dynamic_launcher": self.use_dynamic_launcher,
            "wait_s": raw["wait_s"],
            "summary_magic": int(summary[0]),
            "summary_op_mode": int(summary[1]),
            "active_mask": int(summary[6]),
            "done_count": int(summary[7]),
            "output_hash": int(summary[8]),
            "reference_hash": int(summary[9]),
            "summary_max_abs": float(summary[10]) / 1_000_000.0,
            "summary_mean_abs": float(summary[11]) / 1_000_000.0,
            "host_max_abs": host_max_abs,
            "host_mean_abs": host_mean_abs,
            "threshold": threshold,
            "audit_pass": (
                int(summary[0]) == MAGIC
                and int(summary[7]) == self.active_harts
                and host_max_abs <= threshold
            ),
            "stream_error": raw["stream_error"],
            "kernel_launch_error": raw["kernel_launch_error"],
            "hpm": hpm,
            "node_dir": str(local_node),
        }
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--graph", choices=("decoder", "encoder"), default="decoder")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--op-types", default="Add,Mul,Div,Erf,Softmax")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Skip this many selected nodes before running")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name-filter", default="")
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--mem-size", type=int, default=ARENA_BYTES,
                    help="Arg buffer size in bytes. Values above 16 MiB use the dynamic-memory launcher.")
    ap.add_argument("--tile-elements", type=int, default=0,
                    help="Tile oversized tensors into this many FP32 elements per launch")
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "decoder_scalar_ops_audit_runs")
    args = ap.parse_args()

    op_types = {v.strip() for v in args.op_types.split(",") if v.strip()}
    unknown = op_types - set(OP_MODES)
    if unknown:
        raise SystemExit(f"unsupported op types: {sorted(unknown)}")

    model_path = args.decoder if args.graph == "decoder" else args.encoder
    nodes = selected_nodes(model_path, op_types)
    if args.name_filter:
        rx = re.compile(args.name_filter)
        nodes = [node for node in nodes if rx.search(node.name)]
    total_selected_nodes = len(nodes)
    if args.start_index:
        nodes = nodes[args.start_index:]
    if args.limit:
        nodes = nodes[:args.limit]
    if not nodes:
        raise SystemExit("no scalar nodes selected")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    initializers = value_map_from_initializers(model_path)
    constants = constant_output_names(model_path)
    graph_inputs = model_inputs(model_path)
    output_names = output_names_for_nodes(nodes, initializers, constants, graph_inputs)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    features, input_meta = prepare_features(args.audio, run_dir)
    k_cross = np.asarray([], dtype=np.float32)
    v_cross = np.asarray([], dtype=np.float32)
    tokens: list[int] = []
    if args.graph == "decoder":
        decoder = session_with_outputs(args.decoder, output_names)
        encoder = make_session(args.encoder)
        k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
        k_cross = np.asarray(k_cross, dtype=np.float32)
        v_cross = np.asarray(v_cross, dtype=np.float32)
        outputs, tokens = run_decoder_to_step(
            decoder, tokenizer, k_cross, v_cross, args.step, output_names
        )
    else:
        encoder = session_with_outputs(args.encoder, output_names)
        encoder_outputs = encoder.run(output_names, {"audio": features})
        outputs = {name: np.asarray(value) for name, value in zip(output_names, encoder_outputs)}
        outputs["audio"] = np.asarray(features)

    runner = ScalarOpRunner(run_dir, args.active_harts, args.timeout, args.mem_size)
    results = []
    for idx, node in enumerate(nodes, start=args.start_index):
        a, b, ref, rows, cols = node_tensors(node, outputs, initializers)
        cases = scalar_tile_cases(node, a, b, ref, rows, cols,
                                  args.tile_elements, args.mem_size)
        for case_label, ta, tb, tref, trows, tcols, tile_offset in cases:
            result = runner.run_node(
                idx, node, ta, tb, tref, trows, tcols,
                case_label=case_label,
                full_elem_count=int(ref.size),
                tile_offset=tile_offset,
            )
            results.append(result)
            label = f":{case_label}" if case_label else ""
            print(
                f"{idx:03d} {node.op_type:7s} {node.name}{label}: "
                f"pass={result['audit_pass']} max_abs={result['host_max_abs']:.9g} "
                f"wait={result['wait_s']}",
                flush=True,
            )

    all_pass = all(r["audit_pass"] for r in results)
    report = {
        "timestamp": stamp,
        "scope": f"Real Whisper {args.graph} scalar/Softmax node audit on ET-SoC1",
        "graph": args.graph,
        "audio": input_meta,
        "model": {
            "encoder": str(args.encoder),
            "encoder_sha256": sha256_path(args.encoder),
            "decoder": str(args.decoder),
            "decoder_sha256": sha256_path(args.decoder),
            "tokenizer": "openai/whisper-tiny.en",
        },
        "decoder_step": args.step,
        "tokens_to_step": [int(t) for t in tokens],
        "token_text_to_step": [tokenizer.decode([int(t)]) for t in tokens],
        "active_harts": args.active_harts,
        "mem_size": args.mem_size,
        "dynamic_launcher": args.mem_size > ARENA_BYTES,
        "tile_elements": args.tile_elements,
        "op_types": sorted(op_types),
        "selected_start_index": args.start_index,
        "selected_limit": args.limit,
        "total_selected_nodes_before_slice": total_selected_nodes,
        "selected_nodes": len(nodes),
        "audited_tiles": len(results),
        "all_pass": all_pass,
        "max_abs": max(r["host_max_abs"] for r in results),
        "mean_wait_s": sum(float(r["wait_s"] or 0.0) for r in results) / len(results),
        "feature_sha256": sha256_array(features),
        "k_cache_cross_sha256": sha256_array(k_cross) if k_cross.size else None,
        "v_cache_cross_sha256": sha256_array(v_cross) if v_cross.size else None,
        "results": results,
    }
    report_path = run_dir / f"{args.graph}_scalar_ops_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# Whisper {args.graph.title()} Scalar Ops Audit",
        "",
        f"- Graph: `{args.graph}`",
        f"- Decoder step: `{args.step}`" if args.graph == "decoder" else "- Decoder step: `n/a`",
        f"- Selected nodes: `{len(nodes)}`",
        f"- Audited tiles: `{len(results)}`",
        f"- Tile elements: `{args.tile_elements}`",
        f"- All pass: `{all_pass}`",
        f"- Max abs diff: `{report['max_abs']:.9g}`",
        f"- Mean silicon wait: `{report['mean_wait_s']:.6f} s`",
        "",
        "| index | op | node | tile | elems | max abs | wait s | cycles | pass |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | `{r['op_type']}` | `{r['node']}` | "
            f"`{r['case_label']}` | {r['elem_count']} | {r['host_max_abs']:.9g} | "
            f"{float(r['wait_s'] or 0.0):.6f} | {r['hpm']['hpm3_cycles']} | "
            f"{r['audit_pass']} |"
        )
    (run_dir / f"{args.graph.upper()}_SCALAR_OPS_AUDIT.md").write_text("\n".join(lines) + "\n")

    tsv = PHASE0 / f"whisper_{args.graph}_scalar_ops_audit.tsv"
    new = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "step", "index", "op_type", "node", "case_label",
            "elem_count", "full_elem_count", "tile_offset", "rows", "cols",
            "required_bytes", "wait_s", "host_max_abs", "host_mean_abs",
            "threshold", "audit_pass", "hpm3_cycles", "hpm4_inst0",
            "hpm5_inst1", "hpm6_l2_miss_req", "hpm7_icache_req",
            "hpm8_icache_etlink_req", "report",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": stamp,
                "step": args.step,
                "index": r["index"],
                "op_type": r["op_type"],
                "node": r["node"],
                "case_label": r["case_label"],
                "elem_count": r["elem_count"],
                "full_elem_count": r["full_elem_count"],
                "tile_offset": r["tile_offset"],
                "rows": r["rows"],
                "cols": r["cols"],
                "required_bytes": r["required_bytes"],
                "wait_s": r["wait_s"],
                "host_max_abs": r["host_max_abs"],
                "host_mean_abs": r["host_mean_abs"],
                "threshold": r["threshold"],
                "audit_pass": r["audit_pass"],
                "hpm3_cycles": r["hpm"]["hpm3_cycles"],
                "hpm4_inst0": r["hpm"]["hpm4_inst0"],
                "hpm5_inst1": r["hpm"]["hpm5_inst1"],
                "hpm6_l2_miss_req": r["hpm"]["hpm6_l2_miss_req"],
                "hpm7_icache_req": r["hpm"]["hpm7_icache_req"],
                "hpm8_icache_etlink_req": r["hpm"]["hpm8_icache_etlink_req"],
                "report": str(report_path),
            })

    print()
    print(f"report: {report_path}")
    print(f"all_pass: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
