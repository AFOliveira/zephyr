#!/usr/bin/env python3
"""Run Whisper encoder scalar-node audits natively on the ET-SoC1 host.

The older scalar audit loop generated ONNXRuntime tensors locally and copied
large a/b/ref blobs over Tailscale for every node.  This runner stages the
encoder model, input features, and one runtime-parameterized scalar ELF under
the allowed esperanto-soc6 artifact root, then does the tensor generation,
silicon launch, comparison, and cleanup on esperanto-soc6.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from run_whisper_e2e_audit import (
    REMOTE_BASE,
    REMOTE_HOST,
    REMOTE_PARENT,
    RSYNC_RSH,
    SSH_CMD,
    run,
    run_retry,
    sha256_path,
)


ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLE = PHASE1 / "native_v1_bundles" / "bundle_20260505-205553"
DEFAULT_ENCODER = DEFAULT_BUNDLE / "encoder_fp32.onnx"
DEFAULT_FEATURES = DEFAULT_BUNDLE / "input_features_1x80x3000.bin"
DEFAULT_ELF = PHASE1 / "dyn_scalar_build" / "whisper_scalar_dyn.elf"

REMOTE_RUN_ROOT = f"{REMOTE_PARENT}/whisper-real/remote-native-scalar"
REMOTE_VENV_PY = f"{REMOTE_PARENT}/whisper-real/venv/bin/python"


REMOTE_WORKER = r'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


MAGIC = 0x57534459
PARAM_OFFSET = 0x2000
OUT_OFFSET = 0x4000
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


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("/"))
    return safe[:160] or "unnamed"


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {
        vi.name: vi
        for vi in list(inferred.graph.value_info)
        + list(inferred.graph.output)
        + list(inferred.graph.input)
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


def make_session(model_path: Path, outputs: list[str]) -> ort.InferenceSession:
    patched = add_outputs(onnx.load(model_path), outputs)
    tmp = model_path.parent / f".patched_{os.getpid()}_{time.time_ns()}.onnx"
    onnx.save(patched, tmp)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(str(tmp), sess_options=opts, providers=["CPUExecutionProvider"])
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def selected_nodes(model_path: Path, op_types: set[str]) -> list[onnx.NodeProto]:
    model = onnx.load(model_path)
    return [node for node in model.graph.node if node.op_type in op_types]


def model_inputs(model_path: Path) -> set[str]:
    model = onnx.load(model_path)
    return {i.name for i in model.graph.input}


def value_map_from_initializers(model_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(model_path)
    values = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                values[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return values


def constant_output_names(model_path: Path) -> set[str]:
    model = onnx.load(model_path)
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


def node_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


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
            raise RuntimeError(f"{node.name}: only last-axis Softmax is supported, got axis {axis}")
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


def scalar_layout(elem_count: int, mem_size: int) -> dict[str, int]:
    out_bytes = elem_count * 4
    a_bytes = elem_count * 4
    b_bytes = elem_count * 4
    ref_bytes = elem_count * 4
    a_offset = align_up(OUT_OFFSET + out_bytes)
    b_offset = align_up(a_offset + a_bytes)
    ref_offset = align_up(b_offset + b_bytes)
    required = ref_offset + ref_bytes
    if required > mem_size:
        raise ValueError(f"requires {required} bytes but mem_size is {mem_size}")
    return {
        "out_offset": OUT_OFFSET,
        "a_offset": a_offset,
        "b_offset": b_offset,
        "ref_offset": ref_offset,
        "required_bytes": required,
        "dump_size": OUT_OFFSET + out_bytes,
    }


def scalar_tile_cases(node: onnx.NodeProto, a: np.ndarray, b: np.ndarray,
                      ref: np.ndarray, rows: int, cols: int,
                      tile_elements: int, mem_size: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, int]]:
    if tile_elements <= 0:
        scalar_layout(int(ref.size), mem_size)
        return [("", a, b, ref, rows, cols, 0)]
    try:
        scalar_layout(int(ref.size), mem_size)
        return [("", a, b, ref, rows, cols, 0)]
    except ValueError:
        pass

    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, int]] = []
    if node.op_type == "Softmax":
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

    for _, ta, tb, tref, trows, tcols, _ in cases:
        scalar_layout(int(tref.size), mem_size)
        if int(tref.size) != trows * tcols:
            raise ValueError("bad tile geometry")
        if ta.size != tref.size or tb.size != tref.size:
            raise ValueError("tile input/ref size mismatch")
    return cases


def hpm_from_summary(summary: tuple[int, ...]) -> dict[str, int]:
    return {
        "hpm3_cycles": int(summary[12]) | (int(summary[13]) << 32),
        "hpm4_inst0": int(summary[14]) | (int(summary[15]) << 32),
        "hpm5_inst1": int(summary[16]) | (int(summary[17]) << 32),
        "hpm6_l2_miss_req": int(summary[18]) | (int(summary[19]) << 32),
        "hpm7_icache_req": int(summary[20]) | (int(summary[21]) << 32),
        "hpm8_icache_etlink_req": int(summary[22]) | (int(summary[23]) << 32),
    }


def run_case(args: argparse.Namespace, node_index: int, node: onnx.NodeProto,
             case_label: str, a: np.ndarray, b: np.ndarray, ref: np.ndarray,
             rows: int, cols: int, full_elem_count: int, tile_offset: int) -> dict[str, Any]:
    elem_count = int(ref.size)
    layout = scalar_layout(elem_count, args.mem_size)
    node_safe = f"{node_index:03d}_{safe_name(node.name)}"
    if case_label:
        node_safe += f"_{safe_name(case_label)}"
    work = args.out_dir / node_safe
    work.mkdir(parents=True, exist_ok=True)

    params = struct.pack(
        "<16I",
        MAGIC,
        OP_MODES[node.op_type],
        elem_count,
        rows,
        cols,
        layout["out_offset"],
        layout["a_offset"],
        layout["b_offset"],
        layout["ref_offset"],
        0, 0, 0, 0, 0, 0, 0,
    )
    (work / "params.bin").write_bytes(params)
    (work / "a.bin").write_bytes(a.astype("<f4", copy=False).tobytes())
    (work / "b.bin").write_bytes(b.astype("<f4", copy=False).tobytes())
    (work / "ref.bin").write_bytes(ref.astype("<f4", copy=False).tobytes())

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.remote_base}:{args.remote_parent}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        str(args.launcher),
        "--elf-load", str(args.elf),
        "--shire", str(args.shire),
        "--mem_size", str(args.mem_size),
        "--dump_size", str(layout["dump_size"]),
        "--file_load", f"0x0,{args.zero}",
        "--file_load", f"0x{PARAM_OFFSET:x},{work / 'params.bin'}",
        "--file_load", f"0x{layout['a_offset']:x},{work / 'a.bin'}",
        "--file_load", f"0x{layout['b_offset']:x},{work / 'b.bin'}",
        "--file_load", f"0x{layout['ref_offset']:x},{work / 'ref.bin'}",
        "--dump_after", str(work / "dump.bin"),
        "--timeout", str(args.kernel_timeout),
    ]
    started = time.time()
    with (work / "run.log").open("w") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    wall_s = time.time() - started

    run_log = (work / "run.log").read_text(errors="ignore")
    wait_s = None
    m = re.search(r"Kernel wait seconds:\s*([0-9.eE+-]+)", run_log)
    if m:
        wait_s = float(m.group(1))

    result: dict[str, Any] = {
        "index": node_index,
        "node": node.name,
        "case_label": case_label,
        "op_type": node.op_type,
        "inputs": list(node.input),
        "output": node.output[0],
        "elem_count": elem_count,
        "full_elem_count": full_elem_count,
        "tile_offset": int(tile_offset),
        "rows": int(rows),
        "cols": int(cols),
        "required_bytes": layout["required_bytes"],
        "dump_size": layout["dump_size"],
        "mem_size": args.mem_size,
        "wait_s": wait_s,
        "wall_s": wall_s,
        "returncode": proc.returncode,
        "stream_error": "Stream error" in run_log,
        "kernel_launch_error": "Error on kernel launch" in run_log,
        "node_dir": str(work),
    }

    if proc.returncode != 0 or not (work / "dump.bin").exists():
        result.update({
            "audit_pass": False,
            "host_max_abs": None,
            "host_mean_abs": None,
            "threshold": DEFAULT_THRESHOLDS[node.op_type],
            "error": "launcher failed or dump missing",
        })
    else:
        data = (work / "dump.bin").read_bytes()
        summary = struct.unpack_from("<32I", data, 0x1000)
        out = np.frombuffer(
            data[layout["out_offset"]:layout["out_offset"] + elem_count * 4],
            dtype="<f4",
        ).astype(np.float32, copy=True)
        host_max_abs = float(np.max(np.abs(out - ref))) if ref.size else 0.0
        host_mean_abs = float(np.mean(np.abs(out - ref))) if ref.size else 0.0
        threshold = DEFAULT_THRESHOLDS[node.op_type]
        result.update({
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
                and int(summary[7]) == args.active_harts
                and host_max_abs <= threshold
            ),
            "hpm": hpm_from_summary(summary),
        })

    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.keep_blobs:
        for name in ("a.bin", "b.bin", "ref.bin", "dump.bin", "params.bin"):
            try:
                (work / name).unlink()
            except FileNotFoundError:
                pass
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--elf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--remote-base", default="/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2")
    ap.add_argument("--remote-parent", default="/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe")
    ap.add_argument("--launcher", type=Path, required=True)
    ap.add_argument("--zero", type=Path, required=True)
    ap.add_argument("--shire", type=int, default=0)
    ap.add_argument("--op-types", default="Add,Mul,Div,Erf,Softmax")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name-filter", default="")
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--kernel-timeout", type=int, default=120)
    ap.add_argument("--mem-size", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--tile-elements", type=int, default=0)
    ap.add_argument("--output-batch-size", type=int, default=1)
    ap.add_argument("--keep-blobs", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    op_types = {v.strip() for v in args.op_types.split(",") if v.strip()}
    unknown = op_types - set(OP_MODES)
    if unknown:
        raise SystemExit(f"unsupported op types: {sorted(unknown)}")

    all_nodes = selected_nodes(args.encoder, op_types)
    if args.name_filter:
        rx = re.compile(args.name_filter)
        all_nodes = [node for node in all_nodes if rx.search(node.name)]
    total_selected_nodes = len(all_nodes)
    nodes = all_nodes[args.start_index:]
    if args.limit:
        nodes = nodes[:args.limit]
    if not nodes:
        raise SystemExit("no scalar nodes selected")

    initializers = value_map_from_initializers(args.encoder)
    constants = constant_output_names(args.encoder)
    graph_inputs = model_inputs(args.encoder)
    features = np.fromfile(args.features, dtype="<f4").reshape(1, 80, 3000).astype(np.float32)

    results: list[dict[str, Any]] = []
    indexed = list(enumerate(nodes, start=args.start_index))
    batch_size = max(1, args.output_batch_size)

    for batch0 in range(0, len(indexed), batch_size):
        batch = indexed[batch0:batch0 + batch_size]
        batch_nodes = [node for _, node in batch]
        output_names = output_names_for_nodes(batch_nodes, initializers, constants, graph_inputs)
        session = make_session(args.encoder, output_names)
        outs = session.run(output_names, {"audio": features})
        outputs = {name: np.asarray(value) for name, value in zip(output_names, outs)}
        outputs["audio"] = features

        for idx, node in batch:
            a, b, ref, rows, cols = node_tensors(node, outputs, initializers)
            cases = scalar_tile_cases(node, a, b, ref, rows, cols, args.tile_elements, args.mem_size)
            for case_label, ta, tb, tref, trows, tcols, tile_offset in cases:
                result = run_case(
                    args, idx, node, case_label, ta, tb, tref, trows, tcols,
                    int(ref.size), tile_offset,
                )
                results.append(result)
                label = f":{case_label}" if case_label else ""
                print(
                    f"{idx:03d} {node.op_type:7s} {node.name}{label}: "
                    f"pass={result.get('audit_pass')} max_abs={result.get('host_max_abs')} "
                    f"wait={result.get('wait_s')} wall={result.get('wall_s'):.3f}",
                    flush=True,
                )

    all_pass = bool(results) and all(bool(r.get("audit_pass")) for r in results)
    report = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "scope": "Remote-native Whisper encoder scalar/Softmax node audit on ET-SoC1",
        "graph": "encoder",
        "encoder": str(args.encoder),
        "features": str(args.features),
        "active_harts": args.active_harts,
        "mem_size": args.mem_size,
        "tile_elements": args.tile_elements,
        "output_batch_size": batch_size,
        "op_types": sorted(op_types),
        "selected_start_index": args.start_index,
        "selected_limit": args.limit,
        "total_selected_nodes_before_slice": total_selected_nodes,
        "selected_nodes": len(nodes),
        "audited_tiles": len(results),
        "all_pass": all_pass,
        "max_abs": max(float(r.get("host_max_abs") or 0.0) for r in results),
        "mean_wait_s": sum(float(r.get("wait_s") or 0.0) for r in results) / len(results),
        "mean_wall_s": sum(float(r.get("wall_s") or 0.0) for r in results) / len(results),
        "results": results,
    }
    (args.out_dir / "encoder_scalar_remote_native_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    with (args.out_dir / "encoder_scalar_remote_native.tsv").open("w", newline="") as f:
        cols = [
            "index", "op_type", "node", "case_label", "elem_count",
            "full_elem_count", "tile_offset", "rows", "cols",
            "required_bytes", "wait_s", "wall_s", "host_max_abs",
            "host_mean_abs", "threshold", "audit_pass", "hpm3_cycles",
            "hpm4_inst0", "hpm5_inst1", "hpm6_l2_miss_req",
            "hpm7_icache_req", "hpm8_icache_etlink_req",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for r in results:
            hpm = r.get("hpm") or {}
            writer.writerow({
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
                "wait_s": r.get("wait_s"),
                "wall_s": r.get("wall_s"),
                "host_max_abs": r.get("host_max_abs"),
                "host_mean_abs": r.get("host_mean_abs"),
                "threshold": r.get("threshold"),
                "audit_pass": r.get("audit_pass"),
                "hpm3_cycles": hpm.get("hpm3_cycles"),
                "hpm4_inst0": hpm.get("hpm4_inst0"),
                "hpm5_inst1": hpm.get("hpm5_inst1"),
                "hpm6_l2_miss_req": hpm.get("hpm6_l2_miss_req"),
                "hpm7_icache_req": hpm.get("hpm7_icache_req"),
                "hpm8_icache_etlink_req": hpm.get("hpm8_icache_etlink_req"),
            })

    lines = [
        "# Whisper Encoder Scalar Remote-Native Audit",
        "",
        f"- Selected nodes: `{len(nodes)}`",
        f"- Audited tiles: `{len(results)}`",
        f"- All pass: `{all_pass}`",
        f"- Max abs diff: `{report['max_abs']:.9g}`",
        f"- Mean silicon wait: `{report['mean_wait_s']:.6f} s`",
        "",
        "| index | op | node | tile | elems | max abs | wait s | pass |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | `{r['op_type']}` | `{r['node']}` | "
            f"`{r['case_label']}` | {r['elem_count']} | "
            f"{float(r.get('host_max_abs') or 0.0):.9g} | "
            f"{float(r.get('wait_s') or 0.0):.6f} | {r.get('audit_pass')} |"
        )
    (args.out_dir / "ENCODER_SCALAR_REMOTE_NATIVE_AUDIT.md").write_text("\n".join(lines) + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_remote_worker(local_path: Path) -> None:
    local_path.write_text(textwrap.dedent(REMOTE_WORKER).strip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    ap.add_argument("--start-index", type=int, default=53)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--op-types", default="Add,Mul,Div,Erf,Softmax")
    ap.add_argument("--name-filter", default="")
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--kernel-timeout", type=int, default=120)
    ap.add_argument("--mem-size", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--tile-elements", type=int, default=0)
    ap.add_argument("--output-batch-size", type=int, default=1)
    ap.add_argument("--shire", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=PHASE1 / "remote_native_encoder_scalar_runs")
    args = ap.parse_args()

    for path in (args.encoder, args.features, args.elf):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    local_run = args.out_dir / f"run_{stamp}"
    local_run.mkdir(parents=True, exist_ok=True)
    worker = local_run / "remote_encoder_scalar_worker.py"
    write_remote_worker(worker)

    remote_run = f"{REMOTE_RUN_ROOT}/run_{stamp}"
    remote_encoder = f"{remote_run}/{args.encoder.name}"
    remote_features = f"{remote_run}/{args.features.name}"
    remote_elf = f"{remote_run}/{args.elf.name}"
    remote_worker = f"{remote_run}/{worker.name}"
    remote_out = f"{remote_run}/results"

    metadata = {
        "timestamp": stamp,
        "remote_host": REMOTE_HOST,
        "remote_run": remote_run,
        "encoder": str(args.encoder),
        "encoder_sha256": sha256_path(args.encoder),
        "features": str(args.features),
        "features_sha256": sha256_path(args.features),
        "elf": str(args.elf),
        "elf_sha256": sha256_path(args.elf),
        "start_index": args.start_index,
        "limit": args.limit,
        "mem_size": args.mem_size,
        "tile_elements": args.tile_elements,
        "output_batch_size": args.output_batch_size,
    }
    (local_run / "remote_native_scalar_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(remote_run)} {shlex.quote(remote_out)}"], timeout=30)
    run_retry([
        "rsync", "-e", RSYNC_RSH, "-azq", "--timeout=120", "--partial", "--inplace",
        str(args.encoder), str(args.features), str(args.elf), str(worker),
        f"{REMOTE_HOST}:{remote_run}/",
    ], attempts=8, delay_s=15.0)

    remote_cmd = [
        REMOTE_VENV_PY,
        remote_worker,
        "--encoder", remote_encoder,
        "--features", remote_features,
        "--elf", remote_elf,
        "--out-dir", remote_out,
        "--remote-base", REMOTE_BASE,
        "--remote-parent", REMOTE_PARENT,
        "--launcher", f"{REMOTE_PARENT}/erbium_soc1sim_argbuf_dynmem.remote.board",
        "--zero", f"{REMOTE_PARENT}/zero2m.bin",
        "--shire", str(args.shire),
        "--op-types", args.op_types,
        "--start-index", str(args.start_index),
        "--active-harts", str(args.active_harts),
        "--kernel-timeout", str(args.kernel_timeout),
        "--mem-size", str(args.mem_size),
        "--tile-elements", str(args.tile_elements),
        "--output-batch-size", str(args.output_batch_size),
    ]
    if args.limit:
        remote_cmd.extend(["--limit", str(args.limit)])
    if args.name_filter:
        remote_cmd.extend(["--name-filter", args.name_filter])

    quoted = " ".join(shlex.quote(part) for part in remote_cmd)
    run_log = local_run / "remote_worker.log"
    with run_log.open("w") as log:
        proc = subprocess.run(
            [*SSH_CMD, f"cd {shlex.quote(remote_run)} && {quoted}"],
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(run_log.read_text(errors="ignore"), end="")

    run_retry([
        "rsync", "-e", RSYNC_RSH, "-azq", "--timeout=120", "--partial", "--inplace",
        "--include=*/", "--include=*.json", "--include=*.tsv", "--include=*.md",
        "--include=*.log", "--exclude=*.bin", "--exclude=*.onnx", "--exclude=*",
        f"{REMOTE_HOST}:{remote_out}/",
        str(local_run / "results") + "/",
    ], attempts=8, delay_s=15.0)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
