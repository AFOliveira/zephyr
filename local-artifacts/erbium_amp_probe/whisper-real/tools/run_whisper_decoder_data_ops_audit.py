#!/usr/bin/env python3
"""Audit real Whisper data-movement nodes on ET-SoC1."""
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

from run_whisper_decoder_layernorm_audit import safe_name
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
SRC = ROOT / "whisper_index_copy_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/index-copy-audit"

MAGIC = 0x57494350
OUT_OFFSET = 0x4000
INDEX_OFFSET = 0x200000
SRC_OFFSET = 0x600000
REF_OFFSET = 0xB00000
ARENA_BYTES = 16 * 1024 * 1024

SUPPORTED_OPS = {"Gather", "Slice", "Concat", "Reshape", "Unsqueeze", "Split", "Transpose"}


def align_up(value: int, align: int) -> int:
    return (value + align - 1) & ~(align - 1)


def memory_layout(elem_count: int, src_count: int, index_count: int,
                  mem_size: int) -> dict[str, int]:
    out_bytes = elem_count * 4
    index_bytes = index_count * 4
    src_bytes = src_count * 4
    ref_bytes = elem_count * 4

    index_offset = align_up(OUT_OFFSET + out_bytes, 0x1000)
    src_offset = align_up(index_offset + index_bytes, 0x1000)
    ref_offset = align_up(src_offset + src_bytes, 0x1000)
    required = ref_offset + ref_bytes
    dump_size = align_up(max(0x2000, OUT_OFFSET + out_bytes), 0x1000)

    if required > mem_size:
        raise ValueError(
            f"requires {required} bytes but mem_size is {mem_size} "
            f"(src={src_count} floats, out={elem_count} floats)"
        )

    return {
        "out_offset": OUT_OFFSET,
        "index_offset": index_offset,
        "src_offset": src_offset,
        "ref_offset": ref_offset,
        "required_bytes": required,
        "dump_size": dump_size,
    }


def node_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def selected_nodes(decoder_path: Path, op_types: set[str]) -> list[onnx.NodeProto]:
    model = onnx.load(decoder_path)
    return [node for node in model.graph.node if node.op_type in op_types]


def model_inputs(decoder_path: Path) -> set[str]:
    model = onnx.load(decoder_path)
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


def output_names_for_nodes(nodes: list[onnx.NodeProto], initializers: dict[str, np.ndarray],
                           graph_inputs: set[str]) -> list[str]:
    names: list[str] = []
    seen = set()
    for node in nodes:
        for name in [*node.input, *node.output]:
            if not name or name in initializers or name in graph_inputs:
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


def run_decoder_to_step_with_inputs(decoder, tokenizer: WhisperTokenizer,
                                    k_cross: np.ndarray, v_cross: np.ndarray,
                                    target_step: int, output_names: list[str]
                                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    prompt = decoder_prompt(tokenizer)
    generated = [prompt[0]]
    forced_after_sot = prompt[1:]
    k_self = np.zeros((4, 6, 64, 224), dtype=np.float32)
    v_self = np.zeros((4, 6, 224, 64), dtype=np.float32)
    final_outputs: dict[str, np.ndarray] | None = None
    final_inputs: dict[str, np.ndarray] | None = None

    for step in range(target_step + 1):
        current = generated[-1]
        inputs = {
            "x": np.asarray([[current]], dtype=np.int32),
            "index": np.asarray([[step]], dtype=np.int32),
            "k_cache_cross": k_cross,
            "v_cache_cross": v_cross,
            "k_cache_self": k_self,
            "v_cache_self": v_self,
        }
        names = list(dict.fromkeys([*output_names, "logits", "k_cache", "v_cache"]))
        values = decoder.run(names, inputs)
        outputs = {name: np.asarray(value) for name, value in zip(names, values)}

        logits = np.asarray(outputs["logits"], dtype=np.float32).reshape(-1)
        if step < len(forced_after_sot):
            next_id = forced_after_sot[step]
        else:
            next_id = int(logits.argmax())
        generated.append(next_id)
        if step == target_step:
            final_inputs = {k: np.asarray(v) for k, v in inputs.items()}
            final_outputs = outputs
        k_self = np.asarray(outputs["k_cache"], dtype=np.float32)
        v_self = np.asarray(outputs["v_cache"], dtype=np.float32)

    assert final_outputs is not None and final_inputs is not None
    final_outputs.update(final_inputs)
    return final_outputs, final_inputs, generated


def as_int_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1).astype(np.int64)


def node_index_map(node: onnx.NodeProto, values: dict[str, np.ndarray],
                   initializers: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def get_value(name: str) -> np.ndarray:
        if name in values:
            return np.asarray(values[name])
        if name in initializers:
            return np.asarray(initializers[name])
        raise KeyError(f"{node.name}: missing tensor {name}")

    ref = np.asarray(values[node.output[0]], dtype=np.float32)

    if node.op_type in {"Reshape", "Unsqueeze"}:
        data = np.asarray(get_value(node.input[0]), dtype=np.float32)
        src = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        index = np.arange(ref.size, dtype=np.uint32)
    elif node.op_type == "Transpose":
        data = np.asarray(get_value(node.input[0]), dtype=np.float32)
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = [int(v) for v in attr.ints]
                break
        if perm is None:
            perm = list(reversed(range(data.ndim)))
        src = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        src_index = np.arange(data.size, dtype=np.uint32).reshape(data.shape)
        index = np.ascontiguousarray(np.transpose(src_index, axes=perm).reshape(-1), dtype=np.uint32)
    elif node.op_type == "Gather":
        data = np.asarray(get_value(node.input[0]), dtype=np.float32)
        indices = np.asarray(get_value(node.input[1]))
        axis = node_attr_int(node, "axis", 0)
        if axis < 0:
            axis += data.ndim
        src = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        src_index = np.arange(data.size, dtype=np.uint32).reshape(data.shape)
        index = np.ascontiguousarray(np.take(src_index, indices, axis=axis).reshape(-1), dtype=np.uint32)
    elif node.op_type == "Slice":
        data = np.asarray(get_value(node.input[0]), dtype=np.float32)
        starts = as_int_array(get_value(node.input[1]))
        ends = as_int_array(get_value(node.input[2]))
        if len(node.input) > 3 and node.input[3]:
            axes = as_int_array(get_value(node.input[3]))
        else:
            axes = np.arange(len(starts), dtype=np.int64)
        if len(node.input) > 4 and node.input[4]:
            steps = as_int_array(get_value(node.input[4]))
        else:
            steps = np.ones(len(starts), dtype=np.int64)
        src = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
        src_index = np.arange(data.size, dtype=np.uint32).reshape(data.shape)
        slices: list[slice] = [slice(None)] * data.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            axis = int(axis)
            if axis < 0:
                axis += data.ndim
            slices[axis] = slice(int(start), int(end), int(step))
        index = np.ascontiguousarray(src_index[tuple(slices)].reshape(-1), dtype=np.uint32)
    elif node.op_type == "Concat":
        axis = node_attr_int(node, "axis", 0)
        arrays = [np.asarray(get_value(name), dtype=np.float32) for name in node.input]
        if axis < 0:
            axis += arrays[0].ndim
        src_parts = []
        index_parts = []
        offset = 0
        for arr in arrays:
            src_parts.append(np.ascontiguousarray(arr.reshape(-1), dtype=np.float32))
            index_parts.append((np.arange(arr.size, dtype=np.uint32) + offset).reshape(arr.shape))
            offset += arr.size
        src = np.ascontiguousarray(np.concatenate(src_parts), dtype=np.float32)
        index = np.ascontiguousarray(np.concatenate(index_parts, axis=axis).reshape(-1), dtype=np.uint32)
    else:
        raise ValueError(f"unsupported op {node.op_type}")

    ref_flat = np.ascontiguousarray(ref.reshape(-1), dtype=np.float32)
    if index.size != ref_flat.size:
        raise ValueError(f"{node.name}: index/ref size mismatch {index.size} != {ref_flat.size}")
    return src, index, ref_flat


def node_index_cases(node: onnx.NodeProto, values: dict[str, np.ndarray],
                     initializers: dict[str, np.ndarray]
                     ) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    def get_value(name: str) -> np.ndarray:
        if name in values:
            return np.asarray(values[name])
        if name in initializers:
            return np.asarray(initializers[name])
        raise KeyError(f"{node.name}: missing tensor {name}")

    if node.op_type != "Split":
        src, index, ref = node_index_map(node, values, initializers)
        return [("", node.output[0], src, index, ref)]

    data = np.asarray(get_value(node.input[0]), dtype=np.float32)
    axis = node_attr_int(node, "axis", 0)
    if axis < 0:
        axis += data.ndim

    if len(node.input) > 1 and node.input[1]:
        split_sizes = as_int_array(get_value(node.input[1]))
    else:
        attr_split = None
        for attr in node.attribute:
            if attr.name == "split":
                attr_split = np.asarray(attr.ints, dtype=np.int64)
                break
        if attr_split is not None:
            split_sizes = attr_split
        else:
            if data.shape[axis] % len(node.output) != 0:
                raise ValueError(
                    f"{node.name}: cannot infer equal Split for dimension "
                    f"{data.shape[axis]} and {len(node.output)} outputs"
                )
            split_sizes = np.full(
                len(node.output), data.shape[axis] // len(node.output),
                dtype=np.int64,
            )

    if len(split_sizes) != len(node.output):
        raise ValueError(
            f"{node.name}: split output count mismatch "
            f"{len(split_sizes)} != {len(node.output)}"
        )

    src = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
    src_index = np.arange(data.size, dtype=np.uint32).reshape(data.shape)
    cases: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    for out_no, (out_name, size) in enumerate(zip(node.output, split_sizes)):
        stop = start + int(size)
        slices: list[slice] = [slice(None)] * data.ndim
        slices[axis] = slice(start, stop)
        ref = np.ascontiguousarray(np.asarray(values[out_name], dtype=np.float32).reshape(-1))
        index = np.ascontiguousarray(src_index[tuple(slices)].reshape(-1), dtype=np.uint32)
        if index.size != ref.size:
            raise ValueError(
                f"{node.name}: Split output {out_no} index/ref size mismatch "
                f"{index.size} != {ref.size}"
            )
        cases.append((f"out{out_no}", out_name, src, index, ref))
        start = stop

    return cases


class IndexCopyRunner:
    def __init__(self, run_dir: Path, active_harts: int, timeout: int,
                 launcher: str, mem_size: int, launcher_has_mem_size: bool) -> None:
        self.run_dir = run_dir
        self.active_harts = active_harts
        self.timeout = timeout
        self.launcher = launcher
        self.mem_size = mem_size
        self.launcher_has_mem_size = launcher_has_mem_size
        self.local_static = run_dir / "build"
        self.local_static.mkdir(parents=True, exist_ok=True)
        self.remote = f"{REMOTE_ROOT}/{run_dir.name}"
        run_retry([*SSH_CMD, "true"], timeout=30)
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(self.remote)}"], timeout=30)
        self.elf_cache: dict[tuple[int, int, int, int], Path] = {}

    def build_elf(self, elem_count: int, layout: dict[str, int]) -> Path:
        key = (
            elem_count,
            layout["index_offset"],
            layout["src_offset"],
            layout["ref_offset"],
        )
        if key in self.elf_cache:
            return self.elf_cache[key]

        elf = self.local_static / (
            f"whisper_index_copy_n{elem_count}"
            f"_i{layout['index_offset']:x}"
            f"_s{layout['src_offset']:x}"
            f"_r{layout['ref_offset']:x}.elf"
        )
        cmd = [
            GCC,
            "-O3",
            "-funroll-loops",
            *BASE_FLAGS,
            "-DNUM_HARTS=16",
            f"-DACTIVE_HARTS={self.active_harts}u",
            f"-DELEM_COUNT={elem_count}u",
            f"-DINDEX_OFFSET=0x{layout['index_offset']:x}u",
            f"-DSRC_OFFSET=0x{layout['src_offset']:x}u",
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

    def run_node(self, index_no: int, node: onnx.NodeProto, src: np.ndarray,
                 index_map: np.ndarray, ref: np.ndarray, case_label: str = "",
                 output_name: str | None = None) -> dict[str, Any]:
        elem_count = int(ref.size)
        layout = memory_layout(elem_count, int(src.size), int(index_map.size), self.mem_size)
        elf = self.build_elf(elem_count, layout)
        suffix = f"_{case_label}" if case_label else ""
        node_safe = f"{index_no:03d}_{safe_name(node.name)}{suffix}"
        local_node = self.run_dir / node_safe
        local_node.mkdir(parents=True, exist_ok=True)
        remote_node = f"{self.remote}/{node_safe}"

        (local_node / "src.bin").write_bytes(src.astype("<f4").tobytes())
        (local_node / "index.bin").write_bytes(index_map.astype("<u4").tobytes())
        (local_node / "ref.bin").write_bytes(ref.astype("<f4").tobytes())
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(remote_node)}"], timeout=30)
        run_retry(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
                   str(local_node / "src.bin"), str(local_node / "index.bin"),
                   str(local_node / "ref.bin"), f"{REMOTE_HOST}:{remote_node}/"],
                  attempts=8, delay_s=15.0)

        launcher_options = ""
        if self.launcher_has_mem_size:
            launcher_options = (
                f" --mem_size {self.mem_size}"
                f" --dump_size {layout['dump_size']}"
            )

        script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_node}
LAUNCH={shlex.quote(self.launcher)}
ZERO=$PARENT/zero2m.bin
cd "$WORK"
ln -sf {shlex.quote(self.remote + '/' + elf.name)} {shlex.quote(elf.name)}
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"$LAUNCH" --elf-load ./{elf.name} --shire 0 --file_load 0x0,$ZERO \\
  --file_load 0x{layout['index_offset']:x},index.bin \\
  --file_load 0x{layout['src_offset']:x},src.bin \\
  --file_load 0x{layout['ref_offset']:x},ref.bin \\
  --dump_after dump.bin --timeout {self.timeout}{launcher_options} > run.log 2>&1
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
        "thread_id": raw[3], "i0": raw[4], "i1": raw[5],
        "checksum": raw[6], "done": raw[7],
    }})
Path("out.bin").write_bytes(data[0x{OUT_OFFSET:x}:0x{OUT_OFFSET:x} + {elem_count} * 4])
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
        out = np.fromfile(local_node / "out.bin", dtype="<f4").astype(np.float32)
        host_max_abs = float(np.max(np.abs(out - ref))) if ref.size else 0.0
        host_mean_abs = float(np.mean(np.abs(out - ref))) if ref.size else 0.0
        hpm = {
            "hpm3_cycles": int(summary[9]) | (int(summary[10]) << 32),
            "hpm4_inst0": int(summary[11]) | (int(summary[12]) << 32),
            "hpm5_inst1": int(summary[13]) | (int(summary[14]) << 32),
            "hpm6_l2_miss_req": int(summary[15]) | (int(summary[16]) << 32),
            "hpm7_icache_req": int(summary[17]) | (int(summary[18]) << 32),
            "hpm8_icache_etlink_req": int(summary[19]) | (int(summary[20]) << 32),
        }
        return {
            "index": index_no,
            "node": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "output": output_name or node.output[0],
            "case_label": case_label,
            "elem_count": elem_count,
            "src_count": int(src.size),
            "index_count": int(index_map.size),
            "mem_size": self.mem_size,
            "required_bytes": layout["required_bytes"],
            "dump_size": layout["dump_size"],
            "memory_layout": layout,
            "wait_s": raw["wait_s"],
            "summary_magic": int(summary[0]),
            "active_mask": int(summary[3]),
            "done_count": int(summary[4]),
            "output_hash": int(summary[5]),
            "reference_hash": int(summary[6]),
            "summary_max_abs": float(summary[7]) / 1_000_000.0,
            "summary_mean_abs": float(summary[8]) / 1_000_000.0,
            "host_max_abs": host_max_abs,
            "host_mean_abs": host_mean_abs,
            "audit_pass": int(summary[0]) == MAGIC and int(summary[4]) == self.active_harts and host_max_abs == 0.0,
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
    ap.add_argument("--graph", choices=("decoder", "encoder"), default="decoder")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--op-types", default="Gather,Slice,Concat,Reshape,Unsqueeze,Split")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name-filter", default="")
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--launcher", default=f"{REMOTE_PARENT}/erbium_soc1sim_argbuf")
    ap.add_argument("--mem-size", type=lambda x: int(x, 0), default=ARENA_BYTES)
    ap.add_argument("--launcher-has-mem-size", action="store_true")
    ap.add_argument("--compact-large-gather", action="store_true",
                    help="For oversized Gather sources, upload only selected elements and remap indices")
    ap.add_argument("--tile-elements", type=int, default=0,
                    help="Tile oversized outputs into this many FP32 elements per launch")
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "decoder_data_ops_audit_runs")
    args = ap.parse_args()

    op_types = {v.strip() for v in args.op_types.split(",") if v.strip()}
    unknown = op_types - SUPPORTED_OPS
    if unknown:
        raise SystemExit(f"unsupported op types: {sorted(unknown)}")

    model_path = args.decoder if args.graph == "decoder" else args.encoder
    nodes = selected_nodes(model_path, op_types)
    if args.name_filter:
        rx = re.compile(args.name_filter)
        nodes = [node for node in nodes if rx.search(node.name)]
    if args.start_index:
        nodes = nodes[args.start_index:]
    if args.limit:
        nodes = nodes[:args.limit]
    if not nodes:
        raise SystemExit("no data-movement nodes selected")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    initializers = value_map_from_initializers(model_path)
    graph_inputs = model_inputs(model_path)
    output_names = output_names_for_nodes(nodes, initializers, graph_inputs)
    features, input_meta = prepare_features(args.audio, run_dir)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    k_cross = np.asarray([], dtype=np.float32)
    v_cross = np.asarray([], dtype=np.float32)
    decoder_inputs: dict[str, np.ndarray] = {}
    tokens: list[int] = []
    if args.graph == "decoder":
        decoder = session_with_outputs(args.decoder, output_names)
        encoder = make_session(args.encoder)
        k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
        k_cross = np.asarray(k_cross, dtype=np.float32)
        v_cross = np.asarray(v_cross, dtype=np.float32)
        values, decoder_inputs, tokens = run_decoder_to_step_with_inputs(
            decoder, tokenizer, k_cross, v_cross, args.step, output_names
        )
    else:
        encoder = session_with_outputs(args.encoder, output_names)
        encoder_outputs = encoder.run(output_names, {"audio": features})
        values = {name: np.asarray(value) for name, value in zip(output_names, encoder_outputs)}
        values["audio"] = np.asarray(features)

    runner = IndexCopyRunner(
        run_dir, args.active_harts, args.timeout,
        args.launcher, args.mem_size, args.launcher_has_mem_size,
    )
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        try:
            cases = node_index_cases(node, values, initializers)
            for case_label, output_name, src, index_map, ref in cases:
                display_name = f"{node.name}:{case_label}" if case_label else node.name
                compacted_source = False
                original_src_count = int(src.size)
                try:
                    memory_layout(int(ref.size), int(src.size), int(index_map.size), args.mem_size)
                except ValueError:
                    if not (args.compact_large_gather and node.op_type == "Gather"):
                        pass
                    else:
                        unique_index, remapped = np.unique(index_map, return_inverse=True)
                        src = np.ascontiguousarray(src[unique_index], dtype=np.float32)
                        index_map = np.ascontiguousarray(remapped.astype(np.uint32), dtype=np.uint32)
                        compacted_source = True

                tile_ranges = [(0, int(ref.size))]
                try:
                    memory_layout(int(ref.size), int(src.size), int(index_map.size), args.mem_size)
                except ValueError:
                    if args.tile_elements <= 0:
                        raise
                    tile_ranges = [
                        (start, min(int(ref.size), start + args.tile_elements))
                        for start in range(0, int(ref.size), args.tile_elements)
                    ]

                for tile0, tile1 in tile_ranges:
                    tile_label = case_label
                    tile_ref = ref
                    tile_index = index_map
                    if tile0 != 0 or tile1 != int(ref.size):
                        tile_label = f"{case_label}_elems{tile0}_{tile1}" if case_label else f"elems{tile0}_{tile1}"
                        tile_ref = np.ascontiguousarray(ref[tile0:tile1], dtype=np.float32)
                        tile_index = np.ascontiguousarray(index_map[tile0:tile1], dtype=np.uint32)
                        memory_layout(int(tile_ref.size), int(src.size), int(tile_index.size), args.mem_size)

                    result = runner.run_node(
                        idx, node, src, tile_index, tile_ref,
                        case_label=tile_label, output_name=output_name,
                    )
                    result["compacted_source"] = compacted_source
                    result["original_src_count"] = original_src_count
                    result["full_elem_count"] = int(ref.size)
                    result["tile_offset"] = tile0
                    results.append(result)
                    tile_display = f"{display_name}:{tile_label}" if tile_label and tile_label != case_label else display_name
                    print(
                        f"{idx:03d} {node.op_type:9s} {tile_display}: "
                        f"pass={result['audit_pass']} max_abs={result['host_max_abs']:.9g} "
                        f"wait={result['wait_s']}",
                        flush=True,
                    )
        except Exception as exc:
            item = {
                "index": idx,
                "node": node.name,
                "op_type": node.op_type,
                "reason": str(exc),
            }
            skipped.append(item)
            print(f"{idx:03d} {node.op_type:9s} {node.name}: skip={exc}", flush=True)

    all_pass = bool(results) and all(r["audit_pass"] for r in results)
    report = {
        "timestamp": stamp,
        "scope": f"Real Whisper {args.graph} data-movement node audit on ET-SoC1",
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
        "launcher": args.launcher,
        "mem_size": args.mem_size,
        "compact_large_gather": args.compact_large_gather,
        "tile_elements": args.tile_elements,
        "op_types": sorted(op_types),
        "selected_nodes": len(nodes),
        "audited_nodes": len(results),
        "skipped_nodes": len(skipped),
        "all_pass": all_pass,
        "max_abs": max((r["host_max_abs"] for r in results), default=None),
        "mean_wait_s": (
            sum(float(r["wait_s"] or 0.0) for r in results) / len(results)
            if results else None
        ),
        "feature_sha256": sha256_array(features),
        "k_cache_cross_sha256": sha256_array(k_cross) if k_cross.size else None,
        "v_cache_cross_sha256": sha256_array(v_cross) if v_cross.size else None,
        "decoder_input_hashes": {
            name: sha256_array(np.asarray(value))
            for name, value in decoder_inputs.items()
        },
        "results": results,
        "skipped": skipped,
    }
    report_path = run_dir / f"{args.graph}_data_ops_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# Whisper {args.graph.title()} Data Ops Audit",
        "",
        f"- Graph: `{args.graph}`",
        f"- Decoder step: `{args.step}`" if args.graph == "decoder" else "- Decoder step: `n/a`",
        f"- Selected nodes: `{len(nodes)}`",
        f"- Audited nodes: `{len(results)}`",
        f"- Skipped nodes: `{len(skipped)}`",
        f"- Tile elements: `{args.tile_elements}`",
        f"- All audited pass: `{all_pass}`",
        f"- Max abs diff: `{report['max_abs']}`",
        f"- Mean silicon wait: `{report['mean_wait_s']} s`",
        "",
        "| index | op | node | elems | src elems | max abs | wait s | pass |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | `{r['op_type']}` | `{r['node']}` | "
            f"{r['elem_count']} | {r['src_count']} | {r['host_max_abs']:.9g} | "
            f"{float(r['wait_s'] or 0.0):.6f} | {r['audit_pass']} |"
        )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for s in skipped:
            lines.append(f"- `{s['op_type']}` `{s['node']}`: {s['reason']}")
    (run_dir / f"{args.graph.upper()}_DATA_OPS_AUDIT.md").write_text("\n".join(lines) + "\n")

    tsv = PHASE0 / f"whisper_{args.graph}_data_ops_audit.tsv"
    new = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "step", "index", "op_type", "node", "elem_count",
            "full_elem_count", "tile_offset",
            "src_count", "original_src_count", "compacted_source",
            "wait_s", "host_max_abs", "audit_pass",
            "mem_size", "required_bytes",
            "hpm3_cycles", "hpm4_inst0", "hpm5_inst1", "hpm6_l2_miss_req",
            "hpm7_icache_req", "hpm8_icache_etlink_req", "report",
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
                "elem_count": r["elem_count"],
                "full_elem_count": r.get("full_elem_count", r["elem_count"]),
                "tile_offset": r.get("tile_offset", 0),
                "src_count": r["src_count"],
                "original_src_count": r.get("original_src_count", r["src_count"]),
                "compacted_source": r.get("compacted_source", False),
                "wait_s": r["wait_s"],
                "host_max_abs": r["host_max_abs"],
                "audit_pass": r["audit_pass"],
                "mem_size": r["mem_size"],
                "required_bytes": r["required_bytes"],
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
    print(f"audited_nodes: {len(results)} skipped_nodes: {len(skipped)} all_pass: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
