#!/usr/bin/env python3
"""Run the resident raw-INT8 Whisper decoder tail on ET-SoC1.

This is the first accelerator implementation that consumes the resident
64 MiB raw INT8 weight region instead of per-token/per-tile FP32 weights.
It runs one audited decoder step:

  final decoder LayerNorm input
  -> ET-SoC1 LayerNorm
  -> ET-SoC1 full vocab projection over raw INT8 resident weights
  -> ET-SoC1 argmax

The host still prepares the decoder-step boundary tensor from the raw-INT8
host reference.  This intentionally tests the resident weight transport and the
largest decoder projection before the full resident graph executor exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import subprocess
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
AMP_ROOT = ROOT.parent
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLES = PHASE1 / "native_v1_bundles"
DEFAULT_LAYOUT_ROOT = PHASE1 / "resident_layout_host"
DEFAULT_OUT = PHASE1 / "resident_int8_tail_silicon"
sys.path.insert(0, str(HERE))

from run_whisper_e2e_audit import (  # noqa: E402
    BASE_FLAGS,
    CRT,
    GCC,
    LAYOUT,
    REMOTE_BASE,
    REMOTE_HOST,
    REMOTE_PARENT,
    RSYNC_RSH,
    SSH_CMD,
    decode_loop,
    decoder_prompt,
    make_session,
    run,
    sha256_array,
    sha256_path,
)
from run_whisper_raw_int8_host_executor import (  # noqa: E402
    load_resident_manifest,
    patch_model_from_raw_int8,
)


SRC = ROOT / "whisper_resident_int8_tail_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/resident-int8-tail"
REMOTE_LAUNCHER = f"{REMOTE_PARENT}/erbium_soc1sim_argbuf_dynmem.remote.board"

MAGIC = 0x57524938
PARAM_MAGIC = 0x57504938
RUNTIME_MIB = 16
WEIGHT_MIB = 64
RUNTIME_BYTES = RUNTIME_MIB * 1024 * 1024
WEIGHT_BYTES = WEIGHT_MIB * 1024 * 1024
WEIGHT_REGION_OFFSET = RUNTIME_BYTES

SLOTS_OFFSET = 0x0000
SUMMARY_OFFSET = 0x1000
PARAM_OFFSET = 0x2000
LN_IN_OFFSET = 0x10000
LN_WEIGHT_OFFSET = 0x20000
LN_BIAS_OFFSET = 0x30000
LN_REF_OFFSET = 0x40000


def latest_dir(root: Path, prefix: str) -> Path:
    dirs = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no {prefix} directories found under {root}")
    return dirs[-1]


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def f32_from_bits(v: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(v) & 0xffffffff))[0]


def resident_entry(manifest: dict[str, Any], model_part: str, name: str) -> dict[str, Any]:
    for entry in manifest["entries"]:
        if entry["model_part"] == model_part and entry["name"] == name:
            return entry
    raise SystemExit(f"resident entry not found: {model_part}:{name}")


def bundle_dir_for(manifest: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_BUNDLES / manifest["source_bundle"]


def build_weight_region(bundle_dir: Path, resident_manifest: dict[str, Any], out_path: Path) -> str:
    region = bytearray(WEIGHT_BYTES)
    weight_root = bundle_dir / "weights_int8_symmetric"
    for entry in resident_manifest["entries"]:
        src = weight_root / entry["source_file"]
        data = src.read_bytes()
        if len(data) != int(entry["nbytes"]):
            raise SystemExit(f"size mismatch for {src}: {len(data)} != {entry['nbytes']}")
        start = int(entry["offset"])
        end = start + len(data)
        if end > len(region):
            raise SystemExit(f"resident weight overflows region: {entry['name']}")
        region[start:end] = data
    out_path.write_bytes(region)
    return sha256_path(out_path)


def build_runtime_image(
    *,
    path: Path,
    ln_in: np.ndarray,
    ln_weight: np.ndarray,
    ln_bias: np.ndarray,
    ln_ref: np.ndarray,
    resident_wt_offset: int,
    wt_scale: float,
    host_logits: np.ndarray,
) -> dict[str, Any]:
    image = bytearray(RUNTIME_BYTES)
    argmax = int(np.asarray(host_logits, dtype=np.float32).reshape(-1).argmax())
    best = float(np.asarray(host_logits, dtype=np.float32).reshape(-1)[argmax])
    params = struct.pack(
        "<16I",
        PARAM_MAGIC,
        argmax,
        f32_bits(best),
        int(resident_wt_offset),
        int(host_logits.size),
        int(ln_in.size),
        f32_bits(wt_scale),
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    image[PARAM_OFFSET:PARAM_OFFSET + len(params)] = params

    def put(offset: int, arr: np.ndarray, name: str) -> str:
        data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        end = offset + len(data)
        if end > len(image):
            raise SystemExit(f"{name} overflows runtime image")
        image[offset:end] = data
        return sha256_array(np.asarray(arr, dtype=np.float32))

    shas = {
        "ln_in_sha256": put(LN_IN_OFFSET, ln_in, "ln_in"),
        "ln_weight_sha256": put(LN_WEIGHT_OFFSET, ln_weight, "ln_weight"),
        "ln_bias_sha256": put(LN_BIAS_OFFSET, ln_bias, "ln_bias"),
        "ln_ref_sha256": put(LN_REF_OFFSET, ln_ref, "ln_ref"),
    }
    path.write_bytes(image)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "expected_argmax": argmax,
        "expected_argmax_value": best,
        "resident_wt_offset": resident_wt_offset,
        "wt_scale": wt_scale,
        **shas,
    }


def build_raw_int8_models(
    bundle_dir: Path,
    resident_manifest: dict[str, Any],
    run_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    encoder_model = run_dir / "encoder_raw_int8_dequant.onnx"
    decoder_model = run_dir / "decoder_raw_int8_dequant.onnx"
    enc = patch_model_from_raw_int8(
        part="encoder",
        source_model=bundle_dir / "encoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=encoder_model,
    )
    dec = patch_model_from_raw_int8(
        part="decoder",
        source_model=bundle_dir / "decoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=decoder_model,
    )
    return encoder_model, decoder_model, {"encoder": enc, "decoder": dec}


def export_decoder_step(
    *,
    bundle_dir: Path,
    bundle_manifest: dict[str, Any],
    encoder_model: Path,
    decoder_model: Path,
    step: int,
) -> dict[str, Any]:
    features = np.fromfile(
        bundle_dir / bundle_manifest["native_executor_contract"]["input_tensor"],
        dtype="<f4",
    ).reshape(bundle_manifest["input"]["feature_shape"]).astype(np.float32)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    encoder = make_session(encoder_model)
    decoder = make_session(decoder_model, ["/3/Add_2_output_0", "/ln/LayerNormalization_output_0"])
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    k_self = np.zeros((4, 6, 64, 224), dtype=np.float32)
    v_self = np.zeros((4, 6, 224, 64), dtype=np.float32)
    prompt = decoder_prompt(tokenizer)
    generated = [prompt[0]]
    forced_after_sot = prompt[1:]
    selected = None

    for idx in range(step + 1):
        current = generated[-1]
        raw_ln_in, ln_ref, logits, k_next, v_next = decoder.run(
            ["/3/Add_2_output_0", "/ln/LayerNormalization_output_0", "logits", "k_cache", "v_cache"],
            {
                "x": np.asarray([[current]], dtype=np.int32),
                "index": np.asarray([[idx]], dtype=np.int32),
                "k_cache_cross": k_cross,
                "v_cache_cross": v_cross,
                "k_cache_self": k_self,
                "v_cache_self": v_self,
            },
        )
        logits_1d = np.asarray(logits, dtype=np.float32).reshape(-1)
        host_argmax = int(logits_1d.argmax())
        selected = {
            "step": idx,
            "current_token": int(current),
            "current_text": tokenizer.decode([int(current)]),
            "host_argmax": host_argmax,
            "host_argmax_text": tokenizer.decode([host_argmax]),
            "raw_ln_in": np.asarray(raw_ln_in, dtype=np.float32).reshape(-1),
            "ln_ref": np.asarray(ln_ref, dtype=np.float32).reshape(-1),
            "host_logits": logits_1d,
        }
        if idx < len(forced_after_sot):
            next_id = forced_after_sot[idx]
        else:
            next_id = host_argmax
        generated.append(int(next_id))
        k_self = np.asarray(k_next, dtype=np.float32)
        v_self = np.asarray(v_next, dtype=np.float32)

    assert selected is not None
    return selected


def decoder_ln_params(decoder_model: Path) -> tuple[np.ndarray, np.ndarray]:
    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(decoder_model).graph.initializer}
    return (
        np.asarray(init["ln.weight"], dtype=np.float32).reshape(-1),
        np.asarray(init["ln.bias"], dtype=np.float32).reshape(-1),
    )


def build_elf(
    *,
    run_dir: Path,
    active_harts: int,
    resident_wt_offset: int,
    wt_scale: float,
) -> Path:
    elf = run_dir / "whisper_resident_int8_tail.elf"
    cmd = [
        GCC,
        "-O3",
        "-funroll-loops",
        "-fno-builtin",
        "-fno-tree-loop-distribute-patterns",
        *BASE_FLAGS,
        f"-DNUM_HARTS={active_harts}",
        f"-DACTIVE_HARTS={active_harts}u",
        f"-DRESIDENT_WT_OFFSET={resident_wt_offset}u",
        f"-DWT_SCALE={wt_scale:.12g}f",
        "-o", str(elf),
        str(SRC),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def parse_dump(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    words = struct.unpack_from("<32I", data, SUMMARY_OFFSET)
    slots = []
    for h in range(16):
        s = struct.unpack_from("<16I", data, SLOTS_OFFSET + h * 64)
        slots.append({
            "hart": h,
            "magic": hex(s[0]),
            "hart_id": s[1],
            "minion_id": s[2],
            "thread_id": s[3],
            "col0": s[4],
            "col1": s[5],
            "active_harts": s[6],
            "checksum": s[7],
            "done": s[8],
            "argmax_col": s[9],
            "argmax_value": f32_from_bits(s[10]),
        })
    summary = {
        "magic": hex(words[0]),
        "active_harts": words[1],
        "k_dim": words[2],
        "n_cols": words[3],
        "active_mask": hex(words[4]),
        "done_count": words[5],
        "output_hash": words[6],
        "reference_hash": words[7],
        "ln_max_abs": words[8] / 1_000_000.0,
        "ln_mean_abs": words[9] / 1_000_000.0,
        "argmax_col": words[10],
        "argmax_value": f32_from_bits(words[11]),
        "expected_argmax_col": words[12],
        "argmax_match": bool(words[13]),
        "hpm3": words[14] | (words[15] << 32),
        "hpm4": words[16] | (words[17] << 32),
        "hpm5": words[18] | (words[19] << 32),
        "hpm6": words[20] | (words[21] << 32),
        "hpm7": words[22] | (words[23] << 32),
        "hpm8": words[24] | (words[25] << 32),
        "ops": words[26] | (words[27] << 32),
    }
    return {"summary": summary, "slots": slots}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-manifest", type=Path, default=None)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    resident_manifest_path = args.resident_manifest
    if resident_manifest_path is None:
        resident_manifest_path = latest_dir(DEFAULT_LAYOUT_ROOT, "run_") / "resident_weight_manifest.json"
    resident_manifest = load_resident_manifest(resident_manifest_path)
    bundle_dir = bundle_dir_for(resident_manifest, args.bundle_dir)
    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    projection = resident_entry(resident_manifest, "decoder", "onnx::MatMul_3194")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}_step{args.step}_ah{args.active_harts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder_model, decoder_model, patch_report = build_raw_int8_models(bundle_dir, resident_manifest, run_dir)
    step_data = export_decoder_step(
        bundle_dir=bundle_dir,
        bundle_manifest=bundle_manifest,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        step=args.step,
    )
    ln_weight, ln_bias = decoder_ln_params(decoder_model)

    weights_path = run_dir / "weights_region_64m.bin"
    weights_sha = build_weight_region(bundle_dir, resident_manifest, weights_path)
    runtime_meta = build_runtime_image(
        path=run_dir / "runtime_region_16m.bin",
        ln_in=step_data["raw_ln_in"],
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        ln_ref=step_data["ln_ref"],
        resident_wt_offset=int(projection["offset"]),
        wt_scale=float(projection["scale"]),
        host_logits=step_data["host_logits"],
    )
    elf = build_elf(
        run_dir=run_dir,
        active_harts=args.active_harts,
        resident_wt_offset=int(projection["offset"]),
        wt_scale=float(projection["scale"]),
    )

    remote = f"{REMOTE_ROOT}/{run_dir.name}"
    run([*SSH_CMD, f"test -x {shlex.quote(REMOTE_LAUNCHER)}"], timeout=20)
    run([*SSH_CMD, f"mkdir -p {shlex.quote(remote)}"], timeout=20)
    run([
        "rsync", "-e", RSYNC_RSH, "-aq", "--timeout=180",
        str(elf), str(run_dir / "runtime_region_16m.bin"), str(weights_path),
        f"{REMOTE_HOST}:{remote}/",
    ], timeout=600)

    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote}
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
{shlex.quote(REMOTE_LAUNCHER)} --elf-load ./whisper_resident_int8_tail.elf \\
  --shire 0 --mem_size {RUNTIME_BYTES + WEIGHT_BYTES} --dump_size {RUNTIME_BYTES} \\
  --file_load 0x0,runtime_region_16m.bin \\
  --file_load 0x{WEIGHT_REGION_OFFSET:x},weights_region_64m.bin \\
  --dump_after dump.bin --timeout {args.timeout} > run.log 2>&1
cat run.log
"""
    run([*SSH_CMD, "bash", "-s"], input=script, timeout=args.timeout + 240)
    run([
        "rsync", "-e", RSYNC_RSH, "-aq", "--timeout=120",
        f"{REMOTE_HOST}:{remote}/dump.bin",
        f"{REMOTE_HOST}:{remote}/run.log",
        str(run_dir) + "/",
    ], timeout=300)

    parsed = parse_dump(run_dir / "dump.bin")
    log_text = (run_dir / "run.log").read_text(errors="ignore")
    wait_s = None
    for line in log_text.splitlines():
        if line.startswith("Kernel wait seconds:"):
            wait_s = float(line.split(":", 1)[1].strip())
    summary = parsed["summary"]
    pass_ok = (
        summary["magic"] == hex(MAGIC)
        and summary["done_count"] == args.active_harts
        and summary["active_mask"] == hex((1 << args.active_harts) - 1)
        and summary["argmax_match"]
        and summary["argmax_col"] == runtime_meta["expected_argmax"]
    )
    report = {
        "schema": "whisper-resident-int8-tail-silicon-report",
        "version": 1,
        "timestamp": stamp,
        "pass": pass_ok,
        "bundle_dir": str(bundle_dir),
        "resident_manifest": str(resident_manifest_path),
        "remote_work": remote,
        "launcher": REMOTE_LAUNCHER,
        "step": {
            "index": args.step,
            "current_token": step_data["current_token"],
            "current_text": step_data["current_text"],
            "host_argmax": step_data["host_argmax"],
            "host_argmax_text": step_data["host_argmax_text"],
        },
        "projection_weight": {
            "name": projection["name"],
            "offset": projection["offset"],
            "nbytes": projection["nbytes"],
            "shape": projection["shape"],
            "scale": projection["scale"],
            "sha256": projection["sha256"],
        },
        "artifacts": {
            "elf": str(elf),
            "runtime_region": runtime_meta,
            "weights_region": {
                "path": str(weights_path),
                "sha256": weights_sha,
                "bytes": weights_path.stat().st_size,
            },
            "dump": str(run_dir / "dump.bin"),
            "run_log": str(run_dir / "run.log"),
        },
        "patched_models": patch_report,
        "silicon": {
            "kernel_wait_s": wait_s,
            "summary": summary,
            "slots": parsed["slots"],
            "ops_per_s": (summary["ops"] / wait_s) if wait_s else None,
        },
    }
    (run_dir / "resident_int8_tail_silicon_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Resident INT8 Tail Silicon Report",
        "",
        f"- Result: `{'PASS' if pass_ok else 'FAIL'}`",
        f"- Step: `{args.step}`",
        f"- Host argmax: `{step_data['host_argmax']}` `{step_data['host_argmax_text']}`",
        f"- Silicon argmax: `{summary['argmax_col']}`",
        f"- Kernel wait seconds: `{wait_s}`",
        f"- Ops/s: `{report['silicon']['ops_per_s']}`",
        f"- LN max abs vs host raw-INT8 ref: `{summary['ln_max_abs']}`",
        f"- Remote work: `{remote}`",
        "",
        "This run stages one 16 MiB runtime region plus one 64 MiB raw INT8 weight region and reads the final projection weights from the resident region on ET-SoC1.",
    ]
    (run_dir / "RESIDENT_INT8_TAIL_SILICON_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "pass": pass_ok,
        "report": str(run_dir / "resident_int8_tail_silicon_report.json"),
        "markdown": str(run_dir / "RESIDENT_INT8_TAIL_SILICON_REPORT.md"),
        "host_argmax": step_data["host_argmax"],
        "silicon_argmax": summary["argmax_col"],
        "kernel_wait_s": wait_s,
        "ops_per_s": report["silicon"]["ops_per_s"],
    }, indent=2))
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
