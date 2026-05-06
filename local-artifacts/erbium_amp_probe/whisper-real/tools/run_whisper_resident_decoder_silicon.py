#!/usr/bin/env python3
"""Run a resident raw-INT8 Whisper decoder graph on ET-SoC1.

This executes the decoder side of Whisper in one kernel launch: prompt handling,
all decoder blocks, final LayerNorm, vocab projection, and greedy token
selection.  Encoder cross-attention K/V caches are still supplied by the host as
FP16, so this is decoder-complete but not yet log-mel-to-token-ID complete.
"""
from __future__ import annotations

import argparse
import json
import shlex
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import WhisperTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLES = PHASE1 / "native_v1_bundles"
DEFAULT_LAYOUT_ROOT = PHASE1 / "resident_layout_host"
DEFAULT_OUT = PHASE1 / "resident_decoder_silicon"
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
    make_session,
    run,
    sha256_array,
    sha256_path,
)
from run_whisper_raw_int8_host_executor import (  # noqa: E402
    load_resident_manifest,
    patch_model_from_raw_int8,
)
from run_whisper_resident_int8_tail_silicon import (  # noqa: E402
    build_weight_region,
)


SRC = ROOT / "whisper_resident_decoder_token_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/resident-decoder"
REMOTE_LAUNCHER = f"{REMOTE_PARENT}/erbium_soc1sim_argbuf_dynmem.remote.board"

MAGIC = 0x57524443
PARAM_MAGIC = 0x57524450
RUNTIME_BYTES = 16 * 1024 * 1024
WEIGHT_BYTES = 64 * 1024 * 1024
WEIGHT_REGION_OFFSET = RUNTIME_BYTES

SUMMARY_OFFSET = 0x1000
PARAM_OFFSET = 0x2000
TOKENS_OFFSET = 0x3000
CROSS_K_OFFSET = 0x100000
CROSS_V_OFFSET = 0x580000


def latest_dir(root: Path, prefix: str) -> Path:
    dirs = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no {prefix} directories found under {root}")
    return dirs[-1]


def bundle_dir_for(manifest: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_BUNDLES / manifest["source_bundle"]


def entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry["name"]: entry
        for entry in manifest["entries"]
        if entry["model_part"] == "decoder"
    }


def desc(entries: dict[str, dict[str, Any]], name: str) -> str:
    e = entries.get(name)
    if e is None:
        raise SystemExit(f"missing decoder resident weight: {name}")
    return f"{{ {int(e['offset'])}u, {float(e['scale']):.12g}f }}"


def generate_weight_header(path: Path, manifest: dict[str, Any]) -> None:
    e = entry_map(manifest)
    layers = [
        {
            "attn_ln_weight": "blocks.0.attn_ln.weight",
            "attn_ln_bias": "blocks.0.attn_ln.bias",
            "self_qkv_weight": "_v_2778",
            "self_q_bias": "onnx::Add_2382",
            "self_v_bias": "onnx::Add_2397",
            "self_out_weight": "onnx::MatMul_2490",
            "self_out_bias": "onnx::Add_2489",
            "cross_ln_weight": "blocks.0.cross_attn_ln.weight",
            "cross_ln_bias": "blocks.0.cross_attn_ln.bias",
            "cross_q_weight": "onnx::MatMul_2493",
            "cross_q_bias": "onnx::Add_2492",
            "cross_out_weight": "onnx::MatMul_2577",
            "cross_out_bias": "onnx::Add_2576",
            "mlp_ln_weight": "blocks.0.mlp_ln.weight",
            "mlp_ln_bias": "blocks.0.mlp_ln.bias",
            "mlp_fc1_weight": "onnx::MatMul_2580",
            "mlp_fc1_bias": "onnx::Add_2579",
            "mlp_fc2_weight": "onnx::MatMul_2583",
            "mlp_fc2_bias": "onnx::Add_2582",
        },
        {
            "attn_ln_weight": "blocks.1.attn_ln.weight",
            "attn_ln_bias": "blocks.1.attn_ln.bias",
            "self_qkv_weight": "_v_2785",
            "self_q_bias": "onnx::Add_2585",
            "self_v_bias": "onnx::Add_2600",
            "self_out_weight": "onnx::MatMul_2693",
            "self_out_bias": "onnx::Add_2692",
            "cross_ln_weight": "blocks.1.cross_attn_ln.weight",
            "cross_ln_bias": "blocks.1.cross_attn_ln.bias",
            "cross_q_weight": "onnx::MatMul_2696",
            "cross_q_bias": "onnx::Add_2695",
            "cross_out_weight": "onnx::MatMul_2780",
            "cross_out_bias": "onnx::Add_2779",
            "mlp_ln_weight": "blocks.1.mlp_ln.weight",
            "mlp_ln_bias": "blocks.1.mlp_ln.bias",
            "mlp_fc1_weight": "onnx::MatMul_2783",
            "mlp_fc1_bias": "onnx::Add_2782",
            "mlp_fc2_weight": "onnx::MatMul_2786",
            "mlp_fc2_bias": "onnx::Add_2785",
        },
        {
            "attn_ln_weight": "blocks.2.attn_ln.weight",
            "attn_ln_bias": "blocks.2.attn_ln.bias",
            "self_qkv_weight": "_v_2792",
            "self_q_bias": "onnx::Add_2788",
            "self_v_bias": "onnx::Add_2803",
            "self_out_weight": "onnx::MatMul_2896",
            "self_out_bias": "onnx::Add_2895",
            "cross_ln_weight": "blocks.2.cross_attn_ln.weight",
            "cross_ln_bias": "blocks.2.cross_attn_ln.bias",
            "cross_q_weight": "onnx::MatMul_2899",
            "cross_q_bias": "onnx::Add_2898",
            "cross_out_weight": "onnx::MatMul_2983",
            "cross_out_bias": "onnx::Add_2982",
            "mlp_ln_weight": "blocks.2.mlp_ln.weight",
            "mlp_ln_bias": "blocks.2.mlp_ln.bias",
            "mlp_fc1_weight": "onnx::MatMul_2986",
            "mlp_fc1_bias": "onnx::Add_2985",
            "mlp_fc2_weight": "onnx::MatMul_2989",
            "mlp_fc2_bias": "onnx::Add_2988",
        },
        {
            "attn_ln_weight": "blocks.3.attn_ln.weight",
            "attn_ln_bias": "blocks.3.attn_ln.bias",
            "self_qkv_weight": "_v_2799",
            "self_q_bias": "onnx::Add_2991",
            "self_v_bias": "onnx::Add_3006",
            "self_out_weight": "onnx::MatMul_3099",
            "self_out_bias": "onnx::Add_3098",
            "cross_ln_weight": "blocks.3.cross_attn_ln.weight",
            "cross_ln_bias": "blocks.3.cross_attn_ln.bias",
            "cross_q_weight": "onnx::MatMul_3102",
            "cross_q_bias": "onnx::Add_3101",
            "cross_out_weight": "onnx::MatMul_3186",
            "cross_out_bias": "onnx::Add_3185",
            "mlp_ln_weight": "blocks.3.mlp_ln.weight",
            "mlp_ln_bias": "blocks.3.mlp_ln.bias",
            "mlp_fc1_weight": "onnx::MatMul_3189",
            "mlp_fc1_bias": "onnx::Add_3188",
            "mlp_fc2_weight": "onnx::MatMul_3192",
            "mlp_fc2_bias": "onnx::Add_3191",
        },
    ]
    fields = list(layers[0].keys())
    lines = [
        "#ifndef WHISPER_RESIDENT_DECODER_WEIGHTS_AUTO_H",
        "#define WHISPER_RESIDENT_DECODER_WEIGHTS_AUTO_H",
        "struct weight_desc { unsigned int offset; float scale; };",
        "struct layer_desc {",
    ]
    for f in fields:
        lines.append(f"\tstruct weight_desc {f};")
    lines.extend([
        "};",
        f"static const struct weight_desc token_embedding = {desc(e, 'token_embedding.weight')};",
        f"static const struct weight_desc positional_embedding = {desc(e, 'positional_embedding.weight')};",
        f"static const struct weight_desc final_ln_weight = {desc(e, 'ln.weight')};",
        f"static const struct weight_desc final_ln_bias = {desc(e, 'ln.bias')};",
        f"static const struct weight_desc vocab_proj_weight = {desc(e, 'onnx::MatMul_3194')};",
        "static const struct layer_desc decoder_layers[4] = {",
    ])
    for layer in layers:
        vals = ", ".join(desc(e, layer[f]) for f in fields)
        lines.append(f"\t{{ {vals} }},")
    lines.extend([
        "};",
        "#endif",
        "",
    ])
    path.write_text("\n".join(lines))


def build_raw_int8_models(bundle_dir: Path, resident_manifest: dict[str, Any], run_dir: Path) -> tuple[Path, Path]:
    encoder_model = run_dir / "encoder_raw_int8_dequant.onnx"
    decoder_model = run_dir / "decoder_raw_int8_dequant.onnx"
    patch_model_from_raw_int8(
        part="encoder",
        source_model=bundle_dir / "encoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=encoder_model,
    )
    patch_model_from_raw_int8(
        part="decoder",
        source_model=bundle_dir / "decoder_fp32.onnx",
        bundle_dir=bundle_dir,
        resident_manifest=resident_manifest,
        out_model=decoder_model,
    )
    return encoder_model, decoder_model


def host_reference(bundle_dir: Path, bundle_manifest: dict[str, Any], encoder_model: Path,
                   decoder_model: Path, max_new_tokens: int, run_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    features = np.fromfile(
        bundle_dir / bundle_manifest["native_executor_contract"]["input_tensor"],
        dtype="<f4",
    ).reshape(bundle_manifest["input"]["feature_shape"]).astype(np.float32)
    encoder = make_session(encoder_model)
    decoder = make_session(decoder_model, ["/3/Add_2_output_0", "/ln/LayerNormalization_output_0"])
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    k_half_back = k_cross.astype(np.float16).astype(np.float32)
    v_half_back = v_cross.astype(np.float16).astype(np.float32)
    result = decode_loop(
        "host-only",
        run_dir / "host_fp16_cross_reference",
        decoder,
        WhisperTokenizer.from_pretrained("openai/whisper-tiny.en"),
        k_half_back,
        v_half_back,
        max_new_tokens,
        None,
    )
    return result, k_cross, v_cross


def build_runtime_image(path: Path, max_new_tokens: int, k_cross: np.ndarray, v_cross: np.ndarray) -> dict[str, Any]:
    image = bytearray(RUNTIME_BYTES)
    params = struct.pack("<16I", PARAM_MAGIC, int(max_new_tokens), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    image[PARAM_OFFSET:PARAM_OFFSET + len(params)] = params
    k_half = np.ascontiguousarray(k_cross.astype(np.float16))
    v_half = np.ascontiguousarray(v_cross.astype(np.float16))
    k_bytes = k_half.tobytes()
    v_bytes = v_half.tobytes()
    image[CROSS_K_OFFSET:CROSS_K_OFFSET + len(k_bytes)] = k_bytes
    image[CROSS_V_OFFSET:CROSS_V_OFFSET + len(v_bytes)] = v_bytes
    path.write_bytes(image)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "k_cross_fp32_sha256": sha256_array(k_cross),
        "v_cross_fp32_sha256": sha256_array(v_cross),
        "k_cross_fp16_bytes": len(k_bytes),
        "v_cross_fp16_bytes": len(v_bytes),
    }


def build_elf(run_dir: Path, header: Path) -> Path:
    elf = run_dir / "whisper_resident_decoder_token.elf"
    cmd = [
        GCC,
        "-O3",
        "-funroll-loops",
        "-fno-builtin",
        "-fno-tree-loop-distribute-patterns",
        *BASE_FLAGS,
        f"-I{run_dir}",
        "-DNUM_HARTS=1",
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
    total_steps = words[3]
    tokens = list(struct.unpack_from(f"<{min(total_steps + 1, 225)}I", data, TOKENS_OFFSET))
    summary = {
        "magic": hex(words[0]),
        "hart_id": words[1],
        "max_new_tokens": words[2],
        "total_steps": total_steps,
        "generated_tokens": words[4],
        "eos_seen": bool(words[5]),
        "final_token": words[6],
        "final_argmax": words[7],
        "hpm3": words[8] | (words[9] << 32),
        "hpm4": words[10] | (words[11] << 32),
        "hpm5": words[12] | (words[13] << 32),
        "hpm6": words[14] | (words[15] << 32),
        "hpm7": words[16] | (words[17] << 32),
        "hpm8": words[18] | (words[19] << 32),
        "ops": words[20] | (words[21] << 32),
        "done": bool(words[22]),
    }
    return {"summary": summary, "tokens": tokens}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-manifest", type=Path, default=None)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    resident_manifest_path = args.resident_manifest
    if resident_manifest_path is None:
        resident_manifest_path = latest_dir(DEFAULT_LAYOUT_ROOT, "run_") / "resident_weight_manifest.json"
    resident_manifest = load_resident_manifest(resident_manifest_path)
    bundle_dir = bundle_dir_for(resident_manifest, args.bundle_dir)
    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}_new{args.max_new_tokens}"
    run_dir.mkdir(parents=True, exist_ok=True)
    header = run_dir / "whisper_resident_decoder_weights_auto.h"
    generate_weight_header(header, resident_manifest)
    encoder_model, decoder_model = build_raw_int8_models(bundle_dir, resident_manifest, run_dir)
    ref, k_cross, v_cross = host_reference(
        bundle_dir, bundle_manifest, encoder_model, decoder_model,
        args.max_new_tokens, run_dir,
    )
    runtime_meta = build_runtime_image(run_dir / "runtime_region_16m.bin", args.max_new_tokens, k_cross, v_cross)
    weights_path = run_dir / "weights_region_64m.bin"
    weights_sha = build_weight_region(bundle_dir, resident_manifest, weights_path)
    elf = build_elf(run_dir, header)

    remote = f"{REMOTE_ROOT}/{run_dir.name}"
    run([*SSH_CMD, f"test -x {shlex.quote(REMOTE_LAUNCHER)}"], timeout=20)
    run([*SSH_CMD, f"mkdir -p {shlex.quote(remote)}"], timeout=20)
    run([
        "rsync", "-e", RSYNC_RSH, "-aq", "--timeout=240",
        str(elf), str(run_dir / "runtime_region_16m.bin"), str(weights_path),
        f"{REMOTE_HOST}:{remote}/",
    ], timeout=900)
    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote}
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
{shlex.quote(REMOTE_LAUNCHER)} --elf-load ./whisper_resident_decoder_token.elf \\
  --shire 0 --mem_size {RUNTIME_BYTES + WEIGHT_BYTES} --dump_size {RUNTIME_BYTES} \\
  --file_load 0x0,runtime_region_16m.bin \\
  --file_load 0x{RUNTIME_BYTES:x},weights_region_64m.bin \\
  --dump_after dump.bin --timeout {args.timeout} > run.log 2>&1
cat run.log
"""
    run([*SSH_CMD, "bash", "-s"], input=script, timeout=args.timeout + 300)
    run([
        "rsync", "-e", RSYNC_RSH, "-aq", "--timeout=180",
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
    ref_tokens = [int(x) for x in ref["tokens"][:len(parsed["tokens"])]]
    token_match = parsed["tokens"] == ref_tokens
    report = {
        "schema": "whisper-resident-decoder-silicon-report",
        "version": 1,
        "timestamp": stamp,
        "pass": bool(parsed["summary"]["magic"] == hex(MAGIC) and parsed["summary"]["done"] and token_match),
        "bundle_dir": str(bundle_dir),
        "resident_manifest": str(resident_manifest_path),
        "remote_work": remote,
        "max_new_tokens": args.max_new_tokens,
        "host_reference": {
            "tokens": ref["tokens"],
            "text": ref["text"],
            "generated_nonprompt_tokens": ref["generated_nonprompt_tokens"],
        },
        "silicon": {
            "kernel_wait_s": wait_s,
            "summary": parsed["summary"],
            "tokens": parsed["tokens"],
            "token_match_prefix": token_match,
            "tokens_per_s_wall": (parsed["summary"]["generated_tokens"] / wait_s) if wait_s else None,
            "ops_per_s": (parsed["summary"]["ops"] / wait_s) if wait_s else None,
        },
        "artifacts": {
            "elf": str(elf),
            "header": str(header),
            "runtime": runtime_meta,
            "weights_region": {
                "path": str(weights_path),
                "sha256": weights_sha,
                "bytes": weights_path.stat().st_size,
            },
        },
    }
    (run_dir / "resident_decoder_silicon_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Resident Decoder Silicon Report",
        "",
        f"- Result: `{'PASS' if report['pass'] else 'FAIL'}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        f"- Host tokens: `{ref_tokens}`",
        f"- Silicon tokens: `{parsed['tokens']}`",
        f"- Kernel wait seconds: `{wait_s}`",
        f"- Silicon generated token/s: `{report['silicon']['tokens_per_s_wall']}`",
        f"- Ops/s: `{report['silicon']['ops_per_s']}`",
        f"- Remote work: `{remote}`",
        "",
        "Scope: decoder graph on silicon with host-supplied FP16 encoder cross-cache.",
    ]
    (run_dir / "RESIDENT_DECODER_SILICON_REPORT.md").write_text("\n".join(md) + "\n")
    print(json.dumps({
        "pass": report["pass"],
        "report": str(run_dir / "resident_decoder_silicon_report.json"),
        "markdown": str(run_dir / "RESIDENT_DECODER_SILICON_REPORT.md"),
        "host_tokens": ref_tokens,
        "silicon_tokens": parsed["tokens"],
        "kernel_wait_s": wait_s,
        "tokens_per_s": report["silicon"]["tokens_per_s_wall"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
