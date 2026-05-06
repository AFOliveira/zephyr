#!/usr/bin/env python3
"""Run Whisper Tiny EN log-mel -> token IDs with encoder and decoder on ET-SoC1.

This is still host-launched in two kernels, but the model graph is not executed
by host ONNXRuntime:

1. Encoder kernel consumes log-mel features and resident raw-INT8 weights,
   producing FP16 cross-attention K/V caches in the runtime arena.
2. Decoder kernel consumes that encoder dump plus the same resident weights and
   greedily emits token IDs.

Host ONNXRuntime is used only to build the audit oracle.
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
PHASE1 = ROOT / "phase1"
DEFAULT_BUNDLES = PHASE1 / "native_v1_bundles"
DEFAULT_LAYOUT_ROOT = PHASE1 / "resident_layout_host"
DEFAULT_OUT = PHASE1 / "full_silicon"
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
from run_whisper_resident_decoder_silicon import (  # noqa: E402
    CROSS_K_OFFSET,
    CROSS_V_OFFSET,
    PARAM_MAGIC,
    PARAM_OFFSET,
    RUNTIME_BYTES,
    WEIGHT_BYTES,
    WEIGHT_REGION_OFFSET,
    build_weight_region,
    generate_weight_header as generate_decoder_header,
    parse_dump as parse_decoder_dump,
)


ENCODER_SRC = ROOT / "whisper_resident_encoder_argbuf.c"
DECODER_SRC = ROOT / "whisper_resident_decoder_token_argbuf.c"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/full-silicon"
REMOTE_LAUNCHER = f"{REMOTE_PARENT}/erbium_soc1sim_argbuf_dynmem.remote.board"

ENCODER_MAGIC = 0x5752454E
DECODER_MAGIC = 0x57524443
AUDIO_OFFSET = 0x4000


def latest_dir(root: Path, prefix: str) -> Path:
    dirs = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir())
    if not dirs:
        raise SystemExit(f"no {prefix} directories found under {root}")
    return dirs[-1]


def bundle_dir_for(manifest: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_BUNDLES / manifest["source_bundle"]


def entry_index(manifest: dict[str, Any], part: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        if entry["model_part"] != part:
            continue
        name = entry["name"]
        if name in out:
            raise SystemExit(f"duplicate {part} resident entry: {name}")
        out[name] = entry
    return out


def desc(entries: dict[str, dict[str, Any]], name: str) -> str:
    e = entries.get(name)
    if e is None:
        raise SystemExit(f"missing encoder resident weight: {name}")
    return f"{{ {int(e['offset'])}u, {float(e['scale']):.12g}f }}"


def entry_size(e: dict[str, Any]) -> int:
    for key in ("int8_bytes", "region_bytes", "nbytes", "bytes"):
        if key in e:
            return int(e[key])
    raise SystemExit(f"resident entry has no byte size: {e}")


def align_up(v: int, align: int = 64) -> int:
    return (v + align - 1) & ~(align - 1)


def generate_encoder_header(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    e = entry_index(manifest, "encoder")
    all_entries = manifest["entries"]
    high = max(int(x["offset"]) + entry_size(x) for x in all_entries)
    scratch = align_up(high, 64)
    scratch_need = 1500 * 384 * 4 + 1500 * 1152 * 4
    if scratch + scratch_need > WEIGHT_BYTES:
        raise SystemExit(
            f"encoder scratch does not fit: start={scratch} need={scratch_need} "
            f"limit={WEIGHT_BYTES}"
        )

    layer_names = [
        ("0", "_v_648", "onnx::Add_753", "onnx::Add_758",
         "onnx::MatMul_777", "onnx::Add_776",
         "onnx::MatMul_780", "onnx::Add_779",
         "onnx::MatMul_783", "onnx::Add_782"),
        ("1", "_v_655", "onnx::Add_785", "onnx::Add_790",
         "onnx::MatMul_809", "onnx::Add_808",
         "onnx::MatMul_812", "onnx::Add_811",
         "onnx::MatMul_815", "onnx::Add_814"),
        ("2", "_v_662", "onnx::Add_817", "onnx::Add_822",
         "onnx::MatMul_841", "onnx::Add_840",
         "onnx::MatMul_844", "onnx::Add_843",
         "onnx::MatMul_847", "onnx::Add_846"),
        ("3", "_v_669", "onnx::Add_849", "onnx::Add_854",
         "onnx::MatMul_873", "onnx::Add_872",
         "onnx::MatMul_876", "onnx::Add_875",
         "onnx::MatMul_879", "onnx::Add_878"),
    ]
    fields = [
        "attn_ln_weight", "attn_ln_bias", "self_qkv_weight",
        "self_q_bias", "self_v_bias", "self_out_weight", "self_out_bias",
        "mlp_ln_weight", "mlp_ln_bias", "mlp_fc1_weight", "mlp_fc1_bias",
        "mlp_fc2_weight", "mlp_fc2_bias",
    ]
    key_weights = [f"onnx::MatMul_{880 + i}" for i in range(24)]
    value_weights = [f"onnx::MatMul_{904 + i}" for i in range(24)]
    value_biases = [
        f"cross_attn_value_list.{layer}.split_linears.{head}.bias"
        for layer in range(4)
        for head in range(6)
    ]

    lines = [
        "#ifndef WHISPER_RESIDENT_ENCODER_WEIGHTS_AUTO_H",
        "#define WHISPER_RESIDENT_ENCODER_WEIGHTS_AUTO_H",
        "#include <stdint.h>",
        "struct weight_desc { unsigned int offset; float scale; };",
        "struct encoder_layer_desc {",
    ]
    for f in fields:
        lines.append(f"\tstruct weight_desc {f};")
    lines.extend([
        "};",
        f"#define ENCODER_SCRATCH_OFFSET UINT64_C({scratch})",
        f"static const struct weight_desc enc_conv1_weight = {desc(e, 'onnx::Conv_748')};",
        f"static const struct weight_desc enc_conv1_bias = {desc(e, 'onnx::Conv_749')};",
        f"static const struct weight_desc enc_conv2_weight = {desc(e, 'onnx::Conv_750')};",
        f"static const struct weight_desc enc_conv2_bias = {desc(e, 'onnx::Conv_751')};",
        f"static const struct weight_desc enc_positional_embedding = {desc(e, 'encoder.positional_embedding')};",
        f"static const struct weight_desc enc_attn_scale = {desc(e, '/encoder/blocks.0/attn/Pow_output_0')};",
        f"static const struct weight_desc enc_ln_post_weight = {desc(e, 'encoder.ln_post.weight')};",
        f"static const struct weight_desc enc_ln_post_bias = {desc(e, 'encoder.ln_post.bias')};",
        "static const struct encoder_layer_desc encoder_layers[4] = {",
    ])
    for layer, qkv, qb, vb, out_w, out_b, fc1_w, fc1_b, fc2_w, fc2_b in layer_names:
        vals = [
            desc(e, f"encoder.blocks.{layer}.attn_ln.weight"),
            desc(e, f"encoder.blocks.{layer}.attn_ln.bias"),
            desc(e, qkv),
            desc(e, qb),
            desc(e, vb),
            desc(e, out_w),
            desc(e, out_b),
            desc(e, f"encoder.blocks.{layer}.mlp_ln.weight"),
            desc(e, f"encoder.blocks.{layer}.mlp_ln.bias"),
            desc(e, fc1_w),
            desc(e, fc1_b),
            desc(e, fc2_w),
            desc(e, fc2_b),
        ]
        lines.append(f"\t{{ {', '.join(vals)} }},")
    lines.append("};")
    lines.append("static const struct weight_desc enc_cross_k_weight[4][6] = {")
    for layer in range(4):
        vals = ", ".join(desc(e, key_weights[layer * 6 + h]) for h in range(6))
        lines.append(f"\t{{ {vals} }},")
    lines.append("};")
    lines.append("static const struct weight_desc enc_cross_v_weight[4][6] = {")
    for layer in range(4):
        vals = ", ".join(desc(e, value_weights[layer * 6 + h]) for h in range(6))
        lines.append(f"\t{{ {vals} }},")
    lines.append("};")
    lines.append("static const struct weight_desc enc_cross_v_bias[4][6] = {")
    for layer in range(4):
        vals = ", ".join(desc(e, value_biases[layer * 6 + h]) for h in range(6))
        lines.append(f"\t{{ {vals} }},")
    lines.extend(["};", "#endif", ""])
    path.write_text("\n".join(lines))
    return {
        "scratch_offset": scratch,
        "scratch_bytes": scratch_need,
        "weight_region_high_water": high,
        "weight_region_free_after_scratch": WEIGHT_BYTES - scratch - scratch_need,
    }


def build_raw_int8_models(bundle_dir: Path, resident_manifest: dict[str, Any],
                          run_dir: Path) -> tuple[Path, Path]:
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


def host_reference(bundle_dir: Path, bundle_manifest: dict[str, Any],
                   encoder_model: Path, decoder_model: Path,
                   max_new_tokens: int, run_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
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
        run_dir / "host_raw_int8_fp16_cross_reference",
        decoder,
        WhisperTokenizer.from_pretrained("openai/whisper-tiny.en"),
        k_half_back,
        v_half_back,
        max_new_tokens,
        None,
    )
    return result, features, k_half_back, v_half_back


def build_runtime_image(path: Path, max_new_tokens: int, features: np.ndarray) -> dict[str, Any]:
    image = bytearray(RUNTIME_BYTES)
    params = struct.pack("<16I", PARAM_MAGIC, int(max_new_tokens), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    image[PARAM_OFFSET:PARAM_OFFSET + len(params)] = params
    feature_bytes = np.ascontiguousarray(features.astype("<f4")).tobytes()
    image[AUDIO_OFFSET:AUDIO_OFFSET + len(feature_bytes)] = feature_bytes
    path.write_bytes(image)
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "feature_sha256": sha256_array(features),
        "feature_bytes": len(feature_bytes),
        "audio_offset": AUDIO_OFFSET,
    }


def build_elf(src: Path, out: Path, include_dir: Path, active_harts: int | None = None) -> Path:
    cmd = [
        GCC,
        "-O3",
        "-funroll-loops",
        "-fno-builtin",
        "-fno-tree-loop-distribute-patterns",
        *BASE_FLAGS,
        f"-I{include_dir}",
    ]
    if active_harts is not None:
        cmd.append(f"-DACTIVE_HARTS={active_harts}")
    cmd.extend(["-o", str(out), str(src), str(CRT), str(LAYOUT)])
    run(cmd)
    return out


def parse_wait(log_text: str) -> float | None:
    for line in log_text.splitlines():
        if line.startswith("Kernel wait seconds:"):
            return float(line.split(":", 1)[1].strip())
    return None


def parse_encoder_dump(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    words = struct.unpack_from("<32I", data, 0x1000)
    return {
        "magic": hex(words[0]),
        "hart_id": words[1],
        "active_harts": words[2],
        "src_len": words[3],
        "scratch_offset": words[4] | (words[5] << 32),
        "hpm3": words[6] | (words[7] << 32),
        "hpm4": words[8] | (words[9] << 32),
        "hpm5": words[10] | (words[11] << 32),
        "hpm6": words[12] | (words[13] << 32),
        "hpm7": words[14] | (words[15] << 32),
        "hpm8": words[16] | (words[17] << 32),
        "ops": words[18] | (words[19] << 32),
        "done": bool(words[20]),
    }


def extract_cross(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    k_count = 4 * 6 * 64 * 1500
    v_count = 4 * 6 * 1500 * 64
    k = np.frombuffer(data, dtype="<f2", count=k_count, offset=CROSS_K_OFFSET).astype(np.float32)
    v = np.frombuffer(data, dtype="<f2", count=v_count, offset=CROSS_V_OFFSET).astype(np.float32)
    return k.reshape(4, 6, 64, 1500), v.reshape(4, 6, 1500, 64)


def run_remote_pair(run_dir: Path, remote: str, encoder_timeout: int,
                    decoder_timeout: int) -> tuple[float | None, float | None]:
    run([*SSH_CMD, f"test -x {shlex.quote(REMOTE_LAUNCHER)}"], timeout=20)
    run([*SSH_CMD, f"mkdir -p {shlex.quote(remote)}"], timeout=20)
    run([
        "rsync", "-e", RSYNC_RSH, "-aq", "--timeout=300",
        str(run_dir / "whisper_resident_encoder.elf"),
        str(run_dir / "whisper_resident_decoder_token.elf"),
        str(run_dir / "runtime_initial_16m.bin"),
        str(run_dir / "weights_region_64m.bin"),
        f"{REMOTE_HOST}:{remote}/",
    ], timeout=1200)
    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote}
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
{shlex.quote(REMOTE_LAUNCHER)} --elf-load ./whisper_resident_encoder.elf \\
  --shire 0 --mem_size {RUNTIME_BYTES + WEIGHT_BYTES} --dump_size {RUNTIME_BYTES} \\
  --file_load 0x0,runtime_initial_16m.bin \\
  --file_load 0x{RUNTIME_BYTES:x},weights_region_64m.bin \\
  --dump_after encoder_dump.bin --timeout {encoder_timeout} > encoder.log 2>&1
cat encoder.log
{shlex.quote(REMOTE_LAUNCHER)} --elf-load ./whisper_resident_decoder_token.elf \\
  --shire 0 --mem_size {RUNTIME_BYTES + WEIGHT_BYTES} --dump_size {RUNTIME_BYTES} \\
  --file_load 0x0,encoder_dump.bin \\
  --file_load 0x{RUNTIME_BYTES:x},weights_region_64m.bin \\
  --dump_after decoder_dump.bin --timeout {decoder_timeout} > decoder.log 2>&1
cat decoder.log
"""
    run([*SSH_CMD, "bash", "-s"], input=script, timeout=encoder_timeout + decoder_timeout + 600)
    for name in ("encoder_dump.bin", "decoder_dump.bin", "encoder.log", "decoder.log"):
        with (run_dir / name).open("wb") as out:
            run([*SSH_CMD, "cat", f"{remote}/{name}"], stdout=out, timeout=300)
    enc_wait = parse_wait((run_dir / "encoder.log").read_text(errors="ignore"))
    dec_wait = parse_wait((run_dir / "decoder.log").read_text(errors="ignore"))
    return enc_wait, dec_wait


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-manifest", type=Path, default=None)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=23)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--encoder-timeout", type=int, default=7200)
    ap.add_argument("--decoder-timeout", type=int, default=1200)
    args = ap.parse_args()

    resident_manifest_path = args.resident_manifest
    if resident_manifest_path is None:
        resident_manifest_path = latest_dir(DEFAULT_LAYOUT_ROOT, "run_") / "resident_weight_manifest.json"
    resident_manifest = load_resident_manifest(resident_manifest_path)
    bundle_dir = bundle_dir_for(resident_manifest, args.bundle_dir)
    bundle_manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}_new{args.max_new_tokens}_ah{args.active_harts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder_header_meta = generate_encoder_header(run_dir / "whisper_resident_encoder_weights_auto.h", resident_manifest)
    generate_decoder_header(run_dir / "whisper_resident_decoder_weights_auto.h", resident_manifest)
    encoder_model, decoder_model = build_raw_int8_models(bundle_dir, resident_manifest, run_dir)
    ref, features, k_ref, v_ref = host_reference(
        bundle_dir, bundle_manifest, encoder_model, decoder_model,
        args.max_new_tokens, run_dir,
    )
    runtime_meta = build_runtime_image(run_dir / "runtime_initial_16m.bin", args.max_new_tokens, features)
    weights_sha = build_weight_region(bundle_dir, resident_manifest, run_dir / "weights_region_64m.bin")
    build_elf(ENCODER_SRC, run_dir / "whisper_resident_encoder.elf", run_dir, args.active_harts)
    build_elf(DECODER_SRC, run_dir / "whisper_resident_decoder_token.elf", run_dir, None)

    remote = f"{REMOTE_ROOT}/{run_dir.name}"
    enc_wait, dec_wait = run_remote_pair(run_dir, remote, args.encoder_timeout, args.decoder_timeout)

    enc_summary = parse_encoder_dump(run_dir / "encoder_dump.bin")
    dec_parsed = parse_decoder_dump(run_dir / "decoder_dump.bin")
    k_sil, v_sil = extract_cross(run_dir / "encoder_dump.bin")
    k_diff = np.abs(k_sil - k_ref)
    v_diff = np.abs(v_sil - v_ref)
    ref_tokens = [int(x) for x in ref["tokens"][:len(dec_parsed["tokens"])]]
    token_match = dec_parsed["tokens"] == ref_tokens
    total_wait = (enc_wait or 0.0) + (dec_wait or 0.0) if enc_wait is not None and dec_wait is not None else None
    generated = int(dec_parsed["summary"]["generated_tokens"])
    report = {
        "schema": "whisper-full-silicon-report",
        "version": 1,
        "timestamp": stamp,
        "pass": bool(
            enc_summary["magic"] == hex(ENCODER_MAGIC)
            and enc_summary["done"]
            and dec_parsed["summary"]["magic"] == hex(DECODER_MAGIC)
            and dec_parsed["summary"]["done"]
            and token_match
        ),
        "scope": "log-mel features -> encoder cross-cache -> decoder token IDs on ET-SoC1; host ONNX is audit only",
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
            "encoder_kernel_wait_s": enc_wait,
            "decoder_kernel_wait_s": dec_wait,
            "total_kernel_wait_s": total_wait,
            "generated_tokens_per_s_total_kernel": (generated / total_wait) if total_wait else None,
            "decoder_generated_tokens_per_s": (generated / dec_wait) if dec_wait else None,
            "tokens": dec_parsed["tokens"],
            "token_match": token_match,
            "encoder_summary": enc_summary,
            "decoder_summary": dec_parsed["summary"],
            "cross_cache_audit": {
                "k_max_abs_vs_host_fp16": float(k_diff.max()),
                "k_mean_abs_vs_host_fp16": float(k_diff.mean()),
                "v_max_abs_vs_host_fp16": float(v_diff.max()),
                "v_mean_abs_vs_host_fp16": float(v_diff.mean()),
            },
        },
        "artifacts": {
            "runtime_initial": runtime_meta,
            "weights_region": {
                "path": str(run_dir / "weights_region_64m.bin"),
                "sha256": weights_sha,
                "bytes": (run_dir / "weights_region_64m.bin").stat().st_size,
            },
            "encoder_header_meta": encoder_header_meta,
            "k_silicon_sha256": sha256_array(k_sil.astype("<f4")),
            "v_silicon_sha256": sha256_array(v_sil.astype("<f4")),
            "k_ref_sha256": sha256_array(k_ref.astype("<f4")),
            "v_ref_sha256": sha256_array(v_ref.astype("<f4")),
        },
    }
    (run_dir / "full_silicon_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Full Whisper Silicon Report",
        "",
        f"- Result: `{'PASS' if report['pass'] else 'FAIL'}`",
        "- Scope: log-mel features -> encoder cross-cache -> decoder token IDs on ET-SoC1; host ONNXRuntime is audit only",
        f"- Host text: `{ref['text']}`",
        f"- Host tokens: `{ref_tokens}`",
        f"- Silicon tokens: `{dec_parsed['tokens']}`",
        f"- Encoder wait seconds: `{enc_wait}`",
        f"- Decoder wait seconds: `{dec_wait}`",
        f"- Total kernel wait seconds: `{total_wait}`",
        f"- Generated token/s total kernel: `{report['silicon']['generated_tokens_per_s_total_kernel']}`",
        f"- Decoder-only generated token/s: `{report['silicon']['decoder_generated_tokens_per_s']}`",
        f"- Cross K max abs vs host FP16 oracle: `{report['silicon']['cross_cache_audit']['k_max_abs_vs_host_fp16']}`",
        f"- Cross V max abs vs host FP16 oracle: `{report['silicon']['cross_cache_audit']['v_max_abs_vs_host_fp16']}`",
        f"- Remote work: `{remote}`",
    ]
    (run_dir / "FULL_SILICON_REPORT.md").write_text("\n".join(md) + "\n")
    print(json.dumps({
        "pass": report["pass"],
        "report": str(run_dir / "full_silicon_report.json"),
        "markdown": str(run_dir / "FULL_SILICON_REPORT.md"),
        "host_tokens": ref_tokens,
        "silicon_tokens": dec_parsed["tokens"],
        "encoder_wait_s": enc_wait,
        "decoder_wait_s": dec_wait,
        "total_wait_s": total_wait,
        "tokens_per_s": report["silicon"]["generated_tokens_per_s_total_kernel"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
