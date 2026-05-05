#!/usr/bin/env python3
"""Audit real Whisper decoder LayerNorm nodes on ET-SoC1."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from transformers import WhisperTokenizer

from run_whisper_e2e_audit import (
    DEFAULT_DECODER,
    DEFAULT_ENCODER,
    PHASE0,
    SiliconLogitsRunner,
    decoder_prompt,
    make_session,
    prepare_features,
    sha256_array,
    sha256_path,
)

DEFAULT_AUDIO = Path(__file__).resolve().parent.parent / "audio" / "openai_whisper_jfk_16k.wav"


def safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("/"))
    return out or "root"


def layernorm_nodes(decoder_path: Path) -> list[onnx.NodeProto]:
    model = onnx.load(decoder_path)
    return [node for node in model.graph.node if node.op_type == "LayerNormalization"]


def run_decoder_to_step(
    decoder,
    tokenizer: WhisperTokenizer,
    k_cross: np.ndarray,
    v_cross: np.ndarray,
    target_step: int,
    output_names: list[str],
) -> tuple[dict[str, np.ndarray], list[int]]:
    prompt = decoder_prompt(tokenizer)
    generated = [prompt[0]]
    forced_after_sot = prompt[1:]
    k_self = np.zeros((4, 6, 64, 224), dtype=np.float32)
    v_self = np.zeros((4, 6, 224, 64), dtype=np.float32)

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
        names = [*output_names, "logits", "k_cache", "v_cache"]
        values = decoder.run(names, inputs)
        outputs = {name: np.asarray(value, dtype=np.float32) for name, value in zip(names, values)}

        logits = outputs["logits"].reshape(-1)
        if step < len(forced_after_sot):
            next_id = forced_after_sot[step]
        else:
            next_id = int(logits.argmax())
        generated.append(next_id)
        k_self = outputs["k_cache"]
        v_self = outputs["v_cache"]

    return outputs, generated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--step", type=int, default=3,
                    help="Decoder step whose real tensors are audited")
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0,
                    help="Audit only the first N LayerNorm nodes")
    ap.add_argument("--out-dir", type=Path,
                    default=PHASE0 / "decoder_layernorm_audit_runs")
    args = ap.parse_args()

    if args.step < 0:
        raise SystemExit("--step must be non-negative")
    nodes = layernorm_nodes(args.decoder)
    if args.limit:
        nodes = nodes[:args.limit]
    if not nodes:
        raise SystemExit("no LayerNormalization nodes selected")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(args.decoder).graph.initializer}
    output_names = []
    for node in nodes:
        output_names.extend([node.input[0], node.output[0]])
    decoder = make_session(args.decoder, output_names)
    encoder = make_session(args.encoder)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    features, input_meta = prepare_features(args.audio, run_dir)
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    outputs, tokens = run_decoder_to_step(
        decoder, tokenizer, k_cross, v_cross, args.step, output_names
    )

    dummy_weight = np.zeros((384, 2), dtype=np.float32)
    dummy_logits_ref = np.zeros((2,), dtype=np.float32)
    results = []
    for index, node in enumerate(nodes):
        node_safe = f"{stamp}_{index:02d}_{safe_name(node.name)}"
        node_dir = args.out_dir / node_safe
        ln_in = outputs[node.input[0]].reshape(-1).astype(np.float32)
        ln_ref = outputs[node.output[0]].reshape(-1).astype(np.float32)
        ln_weight = np.asarray(init[node.input[1]], dtype=np.float32).reshape(-1)
        ln_bias = np.asarray(init[node.input[2]], dtype=np.float32).reshape(-1)
        if ln_in.shape != (384,) or ln_ref.shape != (384,):
            raise SystemExit(f"{node.name} has unexpected shape {ln_in.shape} -> {ln_ref.shape}")

        runner = SiliconLogitsRunner(
            node_dir,
            dummy_weight,
            args.active_harts,
            args.timeout,
            tile_cols=2,
            device_argmax_only=True,
            tail_layernorm=True,
            ln_weight=ln_weight,
            ln_bias=ln_bias,
        )
        _, silicon_report = runner.run_logits(
            args.step,
            ln_in,
            dummy_logits_ref,
            ln_ref=ln_ref,
        )
        tile = silicon_report["tiles"][0]
        result = {
            "index": index,
            "node": node.name,
            "input": node.input[0],
            "output": node.output[0],
            "weight": node.input[1],
            "bias": node.input[2],
            "wait_s": silicon_report["total_wait_s"],
            "ln_max_abs": tile["ln_max_abs"],
            "ln_mean_abs": tile["ln_mean_abs"],
            "slot_done_count": tile["slot_done_count"],
            "active_mask": tile["active_mask"],
            "audit_pass": silicon_report["audit_pass"],
            "node_dir": str(node_dir),
        }
        results.append(result)
        print(
            f"{index:02d} {node.name}: pass={result['audit_pass']} "
            f"max_abs={result['ln_max_abs']:.9g} wait={result['wait_s']:.6f}s",
            flush=True,
        )

    all_pass = all(r["audit_pass"] for r in results)
    report = {
        "timestamp": stamp,
        "scope": "Real Whisper decoder LayerNorm node audit on ET-SoC1",
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
        "selected_nodes": len(nodes),
        "all_pass": all_pass,
        "max_ln_abs": max(r["ln_max_abs"] for r in results),
        "mean_wait_s": sum(r["wait_s"] for r in results) / len(results),
        "feature_sha256": sha256_array(features),
        "k_cache_cross_sha256": sha256_array(k_cross),
        "v_cache_cross_sha256": sha256_array(v_cross),
        "results": results,
    }
    (run_dir / "decoder_layernorm_audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = [
        "# Whisper Decoder LayerNorm Audit",
        "",
        f"- Decoder step: `{args.step}`",
        f"- Selected LayerNorm nodes: `{len(nodes)}`",
        f"- All pass: `{all_pass}`",
        f"- Max LayerNorm abs diff: `{report['max_ln_abs']:.9g}`",
        f"- Mean silicon wait: `{report['mean_wait_s']:.6f} s`",
        "",
        "| index | node | max abs | wait s | pass |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['index']} | `{r['node']}` | {r['ln_max_abs']:.9g} | "
            f"{r['wait_s']:.6f} | {r['audit_pass']} |"
        )
    (run_dir / "DECODER_LAYERNORM_AUDIT.md").write_text("\n".join(lines) + "\n")

    tsv = PHASE0 / "whisper_decoder_layernorm_audit.tsv"
    new = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "step", "index", "node", "wait_s", "ln_max_abs",
            "ln_mean_abs", "slot_done_count", "active_mask", "audit_pass",
            "report",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": stamp,
                "step": args.step,
                "index": r["index"],
                "node": r["node"],
                "wait_s": r["wait_s"],
                "ln_max_abs": r["ln_max_abs"],
                "ln_mean_abs": r["ln_mean_abs"],
                "slot_done_count": r["slot_done_count"],
                "active_mask": r["active_mask"],
                "audit_pass": r["audit_pass"],
                "report": str(run_dir / "decoder_layernorm_audit_report.json"),
            })

    print()
    print(f"report: {run_dir / 'decoder_layernorm_audit_report.json'}")
    print(f"all_pass: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
