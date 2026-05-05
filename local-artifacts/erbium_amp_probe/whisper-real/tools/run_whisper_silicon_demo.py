#!/usr/bin/env python3
"""Run the current audited Whisper audio-to-text ET-SoC1 demo."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE0 = ROOT / "phase0"
AUDIT_RUNNER = HERE / "run_whisper_e2e_audit.py"
DEFAULT_AUDIO = ROOT / "audio" / "openai_whisper_jfk_16k.wav"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "demo_runs")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    before = set(args.out_dir.glob("both_*/e2e_audit_report.json"))
    cmd = [
        sys.executable,
        str(AUDIT_RUNNER),
        "--mode", "both",
        "--audio", str(args.audio),
        "--max-new-tokens", str(args.max_new_tokens),
        "--active-harts", "16",
        "--tile-cols", "8192",
        "--silicon-tail-layernorm",
        "--tail-parallel-shires", "0,1,2,3,4,5,6",
        "--timeout", str(args.timeout),
        "--out-dir", str(args.out_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    log_lines: list[str] = []
    for line in proc.stdout:
        log_lines.append(line)
        text = line.rstrip()
        if text.startswith("=== step ") or text.startswith("Parallel tail launcher wall seconds:"):
            print(text, flush=True)
    rc = proc.wait()

    after = set(args.out_dir.glob("both_*/e2e_audit_report.json"))
    new_reports = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_reports:
        (args.out_dir / "whisper_silicon_demo_last.log").write_text("".join(log_lines))
        if rc != 0:
            print("audit runner failed; last output:", file=sys.stderr)
            print("".join(log_lines[-120:]), file=sys.stderr)
            return rc
        raise SystemExit("audit runner completed but no new report was found")

    report_path = new_reports[-1]
    (report_path.parent / "demo_console.log").write_text("".join(log_lines))
    if rc != 0:
        print(f"audit runner failed; log: {report_path.parent / 'demo_console.log'}", file=sys.stderr)
        print("".join(log_lines[-120:]), file=sys.stderr)
        return rc

    report = json.loads(report_path.read_text())
    cmp = report["comparison"]
    hybrid = report["hybrid_silicon"]
    host = report["host_only"]
    passed = (
        cmp["token_sequence_match"]
        and cmp["text_match"]
        and cmp["all_silicon_argmax_match"]
        and cmp["all_silicon_audit_pass"]
    )

    print()
    print("Whisper ET-SoC1 demo")
    print("====================")
    print(f"Transcript: {hybrid['text']}")
    print(f"Audit pass: {passed}")
    print(f"Generated tokens: {hybrid['generated_nonprompt_tokens']}")
    print(f"Host-only wall rate: {host['tokens_per_s_wall']:.6f} token/s")
    print(f"Hybrid wall rate: {hybrid['tokens_per_s_wall']:.6f} token/s")
    print(
        "Silicon tail rate: "
        f"{hybrid['tokens_per_s_silicon_logits_only']:.6f} token/s"
    )
    print(f"Silicon tail wait: {hybrid['silicon_logits_wait_s']:.6f} s")
    print(f"Report: {report_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
