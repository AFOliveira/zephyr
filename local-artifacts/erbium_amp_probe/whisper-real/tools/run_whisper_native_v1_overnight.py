#!/usr/bin/env python3
"""Continuous Whisper native-v1 validation/optimization runner.

This intentionally starts from the current audited silicon path and keeps a
strict ledger while native graph pieces are swapped in.  It never attempts board
resets or driver reloads; infrastructure failures stop the loop after a small
number of consecutive misses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE1 = ROOT / "phase1"
AUDIT_RUNNER = HERE / "run_whisper_e2e_audit.py"
BUNDLE_BUILDER = HERE / "build_whisper_native_v1_bundle.py"
SPLIT_SMOKE = HERE / "run_whisper_native_split_smoke.py"
DEFAULT_AUDIO = ROOT / "audio" / "openai_whisper_jfk_16k.wav"

REMOTE_HOST = "root@esperanto-soc6"
REMOTE_BASE = "/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2"
REMOTE_LOCK = f"{REMOTE_BASE}/erbium-amp-probe/whisper-real/whisper-native-v1.lock"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=12 -o ConnectTimeout=10",
))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]


VARIANTS = [
    {
        "name": "tail_parallel_tile8192",
        "args": [
            "--silicon-tail-layernorm",
            "--tail-parallel-shires", "0,1,2,3,4,5,6",
            "--tile-cols", "8192",
            "--active-harts", "16",
        ],
    },
    {
        "name": "tail_parallel_tile10240",
        "args": [
            "--silicon-tail-layernorm",
            "--tail-parallel-shires", "0,1,2,3,4,5,6",
            "--tile-cols", "10240",
            "--active-harts", "16",
        ],
    },
    {
        "name": "tail_sequential_tile8192",
        "args": [
            "--silicon-tail-layernorm",
            "--tile-cols", "8192",
            "--active-harts", "16",
        ],
    },
]


def run(cmd: list[str], log_path: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, text=True, timeout=timeout)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return proc


def ssh(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH_CMD, "bash", "-s"], input=script, text=True, timeout=timeout)


def acquire_remote_lock(run_root: Path) -> bool:
    meta = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "run_root": str(run_root),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    script = f"""set -e
LOCK={shlex.quote(REMOTE_LOCK)}
if mkdir "$LOCK" 2>/dev/null; then
  cat > "$LOCK/owner.json" <<'JSON'
{json.dumps(meta, indent=2)}
JSON
  exit 0
fi
echo "remote lock already exists: $LOCK" >&2
test -f "$LOCK/owner.json" && cat "$LOCK/owner.json" >&2 || true
exit 17
"""
    proc = ssh(script)
    if proc.returncode == 0:
        return True
    print(proc.stderr or proc.stdout or "failed to acquire lock", file=sys.stderr)
    return False


def release_remote_lock() -> None:
    script = f"""set +e
LOCK={shlex.quote(REMOTE_LOCK)}
rm -f "$LOCK/owner.json"
rmdir "$LOCK" 2>/dev/null
"""
    subprocess.run([*SSH_CMD, "bash", "-s"], input=script, text=True, timeout=30)


def latest_report(out_dir: Path, before: set[Path]) -> Path | None:
    after = set(out_dir.glob("both_*/e2e_audit_report.json"))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return new[-1] if new else None


def summarize_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    cmp = report.get("comparison", {})
    host = report.get("host_only", {})
    hybrid = report.get("hybrid_silicon") or {}
    return {
        "report": str(report_path),
        "pass": bool(
            cmp.get("token_sequence_match")
            and cmp.get("text_match")
            and cmp.get("all_silicon_argmax_match")
            and cmp.get("all_silicon_audit_pass")
        ),
        "token_sequence_match": cmp.get("token_sequence_match"),
        "text_match": cmp.get("text_match"),
        "silicon_argmax_match": cmp.get("all_silicon_argmax_match"),
        "silicon_audit_pass": cmp.get("all_silicon_audit_pass"),
        "generated_nonprompt_tokens": hybrid.get("generated_nonprompt_tokens"),
        "host_tokens_per_s_wall": host.get("tokens_per_s_wall"),
        "hybrid_tokens_per_s_wall": hybrid.get("tokens_per_s_wall"),
        "silicon_tokens_per_s": hybrid.get("tokens_per_s_silicon_logits_only"),
        "silicon_wait_s": hybrid.get("silicon_logits_wait_s"),
        "host_text": host.get("text"),
        "hybrid_text": hybrid.get("text"),
    }


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    cols = [
        "timestamp",
        "iteration",
        "variant",
        "kind",
        "rc",
        "pass",
        "token_sequence_match",
        "text_match",
        "silicon_argmax_match",
        "silicon_audit_pass",
        "generated_nonprompt_tokens",
        "host_tokens_per_s_wall",
        "hybrid_tokens_per_s_wall",
        "silicon_tokens_per_s",
        "silicon_wait_s",
        "report",
        "log",
    ]
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_bundle_once(run_root: Path, max_new_tokens: int, audio: Path) -> int:
    log = run_root / "bundle_builder.log"
    cmd = [
        sys.executable,
        str(BUNDLE_BUILDER),
        "--audio", str(audio),
        "--max-new-tokens", str(max_new_tokens),
        "--out-dir", str(run_root / "bundles"),
    ]
    return run(cmd, log_path=log, timeout=1800).returncode


def run_split_smoke(run_root: Path, iteration: int) -> dict[str, Any]:
    log = run_root / f"split_smoke_i{iteration:04d}.log"
    cmd = [
        sys.executable,
        str(SPLIT_SMOKE),
        "--timeout", "60",
    ]
    proc = run(cmd, log_path=log, timeout=300)
    return {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "iteration": iteration,
        "variant": "native_split_smoke",
        "kind": "smoke",
        "rc": proc.returncode,
        "pass": proc.returncode == 0,
        "log": str(log),
    }


def run_variant(run_root: Path, iteration: int, variant: dict[str, Any], max_new_tokens: int,
                timeout: int, audio: Path) -> dict[str, Any]:
    out_dir = run_root / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("both_*/e2e_audit_report.json"))
    log = run_root / f"{variant['name']}_i{iteration:04d}.log"
    cmd = [
        sys.executable,
        str(AUDIT_RUNNER),
        "--mode", "both",
        "--audio", str(audio),
        "--max-new-tokens", str(max_new_tokens),
        "--timeout", str(timeout),
        "--out-dir", str(out_dir),
        *variant["args"],
    ]
    proc = run(cmd, log_path=log, timeout=max(timeout * max_new_tokens * 8, 900))
    row = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "iteration": iteration,
        "variant": variant["name"],
        "kind": "e2e",
        "rc": proc.returncode,
        "log": str(log),
    }
    report = latest_report(out_dir, before)
    if report is not None:
        row.update(summarize_report(report))
    else:
        row["pass"] = False
    return row


def start_tmux(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--duration-hours", str(args.duration_hours),
        "--max-new-tokens", str(args.max_new_tokens),
        "--timeout", str(args.timeout),
        "--out-dir", str(args.out_dir),
        "--session", args.session,
        "--audio", str(args.audio),
    ]
    if args.once:
        cmd.append("--once")
    if args.skip_bundle:
        cmd.append("--skip-bundle")
    if args.smoke_every:
        cmd.extend(["--smoke-every", str(args.smoke_every)])
    if shutil.which("tmux"):
        tmux_cmd = ["tmux", "new-session", "-d", "-s", args.session, " ".join(shlex.quote(c) for c in cmd)]
        proc = run(tmux_cmd)
        if proc.returncode == 0:
            print(f"started tmux session: {args.session}")
            print(f"attach with: tmux attach -t {args.session}")
        return proc.returncode

    log_dir = Path(args.out_dir) / "background_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.session}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            ["nohup", *cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    pid_path = log_dir / f"{args.session}.pid"
    pid_path.write_text(f"{proc.pid}\n")
    print("tmux not found; started background process with nohup")
    print(f"pid: {proc.pid}")
    print(f"log: {log_path}")
    print(f"pid file: {pid_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-hours", type=float, default=10.0)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out-dir", type=Path, default=PHASE1 / "overnight_runs")
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--skip-bundle", action="store_true")
    ap.add_argument("--smoke-every", type=int, default=6)
    ap.add_argument("--start-tmux", action="store_true")
    ap.add_argument("--session", default="whisper-native-overnight")
    args = ap.parse_args()

    if args.start_tmux:
        return start_tmux(args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = args.out_dir / f"overnight_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    ledger = run_root / "ledger.tsv"
    jsonl = run_root / "ledger.jsonl"
    (run_root / "runner_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    if not acquire_remote_lock(run_root):
        return 17

    consecutive_infra_failures = 0
    iteration = 0
    deadline = time.monotonic() + args.duration_hours * 3600.0
    try:
        if not args.skip_bundle:
            rc = run_bundle_once(run_root, args.max_new_tokens, args.audio)
            row = {
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "iteration": iteration,
                "variant": "native_v1_bundle",
                "kind": "bundle",
                "rc": rc,
                "pass": rc == 0,
                "log": str(run_root / "bundle_builder.log"),
            }
            append_ledger(ledger, row)
            with jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if rc != 0:
                return rc

        while time.monotonic() < deadline:
            variant = VARIANTS[iteration % len(VARIANTS)]
            if args.smoke_every and iteration % args.smoke_every == 0:
                smoke_row = run_split_smoke(run_root, iteration)
                append_ledger(ledger, smoke_row)
                with jsonl.open("a") as f:
                    f.write(json.dumps(smoke_row) + "\n")

            row = run_variant(run_root, iteration, variant, args.max_new_tokens, args.timeout, args.audio)
            append_ledger(ledger, row)
            with jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")

            if row.get("rc") != 0 and not row.get("report"):
                consecutive_infra_failures += 1
            else:
                consecutive_infra_failures = 0

            if consecutive_infra_failures >= 3:
                print("stopping after 3 consecutive infrastructure failures", flush=True)
                return 2

            iteration += 1
            if args.once:
                break
    finally:
        release_remote_lock()

    print(f"ledger: {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
