#!/usr/bin/env python3
"""Validate the board-compatible dynamic-memory Erbium launcher."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
PHASE1 = ROOT / "phase1"

SRC = ROOT / "whisper_dynmem_probe.c"
CRT = AMP_ROOT / "hart-report" / "hart_report_crt.S"
GCC = os.environ.get("GCC", "/home/afonso/et/bin/riscv64-unknown-elf-gcc")
LAYOUT = Path(os.environ.get(
    "LAYOUT", "/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c"))
LD_SCRIPT = Path(os.environ.get(
    "LD_SCRIPT", "/home/afonso/et-platform-vidas/et-common-libs/share/erbium-soc1sim/erbium.ld"))
SOC1SIM_STAGED_INCLUDE = Path(os.environ.get(
    "SOC1SIM_STAGED_INCLUDE",
    "/home/afonso/zephyr/build-etsoc1-halified-emlearn/aifoundry/erbium-soc1sim-staged-include"))

REMOTE_HOST = "root@esperanto-soc6"
REMOTE_BASE = "/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2"
REMOTE_PARENT = f"{REMOTE_BASE}/erbium-amp-probe"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/dynmem-probe"
REMOTE_LAUNCHER = f"{REMOTE_PARENT}/erbium_soc1sim_argbuf_dynmem.remote.board"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=6 -o ConnectTimeout=10"))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]
RSYNC_RSH = "ssh " + shlex.join(SSH_OPTS)

MAGIC = 0x5744594E
BASE_FLAGS = [
    "-march=rv64imfc",
    "-mabi=lp64f",
    "-mcmodel=medany",
    "-nostdlib",
    "-fno-zero-initialized-in-bss",
    "-ffunction-sections",
    "-fdata-sections",
    f"-I{SOC1SIM_STAGED_INCLUDE}",
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-I/home/afonso/et-platform-vidas/hal/platform/erbium/include",
    "-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include",
    "-Wl,--gc-sections",
    "-Wl,--no-warn-rwx-segments",
    "-Wl,--emit-relocs",
    "-T", str(LD_SCRIPT),
]


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def make_blob(size: int) -> bytes:
    return bytes(((i * 37 + 13) & 0xff) for i in range(size))


def host_checksum(blob: bytes) -> tuple[int, int, int, int]:
    total_sum = 0
    total_xor = 0
    active_harts = 16
    for h in range(active_harts):
        i0 = (len(blob) * h) // active_harts
        i1 = (len(blob) * (h + 1)) // active_harts
        part_xor = 0
        part_sum = 0
        for i in range(i0, i1):
            v = blob[i]
            part_sum += v
            part_xor ^= v << ((i & 3) * 8)
        total_sum = (total_sum + part_sum) & 0xffffffff
        total_xor ^= part_xor
    first = struct.unpack_from("<I", blob, 0)[0]
    last = struct.unpack_from("<I", blob, len(blob) - 4)[0]
    return total_sum, total_xor & 0xffffffff, first, last


def build_elf(run_dir: Path, high_offset: int, byte_count: int, active_harts: int) -> Path:
    elf = run_dir / "whisper_dynmem_probe.elf"
    cmd = [
        GCC,
        "-O3",
        "-funroll-loops",
        *BASE_FLAGS,
        f"-DNUM_HARTS={active_harts}",
        f"-DACTIVE_HARTS={active_harts}u",
        f"-DHIGH_OFFSET={high_offset}u",
        f"-DBYTE_COUNT={byte_count}u",
        "-o", str(elf),
        str(SRC),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def parse_dump(dump: bytes) -> dict[str, Any]:
    summary_words = struct.unpack_from("<32I", dump, 0x1000)
    slots = []
    for h in range(16):
        words = struct.unpack_from("<16I", dump, h * 64)
        slots.append({
            "hart": h,
            "magic": hex(words[0]),
            "hart_id": words[1],
            "minion_id": words[2],
            "thread_id": words[3],
            "i0": words[4],
            "i1": words[5],
            "sum": words[6],
            "xorv": words[7],
            "done": words[8],
        })
    return {
        "summary": {
            "magic": hex(summary_words[0]),
            "active_harts": summary_words[1],
            "high_offset": summary_words[2],
            "byte_count": summary_words[3],
            "done_count": summary_words[4],
            "active_mask": hex(summary_words[5]),
            "sum": summary_words[6],
            "xorv": summary_words[7],
            "first_word": summary_words[8],
            "last_word": summary_words[9],
        },
        "slots": slots,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PHASE1 / "dynmem_probe_runs")
    ap.add_argument("--high-offset", type=int, default=20 * 1024 * 1024)
    ap.add_argument("--mem-size", type=int, default=32 * 1024 * 1024)
    ap.add_argument("--byte-count", type=int, default=4096)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    if args.high_offset + args.byte_count > args.mem_size:
        raise SystemExit("--high-offset + --byte-count must fit in --mem-size")
    if args.high_offset < 16 * 1024 * 1024:
        raise SystemExit("probe high offset must be above the historical 16 MiB argbuf")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}_off{args.high_offset}_mem{args.mem_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    blob = make_blob(args.byte_count)
    blob_path = run_dir / "high_blob.bin"
    blob_path.write_bytes(blob)
    expected_sum, expected_xor, expected_first, expected_last = host_checksum(blob)
    elf = build_elf(run_dir, args.high_offset, args.byte_count, args.active_harts)

    remote = f"{REMOTE_ROOT}/{run_dir.name}"
    run([*SSH_CMD, f"test -x {shlex.quote(REMOTE_LAUNCHER)}"], timeout=20)
    run([*SSH_CMD, f"mkdir -p {shlex.quote(remote)}"], timeout=20)
    run(["rsync", "-e", RSYNC_RSH, "-aq", "--timeout=60",
         str(elf), str(blob_path), f"{REMOTE_HOST}:{remote}/"], timeout=120)

    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote}
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
{shlex.quote(REMOTE_LAUNCHER)} --elf-load ./whisper_dynmem_probe.elf \\
  --shire 0 --mem_size {args.mem_size} --dump_size 8192 \\
  --file_load 0x{args.high_offset:x},high_blob.bin \\
  --dump_after dump.bin --timeout {args.timeout} > run.log 2>&1
cat run.log
"""
    run([*SSH_CMD, "bash", "-s"], input=script, timeout=args.timeout + 120)
    run(["rsync", "-e", RSYNC_RSH, "-aq", "--timeout=60",
         f"{REMOTE_HOST}:{remote}/dump.bin",
         f"{REMOTE_HOST}:{remote}/run.log",
         str(run_dir) + "/"], timeout=120)

    parsed = parse_dump((run_dir / "dump.bin").read_bytes())
    summary = parsed["summary"]
    passed = (
        summary["magic"] == hex(MAGIC)
        and summary["active_harts"] == args.active_harts
        and summary["high_offset"] == args.high_offset
        and summary["byte_count"] == args.byte_count
        and summary["done_count"] == args.active_harts
        and summary["active_mask"] == hex((1 << args.active_harts) - 1)
        and summary["sum"] == expected_sum
        and summary["xorv"] == expected_xor
        and summary["first_word"] == expected_first
        and summary["last_word"] == expected_last
    )
    report = {
        "timestamp": stamp,
        "remote_work": remote,
        "launcher": REMOTE_LAUNCHER,
        "mem_size": args.mem_size,
        "high_offset": args.high_offset,
        "byte_count": args.byte_count,
        "active_harts": args.active_harts,
        "expected": {
            "sum": expected_sum,
            "xorv": expected_xor,
            "first_word": expected_first,
            "last_word": expected_last,
        },
        "observed": parsed,
        "pass": passed,
    }
    (run_dir / "dynmem_probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
