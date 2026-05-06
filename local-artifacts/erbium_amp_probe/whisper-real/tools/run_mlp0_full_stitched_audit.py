#!/usr/bin/env python3
"""Run full stitched silicon audit for Whisper encoder MLP0 MatMul."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
SRC = ROOT / "whisper_mlp0_real_audit_vpu_argbuf.c"
CRT = AMP_ROOT / "hart-report" / "hart_report_crt.S"
TILE_SRC = ROOT / "phase0" / "mlp0_tile"

GCC = os.environ.get("GCC", "/home/afonso/et/bin/riscv64-unknown-elf-gcc")
OBJCOPY = os.environ.get("OBJCOPY", "/home/afonso/et/bin/riscv64-unknown-elf-objcopy")
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
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=1 -o ConnectTimeout=10"))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]
RSYNC_RSH = "ssh " + shlex.join(SSH_OPTS)

ROWS = 1500
K_DIM = 384
N_COLS = 1536
TILE_ROWS = 128
OUT_OFFSET = 0x4000
MAGIC = 0x574D4C50


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
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-Wl,--gc-sections",
    "-Wl,--no-warn-rwx-segments",
    "-Wl,--emit-relocs",
    "-T",
    str(LD_SCRIPT),
]

DEFAULT_FLAGS = ["-O3", "-funroll-loops"]


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def objcopy_binary(src: Path, out: Path) -> None:
    run([OBJCOPY, "-I", "binary", "-O", "elf64-littleriscv", "-B", "riscv:rv64",
         src.name, out.name], cwd=src.parent)


def build_tile(tile_id: int, row0: int, row1: int, build_root: Path,
               flags: list[str]) -> Path:
    rows = row1 - row0
    bdir = build_root / f"t{tile_id:02d}"
    bdir.mkdir(parents=True, exist_ok=True)

    act_full = np.fromfile(TILE_SRC / "mlp0_act_full_1500x384.bin",
                           dtype=np.float32).reshape(ROWS, K_DIM)
    ref_full = np.fromfile(TILE_SRC / "mlp0_ref_full_1500x1536.bin",
                           dtype=np.float32).reshape(ROWS, N_COLS)
    weight_t = np.fromfile(TILE_SRC / "mlp0_weight_t_1536x384.bin",
                           dtype=np.float32).reshape(N_COLS, K_DIM)

    act_full[row0:row1].astype("<f4").tofile(bdir / "mlp0_act_128x384.bin")
    ref_full[row0:row1].astype("<f4").tofile(bdir / "mlp0_ref_128x1536.bin")
    weight_t.astype("<f4").tofile(bdir / "mlp0_weight_t_1536x384.bin")

    objcopy_binary(bdir / "mlp0_act_128x384.bin", bdir / "mlp0_act_128x384.o")
    objcopy_binary(bdir / "mlp0_ref_128x1536.bin", bdir / "mlp0_ref_128x1536.o")
    objcopy_binary(bdir / "mlp0_weight_t_1536x384.bin", bdir / "mlp0_weight_t_1536x384.o")

    elf = build_root / f"mlp0_full_t{tile_id:02d}_{row0}_{row1}.elf"
    cmd = [
        GCC,
        *flags,
        *BASE_FLAGS,
        f"-DNUM_HARTS=16",
        f"-DACTIVE_HARTS=16",
        f"-DM_ROWS={rows}u",
        "-o",
        str(elf),
        str(SRC),
        str(bdir / "mlp0_act_128x384.o"),
        str(bdir / "mlp0_weight_t_1536x384.o"),
        str(bdir / "mlp0_ref_128x1536.o"),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def remote_script(tiles: list[tuple[int, int, int]], remote_work: str,
                  timeout: int) -> str:
    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_work}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero2m.bin
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"""
    for tile_id, row0, row1 in tiles:
        rows = row1 - row0
        elf = f"mlp0_full_t{tile_id:02d}_{row0}_{row1}.elf"
        script += f"""
echo "=== tile {tile_id:02d} rows {row0}:{row1} ==="
"$LAUNCH" --elf-load ./{elf} --shire 0 --file_load 0x0,$ZERO \\
  --dump_after dump_t{tile_id:02d}.bin --timeout {timeout} > run_t{tile_id:02d}.log 2>&1
grep -oE 'Kernel wait seconds: [0-9.]+' run_t{tile_id:02d}.log | tail -1 || true
python3 - <<'PY'
import json, re, struct
from pathlib import Path
data = Path("dump_t{tile_id:02d}.bin").read_bytes()
out_off = {OUT_OFFSET}
out_len = {rows} * {N_COLS} * 4
Path("out_t{tile_id:02d}.bin").write_bytes(data[out_off:out_off + out_len])
slot_struct = struct.Struct("<16I")
slots = []
done = 0
active_mask = 0
for h in range(16):
    f = slot_struct.unpack_from(data, h * 64)
    slot = {{
        "magic": f[0],
        "hart_id": f[1],
        "minion_id": f[2],
        "thread_id": f[3],
        "row0": f[4],
        "row1": f[5],
        "active_harts": f[6],
        "checksum": f[7],
        "done": f[8],
    }}
    slots.append(slot)
    if slot["magic"] == {MAGIC} and slot["done"] == 1:
        done += 1
        active_mask |= 1 << slot["hart_id"]
summary = struct.unpack_from("<16I", data, 0x1000)
text = Path("run_t{tile_id:02d}.log").read_text(errors="ignore")
m = re.search(r"Kernel wait seconds:\\s*([0-9.eE+-]+)", text)
report = {{
    "tile_id": {tile_id},
    "row0": {row0},
    "row1": {row1},
    "rows": {rows},
    "wait_s": float(m.group(1)) if m else None,
    "slot_done_count": done,
    "slot_active_mask": active_mask,
    "summary_magic": summary[0],
    "stream_error": "Stream error" in text,
    "kernel_launch_error": "Error on kernel launch" in text,
    "slots": slots,
}}
Path("summary_t{tile_id:02d}.json").write_text(json.dumps(report, indent=2) + "\\n")
PY
"""
    return script


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mlp0_full_stitched_o3_unroll")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS)
    ap.add_argument("--keep-builds", action="store_true")
    args = ap.parse_args()

    if not (TILE_SRC / "mlp0_act_full_1500x384.bin").exists():
        run(["python3", str(HERE / "export_whisper_mlp0_tile.py")])

    tiles = []
    for row0 in range(0, ROWS, TILE_ROWS):
        row1 = min(row0 + TILE_ROWS, ROWS)
        tiles.append((len(tiles), row0, row1))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    build_root = ROOT / "phase0" / "mlp0_full_builds" / f"{args.variant}_{stamp}"
    local_run = ROOT / "phase0" / "runs" / f"{args.variant}_{stamp}"
    local_run.mkdir(parents=True, exist_ok=True)

    print("checking esperanto-soc6 SSH access")
    run([*SSH_CMD, "true"], timeout=30)

    elfs = [build_tile(tile_id, row0, row1, build_root, args.flags)
            for tile_id, row0, row1 in tiles]

    remote_work = f"{REMOTE_ROOT}/{args.variant}_{stamp}"
    run([*SSH_CMD, f"mkdir -p {remote_work}"])
    run(["rsync", "-e", RSYNC_RSH, "-aq", *(str(e) for e in elfs),
         f"{REMOTE_HOST}:{remote_work}/"])

    print("running remote full-node tiled audit")
    subprocess.run([*SSH_CMD, "bash", "-s"], input=remote_script(tiles, remote_work, args.timeout),
                   text=True, check=True)

    run(["rsync", "-e", RSYNC_RSH, "-aq",
         "--include=run_t*.log", "--include=summary_t*.json",
         "--include=out_t*.bin", "--exclude=*",
         f"{REMOTE_HOST}:{remote_work}/", str(local_run) + "/"])

    ref_full = np.fromfile(TILE_SRC / "mlp0_ref_full_1500x1536.bin",
                           dtype=np.float32).reshape(ROWS, N_COLS)
    stitched = np.zeros((ROWS, N_COLS), dtype=np.float32)
    rows = []
    total_wait = 0.0
    tile_ok = True
    for tile_id, row0, row1 in tiles:
        out = np.fromfile(local_run / f"out_t{tile_id:02d}.bin",
                          dtype=np.float32).reshape(row1 - row0, N_COLS)
        stitched[row0:row1] = out
        summary = json.loads((local_run / f"summary_t{tile_id:02d}.json").read_text())
        diff = np.abs(out - ref_full[row0:row1])
        wait = summary.get("wait_s") or 0.0
        total_wait += wait
        ok = (summary["slot_done_count"] == 16 and float(diff.max()) <= 1e-4)
        tile_ok = tile_ok and ok
        rows.append({
            "tile_id": tile_id,
            "row0": row0,
            "row1": row1,
            "wait_s": wait,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "slot_done_count": summary["slot_done_count"],
            "active_mask": hex(summary["slot_active_mask"]),
            "summary_present": summary["summary_magic"] == MAGIC,
            "stream_error": summary["stream_error"],
            "status": "ok_stream_error" if ok and summary["stream_error"] else "ok" if ok else "fail",
        })

    diff = np.abs(stitched - ref_full)
    ops = ROWS * K_DIM * N_COLS * 2
    full = {
        "variant": args.variant,
        "timestamp": stamp,
        "onnx_node": "/encoder/blocks.0/mlp/0/MatMul",
        "rows": ROWS,
        "k_dim": K_DIM,
        "n_cols": N_COLS,
        "tiles": len(tiles),
        "tile_rows": TILE_ROWS,
        "total_wait_s": total_wait,
        "ops": ops,
        "gops": ops / total_wait / 1e9 if total_wait else 0.0,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "allclose_1e4": bool(np.allclose(stitched, ref_full, rtol=1e-4, atol=1e-4)),
        "allclose_1e3": bool(np.allclose(stitched, ref_full, rtol=1e-3, atol=1e-3)),
        "tile_ok": tile_ok,
        "remote_work": remote_work,
        "local_run_dir": str(local_run),
        "tile_rows_report": rows,
    }
    stitched.astype("<f4").tofile(local_run / "stitched_output_1500x1536.bin")
    (local_run / "full_audit_report.json").write_text(json.dumps(full, indent=2) + "\n")

    tsv = ROOT / "phase0" / "whisper_real_full_node_audit.tsv"
    new_file = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = ["timestamp", "variant", "onnx_node", "rows", "k_dim", "n_cols",
                "tiles", "total_wait_s", "ops", "gops", "max_abs", "mean_abs",
                "p99_abs", "allclose_1e4", "allclose_1e3", "tile_ok",
                "remote_work", "local_run_dir"]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow({c: full.get(c, "") for c in cols})

    if not args.keep_builds:
        shutil.rmtree(build_root, ignore_errors=True)

    print(json.dumps(full, indent=2))
    return 0 if full["allclose_1e4"] and tile_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
