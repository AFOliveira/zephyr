#!/usr/bin/env python3
"""Run silicon token-rate audits for one exported Whisper decoder MatMul."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
SRC = ROOT / "whisper_decoder_colmatmul_vpu_argbuf.c"
CRT = AMP_ROOT / "hart-report" / "hart_report_crt.S"

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

MAGIC = 0x57444C47
ALIGN = 0x10000

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


def align_up(v: int, align: int = ALIGN) -> int:
    return (v + align - 1) & ~(align - 1)


def load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name in ("act_full", "weight_t", "ref_full"):
        path = data_dir / manifest["files"][name]
        if not path.exists():
            raise SystemExit(f"missing exported tensor: {path}")
    return manifest


def objcopy_binary(src: Path, out: Path) -> None:
    run([OBJCOPY, "-I", "binary", "-O", "elf64-littleriscv", "-B", "riscv:rv64",
         src.name, out.name], cwd=src.parent)


def align_cols(v: int) -> int:
    return (v + 15) & ~15


def layout_for(k_dim: int, n_cols: int, active_harts: int) -> dict[str, int]:
    out_offset = 0x4000
    out_len = n_cols * 4
    tmp_offset = align_up(out_offset + out_len)
    tmp_len = active_harts * align_cols(n_cols) * 4
    act_offset = align_up(tmp_offset + tmp_len)
    act_len = k_dim * 4
    wt_offset = align_up(act_offset + act_len)
    wt_len = n_cols * k_dim * 4
    ref_offset = align_up(wt_offset + wt_len)
    ref_len = n_cols * 4
    end = ref_offset + ref_len
    if end > 16 * 1024 * 1024:
        raise SystemExit(f"argument buffer layout exceeds 16 MiB: 0x{end:x}")
    return {
        "out_offset": out_offset,
        "tmp_offset": tmp_offset,
        "act_offset": act_offset,
        "wt_offset": wt_offset,
        "ref_offset": ref_offset,
        "end": end,
    }


def build_tile(tile_id: int, col0: int, col1: int, build_root: Path,
               flags: list[str], data_dir: Path, manifest: dict[str, Any],
               mem: dict[str, int], active_harts: int) -> Path:
    n_cols = col1 - col0
    k_dim = int(manifest["k_dim"])
    bdir = build_root / f"c{tile_id:02d}"
    bdir.mkdir(parents=True, exist_ok=True)

    act = np.fromfile(data_dir / manifest["files"]["act_full"],
                      dtype=np.float32).reshape(k_dim)
    ref = np.fromfile(data_dir / manifest["files"]["ref_full"],
                      dtype=np.float32).reshape(int(manifest["n_cols"]))
    weight_t = np.fromfile(data_dir / manifest["files"]["weight_t"],
                           dtype=np.float32).reshape(int(manifest["n_cols"]), k_dim)

    act.astype("<f4").tofile(bdir / "decoder_act.bin")
    ref[col0:col1].astype("<f4").tofile(bdir / "decoder_ref.bin")
    weight_t[col0:col1].astype("<f4").tofile(bdir / "decoder_weight_t.bin")

    objcopy_binary(bdir / "decoder_act.bin", bdir / "decoder_act.o")
    objcopy_binary(bdir / "decoder_ref.bin", bdir / "decoder_ref.o")
    objcopy_binary(bdir / "decoder_weight_t.bin", bdir / "decoder_weight_t.o")

    elf = build_root / f"decoder_colmatmul_t{tile_id:02d}_{col0}_{col1}.elf"
    cmd = [
        GCC,
        *flags,
        *BASE_FLAGS,
        f"-DNUM_HARTS={active_harts}",
        f"-DACTIVE_HARTS={active_harts}u",
        f"-DK_DIM={k_dim}u",
        f"-DN_COLS={n_cols}u",
        f"-DOUT_OFFSET=0x{mem['out_offset']:x}u",
        f"-DTMP_OFFSET=0x{mem['tmp_offset']:x}u",
        f"-DACT_OFFSET=0x{mem['act_offset']:x}u",
        f"-DWT_OFFSET=0x{mem['wt_offset']:x}u",
        f"-DREF_OFFSET=0x{mem['ref_offset']:x}u",
        "-o",
        str(elf),
        str(SRC),
        str(bdir / "decoder_act.o"),
        str(bdir / "decoder_weight_t.o"),
        str(bdir / "decoder_ref.o"),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def remote_script(tiles: list[tuple[int, int, int]], remote_work: str,
                  timeout: int, out_offset: int, active_harts: int) -> str:
    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_work}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero2m.bin
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"""
    for tile_id, col0, col1 in tiles:
        n_cols = col1 - col0
        elf = f"decoder_colmatmul_t{tile_id:02d}_{col0}_{col1}.elf"
        script += f"""
echo "=== tile {tile_id:02d} cols {col0}:{col1} ==="
"$LAUNCH" --elf-load ./{elf} --shire 0 --file_load 0x0,$ZERO \\
  --dump_after dump_t{tile_id:02d}.bin --timeout {timeout} > run_t{tile_id:02d}.log 2>&1
grep -oE 'Kernel wait seconds: [0-9.]+' run_t{tile_id:02d}.log | tail -1 || true
python3 - <<'PY'
import json, re, struct
from pathlib import Path
data = Path("dump_t{tile_id:02d}.bin").read_bytes()
out_off = {out_offset}
out_len = {n_cols} * 4
Path("out_t{tile_id:02d}.bin").write_bytes(data[out_off:out_off + out_len])
slot_struct = struct.Struct("<16I")
slots = []
done = 0
active_mask = 0
for h in range({active_harts}):
    f = slot_struct.unpack_from(data, h * 64)
    slot = {{
        "magic": f[0],
        "hart_id": f[1],
        "minion_id": f[2],
        "thread_id": f[3],
        "col0": f[4],
        "col1": f[5],
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
    "col0": {col0},
    "col1": {col1},
    "n_cols": {n_cols},
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
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--tile-cols", type=int, default=8192)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS)
    ap.add_argument("--cflag", action="append", default=[])
    ap.add_argument("--keep-builds", action="store_true")
    args = ap.parse_args()
    flags = list(args.flags) + list(args.cflag)

    manifest = load_manifest(args.data_dir)
    k_dim = int(manifest["k_dim"])
    n_total = int(manifest["n_cols"])
    if args.active_harts < 1 or args.active_harts > 16:
        raise SystemExit("--active-harts must be 1..16")
    if k_dim % 16 != 0:
        raise SystemExit(f"K dimension must be divisible by 16: {k_dim}")
    if args.tile_cols % 2 != 0:
        raise SystemExit("--tile-cols must be even")

    tiles = []
    for col0 in range(0, n_total, args.tile_cols):
        col1 = min(col0 + args.tile_cols, n_total)
        if (col1 - col0) % 2 != 0:
            raise SystemExit(f"odd tile width at cols {col0}:{col1}")
        tiles.append((len(tiles), col0, col1))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    build_root = ROOT / "phase0" / "decoder_token_builds" / f"{args.variant}_{stamp}"
    local_run = ROOT / "phase0" / "runs" / f"{args.variant}_{stamp}"
    local_run.mkdir(parents=True, exist_ok=True)

    print("checking esperanto-soc6 SSH access")
    run([*SSH_CMD, "true"], timeout=30)

    elfs = []
    for tile_id, col0, col1 in tiles:
        mem = layout_for(k_dim, col1 - col0, args.active_harts)
        elfs.append(build_tile(tile_id, col0, col1, build_root, flags,
                               args.data_dir, manifest, mem, args.active_harts))

    remote_work = f"{REMOTE_ROOT}/{args.variant}_{stamp}"
    run([*SSH_CMD, f"mkdir -p {remote_work}"])
    run(["rsync", "-e", RSYNC_RSH, "-aq", *(str(e) for e in elfs),
         f"{REMOTE_HOST}:{remote_work}/"])

    print("running remote decoder token MatMul audit")
    first_mem = layout_for(k_dim, tiles[0][2] - tiles[0][1], args.active_harts)
    subprocess.run(
        [*SSH_CMD, "bash", "-s"],
        input=remote_script(tiles, remote_work, args.timeout,
                            first_mem["out_offset"], args.active_harts),
        text=True,
        check=True,
    )

    run(["rsync", "-e", RSYNC_RSH, "-aq",
         "--include=run_t*.log", "--include=summary_t*.json",
         "--include=out_t*.bin", "--exclude=*",
         f"{REMOTE_HOST}:{remote_work}/", str(local_run) + "/"])

    ref = np.fromfile(args.data_dir / manifest["files"]["ref_full"],
                      dtype=np.float32).reshape(n_total)
    stitched = np.zeros((n_total,), dtype=np.float32)
    tile_reports = []
    total_wait = 0.0
    tile_ok = True
    for tile_id, col0, col1 in tiles:
        out = np.fromfile(local_run / f"out_t{tile_id:02d}.bin",
                          dtype=np.float32).reshape(col1 - col0)
        stitched[col0:col1] = out
        summary = json.loads((local_run / f"summary_t{tile_id:02d}.json").read_text())
        diff = np.abs(out - ref[col0:col1])
        wait = summary.get("wait_s") or 0.0
        total_wait += wait
        ok = (
            summary["slot_done_count"] == args.active_harts
            and float(diff.max()) <= 1e-4
            and summary["summary_magic"] == MAGIC
        )
        tile_ok = tile_ok and ok
        tile_reports.append({
            "tile_id": tile_id,
            "col0": col0,
            "col1": col1,
            "wait_s": wait,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "slot_done_count": summary["slot_done_count"],
            "active_mask": hex(summary["slot_active_mask"]),
            "summary_present": summary["summary_magic"] == MAGIC,
            "stream_error": summary["stream_error"],
            "status": "ok_stream_error" if ok and summary["stream_error"] else "ok" if ok else "fail",
        })

    diff = np.abs(stitched - ref)
    ops = int(manifest["ops"])
    executed_ops = int(manifest["padded_ops_executed"])
    full = {
        "variant": args.variant,
        "timestamp": stamp,
        "node_name": manifest["node_name"],
        "act_name": manifest["act_name"],
        "weight_name": manifest["weight_name"],
        "out_name": manifest["out_name"],
        "rows": 1,
        "k_dim_unpadded": int(manifest["k_dim_unpadded"]),
        "k_dim": k_dim,
        "n_cols": n_total,
        "tiles": len(tiles),
        "tile_cols": args.tile_cols,
        "active_harts": args.active_harts,
        "total_wait_s": total_wait,
        "token_s_for_this_matmul": (1.0 / total_wait) if total_wait else 0.0,
        "ops": ops,
        "padded_ops_executed": executed_ops,
        "gops_model_ops": ops / total_wait / 1e9 if total_wait else 0.0,
        "gops_executed_ops": executed_ops / total_wait / 1e9 if total_wait else 0.0,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "allclose_1e4": bool(np.allclose(stitched, ref, rtol=1e-4, atol=1e-4)),
        "allclose_1e3": bool(np.allclose(stitched, ref, rtol=1e-3, atol=1e-3)),
        "tile_ok": tile_ok,
        "remote_work": remote_work,
        "local_run_dir": str(local_run),
        "tile_rows_report": tile_reports,
    }
    if n_total > 1:
        full["argmax_silicon"] = int(stitched.argmax())
        full["argmax_ort"] = int(ref.argmax())
        full["argmax_match"] = bool(stitched.argmax() == ref.argmax())
    stitched.astype("<f4").tofile(local_run / "stitched_output.bin")
    (local_run / "full_audit_report.json").write_text(json.dumps(full, indent=2) + "\n")

    tsv = ROOT / "phase0" / "whisper_decoder_token_matmul_audit.tsv"
    new_file = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "variant", "node_name", "act_name", "weight_name",
            "rows", "k_dim_unpadded", "k_dim", "n_cols", "tiles",
            "tile_cols", "active_harts", "total_wait_s",
            "token_s_for_this_matmul", "ops", "padded_ops_executed",
            "gops_model_ops", "gops_executed_ops", "max_abs", "mean_abs",
            "p99_abs", "allclose_1e4", "allclose_1e3", "tile_ok",
            "argmax_match", "remote_work", "local_run_dir",
        ]
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
