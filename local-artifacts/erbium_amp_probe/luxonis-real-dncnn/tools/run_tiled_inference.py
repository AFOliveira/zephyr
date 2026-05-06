#!/usr/bin/env python3
"""Build/run/stitch tiled FP32 Luxonis DnCNN3 on ET-SoC1 silicon.

This is Track A orchestration:

1. Build one tile-specific ELF per tile geometry entry.
2. Stage ELFs under the allowed esperanto-soc6 artifact tree.
3. Run sequentially on shire 0.
4. Extract cropped tile outputs from remote dumps.
5. Stitch a 240x320 output and compare to the full ORT reference.
6. Append tile and full-image rows to audit_master_tiled.tsv.
"""
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
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DN_ROOT = HERE.parent
AMP_ROOT = DN_ROOT.parent

GCC = os.environ.get("GCC", "/home/afonso/et/bin/riscv64-unknown-elf-gcc")
OBJCOPY = os.environ.get("OBJCOPY", "/home/afonso/et/bin/riscv64-unknown-elf-objcopy")
CRT = AMP_ROOT / "hart-report" / "hart_report_crt.S"
LAYOUT = Path(os.environ.get(
    "LAYOUT", "/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c"))
LD_SCRIPT = Path(os.environ.get(
    "LD_SCRIPT", "/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld"))
SRC = DN_ROOT / "dncnn3_luxonis_real_vpu.c"
WEIGHTS_O = DN_ROOT / "dncnn3_luxonis_240x320_weights_packed_f32.o"

REMOTE_HOST = "root@esperanto-soc6"
REMOTE_BASE = "/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2"
REMOTE_PARENT = f"{REMOTE_BASE}/erbium-amp-probe"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=1 -o ConnectTimeout=10"))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]
RSYNC_RSH = "ssh " + shlex.join(SSH_OPTS)

SUMMARY = struct.Struct("<16I")
SLOT = struct.Struct("<16I")
DNCNN_MAGIC = 0xD3C11004
OUTPUT_OFFSET = 0x300000
CH = 64
K = 3
HIDDEN_LAYERS = 18

BASE_FLAGS = [
    "-march=rv64imfc",
    "-mabi=lp64f",
    "-mcmodel=medany",
    "-nostdlib",
    "-fno-zero-initialized-in-bss",
    "-ffunction-sections",
    "-fdata-sections",
    "-I/tmp/et-vidas-install/erbium-soc1sim-umode/include",
    "-I/tmp/et-vidas-install/include/esperanto-fw/erbium_hal",
    "-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include",
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-Wl,--gc-sections",
    "-Wl,--no-warn-rwx-segments",
    "-Wl,--emit-relocs",
    "-T",
    str(LD_SCRIPT),
]

DEFAULT_FLAGS = [
    "-O3",
    "-funroll-loops",
    "-falign-functions=128",
    "-falign-loops=128",
    "-fno-tree-loop-distribute-patterns",
    "-fno-tree-loop-vectorize",
]

DEFAULT_MACROS = [
    "DNCNN_VPU_FUSED9TAP=1",
    "DNCNN_VPU_PREFETCH_READ_WINDOW=1",
]


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def parse_summary_bytes(data: bytes) -> dict[str, Any]:
    fields = SUMMARY.unpack_from(data, 0x1000)
    ops = fields[11] | (fields[12] << 32)
    return {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "width": fields[3],
        "height": fields[4],
        "channels": fields[5],
        "layers": fields[6],
        "active_mask": fields[7],
        "done_count": fields[8],
        "output_hash": fields[9],
        "reference_hash": fields[10],
        "ops": ops,
        "max_abs": fields[13] / 1e9,
        "mean_abs": fields[14] / 1e9,
        "slot_checksum_sum": fields[15],
    }


def wait_seconds(log_text: str) -> float | None:
    match = re.search(r"Kernel wait seconds:\s*([0-9.eE+-]+)", log_text)
    return float(match.group(1)) if match else None


def useful_full_ops(img_h: int, img_w: int) -> int:
    macs_per_pixel = CH * K * K + HIDDEN_LAYERS * CH * CH * K * K + CH * K * K
    return img_h * img_w * macs_per_pixel * 2


def objcopy_binary(src: Path, out: Path) -> None:
    run([OBJCOPY, "-I", "binary", "-O", "elf64-littleriscv", "-B", "riscv:rv64",
         src.name, out.name], cwd=src.parent)


def build_tile(tile: dict[str, Any], tiles_dir: Path, build_root: Path,
               flags: list[str], macros: list[str], harts: int, passes: int,
               variant: str) -> Path:
    tile_id = int(tile["tile_id"])
    tdir = tiles_dir / f"t{tile_id:02d}"
    bdir = build_root / f"t{tile_id:02d}"
    bdir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(tdir / "input.bin", bdir / "dncnn_tile_input.bin")
    shutil.copy2(tdir / "ort_output_full.bin", bdir / "dncnn_tile_ref.bin")
    objcopy_binary(bdir / "dncnn_tile_input.bin", bdir / "dncnn_tile_input.o")
    objcopy_binary(bdir / "dncnn_tile_ref.bin", bdir / "dncnn_tile_ref.o")

    elf = build_root / f"{variant}_t{tile_id:02d}_{tile['input_h']}x{tile['input_w']}.elf"
    cmd = [
        GCC,
        *flags,
        *BASE_FLAGS,
        "-DDNCNN_STATIC_BLOBS",
        "-DDNCNN_TILE_BLOBS",
        f"-DIMG_W={int(tile['input_w'])}",
        f"-DIMG_H={int(tile['input_h'])}",
        f"-DACTIVE_HARTS={harts}u",
        f"-DNUM_HARTS={harts}",
        f"-DDNCNN_PASSES={passes}u",
        *(f"-D{m}" for m in macros),
        "-o",
        str(elf),
        str(SRC),
        str(bdir / "dncnn_tile_input.o"),
        str(bdir / "dncnn_tile_ref.o"),
        str(WEIGHTS_O),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def remote_run_script(tiles: list[dict[str, Any]], variant: str,
                      remote_work: str, timeout_s: int, harts: int,
                      passes: int) -> str:
    script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_work}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero64k.bin
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"""
    for t in tiles:
        tid = int(t["tile_id"])
        elf_name = f"{variant}_t{tid:02d}_{t['input_h']}x{t['input_w']}.elf"
        dump_name = f"dump_t{tid:02d}.bin"
        log_name = f"run_t{tid:02d}.log"
        crop_name = f"crop_t{tid:02d}.bin"
        summary_name = f"summary_t{tid:02d}.json"
        script += f"""
echo "=== tile {tid:02d} ==="
"$LAUNCH" --elf-load ./{elf_name} --shire 0 --file_load 0x0,$ZERO \\
  --dump_after {dump_name} --timeout {timeout_s} > {log_name} 2>&1
grep -oE 'Kernel wait seconds: [0-9.]+' {log_name} | tail -1 || true
python3 - <<'PY'
import json, re, struct
from pathlib import Path
import numpy as np

summary_struct = struct.Struct("<16I")
slot_struct = struct.Struct("<16I")
dump = Path("{dump_name}")
data = dump.read_bytes()
fields = summary_struct.unpack_from(data, 0x1000)
ops = fields[11] | (fields[12] << 32)
slots = []
done_count = 0
active_mask = 0
slot_checksum_sum = 0
for h in range({harts}):
    sf = slot_struct.unpack_from(data, h * 64)
    slot = {{
        "magic": sf[0],
        "hart_id": sf[1],
        "minion_id": sf[2],
        "thread_id": sf[3],
        "row0": sf[4],
        "row1": sf[5],
        "active_harts": sf[6],
        "checksum": sf[7],
        "done": sf[8],
    }}
    slots.append(slot)
    if slot["magic"] == {DNCNN_MAGIC} and slot["done"] == 1:
        done_count += 1
        active_mask |= 1 << slot["hart_id"]
        slot_checksum_sum = (slot_checksum_sum + slot["checksum"]) & 0xffffffff
if fields[0] != {DNCNN_MAGIC}:
    macs_per_pixel = {CH} * {K} * {K} + {HIDDEN_LAYERS} * {CH} * {CH} * {K} * {K} + {CH} * {K} * {K}
    ops = {int(t['input_h'])} * {int(t['input_w'])} * macs_per_pixel * {passes} * 2
summary = {{
    "tile_id": {tid},
    "magic": fields[0],
    "summary_present": fields[0] == {DNCNN_MAGIC},
    "active_harts": fields[1] if fields[0] == {DNCNN_MAGIC} else {harts},
    "passes": fields[2] if fields[0] == {DNCNN_MAGIC} else {passes},
    "width": fields[3] if fields[0] == {DNCNN_MAGIC} else {int(t['input_w'])},
    "height": fields[4] if fields[0] == {DNCNN_MAGIC} else {int(t['input_h'])},
    "channels": fields[5] if fields[0] == {DNCNN_MAGIC} else {CH},
    "layers": fields[6] if fields[0] == {DNCNN_MAGIC} else {HIDDEN_LAYERS + 2},
    "active_mask": fields[7] if fields[0] == {DNCNN_MAGIC} else active_mask,
    "done_count": fields[8] if fields[0] == {DNCNN_MAGIC} else done_count,
    "output_hash": fields[9],
    "reference_hash": fields[10],
    "ops": ops,
    "max_abs": fields[13] / 1e9,
    "mean_abs": fields[14] / 1e9,
    "slot_checksum_sum": fields[15] if fields[0] == {DNCNN_MAGIC} else slot_checksum_sum,
    "slots": slots,
}}
text = Path("{log_name}").read_text(errors="ignore")
m = re.search(r"Kernel wait seconds:\\s*([0-9.eE+-]+)", text)
summary["wait_s"] = float(m.group(1)) if m else None
summary["stream_error"] = "Stream error" in text
summary["kernel_launch_error"] = "Error on kernel launch" in text
out_off = {OUTPUT_OFFSET}
h = {int(t['input_h'])}
w = {int(t['input_w'])}
cy0 = {int(t['crop_y0'])}
cy1 = {int(t['crop_y1'])}
cx0 = {int(t['crop_x0'])}
cx1 = {int(t['crop_x1'])}
arr = np.frombuffer(data, dtype="<f4", count=h*w, offset=out_off).reshape(h, w)
arr[cy0:cy1, cx0:cx1].astype("<f4").tofile("{crop_name}")
Path("{summary_name}").write_text(json.dumps(summary, indent=2) + "\\n")
PY
"""
    return script


def append_master(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "timestamp", "scope", "variant", "tile_id", "tile_h", "tile_w",
        "input_h", "input_w", "halo", "harts", "passes", "wait_s",
        "total_wait_s", "gops_full_image", "gops_executed",
        "ops_useful", "ops_executed", "max_abs", "mean_abs", "allclose",
        "bit_exact", "output_hash", "reference_hash", "status", "remote_work",
        "local_run_dir", "flags", "macros",
    ]
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", type=Path,
                    default=DN_ROOT / "tiled-fp32" / "audit_a1_geometry.json")
    ap.add_argument("--tiles-dir", type=Path,
                    help="Directory containing tXX/input.bin and ORT reference bins")
    ap.add_argument("--full-ort", type=Path,
                    default=DN_ROOT / "dncnn3_luxonis_240x320_ort_output_f32.bin")
    ap.add_argument("--variant", default="a3_tiled64_halo20_f9t_pfw_align128")
    ap.add_argument("--tile-id", type=int, action="append",
                    help="Run only this tile id; repeat for multiple ids")
    ap.add_argument("--harts", type=int, default=16)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS)
    ap.add_argument("--macros", nargs="*", default=DEFAULT_MACROS)
    ap.add_argument("--master", type=Path, default=DN_ROOT / "audit_master_tiled.tsv")
    ap.add_argument("--keep-builds", action="store_true")
    ap.add_argument("--ssh-preflight-timeout", type=int, default=30)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--reuse-remote-work",
                    help="Skip build/upload and run already-staged ELFs from this remote directory")
    args = ap.parse_args()

    geom = json.loads(args.geometry.read_text())
    tiles_dir = args.tiles_dir if args.tiles_dir else args.geometry.parent / "tiles"
    if not tiles_dir.exists():
        raise SystemExit(f"tiles dir does not exist: {tiles_dir}")
    all_tiles = list(geom["tiles"])
    if args.tile_id:
        wanted = set(args.tile_id)
        tiles = [t for t in all_tiles if int(t["tile_id"]) in wanted]
    else:
        tiles = all_tiles
    if not tiles:
        raise SystemExit("no matching tiles")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_root = args.geometry.parent
    build_root = artifact_root / "builds" / f"{args.variant}_{stamp}"
    local_run = artifact_root / "runs" / f"{args.variant}_{stamp}"
    local_run.mkdir(parents=True, exist_ok=True)

    if not args.build_only:
        print("checking esperanto-soc6 SSH access")
        run([*SSH_CMD, "true"], timeout=args.ssh_preflight_timeout)

    if args.reuse_remote_work:
        if args.build_only:
            raise SystemExit("--build-only cannot be combined with --reuse-remote-work")
        remote_work = args.reuse_remote_work.rstrip("/")
        print(f"reusing remote staged ELFs from {remote_work}")
    else:
        print(f"building {len(tiles)} tile ELFs")
        elfs = [build_tile(t, tiles_dir, build_root, args.flags, args.macros,
                           args.harts, args.passes, args.variant) for t in tiles]
        if args.build_only:
            print("build-only ELFs:")
            for elf in elfs:
                print(elf)
            return 0

        remote_family = artifact_root.name
        remote_work = f"{REMOTE_PARENT}/luxonis-real-dncnn/{remote_family}/{args.variant}_{stamp}"
        run([*SSH_CMD, f"mkdir -p {remote_work}"])
        run(["rsync", "-e", RSYNC_RSH, "-aq", *(str(e) for e in elfs),
             f"{REMOTE_HOST}:{remote_work}/"])

    script = remote_run_script(tiles, args.variant, remote_work, args.timeout,
                               args.harts, args.passes)
    print("running remote tiled batch")
    subprocess.run([*SSH_CMD, "bash", "-s"], input=script,
                   text=True, check=True)

    print("pulling cropped outputs, summaries, and logs")
    run(["rsync", "-e", RSYNC_RSH, "-aq",
         "--include=run_t*.log", "--include=summary_t*.json",
         "--include=crop_t*.bin", "--exclude=*",
         f"{REMOTE_HOST}:{remote_work}/", str(local_run) + "/"])

    img_h = int(geom["img_h"])
    img_w = int(geom["img_w"])
    stitched = np.zeros((img_h, img_w), dtype=np.float32)
    full_ort = np.fromfile(args.full_ort, dtype=np.float32).reshape(img_h, img_w)

    rows = []
    total_wait = 0.0
    executed_ops = 0
    tile_status_ok = True
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    for t in tiles:
        tid = int(t["tile_id"])
        summary = json.loads((local_run / f"summary_t{tid:02d}.json").read_text())
        crop = np.fromfile(local_run / f"crop_t{tid:02d}.bin",
                           dtype=np.float32).reshape(
                               int(t["output_h"]), int(t["output_w"]))
        ref_crop = np.fromfile(tiles_dir / f"t{tid:02d}" / "ort_output_crop.bin",
                               dtype=np.float32).reshape(
                                   int(t["output_h"]), int(t["output_w"]))
        tile_diff = np.abs(crop - ref_crop)
        tile_max_abs = float(tile_diff.max())
        tile_mean_abs = float(tile_diff.mean())
        stitched[int(t["output_y0"]):int(t["output_y1"]),
                 int(t["output_x0"]):int(t["output_x1"])] = crop

        wait = summary.get("wait_s")
        if wait is not None:
            total_wait += float(wait)
        executed_ops += int(summary.get("ops", 0))
        allclose = (
            int(summary.get("done_count", 0)) == args.harts
            and tile_max_abs <= args.atol
        )
        bit_exact = bool(np.array_equal(crop, ref_crop))
        tile_status_ok = tile_status_ok and allclose
        rows.append({
            "timestamp": ts_iso,
            "scope": "tile",
            "variant": args.variant,
            "tile_id": tid,
            "tile_h": geom["tile_h"],
            "tile_w": geom["tile_w"],
            "input_h": t["input_h"],
            "input_w": t["input_w"],
            "halo": geom["chosen_halo"],
            "harts": args.harts,
            "passes": args.passes,
            "wait_s": f"{float(wait):.6f}" if wait is not None else "",
            "gops_executed": f"{summary.get('ops', 0) / float(wait) / 1e9:.6f}" if wait else "",
            "ops_executed": summary.get("ops", 0),
            "max_abs": f"{tile_max_abs:.9g}",
            "mean_abs": f"{tile_mean_abs:.9g}",
            "allclose": int(allclose),
            "bit_exact": int(bit_exact),
            "output_hash": f"0x{int(summary.get('output_hash', 0)):08x}",
            "reference_hash": f"0x{int(summary.get('reference_hash', 0)):08x}",
            "status": ("ok_stream_error" if allclose and summary.get("stream_error")
                       else "ok" if allclose else "fail"),
            "remote_work": remote_work,
            "local_run_dir": str(local_run),
            "flags": " ".join(args.flags),
            "macros": ",".join(args.macros),
        })

    stitched_path = local_run / "stitched_output.bin"
    stitched.astype("<f4").tofile(stitched_path)
    diff = np.abs(stitched - full_ort)
    full_max_abs = float(diff.max())
    full_mean_abs = float(diff.mean())
    full_allclose = full_max_abs <= args.atol and tile_status_ok and len(tiles) == len(all_tiles)
    useful_ops = useful_full_ops(img_h, img_w) * args.passes
    full_report = {
        "variant": args.variant,
        "timestamp": stamp,
        "n_tiles_run": len(tiles),
        "n_tiles_total": len(all_tiles),
        "all_tiles_tile_audit_ok": tile_status_ok,
        "stitched_vs_full_ort_max_abs": full_max_abs,
        "stitched_vs_full_ort_mean_abs": full_mean_abs,
        "allclose": full_allclose,
        "total_wait_s": total_wait,
        "gops_full_image_useful": useful_ops / total_wait / 1e9 if total_wait else 0.0,
        "gops_executed": executed_ops / total_wait / 1e9 if total_wait else 0.0,
        "useful_ops": useful_ops,
        "executed_ops": executed_ops,
        "remote_work": remote_work,
        "local_run_dir": str(local_run),
        "geometry": str(args.geometry),
        "tiles_dir": str(tiles_dir),
    }
    (local_run / "full_audit_report.json").write_text(
        json.dumps(full_report, indent=2) + "\n")

    rows.append({
        "timestamp": ts_iso,
        "scope": "full" if len(tiles) == len(all_tiles) else "partial",
        "variant": args.variant,
        "tile_h": geom["tile_h"],
        "tile_w": geom["tile_w"],
        "halo": geom["chosen_halo"],
        "harts": args.harts,
        "passes": args.passes,
        "total_wait_s": f"{total_wait:.6f}",
        "gops_full_image": f"{useful_ops / total_wait / 1e9:.6f}" if total_wait else "",
        "gops_executed": f"{executed_ops / total_wait / 1e9:.6f}" if total_wait else "",
        "ops_useful": useful_ops,
        "ops_executed": executed_ops,
        "max_abs": f"{full_max_abs:.9g}",
        "mean_abs": f"{full_mean_abs:.9g}",
        "allclose": int(full_allclose),
        "bit_exact": int(bool(np.array_equal(stitched, full_ort)) and len(tiles) == len(all_tiles)),
        "status": "ok" if full_allclose else "partial" if len(tiles) != len(all_tiles) else "fail",
        "remote_work": remote_work,
        "local_run_dir": str(local_run),
        "flags": " ".join(args.flags),
        "macros": ",".join(args.macros),
    })
    append_master(args.master, rows)

    if not args.keep_builds:
        shutil.rmtree(build_root, ignore_errors=True)

    print(json.dumps(full_report, indent=2))
    print(f"wrote {local_run / 'full_audit_report.json'}")
    print(f"appended {len(rows)} rows to {args.master}")
    return 0 if full_allclose or len(tiles) != len(all_tiles) else 1


if __name__ == "__main__":
    sys.exit(main())
