#!/usr/bin/env python3
"""DnCNN3 audited experiment orchestrator.

Take a list of experiment specs, build each locally, batch-stage to esperanto-soc6,
run sequentially on silicon, pull dumps back, audit against ORT (via the kernel-side
reference_hash + max_abs), and append to a master TSV.

Each spec is a dict:
  exp_id: int
  batch: str        (e.g. "1_compile")
  variant: str      (short name, slug-friendly)
  tile: int         (1, 4, 16, 64)
  harts: int        (1, 2, 4, 8, 16)
  passes: int       (1, 2, 4, 8, 16, ...)
  flags: list[str]  (gcc flags, e.g. ["-O3", "-funroll-loops"])
  macros: list[str] (-D macros, e.g. ["DNCNN_VPU_OC2=1"])

Usage:
  experiment_runner.py --batch FILE.json [--master master.tsv]

The master TSV is append-only. exp_id is the unique key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
ARTIFACT_ROOT = ROOT
GCC = os.environ.get("GCC", "/home/afonso/et/bin/riscv64-unknown-elf-gcc")
CRT = ROOT / "hart-report" / "hart_report_crt.S"
LAYOUT = Path("/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c")
LD_SCRIPT = Path("/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld")
SRC = DIR / "dncnn3_luxonis_real_vpu.c"
WEIGHTS_O = DIR / "dncnn3_luxonis_240x320_weights_packed_f32.o"

REMOTE_HOST = "root@esperanto-soc6"
REMOTE_BASE = "/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2"
REMOTE_PARENT = f"{REMOTE_BASE}/erbium-amp-probe"
REMOTE_WORK_TMPL = f"{REMOTE_PARENT}/exp_{{stamp}}"

SUMMARY_STRUCT = struct.Struct("<16I")
DNCNN_MAGIC = 0xD3C11004

TILE_INFO = {
    1: ("DNCNN_TILE1_BLOBS", "dncnn3_luxonis_1x1_input_f32.o", "dncnn3_luxonis_1x1_ort_output_f32.o"),
    4: ("DNCNN_TILE4_BLOBS", "dncnn3_luxonis_4x4_input_f32.o", "dncnn3_luxonis_4x4_ort_output_f32.o"),
    16: ("DNCNN_TILE16_BLOBS", "dncnn3_luxonis_16x16_input_f32.o", "dncnn3_luxonis_16x16_ort_output_f32.o"),
    64: ("DNCNN_TILE64_BLOBS", "dncnn3_luxonis_64x64_input_f32.o", "dncnn3_luxonis_64x64_ort_output_f32.o"),
}

COMMON_FLAGS = [
    "-march=rv64imfc", "-mabi=lp64f", "-mcmodel=medany", "-nostdlib",
    "-fno-zero-initialized-in-bss", "-ffunction-sections", "-fdata-sections",
    "-I/tmp/et-vidas-install/erbium-soc1sim-umode/include",
    "-I/tmp/et-vidas-install/include/esperanto-fw/erbium_hal",
    "-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include",
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-Wl,--gc-sections", "-Wl,--no-warn-rwx-segments", "-Wl,--emit-relocs",
    "-T", str(LD_SCRIPT),
]


@dataclass
class Spec:
    exp_id: int
    batch: str
    variant: str
    tile: int
    harts: int
    passes: int = 8
    flags: list[str] = field(default_factory=list)
    macros: list[str] = field(default_factory=list)


@dataclass
class Result:
    exp_id: int
    batch: str
    variant: str
    tile: int
    harts: int
    passes: int
    flags: str
    macros: str
    wait_s: float | None
    gops: float | None
    ops: int
    allclose: bool
    bit_exact: bool
    max_abs: float
    mean_abs: float
    output_hash: str
    reference_hash: str
    done_count: int
    magic_ok: bool
    status: str
    dump: str
    log: str
    build_at: str
    run_at: str


def slug(spec: Spec) -> str:
    return f"e{spec.exp_id:03d}_{spec.variant}_t{spec.tile}_h{spec.harts}_p{spec.passes}"


def build(spec: Spec, out_dir: Path) -> Path | None:
    macro_def, input_o, output_o = TILE_INFO[spec.tile]
    elf = out_dir / f"{slug(spec)}.elf"
    cmd = [GCC, *spec.flags, *COMMON_FLAGS,
           "-DDNCNN_STATIC_BLOBS", f"-D{macro_def}",
           f"-DIMG_W={spec.tile}", f"-DIMG_H={spec.tile}",
           f"-DACTIVE_HARTS={spec.harts}u", f"-DNUM_HARTS={spec.harts}",
           f"-DDNCNN_PASSES={spec.passes}u",
           *(f"-D{m}" for m in spec.macros),
           "-o", str(elf),
           str(SRC), str(DIR / input_o), str(DIR / output_o), str(WEIGHTS_O),
           str(CRT), str(LAYOUT)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  BUILD FAIL {slug(spec)}: {res.stderr[-400:]}")
        return None
    return elf


def stage(elfs: list[Path], remote_work: str) -> None:
    subprocess.run(["ssh", REMOTE_HOST, f"mkdir -p {remote_work}"], check=True)
    cmd = ["rsync", "-aq"] + [str(e) for e in elfs] + [f"{REMOTE_HOST}:{remote_work}/"]
    subprocess.run(cmd, check=True)


def run_remote_batch(elf_names: list[str], remote_work: str, stamp: str) -> dict[str, dict]:
    """Run each ELF via the launcher, return per-elf {wait_s, returncode, dump_size}."""
    script = f"""set -u
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote_work}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero64k.bin
cd "$WORK"
mkdir -p remote-logs-{stamp}
export LD_LIBRARY_PATH=$BASE:$PARENT
"""
    for name in elf_names:
        script += f"""
echo === {name} ===
if "$LAUNCH" --elf-load ./{name}.elf --shire 0 --file_load 0x0,$ZERO \\
    --dump_after remote-logs-{stamp}/dump_{name}.bin --timeout 600 \\
    > remote-logs-{stamp}/run_{name}.log 2>&1; then
    grep -oE 'Kernel wait seconds: [0-9.]+' remote-logs-{stamp}/run_{name}.log | tail -1 || echo 'no_wait'
    echo RC_{name}=0
else
    echo RC_{name}=$?
fi
"""
    res = subprocess.run(["ssh", REMOTE_HOST, "bash", "-s"], input=script,
                         capture_output=True, text=True)
    out = {}
    cur = None
    for line in res.stdout.splitlines():
        m = re.match(r"=== (\S+) ===", line)
        if m:
            cur = m.group(1); out[cur] = {"wait_s": None, "rc": None}
            continue
        m = re.match(r"Kernel wait seconds:\s*([0-9.eE+-]+)", line)
        if m and cur: out[cur]["wait_s"] = float(m.group(1))
        m = re.match(r"RC_(\S+)=(\d+)", line)
        if m: out[m.group(1)]["rc"] = int(m.group(2))
    return out


def pull_dumps(remote_work: str, stamp: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-aq",
        f"{REMOTE_HOST}:{remote_work}/remote-logs-{stamp}/",
        str(local_dir) + "/",
    ], check=True)


def parse_summary(dump: Path) -> dict[str, Any]:
    with dump.open("rb") as f:
        f.seek(0x1000)
        data = f.read(SUMMARY_STRUCT.size)
    if len(data) != SUMMARY_STRUCT.size:
        return {}
    fields = SUMMARY_STRUCT.unpack(data)
    ops = fields[11] | (fields[12] << 32)
    return {
        "magic": fields[0], "active_harts": fields[1], "passes": fields[2],
        "width": fields[3], "height": fields[4], "channels": fields[5],
        "layers": fields[6], "active_mask": fields[7], "done_count": fields[8],
        "output_hash": fields[9], "reference_hash": fields[10], "ops": ops,
        "max_abs": fields[13] / 1e9, "mean_abs": fields[14] / 1e9,
        "slot_checksum_sum": fields[15],
    }


def audit_one(spec: Spec, dump: Path, log: Path, run_state: dict, atol: float = 1e-5) -> Result:
    s = parse_summary(dump) if dump.exists() else {}
    magic_ok = bool(s) and s["magic"] == DNCNN_MAGIC
    bit_exact = magic_ok and s["output_hash"] == s["reference_hash"]
    allclose = (magic_ok and s.get("max_abs", 1.0) <= atol
                and s.get("done_count", 0) == s.get("active_harts", -1))
    wait_s = run_state.get("wait_s")
    ops = s.get("ops", 0)
    gops = (ops / wait_s / 1e9) if (wait_s and wait_s > 0 and ops) else None
    rc = run_state.get("rc")
    if rc != 0 and rc is not None:
        status = f"launcher_rc{rc}"
    elif not magic_ok:
        status = "kernel_no_magic"
    elif s.get("done_count", 0) != s.get("active_harts", -1):
        status = "incomplete_harts"
    elif not allclose:
        status = "wrong_output"
    else:
        status = "ok"
    return Result(
        exp_id=spec.exp_id, batch=spec.batch, variant=spec.variant,
        tile=spec.tile, harts=spec.harts, passes=spec.passes,
        flags=" ".join(spec.flags), macros=",".join(spec.macros),
        wait_s=wait_s, gops=gops, ops=ops,
        allclose=bool(allclose), bit_exact=bool(bit_exact),
        max_abs=s.get("max_abs", float("nan")),
        mean_abs=s.get("mean_abs", float("nan")),
        output_hash=f"0x{s['output_hash']:08x}" if magic_ok else "-",
        reference_hash=f"0x{s['reference_hash']:08x}" if magic_ok else "-",
        done_count=s.get("done_count", 0),
        magic_ok=bool(magic_ok),
        status=status,
        dump=str(dump), log=str(log),
        build_at=run_state.get("build_at", ""),
        run_at=run_state.get("run_at", ""),
    )


def append_master(master_tsv: Path, results: list[Result]) -> None:
    cols = ["exp_id", "batch", "variant", "tile", "harts", "passes",
            "wait_s", "gops", "ops", "allclose", "bit_exact",
            "max_abs", "mean_abs", "output_hash", "reference_hash",
            "done_count", "magic_ok", "status",
            "flags", "macros", "dump", "log", "build_at", "run_at"]
    new = not master_tsv.exists()
    with master_tsv.open("a") as fh:
        if new:
            fh.write("\t".join(cols) + "\n")
        for r in results:
            d = asdict(r)
            row = [d.get(c) for c in cols]
            fh.write("\t".join(
                "" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v))
                for v in row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True, help="JSON file with spec list")
    ap.add_argument("--master", type=Path, default=DIR / "audit_master.tsv")
    ap.add_argument("--build-dir", type=Path, default=DIR / "experiment-builds")
    ap.add_argument("--logs-dir", type=Path, default=DIR / "experiment-logs")
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    raw = json.loads(args.batch.read_text())
    specs = [Spec(**s) for s in raw]
    print(f"loaded {len(specs)} experiments from {args.batch}")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    remote_work = REMOTE_WORK_TMPL.format(stamp=stamp)

    print(f"building {len(specs)} variants under {args.build_dir} ...")
    elfs = []
    build_failures = []
    build_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    for s in specs:
        elf = build(s, args.build_dir)
        if elf:
            elfs.append((s, elf))
        else:
            build_failures.append(s)
    print(f"  built {len(elfs)} ok, {len(build_failures)} failed")

    if not elfs:
        print("no successful builds, abort")
        return 1

    print(f"staging to {remote_work} ...")
    stage([e for _, e in elfs], remote_work)

    print("running on silicon ...")
    run_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    elf_names = [slug(s) for s, _ in elfs]
    run_states = run_remote_batch(elf_names, remote_work, stamp)

    print("pulling dumps ...")
    pull_dumps(remote_work, stamp, args.logs_dir / stamp)

    print("auditing ...")
    results = []
    for s, elf in elfs:
        name = slug(s)
        dump = args.logs_dir / stamp / f"dump_{name}.bin"
        log = args.logs_dir / stamp / f"run_{name}.log"
        rs = run_states.get(name, {"wait_s": None, "rc": None})
        rs["build_at"] = build_at
        rs["run_at"] = run_at
        results.append(audit_one(s, dump, log, rs, atol=args.atol))

    for s in build_failures:
        results.append(Result(
            exp_id=s.exp_id, batch=s.batch, variant=s.variant, tile=s.tile,
            harts=s.harts, passes=s.passes, flags=" ".join(s.flags),
            macros=",".join(s.macros), wait_s=None, gops=None, ops=0,
            allclose=False, bit_exact=False, max_abs=float("nan"),
            mean_abs=float("nan"), output_hash="-", reference_hash="-",
            done_count=0, magic_ok=False, status="build_fail",
            dump="", log="", build_at=build_at, run_at=""))

    append_master(args.master, results)
    print(f"\nappended {len(results)} rows to {args.master}\n")

    # leaderboard
    valid = [r for r in results if r.allclose and r.gops]
    valid.sort(key=lambda r: -(r.gops or 0))
    print(f"{'#':>3} {'exp':>4} {'variant':32s} {'tile':>5s} {'h':>2s} "
          f"{'GOPS':>6s} {'max_abs':>9s} status")
    for i, r in enumerate(valid[:15], 1):
        print(f"{i:>3} {r.exp_id:>4d} {r.variant[:32]:32s} "
              f"{r.tile}x{r.tile:<3d} {r.harts:>2d} "
              f"{(r.gops or 0):>6.3f} {r.max_abs:>9.2e} {r.status}")
    bad = [r for r in results if not r.allclose]
    if bad:
        print(f"\n{len(bad)} non-passing experiments:")
        for r in bad[:10]:
            print(f"  exp {r.exp_id} {r.variant} t{r.tile} h{r.harts} -> {r.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
