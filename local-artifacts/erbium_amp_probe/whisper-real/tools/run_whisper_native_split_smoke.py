#!/usr/bin/env python3
"""Run the Whisper native-on-chip hart split smoke test on ET-SoC1 silicon."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
PHASE0 = ROOT / "phase0"
SRC = ROOT / "whisper_native_hart_split_smoke.c"
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
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/native-split-smoke"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=12 -o ConnectTimeout=10"))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]
SSH_NO_STDIN_CMD = ["ssh", "-n", "-T", *SSH_OPTS, REMOTE_HOST]
RSYNC_RSH = "ssh " + shlex.join(SSH_OPTS)

MAGIC = 0x57485350
K_DIM = 384
N_COLS = 1024
LN_DIM = 384

OFFSETS = {
    "logits_out": 0x4000,
    "act": 0x20000,
    "wt": 0x30000,
    "logits_ref": 0x1B0000,
    "ln_in": 0x1C0000,
    "ln_ref": 0x1D0000,
    "ln_out": 0x1E0000,
    "ln_partial": 0x1F0000,
    "ln_param": 0x1F1000,
}

SMOKE_MODES = {
    "both": 0,
    "vpu-only": 1,
    "scalar-only": 2,
    "barrier-only": 3,
}

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


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def run_retry(cmd: list[str], attempts: int = 5, delay_s: float = 5.0,
              **kwargs: Any) -> subprocess.CompletedProcess:
    last: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None

    for attempt in range(1, attempts + 1):
        try:
            return run(cmd, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last = exc
            print(f"network command failed attempt {attempt}/{attempts}: {exc}",
                  flush=True)
            if attempt < attempts:
                time.sleep(delay_s)

    assert last is not None
    raise last


def f32(x: Any) -> np.float32:
    return np.float32(x)


def fast_inv_sqrtf(x: np.float32) -> np.float32:
    y = np.asarray([x], dtype=np.float32)
    u = y.view(np.uint32)
    u[...] = np.uint32(0x5F3759DF) - (u >> np.uint32(1))
    y = u.view(np.float32)
    y[...] = f32(y[0] * f32(1.5 - f32(0.5) * x * y[0] * y[0]))
    y[...] = f32(y[0] * f32(1.5 - f32(0.5) * x * y[0] * y[0]))
    return f32(y[0])


def layernorm_ref_like_device(x: np.ndarray, scalar_harts: int) -> np.ndarray:
    partial = []
    for h in range(scalar_harts):
        i0 = (LN_DIM * h) // scalar_harts
        i1 = (LN_DIM * (h + 1)) // scalar_harts
        s = f32(0.0)
        ss = f32(0.0)
        for v in x[i0:i1].astype(np.float32):
            s = f32(s + v)
            ss = f32(ss + f32(v * v))
        partial.append((s, ss, i1 - i0))
    total = f32(0.0)
    total_sq = f32(0.0)
    count = 0
    for s, ss, c in partial:
        total = f32(total + s)
        total_sq = f32(total_sq + ss)
        count += c
    del count
    inv_count = f32(1.0 / LN_DIM)
    mean = f32(total * inv_count)
    var = f32(f32(total_sq * inv_count) - f32(mean * mean))
    inv = fast_inv_sqrtf(f32(var + f32(0.00001)))
    out = np.empty_like(x, dtype=np.float32)
    for i, v in enumerate(x.astype(np.float32)):
        out[i] = f32(f32(v - mean) * inv)
    return out


def make_inputs(run_dir: Path, active_harts: int) -> dict[str, Any]:
    rng = np.random.default_rng(0x57484953)
    act = rng.normal(0.0, 0.35, size=(K_DIM,)).astype(np.float32)
    wt = rng.normal(0.0, 0.05, size=(N_COLS, K_DIM)).astype(np.float32)
    logits_ref = (wt.astype(np.float32) @ act.astype(np.float32)).astype(np.float32)
    ln_in = rng.normal(0.0, 0.75, size=(LN_DIM,)).astype(np.float32)
    ln_ref = layernorm_ref_like_device(ln_in, active_harts // 2)

    paths = {
        "act": run_dir / "act.bin",
        "wt": run_dir / "wt.bin",
        "logits_ref": run_dir / "logits_ref.bin",
        "ln_in": run_dir / "ln_in.bin",
        "ln_ref": run_dir / "ln_ref.bin",
    }
    act.astype("<f4").tofile(paths["act"])
    wt.astype("<f4").tofile(paths["wt"])
    logits_ref.astype("<f4").tofile(paths["logits_ref"])
    ln_in.astype("<f4").tofile(paths["ln_in"])
    ln_ref.astype("<f4").tofile(paths["ln_ref"])
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "host_logits_argmax": int(logits_ref.argmax()),
    }


def build_elf(run_dir: Path, active_harts: int, stop_phase: int, mode: str) -> Path:
    elf = run_dir / "whisper_native_hart_split_smoke.elf"
    cmd = [
        GCC,
        "-O3",
        "-funroll-loops",
        *BASE_FLAGS,
        "-DNUM_HARTS=16",
        f"-DACTIVE_HARTS={active_harts}u",
        f"-DK_DIM={K_DIM}u",
        f"-DN_COLS={N_COLS}u",
        f"-DLN_DIM={LN_DIM}u",
        f"-DSMOKE_STOP_PHASE={stop_phase}u",
        f"-DSMOKE_MODE={SMOKE_MODES[mode]}u",
        "-o",
        str(elf),
        str(SRC),
        str(CRT),
        str(LAYOUT),
    ]
    run(cmd)
    return elf


def parse_dump(run_dir: Path, active_harts: int, host_logits_argmax: int) -> dict[str, Any]:
    dump = (run_dir / "dump.bin").read_bytes()
    summary = struct.unpack_from("<28I", dump, 0x1000)
    slot_struct = struct.Struct("<16I")
    slots = []
    for h in range(active_harts):
        raw = slot_struct.unpack_from(dump, h * 64)
        slots.append({
            "magic": hex(raw[0]),
            "hart_id": raw[1],
            "minion_id": raw[2],
            "thread_id": raw[3],
            "role": raw[4],
            "work0": raw[5],
            "work1": raw[6],
            "checksum": raw[7],
            "done": raw[8],
        })
    text = (run_dir / "run.log").read_text(errors="ignore")
    wait = None
    m = re.search(r"Kernel wait seconds:\s*([0-9.eE+-]+)", text)
    if m:
        wait = float(m.group(1))
    report = {
        "magic": hex(summary[0]),
        "active_harts": summary[1],
        "vpu_harts": summary[2],
        "scalar_harts": summary[3],
        "vpu_done": summary[4],
        "scalar_done": summary[5],
        "vpu_active_mask": hex(summary[6]),
        "scalar_active_mask": hex(summary[7]),
        "logits_argmax": summary[8],
        "host_logits_argmax": host_logits_argmax,
        "logits_argmax_match": summary[8] == host_logits_argmax,
        "logits_max_abs": summary[9] / 1_000_000.0,
        "ln_max_abs": summary[10] / 1_000_000.0,
        "ln_mean_abs": summary[11] / 1_000_000.0,
        "ops": summary[12] | (summary[13] << 32),
        "phase": summary[14],
        "mode_id": summary[15],
        "hpmcounter3_cycles": summary[16] | (summary[17] << 32),
        "hpmcounter4_retired_inst0": summary[18] | (summary[19] << 32),
        "hpmcounter5_retired_inst1": summary[20] | (summary[21] << 32),
        "hpmcounter6_l2_miss_request": summary[22] | (summary[23] << 32),
        "hpmcounter7_minion_icache_request": summary[24] | (summary[25] << 32),
        "hpmcounter8_icache_etlink_request": summary[26] | (summary[27] << 32),
        "kernel_wait_s": wait,
        "stream_error": "Stream error" in text,
        "kernel_launch_error": "Error on kernel launch" in text,
        "slots": slots,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--stop-phase", type=int, default=0,
                    help="0 runs the full smoke; 1..8 returns at a checkpoint")
    ap.add_argument("--mode", choices=sorted(SMOKE_MODES), default="both")
    args = ap.parse_args()
    if args.active_harts < 2 or args.active_harts > 16 or args.active_harts % 2:
        raise SystemExit("this role split smoke test expects an even hart count in 2..16")
    if args.stop_phase < 0 or args.stop_phase > 8:
        raise SystemExit("--stop-phase must be in 0..8")

    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{args.mode}_phase{args.stop_phase}_ah{args.active_harts}_{ts}"
    run_dir = PHASE0 / "native_split_smoke_runs" / f"run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs = make_inputs(run_dir, args.active_harts)
    elf = build_elf(run_dir, args.active_harts, args.stop_phase, args.mode)
    remote = f"{REMOTE_ROOT}/{run_dir.name}"
    run_retry([*SSH_NO_STDIN_CMD, "true"], timeout=30)
    run_retry([*SSH_NO_STDIN_CMD, f"mkdir -p {shlex.quote(remote)}"], timeout=30)
    run_retry([
        "rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
        str(elf),
        inputs["paths"]["act"],
        inputs["paths"]["wt"],
        inputs["paths"]["logits_ref"],
        inputs["paths"]["ln_in"],
        inputs["paths"]["ln_ref"],
        f"{REMOTE_HOST}:{remote}/",
    ], attempts=5, delay_s=5.0)

    script = f"""set -uo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={remote}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero2m.bin
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"$LAUNCH" --elf-load ./whisper_native_hart_split_smoke.elf --shire 0 --file_load 0x0,$ZERO \\
  --file_load 0x{OFFSETS['act']:x},act.bin \\
  --file_load 0x{OFFSETS['wt']:x},wt.bin \\
  --file_load 0x{OFFSETS['logits_ref']:x},logits_ref.bin \\
  --file_load 0x{OFFSETS['ln_in']:x},ln_in.bin \\
  --file_load 0x{OFFSETS['ln_ref']:x},ln_ref.bin \\
  --dump_after dump.bin --timeout {args.timeout} > run.log 2>&1
LAUNCH_RC=$?
grep -oE 'Kernel wait seconds: [0-9.]+' run.log | tail -1 || true
exit $LAUNCH_RC
"""
    launch = subprocess.run([*SSH_CMD, "bash", "-s"], input=script,
                            timeout=args.timeout + 120, text=True)
    with (run_dir / "run.log").open("wb") as f:
        fetch_rc = subprocess.run(
            [*SSH_NO_STDIN_CMD, f"cat {shlex.quote(remote)}/run.log"],
            stdout=f,
            timeout=60,
        ).returncode
    with (run_dir / "dump.bin").open("wb") as f:
        dump_fetch_rc = subprocess.run(
            [*SSH_NO_STDIN_CMD,
             f"dd if={shlex.quote(remote)}/dump.bin bs=4096 count=2 2>/dev/null"],
            stdout=f,
            timeout=60,
        ).returncode
    (run_dir / "launch_returncode.txt").write_text(
        f"launch_rc={launch.returncode}\nfetch_rc={fetch_rc}\ndump_fetch_rc={dump_fetch_rc}\n")
    if launch.returncode != 0:
        log_tail = ""
        log_path = run_dir / "run.log"
        if log_path.exists():
            log_tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-80:])
        raise SystemExit(
            f"launcher failed with rc={launch.returncode}; local run dir: {run_dir}\n{log_tail}")
    if dump_fetch_rc != 0:
        raise SystemExit(f"launcher returned success but dump.bin was not fetched; local run dir: {run_dir}")
    report = parse_dump(run_dir, args.active_harts, inputs["host_logits_argmax"])
    report["mode"] = args.mode
    report["stop_phase"] = args.stop_phase
    report["remote_work"] = remote
    report["local_run_dir"] = str(run_dir)
    report["source"] = str(SRC)
    if args.stop_phase:
        report["ok"] = (
            report["magic"] == hex(MAGIC)
            and report["active_harts"] == args.active_harts
            and report["phase"] == args.stop_phase
            and report["mode_id"] == SMOKE_MODES[args.mode]
            and not report["stream_error"]
            and not report["kernel_launch_error"]
        )
    elif args.mode == "both":
        report["ok"] = (
            report["magic"] == hex(MAGIC)
            and report["active_harts"] == args.active_harts
            and report["vpu_done"] == args.active_harts // 2
            and report["scalar_done"] == args.active_harts // 2
            and report["logits_argmax_match"]
            and report["logits_max_abs"] <= 1e-4
            and report["ln_max_abs"] <= 1e-3
        )
    elif args.mode == "vpu-only":
        report["ok"] = (
            report["magic"] == hex(MAGIC)
            and report["active_harts"] == args.active_harts
            and report["vpu_done"] == args.active_harts // 2
            and report["scalar_done"] == 0
            and report["logits_argmax_match"]
            and report["logits_max_abs"] <= 1e-4
        )
    elif args.mode == "scalar-only":
        report["ok"] = (
            report["magic"] == hex(MAGIC)
            and report["active_harts"] == args.active_harts
            and report["scalar_done"] == args.active_harts // 2
            and report["ln_max_abs"] <= 1e-3
        )
    else:
        report["ok"] = report["magic"] == hex(MAGIC) and report["active_harts"] == args.active_harts
    (run_dir / "native_split_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "NATIVE_SPLIT_SMOKE_REPORT.md").write_text(
        "# Whisper Native Hart Split Smoke Report\n\n"
        f"- Local run dir: `{run_dir}`\n"
        f"- Remote work dir: `{remote}`\n"
        f"- Status: `{'ok' if report['ok'] else 'fail'}`\n"
        f"- VPU role harts done: `{report['vpu_done']}/{report['vpu_harts']}` "
        f"mask `{report['vpu_active_mask']}`\n"
        f"- Scalar role harts done: `{report['scalar_done']}/{report['scalar_harts']}` "
        f"mask `{report['scalar_active_mask']}`\n"
        f"- Logits argmax: silicon `{report['logits_argmax']}`, host `{report['host_logits_argmax']}`\n"
        f"- Logits max abs: `{report['logits_max_abs']}`\n"
        f"- LayerNorm-shaped max abs: `{report['ln_max_abs']}`\n"
        f"- Kernel wait seconds: `{report['kernel_wait_s']}`\n"
        f"- hpmcounter3 cycles: `{report['hpmcounter3_cycles']}`\n"
        f"- hpmcounter4 retired inst0: `{report['hpmcounter4_retired_inst0']}`\n"
        f"- hpmcounter5 retired inst1: `{report['hpmcounter5_retired_inst1']}`\n"
        f"- hpmcounter6 L2 miss requests: `{report['hpmcounter6_l2_miss_request']}`\n"
        f"- hpmcounter7 minion icache requests: `{report['hpmcounter7_minion_icache_request']}`\n"
        f"- hpmcounter8 icache etlink requests: `{report['hpmcounter8_icache_etlink_request']}`\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
