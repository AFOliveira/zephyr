#!/usr/bin/env python3
"""Build and run the local Erbium AMP conv3x3 probe."""

from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path


DEVICE_BUFFER_SIZE = 16 * 1024 * 1024
IMG_W = 128
IMG_H = 128
IMG_BYTES = IMG_W * IMG_H
SLOT_BYTES = 64
SUMMARY_BYTES = 64
SUMMARY_OFFSET_FROM_END = 0x40
CONV_MAGIC = 0xC03A3A01


def generate_input() -> bytes:
    data = bytearray(IMG_BYTES)
    state = 0x12345678
    for i in range(IMG_BYTES):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        y = i // IMG_W
        x = i % IMG_W
        data[i] = ((state >> 24) + x * 3 + y * 5) & 0xFF
    return bytes(data)


def pixel(image: bytes, y: int, x: int) -> int:
    y = min(max(y, 0), IMG_H - 1)
    x = min(max(x, 0), IMG_W - 1)
    return image[y * IMG_W + x]


def reference_output(image: bytes) -> bytes:
    out = bytearray(IMG_BYTES)
    for y in range(IMG_H):
        for x in range(IMG_W):
            acc = 0
            acc += pixel(image, y - 1, x - 1)
            acc += 2 * pixel(image, y - 1, x)
            acc += pixel(image, y - 1, x + 1)
            acc += 2 * pixel(image, y, x - 1)
            acc += 4 * pixel(image, y, x)
            acc += 2 * pixel(image, y, x + 1)
            acc += pixel(image, y + 1, x - 1)
            acc += 2 * pixel(image, y + 1, x)
            acc += pixel(image, y + 1, x + 1)
            out[y * IMG_W + x] = (acc + 8) >> 4
    return bytes(out)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def layout(active_harts: int) -> tuple[int, int, int, int]:
    slots_bytes = active_harts * SLOT_BYTES
    slots_offset = SUMMARY_OFFSET_FROM_END + slots_bytes
    output_offset = slots_offset + IMG_BYTES
    input_offset = output_offset + IMG_BYTES
    return slots_bytes, slots_offset, output_offset, input_offset


def build_probe(outdir: Path, args: argparse.Namespace, active_harts: int) -> Path:
    elf = outdir / "conv3x3_amp.elf"
    src = Path(__file__).with_name("conv3x3_amp.c")
    gcc = Path(args.toolchain) / "bin" / "riscv64-unknown-elf-gcc"
    et_install = Path(args.et_install)

    barrier_t1 = "0xff" if active_harts > 8 else "0x00"
    cmd = [
        str(gcc),
        "-march=rv64imfc",
        "-mabi=lp64f",
        "-mcmodel=medany",
        "-nostdlib",
        "-fno-zero-initialized-in-bss",
        "-ffunction-sections",
        "-fdata-sections",
        f"-DACTIVE_HARTS={active_harts}u",
        f"-DBARRIER_MASK_T1={barrier_t1}u",
        "-I",
        str(et_install / "erbium-umode/include"),
        "-I",
        str(et_install / "include/esperanto-fw/erbium_hal"),
        "-Wl,--gc-sections",
        "-Wl,--no-warn-rwx-segments",
        "-T",
        str(et_install / "erbium-umode/share/erbium.ld"),
        "-o",
        str(elf),
        str(src),
        str(Path(args.erbium_examples) / "runtime/erbium/boot.S"),
        str(Path(args.erbium_examples) / "runtime/erbium/crt.S"),
        str(Path(args.erbium_examples) / "runtime/erbium/layout.c"),
    ]
    run(cmd)
    return elf


def parse_summary(dump: bytes) -> tuple[int, ...]:
    start = DEVICE_BUFFER_SIZE - SUMMARY_OFFSET_FROM_END
    return struct.unpack_from("<16I", dump, start)


def parse_slots(dump: bytes, active_harts: int, slots_offset: int) -> list[tuple[int, ...]]:
    start = DEVICE_BUFFER_SIZE - slots_offset
    return [
        struct.unpack_from("<16I", dump, start + h * SLOT_BYTES)
        for h in range(active_harts)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path(__file__).with_name("conv3x3-run"))
    ap.add_argument("--et-install", default="/tmp/et-vidas-install")
    ap.add_argument("--erbium-examples", default="/home/afonso/et-platform-vidas/erbium-examples")
    ap.add_argument("--toolchain", default="/home/afonso/et")
    ap.add_argument("--launcher", default="/home/afonso/et-platform-vidas/soc1sim/scripts/erbium_run")
    ap.add_argument("--emulator", default="/home/afonso/et-platform/build-emu/erbium_emu")
    ap.add_argument("--max-cycles", default="500000000")
    ap.add_argument("--active-harts", type=int, choices=(8, 16), default=8)
    ap.add_argument("--dual-thread", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    active_harts = args.active_harts
    if active_harts == 16 and not args.dual_thread:
        raise SystemExit("--active-harts 16 requires --dual-thread")
    _, slots_offset, output_offset, input_offset_from_end = layout(active_harts)

    image = generate_input()
    ref = reference_output(image)
    input_bin = outdir / "input.bin"
    ref_bin = outdir / "reference_output.bin"
    dump_bin = outdir / "dump_after.bin"
    input_bin.write_bytes(image)
    ref_bin.write_bytes(ref)

    elf = build_probe(outdir, args, active_harts)

    input_offset = DEVICE_BUFFER_SIZE - input_offset_from_end
    cmd = [
        args.launcher,
        "--device",
        "erbium_emu",
        "--elf-load",
        str(elf),
        "--file_load",
        f"0x{input_offset:x},{input_bin}",
        "--dump_after",
        str(dump_bin),
        "--emulator",
        args.emulator,
        "--max-cycles",
        args.max_cycles,
        "-v",
    ]
    if args.dual_thread:
        cmd.append("--dual-thread")
    run(cmd)

    dump = dump_bin.read_bytes()
    output_start = DEVICE_BUFFER_SIZE - output_offset
    got = dump[output_start : output_start + IMG_BYTES]
    mismatches = sum(a != b for a, b in zip(got, ref))

    summary = parse_summary(dump)
    slots = parse_slots(dump, active_harts, slots_offset)
    print(f"summary magic=0x{summary[0]:08x} width={summary[1]} height={summary[2]}")
    print(f"active_mask=0x{summary[3]:08x} done_count={summary[4]} output_sum={summary[5]}")
    for h, slot in enumerate(slots):
        print(
            f"slot[{h}] hart={slot[0]} minion={slot[1]} thread={slot[2]} "
            f"row0={slot[3]} rows={slot[4]} checksum={slot[5]} done={slot[6]}"
        )

    expected_sum = sum(ref) & 0xFFFFFFFF
    expected_mask = 0xFFFF if active_harts == 16 else 0x5555
    errors = []
    if summary[0] != CONV_MAGIC:
        errors.append("bad summary magic")
    if summary[3] != expected_mask:
        errors.append(f"unexpected active mask 0x{summary[3]:x}")
    if summary[4] != active_harts:
        errors.append(f"expected {active_harts} done harts, saw {summary[4]}")
    if summary[5] != expected_sum:
        errors.append(f"summary sum {summary[5]} != reference sum {expected_sum}")
    if mismatches:
        errors.append(f"{mismatches} output bytes differ")

    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    (outdir / "device_output.bin").write_bytes(got)
    print("RESULT: PASS")
    print(f"  output bytes match reference ({IMG_W}x{IMG_H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
