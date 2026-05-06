#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import struct

DIRECT_MAGIC = 0x44495250
SYSCALL_MAGIC = 0x53595350
SUMMARY_MAGIC = 0x53554D50
CSR_START_MAGIC = 0x43535253
CSR_DONE_MAGIC = 0x43535244

DIRECT_OFFSET = 0x0000
SYSCALL_OFFSET = 0x4000
SUMMARY_OFFSET = 0x8000
HPM_COUNT = 6
MAX_HARTS = 16
SC_BANKS = 4
MS_COUNT = 8
PMC_PER_BLOCK = 3
STAGES = 3

DIRECT_STRUCT = struct.Struct("<8I33Q")
SYSCALL_STRUCT = struct.Struct("<8IQ")
SUMMARY_STRUCT = struct.Struct("<8I6Q")
CSR_STRUCT = struct.Struct("<8I4Q")

HPM_NAMES = [
    "hpmcounter3_cycles",
    "hpmcounter4_retired_inst0",
    "hpmcounter5_retired_inst1",
    "hpmcounter6_l2_miss_req",
    "hpmcounter7_minion_icache_req",
    "hpmcounter8_icache_etlink_req",
]


def u64_delta(a, b):
    return (b - a) & ((1 << 64) - 1)


def parse_safe(path, outdir):
    with open(path, "rb") as f:
        data = f.read()

    os.makedirs(outdir, exist_ok=True)
    direct_rows = []
    for hart in range(MAX_HARTS):
        off = DIRECT_OFFSET + hart * DIRECT_STRUCT.size
        fields = DIRECT_STRUCT.unpack_from(data, off)
        magic, hart_id, minion_id, thread_id, active_harts, done, res0, res1 = fields[:8]
        values = fields[8:]
        begin = values[0:6]
        after_instr = values[6:12]
        after_mem = values[12:18]
        after_icache = values[18:24]
        final = values[24:30]
        checksums = values[30:33]
        if magic != DIRECT_MAGIC:
            continue
        for i, name in enumerate(HPM_NAMES):
            direct_rows.append({
                "hart": hart_id,
                "minion": minion_id,
                "thread": thread_id,
                "active_harts": active_harts,
                "done": done,
                "counter": name,
                "begin": begin[i],
                "after_instr": after_instr[i],
                "after_mem": after_mem[i],
                "after_icache": after_icache[i],
                "final": final[i],
                "delta_instr": u64_delta(begin[i], after_instr[i]),
                "delta_mem": u64_delta(after_instr[i], after_mem[i]),
                "delta_icache": u64_delta(after_mem[i], after_icache[i]),
                "delta_total": u64_delta(begin[i], final[i]),
                "instr_checksum": checksums[0],
                "mem_checksum": checksums[1],
                "icache_checksum": checksums[2],
            })

    direct_path = os.path.join(outdir, "safe_direct_hpm.tsv")
    with open(direct_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(direct_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(direct_rows)

    syscall_rows = []
    total_syscalls = STAGES * (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK
    for idx in range(total_syscalls):
        off = SYSCALL_OFFSET + idx * SYSCALL_STRUCT.size
        magic, kind, stage, block_id, pmc_id, shire_id, hart_id, reserved, value = (
            SYSCALL_STRUCT.unpack_from(data, off)
        )
        if magic != SYSCALL_MAGIC:
            continue
        syscall_rows.append({
            "idx": idx,
            "kind": "SC" if kind == 0 else "MS",
            "stage": stage,
            "block": block_id,
            "pmc": pmc_id,
            "shire": shire_id,
            "hart": hart_id,
            "value": value,
        })

    syscall_path = os.path.join(outdir, "safe_syscall_pmc.tsv")
    with open(syscall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(syscall_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(syscall_rows)

    summary_fields = SUMMARY_STRUCT.unpack_from(data, SUMMARY_OFFSET)
    magic = summary_fields[0]
    summary_path = os.path.join(outdir, "safe_summary.tsv")
    with open(summary_path, "w", newline="") as f:
        fields = [
            "magic", "active_harts", "done_count", "shire", "hart0",
            "syscall_records", "sc_banks", "ms_count",
        ] + [f"{name}_delta_sum" for name in HPM_NAMES]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        if magic == SUMMARY_MAGIC:
            row = {
                "magic": hex(summary_fields[0]),
                "active_harts": summary_fields[1],
                "done_count": summary_fields[2],
                "shire": summary_fields[3],
                "hart0": summary_fields[4],
                "syscall_records": summary_fields[5],
                "sc_banks": summary_fields[6],
                "ms_count": summary_fields[7],
            }
            for i, name in enumerate(HPM_NAMES):
                row[f"{name}_delta_sum"] = summary_fields[8 + i]
            writer.writerow(row)

    return direct_path, syscall_path, summary_path


def parse_csr_dumps(pattern, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            data = f.read(CSR_STRUCT.size)
        if len(data) < CSR_STRUCT.size:
            continue
        fields = CSR_STRUCT.unpack(data)
        start_magic, done_magic, csr_num, csr_id, hart, minion, thread, reserved = fields[:8]
        before, after, delta, checksum = fields[8:]
        rows.append({
            "file": os.path.basename(path),
            "start_magic": hex(start_magic),
            "done_magic": hex(done_magic),
            "csr_num": hex(csr_num),
            "csr_id": csr_id,
            "hart": hart,
            "minion": minion,
            "thread": thread,
            "status": "done" if start_magic == CSR_START_MAGIC and done_magic == CSR_DONE_MAGIC
                      else "started_only" if start_magic == CSR_START_MAGIC
                      else "no_start_magic",
            "before": before,
            "after": after,
            "delta": delta,
            "checksum": checksum,
        })
    out = os.path.join(outdir, "single_csr_results.tsv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-dump")
    parser.add_argument("--csr-glob")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if args.safe_dump:
        for path in parse_safe(args.safe_dump, args.outdir):
            print(path)
    if args.csr_glob:
        print(parse_csr_dumps(args.csr_glob, args.outdir))


if __name__ == "__main__":
    main()
