#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import struct
from collections import defaultdict

SNAP_DIRECT_MAGIC = 0x504D5344
SNAP_SYSCALL_MAGIC = 0x504D5359
DIRECT_STRUCT = struct.Struct("<8I6Q")
SYSCALL_STRUCT = struct.Struct("<8IQ")
SUMMARY_STRUCT = struct.Struct("<16I")

CASES = [
    {
        "model": "dncnn",
        "work": "dncnn3-pmc",
        "variant": "v3100_o08_ofast_m09_accfinal_prefw",
        "magic": 0xD3C11003,
        "summary": "dncnn",
    },
    {
        "model": "depth",
        "work": "depth-vit-bench",
        "variant": "d100_o02_o3_unroll_m07_prefb_skipread",
        "magic": 0xD3717001,
        "summary": "transformer",
    },
    {
        "model": "whisper",
        "work": "whisper-bench",
        "variant": "w100_o02_o3_unroll_m04_prefa_prefb",
        "magic": 0x57485350,
        "summary": "transformer",
    },
    {
        "model": "stereo",
        "work": "stereo-bench",
        "variant": "s100_o02_o3_unroll_m09_prefl_lastout",
        "magic": 0x53773001,
        "summary": "stereo",
    },
    {
        "model": "yolo",
        "work": "yolo-bench",
        "variant": "y100_o08_ofast_m03_skipread",
        "magic": 0x10500001,
        "summary": "yolo",
    },
]


def wait_seconds(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"Kernel wait seconds: ([0-9.]+)", text)
    return float(match.group(1)) if match else 0.0


def delta_u64(after, before):
    return (after - before) & ((1 << 64) - 1)


def read_snapshot(path):
    with open(path, "rb") as f:
        data = f.read()

    direct = {}
    for h in range(16):
        fields = DIRECT_STRUCT.unpack_from(data, h * DIRECT_STRUCT.size)
        if fields[0] != SNAP_DIRECT_MAGIC:
            continue
        direct[fields[1]] = {
            "hart": fields[1],
            "minion": fields[2],
            "thread": fields[3],
            "hpm": fields[8:14],
        }

    syscalls = {}
    base = 0x1000
    for i in range(36):
        fields = SYSCALL_STRUCT.unpack_from(data, base + i * SYSCALL_STRUCT.size)
        if fields[0] != SNAP_SYSCALL_MAGIC:
            continue
        _, kind, block, pmc, _, _, _, _, value = fields
        syscalls[(kind, block, pmc)] = value

    return direct, syscalls


def read_summary(path, case):
    with open(path, "rb") as f:
        data = f.read(0x1000 + SUMMARY_STRUCT.size)
    fields = SUMMARY_STRUCT.unpack_from(data, 0x1000)
    kind = case["summary"]

    common = {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
    }

    if kind == "dncnn":
        common.update({
            "active_mask": fields[7],
            "done_count": fields[8],
            "output_sum": fields[9],
            "slot_sum": fields[10],
            "ops": fields[11] | (fields[12] << 32),
        })
    elif kind == "transformer":
        common.update({
            "tokens": fields[3],
            "dim": fields[4],
            "hidden": fields[5],
            "active_mask": fields[6],
            "done_count": fields[7],
            "output_sum": fields[8],
            "slot_sum": fields[9],
            "ops": fields[10] | (fields[11] << 32),
        })
    elif kind == "stereo":
        common.update({
            "width": fields[3],
            "height": fields[4],
            "channels": fields[5],
            "disparities": fields[6],
            "active_mask": fields[7],
            "done_count": fields[8],
            "output_sum": fields[9],
            "slot_sum": fields[10],
            "ops": fields[11] | (fields[12] << 32),
        })
    elif kind == "yolo":
        common.update({
            "width": fields[3],
            "height": fields[4],
            "channels": fields[5],
            "blocks": fields[6],
            "active_mask": fields[7],
            "done_count": fields[8],
            "output_sum": fields[9],
            "slot_sum": fields[10],
            "ops": fields[11] | (fields[12] << 32),
            "head_channels": fields[13],
        })
    return common


def direct_metrics(before, after, active_harts):
    rows = []
    by_minion = defaultdict(dict)
    for h in range(active_harts):
        if h not in before or h not in after:
            continue
        d = [delta_u64(after[h]["hpm"][i], before[h]["hpm"][i]) for i in range(6)]
        rows.append((h, before[h]["minion"], before[h]["thread"], d))
        by_minion[before[h]["minion"]][before[h]["thread"]] = d

    dedup_inst = 0
    for threads in by_minion.values():
        if 0 in threads:
            dedup_inst += threads[0][1]
        if 1 in threads:
            dedup_inst += threads[1][2]

    return {
        "hpm_cycles_max": max((d[0] for _, _, _, d in rows), default=0),
        "hpm_cycles_sum": sum(d[0] for _, _, _, d in rows),
        "hpm_inst_dedup": dedup_inst,
        "hpm_l2_miss_sum": sum(d[3] for _, _, _, d in rows),
        "hpm_icache_req_sum": sum(d[4] for _, _, _, d in rows),
        "hpm_icache_etlink_sum": sum(d[5] for _, _, _, d in rows),
    }


def syscall_metrics(before, after):
    out = {}
    for label, kind in (("sc", 0), ("ms", 1)):
        for pmc_label, pmc in (("cycles", 0), ("reads", 1), ("writes", 2)):
            vals = [
                delta_u64(after[k], v)
                for k, v in before.items()
                if k[0] == kind and k[2] == pmc and k in after
            ]
            out[f"{label}_{pmc_label}_sum"] = sum(vals)
            out[f"{label}_{pmc_label}_max"] = max(vals) if vals else 0
    return out


def latest_case_paths(root, case):
    work = os.path.join(root, case["work"])
    pattern = os.path.join(work, f"run_pmc_{case['model']}_{case['variant']}_*.log")
    logs = sorted(glob.glob(pattern))
    if not logs:
        return None

    log = logs[-1]
    stamp = os.path.basename(log)
    stamp = stamp.removeprefix(f"run_pmc_{case['model']}_{case['variant']}_")
    stamp = stamp.removesuffix(".log")

    return {
        "log": log,
        "dump": os.path.join(work, f"dump_pmc_{case['model']}_{case['variant']}_{stamp}.bin"),
        "before": os.path.join(work, f"pmc_before_{case['model']}_{case['variant']}_{stamp}.bin"),
        "after": os.path.join(work, f"pmc_after_{case['model']}_{case['variant']}_{stamp}.bin"),
        "stamp": stamp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="local-artifacts/erbium_amp_probe")
    parser.add_argument("--out", default="local-artifacts/erbium_amp_probe/optimization-kb/best_pmc_results.tsv")
    args = parser.parse_args()

    rows = []
    for case in CASES:
        paths = latest_case_paths(args.root, case)
        if paths is None or not all(os.path.exists(paths[k]) for k in ("dump", "before", "after")):
            continue

        summary = read_summary(paths["dump"], case)
        before_direct, before_sys = read_snapshot(paths["before"])
        after_direct, after_sys = read_snapshot(paths["after"])
        wait = wait_seconds(paths["log"])
        ops = summary["ops"]

        row = {
            "model": case["model"],
            "variant": case["variant"],
            "stamp": paths["stamp"],
            "wait_s": f"{wait:.6f}",
            "gops": f"{ops / wait / 1e9:.6f}" if wait else "0.000000",
            "valid": int(
                summary["magic"] == case["magic"]
                and summary["active_harts"] == summary["done_count"]
                and summary["output_sum"] == summary["slot_sum"]
            ),
            "passes": summary["passes"],
            "harts": summary["active_harts"],
            "mask": hex(summary["active_mask"]),
            "output_sum": summary["output_sum"],
            "ops": ops,
            "log": os.path.relpath(paths["log"], args.root),
            "dump": os.path.relpath(paths["dump"], args.root),
        }
        row.update(direct_metrics(before_direct, after_direct, summary["active_harts"]))
        row.update(syscall_metrics(before_sys, after_sys))
        if ops:
            row["cycles_per_op_max"] = f"{row['hpm_cycles_max'] / ops:.9f}"
            row["inst_per_op_dedup"] = f"{row['hpm_inst_dedup'] / ops:.9f}"
            row["l2_miss_per_kop"] = f"{row['hpm_l2_miss_sum'] * 1000.0 / ops:.6f}"
            row["sc_reads_per_kop"] = f"{row['sc_reads_sum'] * 1000.0 / ops:.6f}"
            row["sc_writes_per_kop"] = f"{row['sc_writes_sum'] * 1000.0 / ops:.6f}"
            row["ms_reads_per_kop"] = f"{row['ms_reads_sum'] * 1000.0 / ops:.6f}"
            row["ms_writes_per_kop"] = f"{row['ms_writes_sum'] * 1000.0 / ops:.6f}"
        rows.append(row)

    if not rows:
        raise SystemExit("no PMC rows parsed")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
