#!/usr/bin/env python3
import argparse
import csv
import glob
import os

from parse_best_pmc_suite import (
    direct_metrics,
    read_snapshot,
    read_summary,
    syscall_metrics,
    wait_seconds,
)

CASES = [
    {
        "model": "dncnn",
        "work": "optimization-kb/focused20b",
        "variant": "f20b_dncnn_ofast_rename",
        "magic": 0xD3C11003,
        "summary": "dncnn",
    },
    {
        "model": "depth",
        "work": "optimization-kb/focused20b",
        "variant": "f20b_depth_o3_unroll_noipa_prefb_skipread",
        "magic": 0xD3717001,
        "summary": "transformer",
    },
    {
        "model": "whisper",
        "work": "optimization-kb/focused20",
        "variant": "f20_whisper_o3_unroll_prefa_prefb_skipread",
        "magic": 0x57485350,
        "summary": "transformer",
    },
    {
        "model": "stereo",
        "work": "optimization-kb/focused20",
        "variant": "f20_stereo_ofast_unroll_prefl_lastout",
        "magic": 0x53773001,
        "summary": "stereo",
    },
    {
        "model": "yolo",
        "work": "optimization-kb/focused20b",
        "variant": "f20b_yolo_ofast_unroll_align32_skipread",
        "magic": 0x10500001,
        "summary": "yolo",
    },
]


def latest_case_paths(root, case):
    work = os.path.join(root, case["work"])
    pattern = os.path.join(work, f"run_pmc_current_{case['model']}_{case['variant']}_*.log")
    logs = sorted(glob.glob(pattern))
    if not logs:
        return None

    log = logs[-1]
    stamp = os.path.basename(log)
    prefix = f"run_pmc_current_{case['model']}_{case['variant']}_"
    if stamp.startswith(prefix):
        stamp = stamp[len(prefix):]
    if stamp.endswith(".log"):
        stamp = stamp[:-4]

    return {
        "log": log,
        "dump": os.path.join(work, f"dump_pmc_current_{case['model']}_{case['variant']}_{stamp}.bin"),
        "before": os.path.join(work, f"pmc_before_current_{case['model']}_{case['variant']}_{stamp}.bin"),
        "after": os.path.join(work, f"pmc_after_current_{case['model']}_{case['variant']}_{stamp}.bin"),
        "stamp": stamp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="local-artifacts/erbium_amp_probe")
    parser.add_argument("--out", default="local-artifacts/erbium_amp_probe/optimization-kb/current_best_pmc_results.tsv")
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
        raise SystemExit("no current best PMC rows parsed")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
