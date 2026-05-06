#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import struct

SUMMARY = struct.Struct("<16I")

MODEL_INFO = {
    "dncnn": {"magic": 0xD3C11003, "kind": "dncnn"},
    "depth": {"magic": 0xD3717001, "kind": "transformer"},
    "whisper": {"magic": 0x57485350, "kind": "transformer"},
    "stereo": {"magic": 0x53773001, "kind": "stereo"},
    "yolo": {"magic": 0x10500001, "kind": "yolo"},
}


def wait_seconds(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"Kernel wait seconds: ([0-9.]+)", text)
    return float(match.group(1)) if match else 0.0


def read_variants(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            model, variant = line.split("\t")
            out.append((model, variant))
    return out


def read_summary(path, model):
    fields = SUMMARY.unpack_from(open(path, "rb").read(0x1000 + SUMMARY.size), 0x1000)
    kind = MODEL_INFO[model]["kind"]
    out = {"magic": fields[0], "active_harts": fields[1], "passes": fields[2]}
    if kind == "dncnn":
        out.update({"active_mask": fields[7], "done_count": fields[8], "output_sum": fields[9],
                    "slot_sum": fields[10], "ops": fields[11] | (fields[12] << 32)})
    elif kind == "transformer":
        out.update({"active_mask": fields[6], "done_count": fields[7], "output_sum": fields[8],
                    "slot_sum": fields[9], "ops": fields[10] | (fields[11] << 32)})
    elif kind == "stereo":
        out.update({"active_mask": fields[7], "done_count": fields[8], "output_sum": fields[9],
                    "slot_sum": fields[10], "ops": fields[11] | (fields[12] << 32)})
    elif kind == "yolo":
        out.update({"active_mask": fields[7], "done_count": fields[8], "output_sum": fields[9],
                    "slot_sum": fields[10], "ops": fields[11] | (fields[12] << 32)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="local-artifacts/erbium_amp_probe/optimization-kb/focused20")
    parser.add_argument("--out", default="local-artifacts/erbium_amp_probe/optimization-kb/focused20_results.tsv")
    args = parser.parse_args()

    rows = []
    for model, variant in read_variants(os.path.join(args.root, "focused20_variants.tsv")):
        logs = sorted(glob.glob(os.path.join(args.root, f"run_{model}_{variant}_*.log")))
        dumps = sorted(glob.glob(os.path.join(args.root, f"dump_{model}_{variant}_*.bin")))
        if not logs or not dumps:
            continue
        wait = wait_seconds(logs[-1])
        summary = read_summary(dumps[-1], model)
        ops = summary["ops"]
        valid = (
            summary["magic"] == MODEL_INFO[model]["magic"]
            and summary["active_harts"] == summary["done_count"]
            and summary["output_sum"] == summary["slot_sum"]
        )
        rows.append({
            "model": model,
            "variant": variant,
            "wait_s": f"{wait:.6f}",
            "gops": f"{ops / wait / 1e9:.6f}" if wait else "0.000000",
            "valid": int(valid),
            "passes": summary["passes"],
            "harts": summary["active_harts"],
            "mask": hex(summary["active_mask"]),
            "output_sum": summary["output_sum"],
            "ops": ops,
            "log": os.path.relpath(logs[-1], os.path.dirname(args.out)),
            "dump": os.path.relpath(dumps[-1], os.path.dirname(args.out)),
        })

    rows.sort(key=lambda row: (row["model"], -float(row["gops"])))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
