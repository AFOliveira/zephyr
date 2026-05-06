#!/usr/bin/env python3
"""Tabulate the audited DnCNN3 sweep: per-variant correctness vs ORT + silicon throughput.

For each dump_a_tile<T>_<variant>_<harts>h_<stamp>.bin under audited-sweep/remote-logs/,
extract:
  - silicon summary at offset 0x1000 (magic, dims, output_hash, reference_hash, ops, max_abs, mean_abs)
  - wait_s from the run log
and emit one TSV with: variant, tile, harts, wait_s, gops, allclose, bit_exact,
output_hash, reference_hash, ops, max_abs, mean_abs, status.

The reference_hash in the summary is computed on-device from the static ORT output
blob (DNCNN_REF_BLOB) — so equality with output_hash is the silicon-vs-ORT bit-exact
gate already baked into the kernel. No external onnxruntime call is needed for this audit.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

SUMMARY_STRUCT = struct.Struct("<16I")
DNCNN_MAGIC = 0xD3C11004
NAME_RE = re.compile(r"dump_[ah]_tile(\d+)_(.+)_(\d+)h_(\d{8}-\d{6})\.bin$")


def parse_summary(dump: Path) -> dict:
    with dump.open("rb") as f:
        f.seek(0x1000)
        data = f.read(SUMMARY_STRUCT.size)
    fields = SUMMARY_STRUCT.unpack(data)
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


def parse_log(log: Path) -> tuple[float | None, str]:
    if not log.exists():
        return None, "log_missing"
    text = log.read_text(errors="ignore")
    m = re.search(r"Kernel wait seconds:\s*([0-9.eE+-]+)", text)
    if not m:
        if "Segmentation fault" in text or "FATAL_SIGNAL" in text:
            return None, "host_segv"
        return None, "no_wait_s"
    return float(m.group(1)), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path,
                    default=Path(__file__).parent / "audited-sweep" / "remote-logs")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--out-tsv", type=Path,
                    default=Path(__file__).parent / "audited-sweep" / "audit_results.tsv")
    args = ap.parse_args()

    rows: list[dict] = []
    for dump in sorted(args.logs_dir.glob("dump_*_tile*.bin")):
        m = NAME_RE.match(dump.name)
        if not m:
            continue
        tile, variant, harts, stamp = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        prefix = dump.name.split("_", 2)[1]  # 'a' or 'h'
        log = args.logs_dir / f"run_{prefix}_tile{tile}_{variant}_{harts}h_{stamp}.log"
        wait_s, log_status = parse_log(log)

        s = parse_summary(dump)
        magic_ok = s["magic"] == DNCNN_MAGIC
        bit_exact = magic_ok and s["output_hash"] == s["reference_hash"]
        allclose = magic_ok and s["max_abs"] <= args.atol and s["done_count"] == s["active_harts"]

        gops = (s["ops"] / wait_s / 1e9) if (wait_s and wait_s > 0 and s["ops"]) else 0.0

        rows.append({
            "variant": variant,
            "tile": f"{tile}x{tile}",
            "harts": harts,
            "wait_s": f"{wait_s:.6f}" if wait_s is not None else "-",
            "gops": f"{gops:.4f}" if gops else "-",
            "allclose": str(allclose),
            "bit_exact": str(bit_exact),
            "output_hash": f"0x{s['output_hash']:08x}",
            "reference_hash": f"0x{s['reference_hash']:08x}",
            "ops": s["ops"],
            "max_abs": f"{s['max_abs']:.3e}",
            "mean_abs": f"{s['mean_abs']:.3e}",
            "magic_ok": str(magic_ok),
            "log_status": log_status,
            "dump": dump.name,
        })

    if not rows:
        print("no dumps found", file=sys.stderr)
        return 1

    cols = ["variant", "tile", "harts", "wait_s", "gops", "allclose", "bit_exact",
            "max_abs", "mean_abs", "output_hash", "reference_hash", "ops",
            "magic_ok", "log_status", "dump"]
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Pretty-print sorted by tile then GOPS desc
    def sort_key(r):
        try: g = float(r["gops"])
        except: g = 0.0
        return (r["tile"], r["harts"], -g)

    rows.sort(key=sort_key)
    print(f"{'variant':24s} {'tile':>6s} {'h':>2s} {'wait_s':>10s} {'GOPS':>8s} "
          f"{'allclose':>9s} {'bit_exact':>10s} {'max_abs':>10s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['variant']:24s} {r['tile']:>6s} {r['harts']:>2d} {r['wait_s']:>10s} "
              f"{r['gops']:>8s} {r['allclose']:>9s} {r['bit_exact']:>10s} {r['max_abs']:>10s}")

    print(f"\nwrote {args.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
