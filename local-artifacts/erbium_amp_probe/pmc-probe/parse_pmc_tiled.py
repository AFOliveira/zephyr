#!/usr/bin/env python3
"""Diff before/after pmc_snapshot_16h dumps and combine with tile geometry.

The kernel summary lives at heap0_end-64MB (off-dump for the raw launcher), so
ops are computed from the tile geometry instead of being read from dump.bin.

Reliable signals:
  - HPM[0] (cycles) on T0 harts on minions 0/2/4/6 (and only those — minions
    1/3/5/7 underflow on this snapshot kernel; same artefact as focused20).
  - SYSCALL PMCs: kind 0 = SC (L2/shire-cache), kind 1 = MS (DRAM/main-store);
    pmc 0 = cycles, 1 = reads, 2 = writes.

HPM[1..5] per-hart values are unreliable (overflow / not free-running) — same
issue documented in optimization-kb/current_best_pmc_results.tsv.
"""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path

SNAP_DIRECT_MAGIC  = 0x504D5344
SNAP_SYSCALL_MAGIC = 0x504D5359
DIRECT_STRUCT  = struct.Struct("<8I6Q")
SYSCALL_STRUCT = struct.Struct("<8IQ")

CH = 64
HIDDEN_LAYERS = 18
K = 3


def delta(a: int, b: int) -> int:
    return (a - b) & ((1 << 64) - 1)


def read_snapshot(path: Path):
    data = path.read_bytes()
    direct = {}
    for h in range(16):
        f = DIRECT_STRUCT.unpack_from(data, h * DIRECT_STRUCT.size)
        if f[0] != SNAP_DIRECT_MAGIC:
            continue
        direct[f[1]] = {
            "hart":   f[1], "minion": f[2], "thread": f[3],
            "hpm":    f[8:14],
        }
    syscalls = {}
    base = 0x1000
    for i in range(36):
        f = SYSCALL_STRUCT.unpack_from(data, base + i * SYSCALL_STRUCT.size)
        if f[0] != SNAP_SYSCALL_MAGIC:
            continue
        _, kind, block, pmc, _, _, _, _, value = f
        syscalls[(kind, block, pmc)] = value
    return direct, syscalls


def tile_ops(input_h: int, input_w: int, passes: int = 1) -> int:
    macs = (input_h * input_w) * (CH * K * K + HIDDEN_LAYERS * CH * CH * K * K + CH * K * K)
    return macs * 2 * passes


def metrics_for_variant(work: Path, label: str, geom_root: str) -> dict | None:
    manifests = sorted(work.glob("manifest_*.json"))
    if not manifests:
        return None
    man = json.loads(manifests[-1].read_text())
    stamp = man["stamp"]
    before = work / f"pmc_before_{stamp}.bin"
    after  = work / f"pmc_after_{stamp}.bin"
    if not (before.exists() and after.exists()):
        return None

    bd, bs = read_snapshot(before)
    ad, as_ = read_snapshot(after)

    # Cycles per hart (delta). Only minions 0/2/4/6 give clean readings on this
    # snapshot kernel; the others underflow (huge ~2^47+ values). Filter those.
    per_hart = []
    for h in sorted(set(bd) & set(ad)):
        d_cyc = delta(ad[h]["hpm"][0], bd[h]["hpm"][0])
        per_hart.append({
            "hart":   h,
            "minion": bd[h]["minion"],
            "thread": bd[h]["thread"],
            "cycles_delta": d_cyc,
        })
    valid_cycles = [r["cycles_delta"] for r in per_hart if r["cycles_delta"] < (1 << 40)]
    cycles_max = max(valid_cycles) if valid_cycles else 0
    cycles_min = min(valid_cycles) if valid_cycles else 0

    # syscall PMC deltas
    sc = {"reads": 0, "writes": 0, "cycles": 0}
    ms = {"reads": 0, "writes": 0, "cycles": 0}
    for (kind, block, pmc), v_before in bs.items():
        if (kind, block, pmc) not in as_:
            continue
        d = delta(as_[(kind, block, pmc)], v_before)
        if pmc == 0: name = "cycles"
        elif pmc == 1: name = "reads"
        elif pmc == 2: name = "writes"
        else: continue
        if kind == 0: sc[name] += d
        elif kind == 1: ms[name] += d

    # Compute ops from geometry
    label_to_dir = {
        "tg01_h20_t64_boundary_t06": "tiled_v2_tg01_h20_t64_boundary",
        "tg02_h24_t64_boundary_t06": "tiled_v2_tg02_h24_t64_boundary",
        "tg09_h20_t64_skip_read_t06": "tiled_v2_tg09_h20_t64_skip_read",
    }
    info_path = Path(geom_root) / label_to_dir.get(label, "") / "info.json"
    tile_idx = int(man["tile_idx"])
    if info_path.exists():
        info = json.loads(info_path.read_text())
        t = info["tiles"][tile_idx]
        ops = tile_ops(t["input_h"], t["input_w"], info.get("passes", 1))
        ih, iw = t["input_h"], t["input_w"]
    else:
        ops = 0; ih = iw = 0

    wait_s = man.get("kernel_wait_s") or 0
    return {
        "label": label,
        "stamp": stamp,
        "input_h": ih, "input_w": iw,
        "kernel_wait_s": wait_s,
        "ops": ops,
        "tile_gops": (ops / wait_s / 1e9) if wait_s else 0.0,
        "cycles_max_t0": cycles_max,
        "cycles_min_t0": cycles_min,
        "freq_mhz_implied": (cycles_max / wait_s / 1e6) if wait_s else 0,
        "cycles_per_op": (cycles_max / ops) if ops else 0,
        "sc_reads":  sc["reads"],  "sc_writes":  sc["writes"],  "sc_cycles":  sc["cycles"],
        "ms_reads":  ms["reads"],  "ms_writes":  ms["writes"],  "ms_cycles":  ms["cycles"],
        "sc_reads_per_kop":   sc["reads"]  * 1000.0 / ops if ops else 0,
        "sc_writes_per_kop":  sc["writes"] * 1000.0 / ops if ops else 0,
        "ms_reads_per_kop":   ms["reads"]  * 1000.0 / ops if ops else 0,
        "ms_writes_per_kop":  ms["writes"] * 1000.0 / ops if ops else 0,
        # bandwidth implied by syscall counts (assume 64B cache lines)
        "sc_read_GBs":  (sc["reads"] * 64.0) / wait_s / 1e9 if wait_s else 0,
        "sc_write_GBs": (sc["writes"] * 64.0) / wait_s / 1e9 if wait_s else 0,
        "ms_read_GBs":  (ms["reads"] * 64.0) / wait_s / 1e9 if wait_s else 0,
        "ms_write_GBs": (ms["writes"] * 64.0) / wait_s / 1e9 if wait_s else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--geom-root", default="/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn",
                    help="local dir holding tiled_v2_* with info.json — actually we need soc4 paths, override below")
    ap.add_argument("--out",  default="pmc_results.tsv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = metrics_for_variant(d, d.name, args.geom_root)
        if m:
            rows.append(m)
        else:
            print(f"  skip {d.name} (incomplete)")

    if not rows:
        print("no rows"); return 1

    # Print compact comparison
    print("=" * 96)
    print("PMC summary on interior tile_06 (3 best tiled-v2 variants)")
    print("=" * 96)
    hdr = ["", "tg01 (h20 base)", "tg02 (h24 halo)", "tg09 (skip_read)"]
    rows_by_label = {r["label"]: r for r in rows}
    keys = ["tg01_h20_t64_boundary_t06","tg02_h24_t64_boundary_t06","tg09_h20_t64_skip_read_t06"]
    cols = [rows_by_label.get(k) for k in keys]

    def fmt(metric, fn=lambda v: v, suffix=""):
        vals = [fn(c[metric]) if c else "—" for c in cols]
        print(f"  {metric:<24s}  " + "  ".join(f"{v:>16}" for v in vals) + (f"  {suffix}" if suffix else ""))

    print(f"\n{'metric':<26s}  {'tg01 h20 base':>16s}  {'tg02 h24 halo':>16s}  {'tg09 skipread':>16s}")
    print("-"*92)
    for c in cols:
        if c: print(f"  shape input            {c['input_h']}x{c['input_w']}", end="  ")
    print()
    fmt("kernel_wait_s",    lambda v: f"{v:.4f}",  "(s for one tile)")
    fmt("tile_gops",        lambda v: f"{v:.3f}",  "(GOPS, single tile)")
    fmt("ops",              lambda v: f"{v:,}")
    fmt("cycles_max_t0",    lambda v: f"{v:,}",    "(max T0 cycles, even minions)")
    fmt("freq_mhz_implied", lambda v: f"{v:.1f}",  "MHz")
    fmt("cycles_per_op",    lambda v: f"{v:.4f}")
    print()
    print("  -- L2 (shire cache) --")
    fmt("sc_reads",          lambda v: f"{v:,}")
    fmt("sc_writes",         lambda v: f"{v:,}")
    fmt("sc_reads_per_kop",  lambda v: f"{v:.3f}")
    fmt("sc_writes_per_kop", lambda v: f"{v:.3f}")
    fmt("sc_read_GBs",       lambda v: f"{v:.2f}",  "GB/s read")
    fmt("sc_write_GBs",      lambda v: f"{v:.2f}",  "GB/s write")
    print()
    print("  -- DRAM (main store) --")
    fmt("ms_reads",          lambda v: f"{v:,}")
    fmt("ms_writes",         lambda v: f"{v:,}")
    fmt("ms_reads_per_kop",  lambda v: f"{v:.4f}")
    fmt("ms_writes_per_kop", lambda v: f"{v:.4f}")
    fmt("ms_read_GBs",       lambda v: f"{v:.4f}",  "GB/s read")
    fmt("ms_write_GBs",      lambda v: f"{v:.4f}",  "GB/s write")
    print("=" * 96)

    if rows:
        keys_all = list(rows[0].keys())
        with open(args.out, "w") as f:
            f.write("\t".join(keys_all) + "\n")
            for r in rows:
                f.write("\t".join(str(r[k]) for k in keys_all) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
