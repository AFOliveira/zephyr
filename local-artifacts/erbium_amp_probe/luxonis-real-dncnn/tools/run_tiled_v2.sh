#!/usr/bin/env bash
# Run a built-tiled-v2 variant on soc4: stage, run all tiles via board_run.sh, audit.
# Usage: run_tiled_v2.sh <label>
set -euo pipefail

LABEL="${1:?need label}"
LOCAL=/tmp/tiled_v2/$LABEL
[ -d "$LOCAL" ] || { echo "missing local build $LOCAL"; exit 1; }
REMOTE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/tiled_v2_$LABEL
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"

echo "=== stage to soc4 ==="
ssh root@esperanto-soc4 "mkdir -p $REMOTE" >/dev/null
rsync -aq "$LOCAL/" "root@esperanto-soc4:$REMOTE/"

echo "=== run all tiles ==="
RUN_SCRIPT=/tmp/run_${LABEL}.sh
N_TILES=$(jq '.tiles | length' "$LOCAL/info.json")
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT=\$BASE/erbium-amp-probe
DIR=$REMOTE
EOF
for i in $(seq -w 0 $((N_TILES-1))); do
cat >> "$RUN_SCRIPT" <<EOF
\$BASE/board_run.sh --elf "\$DIR/tile_${i}.elf" --shire 0 \\
    --file-load 0x0,\$PARENT/zero64k.bin \\
    --agent tiled-v2-${LABEL} --purpose "tile $i" \\
    --run-id "tv2-${LABEL}-tile${i}-${STAMP}" \\
    --timeout 60 --lock-timeout 30 || true
EOF
done
echo "echo STAMP=$STAMP" >> "$RUN_SCRIPT"
scp -q "$RUN_SCRIPT" "root@esperanto-soc4:$REMOTE/run.sh"
ssh root@esperanto-soc4 "bash $REMOTE/run.sh" 2>&1 | grep -E "ok rc|FAIL|STAMP" | tail -25

echo "=== pull dumps ==="
PULL_DIR=/tmp/tiled_v2_runs/$LABEL
mkdir -p "$PULL_DIR"
for i in $(seq -w 0 $((N_TILES-1))); do
    rsync -aq "root@esperanto-soc4:/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/runs/tv2-${LABEL}-tile${i}-${STAMP}/" \
        "$PULL_DIR/tile_${i}/" 2>/dev/null || true
done
echo "  pulled $(ls -d $PULL_DIR/tile_* 2>/dev/null | wc -l) tile dirs"

echo "=== audit + stitch ==="
GEOM=/tmp/tiled_v2/$LABEL/info.json
python3 - "$GEOM" "$PULL_DIR" "$LABEL" <<'PY'
import json, struct, sys
import numpy as np
from pathlib import Path

geom_path, runs_dir, label = sys.argv[1:4]
g = json.load(open(geom_path))
tiles = g["tiles"]
img_h, img_w = 240, 320
SUMMARY = struct.Struct("<16I")

full_ort = np.fromfile(
    "/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/dncnn3_luxonis_240x320_ort_output_f32.bin",
    dtype=np.float32).reshape(img_h, img_w)
stitched = np.zeros_like(full_ort)
total_wait = 0.0; total_ops = 0; n_pass = 0

for t in tiles:
    rd = Path(runs_dir) / f"tile_{t['tile_id']:02d}"
    if not (rd / "dump.bin").exists():
        print(f"  tile {t['tile_id']:02d}: MISSING dump"); continue
    man = json.load(open(rd / "manifest.json"))
    wait = man.get("kernel_wait_s") or 0
    total_wait += wait
    with open(rd / "dump.bin", "rb") as f:
        f.seek(0x300000)
        out = np.frombuffer(f.read(t["input_h"]*t["input_w"]*4), dtype=np.float32).reshape(t["input_h"], t["input_w"])
    cropped = out[t["crop_y0"]:t["crop_y1"], t["crop_x0"]:t["crop_x1"]]
    stitched[t["output_y0"]:t["output_y1"], t["output_x0"]:t["output_x1"]] = cropped
    if (rd / "manifest.json").exists() and man.get("status") == "ok":
        n_pass += 1

# Compute approximate ops per tile (matches kernel's accounting)
HIDDEN_LAYERS = 18; CH = 64; K = 3
for t in tiles:
    macs = (t["input_h"] * t["input_w"]) * (CH*K*K + HIDDEN_LAYERS*CH*CH*K*K + CH*K*K)
    total_ops += macs * 2 * g.get("passes", 1)

diff = np.abs(stitched - full_ort)
max_abs = float(diff.max()); mean_abs = float(diff.mean())
gops = total_ops / total_wait / 1e9 if total_wait else 0
inf_per_s = 1 / total_wait if total_wait else 0
print(f"label={label}")
print(f"  tiles={len(tiles)}  per-tile-pass={n_pass}/{len(tiles)}")
print(f"  total_wait_s={total_wait:.4f}  inf/s={inf_per_s:.4f}")
print(f"  total_ops={total_ops:,}  full-image GOPS={gops:.3f}")
print(f"  stitched vs ORT: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}  pass@1e-5={max_abs <= 1e-5}")

# Append to master TSV
master = "/home/afonso/etsoc1-luxonis-experiments/results/audit_master_tiled_v2.tsv"
import os
new = not os.path.exists(master)
with open(master, "a") as f:
    if new:
        f.write("label\ttile_h\ttile_w\thalo\tpasses\textra\tn_tiles\tn_pass\twait_s\tgops\tinf_per_s\tmax_abs\tmean_abs\tallclose_1e5\n")
    f.write(f"{label}\t{g.get('tile_h','')}\t{g.get('tile_w','')}\t{g.get('halo','')}\t{g.get('passes','')}\t{','.join(g.get('extra',[]))}\t{len(tiles)}\t{n_pass}\t{total_wait:.4f}\t{gops:.4f}\t{inf_per_s:.4f}\t{max_abs:.3e}\t{mean_abs:.3e}\t{max_abs <= 1e-5}\n")
PY

echo "=== done ==="
