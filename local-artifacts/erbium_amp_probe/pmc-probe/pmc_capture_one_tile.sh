#!/usr/bin/env bash
# pmc_capture_one_tile.sh <variant_label> <tile_idx_2digit>
# Runs (under a single flock on shire 0):
#   1. pmc_snapshot_16h.elf   -> pmc_before_<stamp>.bin
#   2. tiled_v2_<label>/tile_<idx>.elf  -> dump_<stamp>.bin
#   3. pmc_snapshot_16h.elf   -> pmc_after_<stamp>.bin
# Run on esperanto-soc4. Output dir: /root/.../erbium-amp-probe/pmc-tiled-v2/<label>_t<idx>/
set -uo pipefail

LABEL="${1:?need label like tg01_h20_t64_boundary}"
TILE_IDX="${2:?need tile idx like 06}"

BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT="$BASE/erbium-amp-probe"
LAUNCH="$PARENT/erbium_soc1sim_argbuf"
SNAP_ELF="$PARENT/dncnn3-pmc/pmc_snapshot_16h.elf"
TILE_ELF="$PARENT/tiled_v2_${LABEL}/tile_${TILE_IDX}.elf"
ZERO64K="$PARENT/zero64k.bin"
ZERO_BIG="$PARENT/zero2m.bin"

OUT_DIR="$PARENT/pmc-tiled-v2/${LABEL}_t${TILE_IDX}"
mkdir -p "$OUT_DIR"
STAMP=$(date -u +%Y%m%d-%H%M%SZ)

[ -f "$SNAP_ELF" ]  || { echo "missing $SNAP_ELF"; exit 2; }
[ -f "$TILE_ELF" ]  || { echo "missing $TILE_ELF"; exit 2; }
[ -f "$ZERO64K" ]   || { echo "missing $ZERO64K"; exit 2; }
[ -f "$ZERO_BIG" ]  || { echo "missing $ZERO_BIG"; exit 2; }

export LD_LIBRARY_PATH="$BASE:$PARENT:${LD_LIBRARY_PATH:-}"

LOCK=/var/lock/etsoc-shire0.lock

(
  exec 9>"$LOCK"
  if ! flock -x -w 600 9; then
    echo "FAIL: flock timeout"; exit 3
  fi
  cd "$OUT_DIR" || exit 4
  echo "=== ${LABEL} tile_${TILE_IDX} stamp=${STAMP} ==="

  echo "--- snap before ---"
  "$LAUNCH" --elf-load "$SNAP_ELF" --shire 0 \
    --file_load 0x0,"$ZERO_BIG" \
    --dump_after "pmc_before_${STAMP}.bin" \
    --timeout 60 > "run_before_${STAMP}.log" 2>&1
  rc_before=$?
  echo "  rc=${rc_before} dump=$(stat -c%s pmc_before_${STAMP}.bin 2>/dev/null || echo 0)B"

  echo "--- kernel ---"
  "$LAUNCH" --elf-load "$TILE_ELF" --shire 0 \
    --file_load 0x0,"$ZERO64K" \
    --dump_after "dump_${STAMP}.bin" \
    --timeout 60 > "run_kernel_${STAMP}.log" 2>&1
  rc_kernel=$?
  wait_s=$(grep -oE "Kernel wait seconds: [0-9.]+" "run_kernel_${STAMP}.log" | tail -1 | awk '{print $NF}')
  echo "  rc=${rc_kernel} kernel_wait_s=${wait_s:-?} dump=$(stat -c%s dump_${STAMP}.bin 2>/dev/null || echo 0)B"

  echo "--- snap after ---"
  "$LAUNCH" --elf-load "$SNAP_ELF" --shire 0 \
    --file_load 0x0,"$ZERO_BIG" \
    --dump_after "pmc_after_${STAMP}.bin" \
    --timeout 60 > "run_after_${STAMP}.log" 2>&1
  rc_after=$?
  echo "  rc=${rc_after} dump=$(stat -c%s pmc_after_${STAMP}.bin 2>/dev/null || echo 0)B"

  cat > "manifest_${STAMP}.json" <<EOF
{
  "label": "${LABEL}",
  "tile_idx": "${TILE_IDX}",
  "stamp": "${STAMP}",
  "rc_before": ${rc_before},
  "rc_kernel": ${rc_kernel},
  "rc_after": ${rc_after},
  "kernel_wait_s": ${wait_s:-null},
  "snap_before": "pmc_before_${STAMP}.bin",
  "snap_after":  "pmc_after_${STAMP}.bin",
  "kernel_dump": "dump_${STAMP}.bin",
  "tile_elf": "${TILE_ELF}"
}
EOF
  echo "STAMP=${STAMP}"
)
