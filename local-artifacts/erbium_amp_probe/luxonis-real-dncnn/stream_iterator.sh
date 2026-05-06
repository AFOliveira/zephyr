#!/usr/bin/env bash
# Host-side streaming iterator. Stage ELF once, then per-frame:
#   stage image  →  ssh launch  →  rsync dump  →  audit  →  render.
#
# Per-frame wall time is dominated by ssh/rsync (~30-50s), NOT the kernel
# (which takes 2.5s). True 3-second intervals are not achievable through
# the stock argbuf launcher — that needs a libetrt custom launcher with
# kernel-side polling. This iterator demonstrates the principle with the
# interval bounded by host overhead.
#
# Usage:  ./stream_iterator.sh [MIN_INTERVAL_SECS]
set -uo pipefail

INTERVAL="${1:-5}"
DIR=/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn

# Build the streaming kernel if missing
if [ ! -f /tmp/dncnn_tfma/int8_tfma_v2_stream.elf ]; then
    echo ">>> building streaming kernel ..."
    EXTRA_FLAGS=-DDNCNN_STREAMING=1 LABEL=int8_tfma_v2_stream \
        bash $DIR/build_tfma_int8_v2.sh > /tmp/build.log 2>&1
fi

# Stage the ELF once (saves ~1 sec/frame)
SOC4=root@esperanto-soc4
PARENT=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe
echo ">>> staging kernel to soc4 (one-time) ..."
scp -q /tmp/dncnn_tfma/int8_tfma_v2_stream.elf "$SOC4:$PARENT/stream_iter.elf"

QUEUE=(
    "luxonis_clean    $DIR/dncnn3_stream_input_0_f32.bin   $DIR/dncnn3_stream_ort_output_0_f32.bin"
    "luxonis_noisy05  $DIR/dncnn3_stream_input_1_f32.bin   $DIR/dncnn3_stream_ort_output_1_f32.bin"
    "luxonis_noisy10  $DIR/dncnn3_stream_input_2_f32.bin   $DIR/dncnn3_stream_ort_output_2_f32.bin"
    "chessboard       $DIR/dncnn3_other_input_0_f32.bin    $DIR/dncnn3_other_ort_0_f32.bin"
    "gradient         $DIR/dncnn3_other_input_1_f32.bin    $DIR/dncnn3_other_ort_1_f32.bin"
    "circles          $DIR/dncnn3_other_input_2_f32.bin    $DIR/dncnn3_other_ort_2_f32.bin"
    "horiz_lines      $DIR/dncnn3_other_input_3_f32.bin    $DIR/dncnn3_other_ort_3_f32.bin"
)

LOG=/tmp/stream_log.tsv
echo -e "frame\tlabel\twall_s\twait_s\tmax_abs\tpass" > "$LOG"

START=$(date +%s)
echo
echo ">>> streaming iterator (min_interval=${INTERVAL}s, ${#QUEUE[@]} frames)"
echo

for i in "${!QUEUE[@]}"; do
    read -r LABEL INPUT_BIN REF_BIN <<< "${QUEUE[$i]}"
    FT0=$(date +%s.%N)
    STAMP=$(date -u +%Y%m%d-%H%M%SZ)
    REMOTE_DIR="/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/runs/iter${i}-${STAMP}"
    LOCAL_DIR="/tmp/dncnn_tfma/iter${i}-${STAMP}"

    echo "[$(date +%H:%M:%S)] frame $i: $LABEL"

    # 1. Stage the image (per-frame; fast since 308 KB)
    scp -q "$INPUT_BIN" "$SOC4:$PARENT/stream_iter_input.bin"

    # 2. Run the kernel under flock; ssh once, do everything inside
    ssh -q "$SOC4" "
        export LD_LIBRARY_PATH='/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2:$PARENT'
        mkdir -p '$REMOTE_DIR'
        cd '$REMOTE_DIR'
        flock -x -w 60 /var/lock/etsoc-shire0.lock \
          '$PARENT/erbium_soc1sim_argbuf' \
            --elf-load '$PARENT/stream_iter.elf' --shire 0 \
            --file_load 0x0,'$PARENT/zero64k.bin' \
            --file_load 0x10000,'$PARENT/stream_iter_input.bin' \
            --dump_after dump.bin --timeout 60 > run.log 2>&1
        rc=\$?
        echo \"rc=\$rc\"
        grep -oE 'Kernel wait seconds: [0-9.]+' run.log | tail -1
    " > /tmp/stream_run_$i.log 2>&1

    # 3. Pull the dump back (with retry)
    mkdir -p "$LOCAL_DIR"
    for attempt in 1 2 3; do
        if rsync -aqz --timeout=20 "$SOC4:$REMOTE_DIR/" "$LOCAL_DIR/" 2>/dev/null; then
            break
        fi
        echo "  rsync attempt $attempt failed, retrying ..."
        sleep 1
    done

    if [ ! -f "$LOCAL_DIR/dump.bin" ]; then
        echo "  ERROR: dump.bin not retrieved"
        echo -e "$i\t$LABEL\t-\tFAIL\t-\tFAIL" >> "$LOG"
    else
        WAIT_S=$(grep -oE 'Kernel wait seconds: [0-9.]+' "$LOCAL_DIR/run.log" | tail -1 | awk '{print $NF}')
        STATS=$(python3 - "$INPUT_BIN" "$LOCAL_DIR/dump.bin" "$REF_BIN" <<'PY'
import sys, numpy as np
inp = np.fromfile(sys.argv[1], dtype=np.float32).reshape(240, 320)
out = np.frombuffer(open(sys.argv[2],"rb").read()[0x300000:0x300000+240*320*4],
                   dtype=np.float32).reshape(240, 320)
ref = np.fromfile(sys.argv[3], dtype=np.float32).reshape(240, 320)
mx = float(np.abs(out - ref).max())
print(f"{mx:.4e}\t{'PASS' if mx < 5e-2 else 'FAIL'}")
PY
)
        FT1=$(date +%s.%N)
        WALL=$(echo "$FT1 - $FT0" | bc -l)
        echo -e "$i\t$LABEL\t${WALL%.???}\t$WAIT_S\t$STATS" >> "$LOG"
        echo "  wall=$(printf %.1f $WALL)s  kernel=${WAIT_S}s  $(echo "$STATS" | awk '{print "max_abs="$1" verdict="$2}')"
        python3 $DIR/tools/render_dncnn.py \
            "$INPUT_BIN" "$LOCAL_DIR/dump.bin" --ref "$REF_BIN" \
            --out /tmp/stream_frame_$i.png \
            --title "frame $i: $LABEL" 2>/dev/null
    fi

    # Sleep to enforce min interval (will be 0 if frame took longer than interval)
    if [ "$i" -lt "$((${#QUEUE[@]} - 1))" ]; then
        FT_NOW=$(date +%s.%N)
        ELAPSED_FRAME=$(echo "$FT_NOW - $FT0" | bc -l)
        SLEEP_S=$(echo "$INTERVAL - $ELAPSED_FRAME" | bc -l)
        if (( $(echo "$SLEEP_S > 0" | bc -l) )); then
            sleep "$SLEEP_S"
        fi
    fi
done

# Update HTML
python3 $DIR/tools/build_viz_html.py > /dev/null 2>&1 || true

ELAPSED=$(($(date +%s) - START))
echo
echo ">>> done. ${#QUEUE[@]} frames in ${ELAPSED}s"
echo
column -t -s$'\t' "$LOG"
echo
echo ">>> per-frame PNGs:    /tmp/stream_frame_*.png"
echo ">>> HTML viz:          /tmp/dncnn_visualization.html"
echo ">>> log:               $LOG"
