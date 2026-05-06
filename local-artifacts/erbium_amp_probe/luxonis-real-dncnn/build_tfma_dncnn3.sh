#!/usr/bin/env bash
# Build TFMA-based DnCNN3 kernel (multi-minion).
#  VARIANT={A,B,C}  partition strategy
#  DTYPE={fp32,int8} (int8 added in Phase 8)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
INC=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc
OBJCOPY=/home/afonso/et/bin/riscv64-unknown-elf-objcopy

OUT="${OUT:-/tmp/dncnn_tfma}"
VARIANT="${VARIANT:-A}"
DTYPE="${DTYPE:-fp32}"
LABEL="${LABEL:-tfma_${DTYPE}_${VARIANT}}"
REGION_SIZE="${REGION_SIZE:-0x04000000}"  # 64 MB so two 19.95 MB padded act banks fit
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

mkdir -p "$OUT"
ELF="$OUT/${LABEL}.elf"

# Repack weights once if blobs missing
if [ ! -f "$DIR/dncnn3_tfma_hidden_weights_fp32.bin" ]; then
    echo "[build_tfma_dncnn3] regenerating weight pack ..."
    python3 "$DIR/tools/pack_tfma_weights.py"
fi

# Build .o blobs from .bin files (idempotent; --rename-section makes them .rodata)
for f in dncnn3_luxonis_240x320_input_f32 dncnn3_luxonis_240x320_ort_output_f32 \
         dncnn3_tfma_hidden_weights_fp32 dncnn3_tfma_hidden_biases_fp32 \
         dncnn3_tfma_hidden_orig_weights_fp32 \
         dncnn3_tfma_first_weights_fp32 dncnn3_tfma_first_biases_fp32 \
         dncnn3_tfma_final_weights_fp32 dncnn3_tfma_final_biases_fp32; do
    if [ ! -f "$DIR/${f}.o" ] || [ "$DIR/${f}.bin" -nt "$DIR/${f}.o" ]; then
        ( cd "$DIR" && "$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
              --rename-section .data=.rodata,alloc,load,readonly,data,contents \
              "${f}.bin" "${f}.o" )
    fi
done

# Pick source file based on dtype
case "$DTYPE" in
    fp32) SRC="$DIR/dncnn3_tfma_v1.c" ;;
    int8) SRC="$DIR/dncnn3_tfma_int8_v1.c" ;;
    *) echo "unknown DTYPE=$DTYPE" >&2; exit 2 ;;
esac

# Pick partition flag
PART_FLAG="-DTFMA_PARTITION='\"$VARIANT\"'"

set -x
"$GCC" \
    -O3 -funroll-loops \
    -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize \
    -march=rv64imfc -mabi=lp64f -mcmodel=medany -nostdlib \
    -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections \
    -I"$INC/et-common-libs/build-headers/erbium-soc1sim-staged-include" \
    -I"$INC/hal/platform/erbium/include" \
    -I"$INC/hal/platform/etsoc/include" \
    -I"$INC/et-common-libs/include" \
    -Wl,--gc-sections -Wl,--no-warn-rwx-segments -Wl,--emit-relocs \
    -Wl,--defsym=region0_size="$REGION_SIZE" \
    -T "$INC/et-common-libs/share/erbium-soc1sim/erbium.ld" \
    -DACTIVE_HARTS=8u -DNUM_HARTS=16 \
    -DTFMA_PARTITION="'$VARIANT'" \
    $EXTRA_FLAGS \
    -o "$ELF" \
    "$SRC" \
    "$DIR/dncnn3_luxonis_240x320_input_f32.o" \
    "$DIR/dncnn3_luxonis_240x320_ort_output_f32.o" \
    "$DIR/dncnn3_tfma_hidden_weights_fp32.o" \
    "$DIR/dncnn3_tfma_hidden_biases_fp32.o" \
    "$DIR/dncnn3_tfma_hidden_orig_weights_fp32.o" \
    "$DIR/dncnn3_tfma_first_weights_fp32.o" \
    "$DIR/dncnn3_tfma_first_biases_fp32.o" \
    "$DIR/dncnn3_tfma_final_weights_fp32.o" \
    "$DIR/dncnn3_tfma_final_biases_fp32.o" \
    "$ROOT/hart-report/hart_report_crt.S" \
    "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
set +x

ls -la "$ELF"
"$GCC"-elf-readelf -h "$ELF" 2>/dev/null | grep -E "^  Entry|^  Type" || true
echo "OK: $ELF (VARIANT=$VARIANT DTYPE=$DTYPE region=$REGION_SIZE)"
