#!/usr/bin/env bash
# Build TFMA-INT8 multi-minion DnCNN3 v2 (Tier 1.1 — TFMA hardware path).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
INC=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc
OBJCOPY=/home/afonso/et/bin/riscv64-unknown-elf-objcopy

OUT="${OUT:-/tmp/dncnn_tfma}"
LABEL="${LABEL:-int8_tfma_v2}"
REGION_SIZE="${REGION_SIZE:-0x04000000}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

mkdir -p "$OUT"
ELF="$OUT/${LABEL}.elf"

# Make sure the TFMA INT8 A-pack blob is current
if [ ! -f "$DIR/dncnn3_tfma_int8_a_pack.bin" ] || \
   [ "$DIR/dncnn3_int8_hidden_weights_int8.bin" -nt "$DIR/dncnn3_tfma_int8_a_pack.bin" ]; then
    echo "[build] regenerating TFMA INT8 A-pack..."
    python3 "$DIR/tools/pack_tfma_int8_a.py"
fi

# Convert all referenced .bin to .o
for f in dncnn3_luxonis_240x320_input_f32 \
         dncnn3_luxonis_240x320_ort_output_f32 \
         dncnn3_tfma_first_weights_fp32 dncnn3_tfma_first_biases_fp32 \
         dncnn3_tfma_final_weights_fp32 dncnn3_tfma_final_biases_fp32 \
         dncnn3_tfma_hidden_biases_fp32 \
         dncnn3_tfma_int8_a_pack \
         dncnn3_int8_hidden_w_scales_fp32 \
         dncnn3_int8_act_scales_fp32 \
         dncnn3_int8_inv_act_scales_fp32; do
    if [ ! -f "$DIR/${f}.o" ] || [ "$DIR/${f}.bin" -nt "$DIR/${f}.o" ] || [ "$0" -nt "$DIR/${f}.o" ]; then
        # tensor_load encodes addr & ~0x3F → loaded addresses must be cache-line
        # (64-byte) aligned. objcopy default is 4-byte; force 64.
        ( cd "$DIR" && "$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
              --rename-section .data=.rodata,alloc,load,readonly,data,contents \
              --set-section-alignment .rodata=64 \
              "${f}.bin" "${f}.o" )
    fi
done

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
    $EXTRA_FLAGS \
    -o "$ELF" \
    "$DIR/dncnn3_tfma_int8_v2.c" \
    "$DIR/dncnn3_luxonis_240x320_input_f32.o" \
    "$DIR/dncnn3_luxonis_240x320_ort_output_f32.o" \
    "$DIR/dncnn3_tfma_first_weights_fp32.o" \
    "$DIR/dncnn3_tfma_first_biases_fp32.o" \
    "$DIR/dncnn3_tfma_final_weights_fp32.o" \
    "$DIR/dncnn3_tfma_final_biases_fp32.o" \
    "$DIR/dncnn3_tfma_hidden_biases_fp32.o" \
    "$DIR/dncnn3_tfma_int8_a_pack.o" \
    "$DIR/dncnn3_int8_hidden_w_scales_fp32.o" \
    "$DIR/dncnn3_int8_act_scales_fp32.o" \
    "$DIR/dncnn3_int8_inv_act_scales_fp32.o" \
    "$ROOT/hart-report/hart_report_crt.S" \
    "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
set +x

ls -la "$ELF"
echo "OK: $ELF"
