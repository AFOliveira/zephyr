#!/usr/bin/env bash
# Build untiled 240x320 DnCNN3 — single ELF, single launch, no halo redundancy.
# Requires region0_size > 16 MB so .bss (act0+act1 = ~40 MB) fits.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
INC=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc

OUT="${OUT:-/tmp/dncnn_untiled}"
LABEL="${LABEL:-u01_baseline}"
REGION_SIZE="${REGION_SIZE:-0x04000000}"   # 64 MB by default
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
PASSES="${PASSES:-1}"
ACTIVE_HARTS="${ACTIVE_HARTS:-16u}"
NUM_HARTS="${NUM_HARTS:-16}"

mkdir -p "$OUT"
ELF="$OUT/${LABEL}.elf"

COMMON=(
    -O3 -funroll-loops
    -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize
    -march=rv64imfc -mabi=lp64f -mcmodel=medany -nostdlib
    -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections
    -I"$INC/et-common-libs/build-headers/erbium-soc1sim-staged-include"
    -I"$INC/hal/platform/erbium/include"
    -I"$INC/hal/platform/etsoc/include"
    -I"$INC/et-common-libs/include"
    -Wl,--gc-sections -Wl,--no-warn-rwx-segments -Wl,--emit-relocs
    -Wl,--defsym=region0_size="$REGION_SIZE"
    -T "$INC/et-common-libs/share/erbium-soc1sim/erbium.ld"
    -DDNCNN_STATIC_BLOBS
    -DACTIVE_HARTS="$ACTIVE_HARTS" -DNUM_HARTS="$NUM_HARTS"
    -DDNCNN_VPU_FUSED9TAP=1 -DDNCNN_VPU_PREFETCH_READ_WINDOW=1
    -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1
)

set -x
"$GCC" "${COMMON[@]}" $EXTRA_FLAGS \
    -DIMG_W=320u -DIMG_H=240u \
    -DDNCNN_PASSES="${PASSES}u" \
    -o "$ELF" \
    "$DIR/dncnn3_luxonis_real_vpu.c" \
    "$DIR/dncnn3_luxonis_240x320_input_f32.o" \
    "$DIR/dncnn3_luxonis_240x320_ort_output_f32.o" \
    "$DIR/dncnn3_luxonis_240x320_weights_packed_f32.o" \
    "$ROOT/hart-report/hart_report_crt.S" \
    "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
set +x

ls -la "$ELF"
echo "OK: $ELF (region0_size=$REGION_SIZE)"
