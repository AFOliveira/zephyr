#!/usr/bin/env bash
# Build the minimal TFMA throughput probe.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
INC=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc

OUT="${OUT:-/tmp/tfma_probe}"
LABEL="${LABEL:-p01_fp32_b3a15c15}"
TFMA_TYPE="${TFMA_TYPE:-0}"
TFMA_AROWS="${TFMA_AROWS:-0xF}"
TFMA_ACOLS="${TFMA_ACOLS:-0xF}"
TFMA_BCOLS="${TFMA_BCOLS:-0x3}"
TFMA_ITERS="${TFMA_ITERS:-1000}"

mkdir -p "$OUT"
ELF="$OUT/${LABEL}.elf"

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
    -T "$INC/et-common-libs/share/erbium-soc1sim/erbium.ld" \
    -DTFMA_TYPE="$TFMA_TYPE" \
    -DTFMA_AROWS="$TFMA_AROWS" \
    -DTFMA_ACOLS="$TFMA_ACOLS" \
    -DTFMA_BCOLS="$TFMA_BCOLS" \
    -DTFMA_ITERS="$TFMA_ITERS" \
    -DACTIVE_HARTS=1u -DNUM_HARTS=1 \
    -o "$ELF" \
    "$DIR/tfma_probe.c" \
    "$ROOT/hart-report/hart_report_crt.S" \
    "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
set +x

ls -la "$ELF"
echo "OK: $ELF (TFMA_TYPE=$TFMA_TYPE arows=$TFMA_AROWS acols=$TFMA_ACOLS bcols=$TFMA_BCOLS iters=$TFMA_ITERS)"
