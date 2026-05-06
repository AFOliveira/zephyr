#!/usr/bin/env bash
# Build all 20 per-tile ELFs for full 240x320 tiled DnCNN3 inference.
# Each ELF uses DNCNN_TILE_BLOBS mode + a per-tile pair of .o files renamed
# to consistent symbol names: _binary_dncnn_tile_input_bin_start,
# _binary_dncnn_tile_ref_bin_start.
set -euo pipefail

DIR=/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn
ROOT=/home/afonso/zephyr/local-artifacts/erbium_amp_probe
INC_BASE=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc
OBJCOPY=/home/afonso/et/bin/riscv64-unknown-elf-objcopy
GEOM=$DIR/tiled-fp32/audit_a1_geometry.json
TILES_DIR=$DIR/tiled-fp32/tiles
OUT=/tmp/tiled_240x320
mkdir -p "$OUT"

COMMON=(
    -O3 -funroll-loops -falign-functions=256 -falign-loops=256
    -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize
    -march=rv64imfc -mabi=lp64f -mcmodel=medany -nostdlib
    -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections
    -I$INC_BASE/et-common-libs/build-headers/erbium-soc1sim-staged-include
    -I$INC_BASE/hal/platform/erbium/include
    -I$INC_BASE/hal/platform/etsoc/include
    -I$INC_BASE/et-common-libs/include
    -Wl,--gc-sections -Wl,--no-warn-rwx-segments -Wl,--emit-relocs
    -T $INC_BASE/et-common-libs/share/erbium-soc1sim/erbium.ld
    -DDNCNN_STATIC_BLOBS -DDNCNN_TILE_BLOBS
    -DACTIVE_HARTS=16u -DNUM_HARTS=16 -DDNCNN_PASSES=1u
    -DDNCNN_VPU_FUSED9TAP=1 -DDNCNN_VPU_PREFETCH_READ_WINDOW=1
    -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1
)

WEIGHTS_O=$DIR/dncnn3_luxonis_240x320_weights_packed_f32.o
CRT=$ROOT/hart-report/hart_report_crt.S
LAYOUT=$INC_BASE/erbium-examples/runtime/erbium-soc1sim/layout.c
SRC=$DIR/dncnn3_luxonis_real_vpu.c

# Read geometry and build per-tile
python3 - <<PY > /tmp/tiles_to_build.tsv
import json
g = json.load(open("$GEOM"))
for t in g["tiles"]:
    print(f"{t['tile_id']}\t{t['input_h']}\t{t['input_w']}\t{t['output_h']}\t{t['output_w']}\t{t['crop_y0']}\t{t['crop_y1']}\t{t['crop_x0']}\t{t['crop_x1']}")
PY

ok=0
fail=0
while IFS=$'\t' read tid ih iw oh ow cy0 cy1 cx0 cx1; do
    name=$(printf "tile_%02d" "$tid")
    src_dir="$TILES_DIR/t$(printf %02d $tid)"
    in_bin="$src_dir/input.bin"
    ref_bin="$src_dir/ort_output_full.bin"
    [ -f "$in_bin" ] && [ -f "$ref_bin" ] || { echo "MISS bins for tile $tid"; fail=$((fail+1)); continue; }

    # Copy bins to build dir under fixed names so objcopy emits the right symbols
    work="$OUT/$name"
    mkdir -p "$work"
    cp "$in_bin"  "$work/dncnn_tile_input.bin"
    cp "$ref_bin" "$work/dncnn_tile_ref.bin"

    # objcopy derives symbol names from input path; cd into work dir so symbols
    # become _binary_dncnn_tile_input_bin_{start,end,size} (no path prefix).
    (cd "$work" && \
        "$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
            --rename-section .data=.rodata,alloc,load,readonly,data,contents \
            "dncnn_tile_input.bin" "dncnn_tile_input.o" && \
        "$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
            --rename-section .data=.rodata,alloc,load,readonly,data,contents \
            "dncnn_tile_ref.bin" "dncnn_tile_ref.o")

    "$GCC" "${COMMON[@]}" \
        -DIMG_W=${iw}u -DIMG_H=${ih}u \
        -o "$OUT/${name}.elf" \
        "$SRC" "$work/dncnn_tile_input.o" "$work/dncnn_tile_ref.o" "$WEIGHTS_O" \
        "$CRT" "$LAYOUT" 2> "$work/build.log"
    if [ $? -eq 0 ] && [ -f "$OUT/${name}.elf" ]; then
        size=$(stat -c %s "$OUT/${name}.elf")
        echo "OK  ${name}.elf  ${ih}x${iw} -> output ${oh}x${ow} crop[$cy0:$cy1,$cx0:$cx1]  ${size} bytes"
        ok=$((ok+1))
    else
        echo "FAIL ${name}.elf — see $work/build.log"
        fail=$((fail+1))
    fi
done < /tmp/tiles_to_build.tsv

echo
echo "=== built $ok ELFs, $fail failures ==="
ls -la "$OUT"/*.elf | head -25
