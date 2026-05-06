#!/usr/bin/env bash
# 16-hart sweep of the verified Luxonis DnCNN3 port at the working tile sizes
# above 4×4 (where the row partition is non-degenerate at 16 harts).
#
# Skips -Ofast variants because they expose a host-side libetrt segfault on
# multi-hart runs of this kernel. -O3 -funroll-loops is the empirical winner
# for transformer/stereo proxies and runs cleanly here.
set -euo pipefail
export OUT_SUBDIR=audited-sweep-h16-p8
export PASSES=8

# We use a custom inline build instead of overriding VARIANTS in the parent
# script, since the parent currently runs tile1 + tile4 only.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
OUT="$DIR/$OUT_SUBDIR"
mkdir -p "$OUT"
: > "$OUT/manifest.tsv"

GCC="${GCC:-/home/afonso/et/bin/riscv64-unknown-elf-gcc}"
CRT="$ROOT/hart-report/hart_report_crt.S"
LAYOUT="${LAYOUT:-/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c}"
LD_SCRIPT="${LD_SCRIPT:-/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld}"
SRC="$DIR/dncnn3_luxonis_real_vpu.c"
WEIGHTS_O="$DIR/dncnn3_luxonis_240x320_weights_packed_f32.o"

COMMON=(
	-march=rv64imfc -mabi=lp64f -mcmodel=medany -nostdlib
	-fno-zero-initialized-in-bss -ffunction-sections -fdata-sections
	-I/tmp/et-vidas-install/erbium-soc1sim-umode/include
	-I/tmp/et-vidas-install/include/esperanto-fw/erbium_hal
	-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include
	-I/home/afonso/et-platform-vidas/et-common-libs/include
	-Wl,--gc-sections -Wl,--no-warn-rwx-segments -Wl,--emit-relocs
	-T "$LD_SCRIPT"
)

declare -A VARIANTS=(
	[o3]="-O3"
	[o3_unroll]="-O3 -funroll-loops"
	[o3_unroll_noipa]="-O3 -funroll-loops -fno-ipa-cp -fno-ipa-sra"
	[o3_unroll_align64]="-O3 -funroll-loops -falign-functions=64 -falign-loops=64"
)

build_one() {
	local tile="$1" harts="$2" variant="$3"
	local img_w img_h tile_macro input_o output_o
	case "$tile" in
		16)  img_w=16;  img_h=16;  tile_macro=DNCNN_TILE16_BLOBS;  input_o="$DIR/dncnn3_luxonis_16x16_input_f32.o";  output_o="$DIR/dncnn3_luxonis_16x16_ort_output_f32.o" ;;
		64)  img_w=64;  img_h=64;  tile_macro=DNCNN_TILE64_BLOBS;  input_o="$DIR/dncnn3_luxonis_64x64_input_f32.o";  output_o="$DIR/dncnn3_luxonis_64x64_ort_output_f32.o" ;;
		*)   echo "unsupported tile $tile" >&2; return 1 ;;
	esac

	local name="h_tile${tile}_${variant}_${harts}h"
	# shellcheck disable=SC2206
	local flags=( ${VARIANTS[$variant]} )

	"$GCC" "${flags[@]}" "${COMMON[@]}" \
		-DDNCNN_STATIC_BLOBS \
		-D${tile_macro} \
		-DIMG_W=${img_w} -DIMG_H=${img_h} \
		-DACTIVE_HARTS=${harts}u -DNUM_HARTS=${harts} \
		-DDNCNN_PASSES=${PASSES}u \
		-o "$OUT/${name}.elf" \
		"$SRC" "$input_o" "$output_o" "$WEIGHTS_O" \
		"$CRT" "$LAYOUT"

	printf '%s\t%s\t%d\t%d\n' "$name" "$variant" "$tile" "$harts" >> "$OUT/manifest.tsv"
	echo "built $name"
}

# 16x16 at 16 harts (1 row per hart) — clean row partition
for v in o3 o3_unroll o3_unroll_noipa o3_unroll_align64; do
	build_one 16 16 "$v"
done

# 64x64 at 16 harts (4 rows per hart) — closest to proxy 16h geometry
for v in o3 o3_unroll o3_unroll_noipa o3_unroll_align64; do
	build_one 64 16 "$v"
done

# 16x16 at 8 harts (even-only mask) — VPU-only baseline
for v in o3 o3_unroll; do
	build_one 16 8 "$v"
done

echo
echo "=== built $(wc -l < "$OUT/manifest.tsv") variants under $OUT ==="
ls -la "$OUT"/*.elf | head
