#!/usr/bin/env bash
# Build optimization-flag variants of the AUDITED Luxonis DnCNN3 port at the
# launcher-stable tile sizes (1x1, 4x4). All variants share the same C source
# and ORT-comparable bin blobs as the verified port — only compile flags vary.
#
# Output ELFs go under audited-sweep/. Names: a_tile<T>_<flags>_<harts>h.elf
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
OUT="$DIR/${OUT_SUBDIR:-audited-sweep}"
PASSES="${PASSES:-1}"
mkdir -p "$OUT"

GCC="${GCC:-/home/afonso/et/bin/riscv64-unknown-elf-gcc}"
CRT="$ROOT/hart-report/hart_report_crt.S"
LAYOUT="${LAYOUT:-/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c}"
LD_SCRIPT="${LD_SCRIPT:-/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld}"
SRC="$DIR/dncnn3_luxonis_real_vpu.c"
WEIGHTS_O="$DIR/dncnn3_luxonis_240x320_weights_packed_f32.o"

COMMON=(
	-march=rv64imfc
	-mabi=lp64f
	-mcmodel=medany
	-nostdlib
	-fno-zero-initialized-in-bss
	-ffunction-sections
	-fdata-sections
	-I/tmp/et-vidas-install/erbium-soc1sim-umode/include
	-I/tmp/et-vidas-install/include/esperanto-fw/erbium_hal
	-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include
	-I/home/afonso/et-platform-vidas/et-common-libs/include
	-Wl,--gc-sections
	-Wl,--no-warn-rwx-segments
	-Wl,--emit-relocs
	-T "$LD_SCRIPT"
)

# variant_name : flag list
declare -A VARIANTS=(
	[baseline]="-O2"
	[o3]="-O3"
	[ofast]="-Ofast"
	[ofast_rename]="-Ofast -frename-registers"
	[ofast_rename_unroll]="-Ofast -frename-registers -funroll-loops"
	[o3_unroll]="-O3 -funroll-loops"
	[ofast_noipa]="-Ofast -fno-ipa-cp -fno-ipa-sra"
	[ofast_align64]="-Ofast -falign-functions=64 -falign-loops=64"
)

build_one() {
	local tile="$1" harts="$2" variant="$3"
	local img_w img_h tile_macro input_o output_o
	case "$tile" in
		1)   img_w=1;   img_h=1;   tile_macro=DNCNN_TILE1_BLOBS;   input_o="$DIR/dncnn3_luxonis_1x1_input_f32.o";    output_o="$DIR/dncnn3_luxonis_1x1_ort_output_f32.o" ;;
		4)   img_w=4;   img_h=4;   tile_macro=DNCNN_TILE4_BLOBS;   input_o="$DIR/dncnn3_luxonis_4x4_input_f32.o";    output_o="$DIR/dncnn3_luxonis_4x4_ort_output_f32.o" ;;
		16)  img_w=16;  img_h=16;  tile_macro=DNCNN_TILE16_BLOBS;  input_o="$DIR/dncnn3_luxonis_16x16_input_f32.o";  output_o="$DIR/dncnn3_luxonis_16x16_ort_output_f32.o" ;;
		64)  img_w=64;  img_h=64;  tile_macro=DNCNN_TILE64_BLOBS;  input_o="$DIR/dncnn3_luxonis_64x64_input_f32.o";  output_o="$DIR/dncnn3_luxonis_64x64_ort_output_f32.o" ;;
		*)   echo "unknown tile $tile" >&2; return 1 ;;
	esac

	local name="a_tile${tile}_${variant}_${harts}h"
	local elf="$OUT/${name}.elf"
	# shellcheck disable=SC2206
	local flags=( ${VARIANTS[$variant]} )

	"$GCC" "${flags[@]}" "${COMMON[@]}" \
		-DDNCNN_STATIC_BLOBS \
		-D${tile_macro} \
		-DIMG_W=${img_w} -DIMG_H=${img_h} \
		-DACTIVE_HARTS=${harts}u -DNUM_HARTS=${harts} \
		-DDNCNN_PASSES=${PASSES}u \
		-o "$elf" \
		"$SRC" "$input_o" "$output_o" "$WEIGHTS_O" \
		"$CRT" "$LAYOUT"

	printf '%s\t%s\t%d\t%d\n' "$name" "$variant" "$tile" "$harts" >> "$OUT/manifest.tsv"
	echo "built $name"
}

: > "$OUT/manifest.tsv"

# 1x1 with 1 hart (matches existing tile1_static_1h success)
for v in baseline o3 ofast ofast_rename ofast_rename_unroll o3_unroll ofast_noipa ofast_align64; do
	build_one 1 1 "$v"
done

# 4x4 with 1 hart only (4-hart path has a coherence bug — see task #6)
for v in baseline o3 ofast ofast_rename ofast_rename_unroll o3_unroll ofast_noipa ofast_align64; do
	build_one 4 1 "$v"
done

echo
echo "=== built $(wc -l < "$OUT/manifest.tsv") variants under $OUT ==="
ls -la "$OUT"/*.elf | head -30
