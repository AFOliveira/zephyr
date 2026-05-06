#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRT="${ROOT}/erbium_amp_probe/hart-report/hart_report_crt.S"
OUT="${ROOT}/erbium_amp_probe/optimization-kb/focused20b"

GCC="${GCC:-/home/afonso/et/bin/riscv64-unknown-elf-gcc}"
LAYOUT="${LAYOUT:-/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c}"
LD_SCRIPT="${LD_SCRIPT:-/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld}"

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
	-T "${LD_SCRIPT}"
	-DNUM_HARTS=16
	-DACTIVE_HARTS=16
)

mkdir -p "${OUT}"
: > "${OUT}/focused20_variants.tsv"

build_one() {
	local model="$1"
	local name="$2"
	local src="$3"
	shift 3

	echo -e "${model}\t${name}" >> "${OUT}/focused20_variants.tsv"
	"${GCC}" "$@" "${COMMON[@]}" -o "${OUT}/${name}.elf" "${src}" "${CRT}" "${LAYOUT}"
}

DNCNN_SRC="${ROOT}/erbium_amp_probe/dncnn3-pmc/dncnn3_vpu_fp_argbuf.c"
DNCNN_BASE=(-DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1 -DDNCNN_VPU_ACCUM_3X3=1 -DDNCNN_VPU_PREFETCH_WPACK=1)
build_one dncnn f20b_dncnn_ofast_noipa_align64 "${DNCNN_SRC}" -Ofast -fno-ipa-cp -fno-ipa-sra -falign-functions=64 -falign-loops=64 "${DNCNN_BASE[@]}"
build_one dncnn f20b_dncnn_ofast_noipa_unroll "${DNCNN_SRC}" -Ofast -fno-ipa-cp -fno-ipa-sra -funroll-loops "${DNCNN_BASE[@]}"
build_one dncnn f20b_dncnn_ofast_notreecopy "${DNCNN_SRC}" -Ofast -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize "${DNCNN_BASE[@]}"
build_one dncnn f20b_dncnn_ofast_rename "${DNCNN_SRC}" -Ofast -frename-registers "${DNCNN_BASE[@]}"

DEPTH_SRC="${ROOT}/erbium_amp_probe/depth-vit-bench/depth_vit_vpu_argbuf.c"
DEPTH_BASE=(-DDEPTH_PASSES=32 -DDEPTH_VPU_OC2=1 -DDEPTH_PREFETCH_B=1 -DDEPTH_SKIP_READ_EVICT=1)
build_one depth f20b_depth_o3_unroll_noipa_prefb_skipread "${DEPTH_SRC}" -O3 -funroll-loops -fno-ipa-cp -fno-ipa-sra "${DEPTH_BASE[@]}"
build_one depth f20b_depth_o3_unroll_nosched_prefb_skipread "${DEPTH_SRC}" -O3 -funroll-loops -fno-schedule-insns -fno-schedule-insns2 "${DEPTH_BASE[@]}"
build_one depth f20b_depth_o3_unroll_rename_prefb_skipread "${DEPTH_SRC}" -O3 -funroll-loops -frename-registers "${DEPTH_BASE[@]}"
build_one depth f20b_depth_o3_unroll_notreecopy_prefb_skipread "${DEPTH_SRC}" -O3 -funroll-loops -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize "${DEPTH_BASE[@]}"

WHISPER_SRC="${ROOT}/erbium_amp_probe/whisper-bench/whisper_transformer_vpu_argbuf.c"
WHISPER_BASE=(-DDEPTH_PASSES=16 -DDEPTH_VPU_OC2=1 -DDEPTH_PREFETCH_A_ROWS=1 -DDEPTH_PREFETCH_B=1 -DDEPTH_SKIP_READ_EVICT=1)
build_one whisper f20b_whisper_o3_unroll_align32_prefab_skipread "${WHISPER_SRC}" -O3 -funroll-loops -falign-functions=32 -falign-loops=32 "${WHISPER_BASE[@]}"
build_one whisper f20b_whisper_o3_unroll_align64_prefab_skipread "${WHISPER_SRC}" -O3 -funroll-loops -falign-functions=64 -falign-loops=64 "${WHISPER_BASE[@]}"
build_one whisper f20b_whisper_o3_unroll_noipa_prefab_skipread "${WHISPER_SRC}" -O3 -funroll-loops -fno-ipa-cp -fno-ipa-sra "${WHISPER_BASE[@]}"
build_one whisper f20b_whisper_ofast_unroll_prefab_skipread "${WHISPER_SRC}" -Ofast -funroll-loops "${WHISPER_BASE[@]}"

STEREO_SRC="${ROOT}/erbium_amp_probe/stereo-bench/stereo_corr_vpu_argbuf.c"
STEREO_BASE=(-DSTEREO_PASSES=64 -DSTEREO_DISP2=1 -DSTEREO_PREFETCH_LEFT_ROWS=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1)
build_one stereo f20b_stereo_ofast_unroll_align32_prefl_lastout "${STEREO_SRC}" -Ofast -funroll-loops -falign-functions=32 -falign-loops=32 "${STEREO_BASE[@]}"
build_one stereo f20b_stereo_ofast_unroll_align64_prefl_lastout "${STEREO_SRC}" -Ofast -funroll-loops -falign-functions=64 -falign-loops=64 "${STEREO_BASE[@]}"
build_one stereo f20b_stereo_ofast_unroll_noipa_prefl_lastout "${STEREO_SRC}" -Ofast -funroll-loops -fno-ipa-cp -fno-ipa-sra "${STEREO_BASE[@]}"
build_one stereo f20b_stereo_ofast_unroll_skipread_prefl_lastout "${STEREO_SRC}" -Ofast -funroll-loops "${STEREO_BASE[@]}" -DSTEREO_SKIP_READ_EVICT=1

YOLO_SRC="${ROOT}/erbium_amp_probe/yolo-bench/yolo_vpu_argbuf.c"
YOLO_BASE=(-DYOLO_PASSES=4 -DYOLO_VPU_OC2=1 -DYOLO_SKIP_READ_EVICT=1)
build_one yolo f20b_yolo_ofast_unroll_align32_skipread "${YOLO_SRC}" -Ofast -funroll-loops -falign-functions=32 -falign-loops=32 "${YOLO_BASE[@]}"
build_one yolo f20b_yolo_ofast_unroll_align64_skipread "${YOLO_SRC}" -Ofast -funroll-loops -falign-functions=64 -falign-loops=64 "${YOLO_BASE[@]}"
build_one yolo f20b_yolo_ofast_unroll_noipa_skipread "${YOLO_SRC}" -Ofast -funroll-loops -fno-ipa-cp -fno-ipa-sra "${YOLO_BASE[@]}"
build_one yolo f20b_yolo_ofast_unroll_skipfinal_skipread "${YOLO_SRC}" -Ofast -funroll-loops "${YOLO_BASE[@]}" -DYOLO_SKIP_FINAL_BARRIER=1
