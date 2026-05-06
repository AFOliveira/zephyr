#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/erbium_amp_probe/stereo-bench/stereo_corr_vpu_argbuf.c"
OUT="${ROOT}/erbium_amp_probe/stereo-bench"
CRT="${ROOT}/erbium_amp_probe/hart-report/hart_report_crt.S"

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
	-DSTEREO_PASSES=64
	-DSTEREO_DISP2=1
)

opt_names=(
	o00_o3
	o01_o3_nosched
	o02_o3_unroll
	o03_o3_nounroll
	o04_o3_align32
	o05_o3_align64
	o06_o3_noipa
	o07_o3_notreecopy
	o08_ofast
	o09_o2
)

opt_flags=(
	"-O3"
	"-O3 -fno-schedule-insns -fno-schedule-insns2"
	"-O3 -funroll-loops"
	"-O3 -fno-unroll-loops"
	"-O3 -falign-functions=32 -falign-loops=32"
	"-O3 -falign-functions=64 -falign-loops=64"
	"-O3 -fno-ipa-cp -fno-ipa-sra"
	"-O3 -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize"
	"-Ofast"
	"-O2"
)

macro_names=(
	m00_base
	m01_prefr
	m02_prefl
	m03_skipread
	m04_lastout
	m05_skipread_lastout
	m06_prefr_skipread
	m07_prefl_skipread
	m08_prefr_lastout
	m09_prefl_lastout
)

macro_flags=(
	""
	"-DSTEREO_PREFETCH_RIGHT=1"
	"-DSTEREO_PREFETCH_LEFT_ROWS=1"
	"-DSTEREO_SKIP_READ_EVICT=1"
	"-DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
	"-DSTEREO_SKIP_READ_EVICT=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
	"-DSTEREO_PREFETCH_RIGHT=1 -DSTEREO_SKIP_READ_EVICT=1"
	"-DSTEREO_PREFETCH_LEFT_ROWS=1 -DSTEREO_SKIP_READ_EVICT=1"
	"-DSTEREO_PREFETCH_RIGHT=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
	"-DSTEREO_PREFETCH_LEFT_ROWS=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
)

: > "${OUT}/stereo_100_variants.txt"

for oi in "${!opt_names[@]}"; do
	for mi in "${!macro_names[@]}"; do
		name="s100_${opt_names[$oi]}_${macro_names[$mi]}"
		read -r -a opt_array <<< "${opt_flags[$oi]}"
		read -r -a macro_array <<< "${macro_flags[$mi]}"
		echo "${name}" >> "${OUT}/stereo_100_variants.txt"
		"${GCC}" "${opt_array[@]}" "${COMMON[@]}" "${macro_array[@]}" \
			-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
	done
done
