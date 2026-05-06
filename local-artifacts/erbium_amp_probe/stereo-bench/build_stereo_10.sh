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
)

names=(
	s10_00_base
	s10_01_disp2
	s10_02_disp2_prefr
	s10_03_disp2_prefl
	s10_04_disp2_skipread
	s10_05_disp2_lastout
	s10_06_disp2_skipread_lastout
	s10_07_disp2_prefr_skipread
	s10_08_disp2_unroll
	s10_09_disp2_ofast
)

opts=(
	"-O3"
	"-O3 -DSTEREO_DISP2=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_PREFETCH_RIGHT=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_PREFETCH_LEFT_ROWS=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_SKIP_READ_EVICT=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_SKIP_READ_EVICT=1 -DSTEREO_EVICT_OUTPUT_LAST_ONLY=1"
	"-O3 -DSTEREO_DISP2=1 -DSTEREO_PREFETCH_RIGHT=1 -DSTEREO_SKIP_READ_EVICT=1"
	"-O3 -funroll-loops -DSTEREO_DISP2=1"
	"-Ofast -DSTEREO_DISP2=1"
)

: > "${OUT}/stereo_10_variants.txt"

for i in "${!names[@]}"; do
	name="${names[$i]}"
	read -r -a opt_array <<< "${opts[$i]}"
	echo "${name}" >> "${OUT}/stereo_10_variants.txt"
	"${GCC}" "${opt_array[@]}" "${COMMON[@]}" \
		-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
done
