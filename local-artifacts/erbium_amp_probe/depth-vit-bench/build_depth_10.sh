#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/erbium_amp_probe/depth-vit-bench/depth_vit_vpu_argbuf.c"
OUT="${ROOT}/erbium_amp_probe/depth-vit-bench"
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
	-DDEPTH_PASSES=32
)

names=(
	d10_00_base
	d10_01_oc2
	d10_02_oc2_prefstatic
	d10_03_oc2_prefb
	d10_04_oc2_prefa
	d10_05_oc2_skipread
	d10_06_oc2_lastout
	d10_07_oc2_skipread_lastout
	d10_08_oc2_unroll
	d10_09_oc2_ofast
)

opts=(
	"-O3"
	"-O3 -DDEPTH_VPU_OC2=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_PREFETCH_STATIC=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_PREFETCH_B=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_PREFETCH_A_ROWS=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_SKIP_READ_EVICT=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_EVICT_OUTPUT_LAST_ONLY=1"
	"-O3 -DDEPTH_VPU_OC2=1 -DDEPTH_SKIP_READ_EVICT=1 -DDEPTH_EVICT_OUTPUT_LAST_ONLY=1"
	"-O3 -funroll-loops -DDEPTH_VPU_OC2=1"
	"-Ofast -DDEPTH_VPU_OC2=1 -DDEPTH_SKIP_READ_EVICT=1"
)

: > "${OUT}/depth_10_variants.txt"

for i in "${!names[@]}"; do
	name="${names[$i]}"
	read -r -a opt_array <<< "${opts[$i]}"
	echo "${name}" >> "${OUT}/depth_10_variants.txt"
	"${GCC}" "${opt_array[@]}" "${COMMON[@]}" \
		-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
done
