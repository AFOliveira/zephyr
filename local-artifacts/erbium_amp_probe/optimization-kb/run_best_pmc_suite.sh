#!/usr/bin/env bash
set -euo pipefail

BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT="${BASE}/erbium-amp-probe"
LAUNCH="${PARENT}/erbium_soc1sim_argbuf"
SNAP="${PARENT}/dncnn3-pmc/pmc_snapshot_16h.elf"
STAMP="$(date +%Y%m%d-%H%M%S)"

export LD_LIBRARY_PATH="${BASE}:${PARENT}:${LD_LIBRARY_PATH:-}"

run_case() {
	local model="$1"
	local work="$2"
	local variant="$3"
	shift 3

	cd "${PARENT}/${work}"
	echo "=== ${model} ${variant} ==="

	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--dump_after "pmc_before_${model}_${variant}_${STAMP}.bin" \
		--timeout 60 \
		> "run_pmc_before_${model}_${variant}_${STAMP}.log" 2>&1

	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		"$@" \
		--dump_after "dump_pmc_${model}_${variant}_${STAMP}.bin" \
		--timeout 240 \
		> "run_pmc_${model}_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_pmc_${model}_${variant}_${STAMP}.log" || true

	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--dump_after "pmc_after_${model}_${variant}_${STAMP}.bin" \
		--timeout 60 \
		> "run_pmc_after_${model}_${variant}_${STAMP}.log" 2>&1
}

run_case dncnn dncnn3-pmc v3100_o08_ofast_m09_accfinal_prefw \
	--file_load 0x0,../zero2m.bin \
	--file_load 0x2000,../dncnn3_input.bin \
	--file_load 0x4000,../dncnn3_weights.bin

run_case depth depth-vit-bench d100_o02_o3_unroll_m07_prefb_skipread \
	--file_load 0x0,../zero2m.bin

run_case whisper whisper-bench w100_o02_o3_unroll_m04_prefa_prefb \
	--file_load 0x0,../zero2m.bin

run_case stereo stereo-bench s100_o02_o3_unroll_m09_prefl_lastout \
	--file_load 0x0,../zero2m.bin

run_case yolo yolo-bench y100_o08_ofast_m03_skipread \
	--file_load 0x0,../zero2m.bin

echo "${STAMP}"
