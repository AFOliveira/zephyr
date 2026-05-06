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
	local zero_image="$1"
	shift

	cd "${PARENT}/${work}"
	echo "=== ${model} ${variant} ==="

	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,${zero_image} \
		--dump_after "pmc_before_current_${model}_${variant}_${STAMP}.bin" \
		--timeout 60 \
		> "run_pmc_before_current_${model}_${variant}_${STAMP}.log" 2>&1

	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		"$@" \
		--dump_after "dump_pmc_current_${model}_${variant}_${STAMP}.bin" \
		--timeout 240 \
		> "run_pmc_current_${model}_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_pmc_current_${model}_${variant}_${STAMP}.log" || true

	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,${zero_image} \
		--dump_after "pmc_after_current_${model}_${variant}_${STAMP}.bin" \
		--timeout 60 \
		> "run_pmc_after_current_${model}_${variant}_${STAMP}.log" 2>&1
}

run_case dncnn optimization-kb/focused20b f20b_dncnn_ofast_rename \
	../../zero2m.bin \
	--file_load 0x0,../../zero2m.bin \
	--file_load 0x2000,../../dncnn3_input.bin \
	--file_load 0x4000,../../dncnn3_weights.bin

run_case depth optimization-kb/focused20b f20b_depth_o3_unroll_noipa_prefb_skipread \
	../../zero2m.bin \
	--file_load 0x0,../../zero2m.bin

run_case whisper optimization-kb/focused20 f20_whisper_o3_unroll_prefa_prefb_skipread \
	../../zero2m.bin \
	--file_load 0x0,../../zero2m.bin

run_case stereo optimization-kb/focused20 f20_stereo_ofast_unroll_prefl_lastout \
	../../zero2m.bin \
	--file_load 0x0,../../zero2m.bin

run_case yolo optimization-kb/focused20b f20b_yolo_ofast_unroll_align32_skipread \
	../../zero2m.bin \
	--file_load 0x0,../../zero2m.bin

echo "${STAMP}"
