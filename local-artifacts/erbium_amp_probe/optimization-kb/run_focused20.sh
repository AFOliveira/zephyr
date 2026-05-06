#!/usr/bin/env bash
set -euo pipefail

BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT="${BASE}/erbium-amp-probe"
WORK="${PARENT}/optimization-kb/focused20"
LAUNCH="${PARENT}/erbium_soc1sim_argbuf"
STAMP="$(date +%Y%m%d-%H%M%S)"

cd "${WORK}"
export LD_LIBRARY_PATH="${BASE}:${PARENT}:${LD_LIBRARY_PATH:-}"

while IFS=$'\t' read -r model variant; do
	[ -n "${model}" ] || continue
	echo "=== ${model} ${variant} ==="

	args=(--file_load 0x0,../../zero2m.bin)
	if [ "${model}" = "dncnn" ]; then
		args+=(--file_load 0x2000,../../dncnn3_input.bin)
		args+=(--file_load 0x4000,../../dncnn3_weights.bin)
	fi

	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		"${args[@]}" \
		--dump_after "dump_${model}_${variant}_${STAMP}.bin" \
		--timeout 240 \
		> "run_${model}_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_${model}_${variant}_${STAMP}.log" || true
done < focused20_variants.tsv

echo "${STAMP}"
