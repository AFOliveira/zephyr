#!/usr/bin/env bash
# Verify migration mnemonic object with patched Binutils objdump (9/9).
set -euo pipefail

OBJ="${1:?object file}"
AS="${AIF_PATCHED_AS:?set AIF_PATCHED_AS}"
OD="${AIF_PATCHED_OBJDUMP:?set AIF_PATCHED_OBJDUMP}"
LOG="${2:-}"

disasm() {
	"$OD" -d -j .text.aif_mnemonic_emitters "$1" 2>/dev/null || \
		"$OD" -d "$1" 2>/dev/null
}

out=$(disasm "$OBJ")
log() { echo "$@" | tee -a "${LOG:-/dev/stderr}"; }

log "=== patched Binutils mnemonic verify: $OBJ ==="
log "AS=$AS"
log "OBJDUMP=$OD"
log "$out"

targets=(flq2 fsq2 faddi.pi fandi.pi fcmov.ps fcmovm.ps packb bitmixb europeriscvsummit)
fail=0
for t in "${targets[@]}"; do
	pat="${t//./\\.}"
	if echo "$out" | grep -Eiq "(aif\.)?${pat}([[:space:]]|,|\$)"; then
		log "PASS $t: mnemonic in patched objdump"
	else
		log "FAIL $t: mnemonic missing in patched objdump"
		fail=1
	fi
done

exit "$fail"
