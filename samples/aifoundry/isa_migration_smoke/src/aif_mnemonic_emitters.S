/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Migration mnemonics assembled with patched Binutils (rv64i_xaifet).
 * Canonical assembler forms match agentws/jobs/isa-toolchain round-trip tests.
 * Retained in the ELF but not called from main().
 */

	.section .text.aif_mnemonic_emitters, "ax", @progbits
	.align 2

	.globl aif_mn_emit_flq2
	.type aif_mn_emit_flq2, %function
aif_mn_emit_flq2:
	flq2 ft0, 0(a0)
	ret
	.size aif_mn_emit_flq2, . - aif_mn_emit_flq2

	.globl aif_mn_emit_fsq2
	.type aif_mn_emit_fsq2, %function
aif_mn_emit_fsq2:
	fsq2 ft0, 0(a0)
	ret
	.size aif_mn_emit_fsq2, . - aif_mn_emit_fsq2

	.globl aif_mn_emit_faddi_pi
	.type aif_mn_emit_faddi_pi, %function
aif_mn_emit_faddi_pi:
	faddi.pi ft0, ft1, 0
	ret
	.size aif_mn_emit_faddi_pi, . - aif_mn_emit_faddi_pi

	.globl aif_mn_emit_fandi_pi
	.type aif_mn_emit_fandi_pi, %function
aif_mn_emit_fandi_pi:
	fandi.pi ft0, ft1, 0
	ret
	.size aif_mn_emit_fandi_pi, . - aif_mn_emit_fandi_pi

	.globl aif_mn_emit_fcmov_ps
	.type aif_mn_emit_fcmov_ps, %function
aif_mn_emit_fcmov_ps:
	fcmov.ps ft0, ft1, ft2, ft3
	ret
	.size aif_mn_emit_fcmov_ps, . - aif_mn_emit_fcmov_ps

	.globl aif_mn_emit_fcmovm_ps
	.type aif_mn_emit_fcmovm_ps, %function
aif_mn_emit_fcmovm_ps:
	fcmovm.ps ft0, ft1, ft2
	ret
	.size aif_mn_emit_fcmovm_ps, . - aif_mn_emit_fcmovm_ps

	.globl aif_mn_emit_packb
	.type aif_mn_emit_packb, %function
aif_mn_emit_packb:
	packb a0, a1, a2
	ret
	.size aif_mn_emit_packb, . - aif_mn_emit_packb

	.globl aif_mn_emit_bitmixb
	.type aif_mn_emit_bitmixb, %function
aif_mn_emit_bitmixb:
	bitmixb a0, a1, a2
	ret
	.size aif_mn_emit_bitmixb, . - aif_mn_emit_bitmixb

	.globl aif_mn_emit_summit
	.type aif_mn_emit_summit, %function
aif_mn_emit_summit:
	aif.europeriscvsummit a0, a1, 42
	ret
	.size aif_mn_emit_summit, . - aif_mn_emit_summit
