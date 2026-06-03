/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compile-time encoder checks (kept out of main()).
 */

#include <zephyr/arch/riscv/aif/xaifet.h>

typedef void (*aif_emit_fn)(void);

extern const aif_emit_fn aif_compile_only_retain_table[];

enum {
	_aif_v_flq2 = AIF_VECTOR_ENC_FLQ2,
	_aif_v_fsq2 = AIF_VECTOR_ENC_FSQ2,
	_aif_v_faddi_pi = AIF_VECTOR_ENC_FADDI_PI,
	_aif_v_fandi_pi = AIF_VECTOR_ENC_FANDI_PI,
	_aif_v_fcmov_ps = AIF_VECTOR_ENC_FCMOV_PS,
	_aif_v_fcmovm_ps = AIF_VECTOR_ENC_FCMOVM_PS,
	_aif_v_packb = AIF_VECTOR_ENC_PACKB,
	_aif_v_bitmixb = AIF_VECTOR_ENC_BITMIXB,
	_aif_v_summit = AIF_VECTOR_ENC_SUMMIT,
};

_Static_assert(_aif_v_flq2 == AIF_VECTOR_WORD_FLQ2, "flq2 enc");
_Static_assert(_aif_v_fsq2 == AIF_VECTOR_WORD_FSQ2, "fsq2 enc");
_Static_assert(_aif_v_faddi_pi == AIF_VECTOR_WORD_FADDI_PI, "faddi.pi enc");
_Static_assert(_aif_v_fandi_pi == AIF_VECTOR_WORD_FANDI_PI, "fandi.pi enc");
_Static_assert(_aif_v_fcmov_ps == AIF_VECTOR_WORD_FCMOV_PS, "fcmov.ps enc");
_Static_assert(_aif_v_fcmovm_ps == AIF_VECTOR_WORD_FCMOVM_PS, "fcmovm.ps enc");
_Static_assert(_aif_v_packb == AIF_VECTOR_WORD_PACKB, "packb enc");
_Static_assert(_aif_v_bitmixb == AIF_VECTOR_WORD_BITMIXB, "bitmixb enc");
_Static_assert(_aif_v_summit == AIF_VECTOR_WORD_AIF_EUROPERISCVSUMMIT, "summit enc");

__attribute__((used))
void aif_encode_verify(void)
{
	/* Retain compile-only emitters without executing them from main(). */
	volatile const void *retain = aif_compile_only_retain_table;

	(void)retain;
}
