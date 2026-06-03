/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compile-only retained raw-word emitters for all nine approved targets.
 * Not referenced from main(); linker retains via the table below.
 */

#include <zephyr/arch/riscv/aif/xaifet.h>

typedef void (*aif_emit_fn)(void);

#define AIF_VECTOR_RS1_SETUP()                                                 \
	do {                                                                   \
		__asm__ volatile("mv a1, %0" : : "r"(AIF_VECTOR_GPR_RS1_VAL)     \
				 : "a1");                                      \
		__asm__ volatile("mv a2, %0" : : "r"(AIF_VECTOR_GPR_RS2_VAL)     \
				 : "a2");                                      \
	} while (0)

__attribute__((noinline, used))
void aif_co_emit_flq2(void)
{
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FLQ2);
}

__attribute__((noinline, used))
void aif_co_emit_fsq2(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FSQ2);
}

__attribute__((noinline, used))
void aif_co_emit_faddi_pi(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FADDI_PI);
}

__attribute__((noinline, used))
void aif_co_emit_fandi_pi(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FANDI_PI);
}

__attribute__((noinline, used))
void aif_co_emit_fcmov_ps(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FCMOV_PS);
}

__attribute__((noinline, used))
void aif_co_emit_fcmovm_ps(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_FCMOVM_PS);
}

__attribute__((noinline, used))
void aif_co_emit_packb(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_PACKB);
}

__attribute__((noinline, used))
void aif_co_emit_bitmixb(void)
{
	AIF_VECTOR_RS1_SETUP();
	AIF_EMIT_WORD(AIF_VECTOR_WORD_BITMIXB);
}

__attribute__((noinline, used))
void aif_co_emit_summit(void)
{
	__asm__ volatile("mv a1, %0" : : "r"(AIF_VECTOR_GPR_RS1_VAL) : "a1");
	AIF_EMIT_WORD(AIF_VECTOR_WORD_AIF_EUROPERISCVSUMMIT);
}

__attribute__((used))
const aif_emit_fn aif_compile_only_retain_table[] = {
	aif_co_emit_flq2,
	aif_co_emit_fsq2,
	aif_co_emit_faddi_pi,
	aif_co_emit_fandi_pi,
	aif_co_emit_fcmov_ps,
	aif_co_emit_fcmovm_ps,
	aif_co_emit_packb,
	aif_co_emit_bitmixb,
	aif_co_emit_summit,
};
