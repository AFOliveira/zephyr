/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Smoke sample for the nine approved ISA migration targets.  Integer targets
 * execute on the target hart and check GPR results against the simulator
 * vectors; FP/load/store/PI targets compile raw emission and BUILD_ASSERT the
 * encoded word matches the vector target (runtime FP semantics are not checked
 * in this Zephyr image).
 */

#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

#if defined(CONFIG_AIF_ET_ISA_MIGRATION)
#include <zephyr/arch/riscv/aif/xaifet.h>
#endif

#define RS1_VAL AIF_VECTOR_GPR_RS1_VAL
#define RS2_VAL AIF_VECTOR_GPR_RS2_VAL
#define IMM_VAL AIF_VECTOR_GPR_IMM

#if defined(CONFIG_AIF_ET_ISA_MIGRATION)

BUILD_ASSERT(AIF_ENCODE_I_TYPE(AIF_MATCH_FLQ2, AIF_MASK_FLQ2, 10, 11, IMM_VAL) ==
	     AIF_VECTOR_WORD_FLQ2);
BUILD_ASSERT(AIF_ENCODE_FSQ2(AIF_MATCH_FSQ2, AIF_MASK_FSQ2, 11, 12, IMM_VAL) ==
	     AIF_VECTOR_WORD_FSQ2);
BUILD_ASSERT(AIF_ENCODE_FADDI_PI(AIF_MATCH_FADDI_PI, AIF_MASK_FADDI_PI, 10, 11,
				 IMM_VAL) == AIF_VECTOR_WORD_FADDI_PI);
BUILD_ASSERT(AIF_ENCODE_FADDI_PI(AIF_MATCH_FANDI_PI, AIF_MASK_FANDI_PI, 10, 11,
				 IMM_VAL) == AIF_VECTOR_WORD_FANDI_PI);
BUILD_ASSERT(AIF_ENCODE_FCMOV_PS(AIF_MATCH_FCMOV_PS, AIF_MASK_FCMOV_PS, 10, 11, 12,
				 13) == AIF_VECTOR_WORD_FCMOV_PS);
BUILD_ASSERT(AIF_ENCODE_R_TYPE(AIF_MATCH_FCMOVM_PS, AIF_MASK_FCMOVM_PS, 10, 11,
			       12) == AIF_VECTOR_WORD_FCMOVM_PS);
BUILD_ASSERT(AIF_ENCODE_R_TYPE(AIF_MATCH_PACKB, AIF_MASK_PACKB, 10, 11, 12) ==
	     AIF_VECTOR_WORD_PACKB);
BUILD_ASSERT(AIF_ENCODE_R_TYPE(AIF_MATCH_BITMIXB, AIF_MASK_BITMIXB, 10, 11, 12) ==
	     AIF_VECTOR_WORD_BITMIXB);
BUILD_ASSERT(AIF_ENCODE_I_TYPE(AIF_MATCH_AIF_EUROPERISCVSUMMIT,
			       AIF_MASK_AIF_EUROPERISCVSUMMIT,
			       10, 11, IMM_VAL) == AIF_VECTOR_WORD_AIF_EUROPERISCVSUMMIT);

static void emit_fp_migration_smoke(void)
{
	/* Keep emission paths in the linked image; operands match vector metadata. */
	aif_emit_flq2(10, 11, IMM_VAL);
	aif_emit_fsq2(11, 12, IMM_VAL);
	aif_emit_faddi_pi(10, 11, IMM_VAL);
	aif_emit_fandi_pi(10, 11, IMM_VAL);
	aif_emit_fcmov_ps(10, 11, 12, 13);
	aif_emit_fcmovm_ps(10, 11, 12);
}

#endif /* CONFIG_AIF_ET_ISA_MIGRATION */

int main(void)
{
#if !defined(CONFIG_AIF_ET_ISA_MIGRATION)
	printk("isa-migration-smoke: CONFIG_AIF_ET_ISA_MIGRATION disabled\n");
	return 1;
#else
	uint32_t packb_out;
	uint32_t bitmixb_out;
	uint32_t summit_out;

	emit_fp_migration_smoke();

	packb_out = aif_packb(RS1_VAL, RS2_VAL);
	if (packb_out != AIF_VECTOR_EXPECT_PACKB) {
		printk("isa-migration-smoke: packb fail got=0x%x expect=0x%x\n",
		       packb_out, AIF_VECTOR_EXPECT_PACKB);
		return 2;
	}

	bitmixb_out = aif_bitmixb(RS1_VAL, RS2_VAL);
	if (bitmixb_out != AIF_VECTOR_EXPECT_BITMIXB) {
		printk("isa-migration-smoke: bitmixb fail got=0x%x expect=0x%x\n",
		       bitmixb_out, AIF_VECTOR_EXPECT_BITMIXB);
		return 3;
	}

	summit_out = aif_europeriscvsummit(RS1_VAL, IMM_VAL);
	if (summit_out != AIF_VECTOR_EXPECT_SUMMIT) {
		printk("isa-migration-smoke: summit fail got=0x%x expect=0x%x\n",
		       summit_out, AIF_VECTOR_EXPECT_SUMMIT);
		return 4;
	}

	printk("isa-migration-smoke: PASS packb=0x%x bitmixb=0x%x summit=0x%x\n",
	       packb_out, bitmixb_out, summit_out);
	return 0;
#endif
}
