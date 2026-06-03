/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Strict-contract smoke sample: main() performs no raw custom instruction
 * execution.  Encoder checks live in aif_encode_verify.c; raw emission lives
 * in aif_compile_only_emitters.c (retained, not called from main()).
 */

#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

#if defined(CONFIG_AIF_ET_ISA_MIGRATION)
#include <zephyr/arch/riscv/aif/xaifet.h>
#endif

int main(void)
{
#if !defined(CONFIG_AIF_ET_ISA_MIGRATION)
	printk("isa-migration-smoke: CONFIG_AIF_ET_ISA_MIGRATION disabled\n");
	return 1;
#else
	printk("isa-migration-smoke: strict compile-only PASS (no raw insn in main)\n");
	printk("  expected packb x10=0x%x bitmixb x10=0x%x summit x10=0x%x\n",
	       AIF_VECTOR_EXPECT_PACKB, AIF_VECTOR_EXPECT_BITMIXB,
	       AIF_VECTOR_EXPECT_SUMMIT);
	return 0;
#endif
}
