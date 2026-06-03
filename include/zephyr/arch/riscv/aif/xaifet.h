/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Static inline emitters for the nine approved AIFoundry ISA migration targets
 * in harness/config/encoding_plan.yaml.  Words are built from MATCH/MASK plus
 * register/immediate fields; when the Zephyr SDK assembler lacks a migrated
 * mnemonic, emission uses a raw .word.
 */

#ifndef ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_
#define ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_

#include <stdint.h>

#include <zephyr/toolchain.h>

/* MATCH/MASK from encoding_plan.yaml (approved proposed entries). */
#define AIF_MATCH_FLQ2                  0x0000500bU
#define AIF_MASK_FLQ2                   0x0000707fU
#define AIF_MATCH_FSQ2                  0x0000502bU
#define AIF_MASK_FSQ2                   0x0000707fU
#define AIF_MATCH_FADDI_PI              0x0400002bU
#define AIF_MASK_FADDI_PI               0x0600707fU
#define AIF_MATCH_FANDI_PI              0x0400102bU
#define AIF_MASK_FANDI_PI               0x0600707fU
#define AIF_MATCH_FCMOV_PS              0x0400202bU
#define AIF_MASK_FCMOV_PS               0x0600707fU
#define AIF_MATCH_FCMOVM_PS             0x0000002bU
#define AIF_MASK_FCMOVM_PS              0xfe00707fU
#define AIF_MATCH_PACKB                 0x8000602bU
#define AIF_MASK_PACKB                  0xfe00707fU
#define AIF_MATCH_BITMIXB               0x8000702bU
#define AIF_MASK_BITMIXB                0xfe00707fU
#define AIF_MATCH_AIF_EUROPERISCVSUMMIT  0x0000402bU
#define AIF_MASK_AIF_EUROPERISCVSUMMIT  0x0000707fU

/* Simulator vector target words (rd=x10, rs1=x11, rs2=x12, rs3=x13, imm=7). */
#define AIF_VECTOR_WORD_FLQ2                  0x0075d50bU
#define AIF_VECTOR_WORD_FSQ2                  0x00c5d3abU
#define AIF_VECTOR_WORD_FADDI_PI              0x0475852bU
#define AIF_VECTOR_WORD_FANDI_PI              0x0475952bU
#define AIF_VECTOR_WORD_FCMOV_PS              0x6cc5a52bU
#define AIF_VECTOR_WORD_FCMOVM_PS             0x00c5852bU
#define AIF_VECTOR_WORD_PACKB                 0x80c5e52bU
#define AIF_VECTOR_WORD_BITMIXB               0x80c5f52bU
#define AIF_VECTOR_WORD_AIF_EUROPERISCVSUMMIT 0x0075c52bU

#define AIF_VECTOR_GPR_RS1_VAL 0xaa11U
#define AIF_VECTOR_GPR_RS2_VAL 0xbb22U
#define AIF_VECTOR_GPR_IMM     7U
#define AIF_VECTOR_EXPECT_PACKB 0x2211U
#define AIF_VECTOR_EXPECT_BITMIXB 0xf895U
#define AIF_VECTOR_EXPECT_SUMMIT 0xaa18U

#define AIF_MERGE_INSN(match, mask, fields) \
	(((match) & (mask)) | ((fields) & ~(mask)))

#define AIF_ENCODE_R_TYPE(match, mask, rd, rs1, rs2)                           \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rd) & 0x1fU) << 7) | (((rs1) & 0x1fU) << 15) |       \
		       (((rs2) & 0x1fU) << 20))

#define AIF_ENCODE_I_TYPE(match, mask, rd, rs1, imm12)                         \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rd) & 0x1fU) << 7) | (((rs1) & 0x1fU) << 15) |       \
		       (((imm12) & 0xfffU) << 20))

#define AIF_ENCODE_FADDI_PI(match, mask, rd, rs1, imm10)                       \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rd) & 0x1fU) << 7) | (((rs1) & 0x1fU) << 15) |       \
		       (((imm10) & 0x1fU) << 20) |                               \
		       ((((imm10) >> 5) & 0x1fU) << 27))

#define AIF_ENCODE_FCMOV_PS(match, mask, rd, rs1, rs2, rs3)                  \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rd) & 0x1fU) << 7) | (((rs1) & 0x1fU) << 15) |       \
		       (((rs2) & 0x1fU) << 20) | (((rs3) & 0x1fU) << 27))

#define AIF_ENCODE_FSQ2(match, mask, rs1, rs2, imm12)                          \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rs2) & 0x1fU) << 20) | (((rs1) & 0x1fU) << 15) |       \
		       (((imm12) & 0x1fU) << 7) | (((imm12) >> 5) & 0x7fU) << 25)

static inline void aif_emit_word(uint32_t word)
{
	__asm__ volatile(".word %0" : : "i"(word) : "memory");
}

static inline uint32_t aif_exec_word_r10(uint32_t word, uint32_t rs1, uint32_t rs2)
{
	uint32_t rd;

	__asm__ volatile(
		"mv a1, %1\n"
		"mv a2, %2\n"
		".word %3\n"
		"mv %0, a0\n"
		: "=r"(rd)
		: "r"(rs1), "r"(rs2), "i"(word)
		: "a0", "a1", "a2", "memory");
	return rd;
}

static inline uint32_t aif_packb(uint32_t rs1, uint32_t rs2)
{
	return aif_exec_word_r10(AIF_ENCODE_R_TYPE(AIF_MATCH_PACKB, AIF_MASK_PACKB,
						    10, 11, 12),
				 rs1, rs2);
}

static inline uint32_t aif_bitmixb(uint32_t rs1, uint32_t rs2)
{
	return aif_exec_word_r10(AIF_ENCODE_R_TYPE(AIF_MATCH_BITMIXB, AIF_MASK_BITMIXB,
						    10, 11, 12),
				 rs1, rs2);
}

static inline uint32_t aif_europeriscvsummit(uint32_t rs1, unsigned int imm12)
{
	uint32_t rd;
	uint32_t word = AIF_ENCODE_I_TYPE(AIF_MATCH_AIF_EUROPERISCVSUMMIT,
					  AIF_MASK_AIF_EUROPERISCVSUMMIT,
					  10, 11, imm12);

	__asm__ volatile(
		"mv a1, %1\n"
		".word %2\n"
		"mv %0, a0\n"
		: "=r"(rd)
		: "r"(rs1), "i"(word)
		: "a0", "a1", "memory");
	return rd;
}

static inline void aif_emit_flq2(unsigned int rd, unsigned int rs1, unsigned int imm12)
{
	aif_emit_word(AIF_ENCODE_I_TYPE(AIF_MATCH_FLQ2, AIF_MASK_FLQ2, rd, rs1, imm12));
}

static inline void aif_emit_fsq2(unsigned int rs1, unsigned int rs2, unsigned int imm12)
{
	aif_emit_word(AIF_ENCODE_FSQ2(AIF_MATCH_FSQ2, AIF_MASK_FSQ2, rs1, rs2, imm12));
}

static inline void aif_emit_faddi_pi(unsigned int rd, unsigned int rs1, unsigned int imm10)
{
	aif_emit_word(AIF_ENCODE_FADDI_PI(AIF_MATCH_FADDI_PI, AIF_MASK_FADDI_PI,
					    rd, rs1, imm10));
}

static inline void aif_emit_fandi_pi(unsigned int rd, unsigned int rs1, unsigned int imm10)
{
	aif_emit_word(AIF_ENCODE_FADDI_PI(AIF_MATCH_FANDI_PI, AIF_MASK_FANDI_PI,
					    rd, rs1, imm10));
}

static inline void aif_emit_fcmov_ps(unsigned int rd, unsigned int rs1, unsigned int rs2,
				     unsigned int rs3)
{
	aif_emit_word(AIF_ENCODE_FCMOV_PS(AIF_MATCH_FCMOV_PS, AIF_MASK_FCMOV_PS,
					    rd, rs1, rs2, rs3));
}

static inline void aif_emit_fcmovm_ps(unsigned int rd, unsigned int rs1, unsigned int rs2)
{
	aif_emit_word(AIF_ENCODE_R_TYPE(AIF_MATCH_FCMOVM_PS, AIF_MASK_FCMOVM_PS,
					rd, rs1, rs2));
}

#endif /* ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_ */
