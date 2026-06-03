/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Constant-only encoders and emitters for the nine approved ISA migration
 * targets in harness/config/encoding_plan.yaml.  Runtime-variable instruction
 * words must not be passed to inline asm immediates; use the vector word
 * constants or the AIF_ENCODE_* macros at compile time.
 */

#ifndef ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_
#define ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_

#include <stdint.h>

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
#define AIF_MATCH_AIF_EUROPERISCVSUMMIT 0x0000402bU
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

#define AIF_ENCODE_FCMOV_PS(match, mask, rd, rs1, rs2, rs3)                   \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rd) & 0x1fU) << 7) | (((rs1) & 0x1fU) << 15) |       \
		       (((rs2) & 0x1fU) << 20) | (((rs3) & 0x1fU) << 27))

#define AIF_ENCODE_FSQ2(match, mask, rs1, rs2, imm12)                          \
	AIF_MERGE_INSN((match), (mask),                                          \
		       (((rs2) & 0x1fU) << 20) | (((rs1) & 0x1fU) << 15) |       \
		       (((imm12) & 0x1fU) << 7) | ((((imm12) >> 5) & 0x7fU) << 25))

#define AIF_EMIT_WORD(word)                                                    \
	do {                                                                   \
		__asm__ volatile(".word %0" : : "i"((uint32_t)(word)) : "memory"); \
	} while (0)

/* Pre-encoded vector words for BUILD_ASSERT (no commas in macro use sites). */
#define AIF_VECTOR_ENC_FLQ2                                                  \
	AIF_ENCODE_I_TYPE(AIF_MATCH_FLQ2, AIF_MASK_FLQ2, 10, 11,             \
			  AIF_VECTOR_GPR_IMM)
#define AIF_VECTOR_ENC_FSQ2                                                  \
	AIF_ENCODE_FSQ2(AIF_MATCH_FSQ2, AIF_MASK_FSQ2, 11, 12,               \
			AIF_VECTOR_GPR_IMM)
#define AIF_VECTOR_ENC_FADDI_PI                                              \
	AIF_ENCODE_FADDI_PI(AIF_MATCH_FADDI_PI, AIF_MASK_FADDI_PI, 10, 11,   \
			    AIF_VECTOR_GPR_IMM)
#define AIF_VECTOR_ENC_FANDI_PI                                              \
	AIF_ENCODE_FADDI_PI(AIF_MATCH_FANDI_PI, AIF_MASK_FANDI_PI, 10, 11,   \
			    AIF_VECTOR_GPR_IMM)
#define AIF_VECTOR_ENC_FCMOV_PS                                              \
	AIF_ENCODE_FCMOV_PS(AIF_MATCH_FCMOV_PS, AIF_MASK_FCMOV_PS, 10, 11, 12, 13)
#define AIF_VECTOR_ENC_FCMOVM_PS                                             \
	AIF_ENCODE_R_TYPE(AIF_MATCH_FCMOVM_PS, AIF_MASK_FCMOVM_PS, 10, 11, 12)
#define AIF_VECTOR_ENC_PACKB                                                 \
	AIF_ENCODE_R_TYPE(AIF_MATCH_PACKB, AIF_MASK_PACKB, 10, 11, 12)
#define AIF_VECTOR_ENC_BITMIXB                                               \
	AIF_ENCODE_R_TYPE(AIF_MATCH_BITMIXB, AIF_MASK_BITMIXB, 10, 11, 12)
#define AIF_VECTOR_ENC_SUMMIT                                                \
	AIF_ENCODE_I_TYPE(AIF_MATCH_AIF_EUROPERISCVSUMMIT,                   \
			  AIF_MASK_AIF_EUROPERISCVSUMMIT, 10, 11,             \
			  AIF_VECTOR_GPR_IMM)

#endif /* ZEPHYR_INCLUDE_ARCH_RISCV_AIF_XAIFET_H_ */
