/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * cm-umode syscall dispatcher (M-mode side).
 *
 * U-mode kernels ecall with SYSCALL_CACHE_OPS_* IDs that carry a pre-packed
 * CSR encoding in a1 (produced by cache_ops_priv_* wrappers from
 * isa/common/cacheops-umode.h).  We decompose that encoding and call the
 * cm-umode M-mode direct-CSR helpers from <isa/erbium/cacheops.h> so the
 * actual CSR write lives in one place -- shared between Zephyr-M, cm-umode
 * consumers, and (via the same custom CSR numbers) MachineMinion on
 * ET-SoC1.
 *
 * SYSCALL_RETURN_FROM_KERNEL is intercepted in umode_fault.c before
 * reaching here, so the case below only runs for out-of-band dispatch.
 *
 * Design note — "the tricky bit" the CMakeLists refers to: cm-umode's
 * flush_sw/evict_sw/... take *decomposed* fields (use_tmask, dst, way,
 * set, num_lines).  The U-mode wrapper packs those fields into a 64-bit
 * CSR encoding that we receive in a1.  We reverse the packing here.
 * This is what MachineMinion effectively does too (via its private
 * cache_ops_flush_sw(csr_enc) helper that just csrw's the value).  The
 * cleanest long-term fix would be to publish csr_enc-taking raw variants
 * in isa/common/cacheops.h so both M-mode servicers can share them
 * verbatim -- see TODO at the bottom of this file.
 */

#include <stdint.h>

#include "umode_abi.h"

/*
 * cm-umode M-mode direct-CSR helpers (csrw 0x7f9/0x7fb/0x7fd/0x7ff/0x7d0).
 * Erbium's slice forwards everything non-cb_drain to isa/common/cacheops.h
 * so these are the exact same helpers the ET-SoC1 path uses.
 */
#include <isa/erbium/cacheops.h>

/*
 * Decompose a U-mode ecall's packed CSR encoding back into the five fields
 * that cache_ops_priv_evict_sw / flush_sw produce.  Packing layout (from
 * isa/common/cacheops-umode.h):
 *   bit  63    : use_tmask
 *   bits 59:58 : dst (to_L1/L2/L3/Mem)
 *   bits 17:14 : set
 *   bits  7:6  : way
 *   bits  3:0  : num_lines
 */
static inline void decompose_sw(uint64_t csr_enc,
                                uint64_t *use_tmask, uint64_t *dst,
                                uint64_t *way, uint64_t *set,
                                uint64_t *num_lines)
{
	*use_tmask = (csr_enc >> 63) & 0x1;
	*dst       = (csr_enc >> 58) & 0x3;
	*set       = (csr_enc >> 14) & 0xF;
	*way       = (csr_enc >> 6)  & 0x3;
	*num_lines = csr_enc         & 0xF;
}

int64_t aifoundry_umode_dispatch(uint64_t id, uint64_t a1, uint64_t a2, uint64_t a3)
{
	(void)a2;
	(void)a3;

	uint64_t use_tmask, dst, way, set, num_lines;

	switch (id) {
	case SYSCALL_CACHE_OPS_EVICT_SW:
		decompose_sw(a1, &use_tmask, &dst, &way, &set, &num_lines);
		evict_sw(use_tmask, dst, way, set, num_lines);
		return SYSCALL_SUCCESS;

	case SYSCALL_CACHE_OPS_FLUSH_SW:
		decompose_sw(a1, &use_tmask, &dst, &way, &set, &num_lines);
		flush_sw(use_tmask, dst, way, set, num_lines);
		return SYSCALL_SUCCESS;

	case SYSCALL_CACHE_OPS_LOCK_SW:
		/* a1 packs way in [56:55] and paddr in [39:6] (64B-aligned). */
		lock_sw((a1 >> 55) & 0x3, a1 & 0xFFFFFFFFC0ULL);
		return SYSCALL_SUCCESS;

	case SYSCALL_CACHE_OPS_UNLOCK_SW:
		/* a1 packs way in [62:55] and set in [9:6]. */
		unlock_sw((a1 >> 55) & 0xFF, (a1 >> 6) & 0xF);
		return SYSCALL_SUCCESS;

	case SYSCALL_CACHE_OPS_INVALIDATE:
		/* a1 packing (from cache_ops_priv_cache_invalidate):
		 *   bit 0 : inval_TLBs_and_PTW
		 *   bit 1 : inval_instr_cache
		 */
		cache_invalidate((a1 >> 1) & 0x1, a1 & 0x1);
		return SYSCALL_SUCCESS;

	case SYSCALL_RETURN_FROM_KERNEL:
		/* Normally intercepted in umode_fault.c before reaching here. */
		return SYSCALL_SUCCESS;

	/* Not serviced on Erbium M-mode yet:
	 *   SYSCALL_CACHE_OPS_EVICT_L1      - needs the multi-set loop wrapper
	 *                                     from MachineMinion::syscall.c
	 *   SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2 - ditto, plus no L2 SCP on Erbium
	 *   SYSCALL_SHIRE_CACHE_BANK_OP     - no shire-cache banks on Erbium
	 *   SYSCALL_PMC_SC_SAMPLE/MS_SAMPLE - PMU skipped per project decision
	 */
	case SYSCALL_CACHE_OPS_EVICT_L1:
	case SYSCALL_SHIRE_CACHE_BANK_OP:
	case SYSCALL_PMC_SC_SAMPLE:
	case SYSCALL_PMC_MS_SAMPLE:
	case SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2:
	default:
		return SYSCALL_INVALID_ID;
	}
}

/*
 * TODO (clean-up candidate for et-common-libs): publish csr_enc-taking
 * raw variants of each cache-op in isa/common/cacheops.h so both this
 * dispatcher and MachineMinion::syscall.c can call them directly without
 * decomposing.  Shape:
 *
 *   inline void evict_sw_raw(uint64_t csr_enc)  { csrw 0x7f9, csr_enc; }
 *   inline void flush_sw_raw(uint64_t csr_enc)  { csrw 0x7fb, csr_enc; }
 *   inline void lock_sw_raw (uint64_t csr_enc)  { csrw 0x7fd, csr_enc; }
 *   inline void unlock_sw_raw(uint64_t csr_enc) { csrw 0x7ff, csr_enc; }
 *
 * Then the five cache-op cases above collapse to one-line raw calls, and
 * MachineMinion's private static cache_ops_* helpers (which today do the
 * exact same csrw) get replaced by shared cm-umode helpers.  One source of
 * truth; zero inlined duplication across the two M-mode servicers.
 */
