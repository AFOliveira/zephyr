/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * cm-umode syscall dispatcher (M-mode side).
 *
 * Only the two syscalls required by the minimal U-mode round-trip demo are
 * serviced:
 *
 *   SYSCALL_CACHE_OPS_INVALIDATE  - write the cache-invalidate CSR (0x7d0)
 *                                   with the caller-supplied encoding.
 *   SYSCALL_RETURN_FROM_KERNEL    - tear down to the launcher; handled by
 *                                   the fault glue (see umode_fault.c) and
 *                                   therefore should not reach here via the
 *                                   regular dispatch path.
 *
 * Everything else returns SYSCALL_INVALID_ID.  A TODO is left against each
 * skipped cache-op so future work can fill them in against the Erbium CSRs.
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>

#include "umode_abi.h"

/*
 * Erbium cache-invalidate CSR.  Matches the ET-SoC1 encoding used by
 * device-minion-runtime/src/MachineMinion/src/syscall.c :: cache_ops_cache_invalidate().
 */
static inline void erbium_csr_cache_invalidate(uint64_t csr_enc)
{
	__asm__ __volatile__("csrw 0x7d0, %[v]\n" : : [v] "r"(csr_enc));
}

int64_t aifoundry_umode_dispatch(uint64_t id, uint64_t a1, uint64_t a2, uint64_t a3)
{
	(void)a2;
	(void)a3;

	switch (id) {
	case SYSCALL_CACHE_OPS_INVALIDATE:
		erbium_csr_cache_invalidate(a1);
		return SYSCALL_SUCCESS;

	/*
	 * RETURN_FROM_KERNEL never reaches this function: the fault glue
	 * peels it off and calls aifoundry_resume_from_kernel_asm() directly
	 * so we can longjmp back to the launcher's caller.  We still list it
	 * here so an out-of-band dispatch returns a sensible code.
	 */
	case SYSCALL_RETURN_FROM_KERNEL:
		return SYSCALL_SUCCESS;

	/* TODO: plumb through shire/neigh-aware implementations. */
	case SYSCALL_CACHE_OPS_EVICT_SW:
	case SYSCALL_CACHE_OPS_FLUSH_SW:
	case SYSCALL_CACHE_OPS_LOCK_SW:
	case SYSCALL_CACHE_OPS_UNLOCK_SW:
	case SYSCALL_CACHE_OPS_EVICT_L1:
	case SYSCALL_SHIRE_CACHE_BANK_OP:
	case SYSCALL_PMC_SC_SAMPLE:
	case SYSCALL_PMC_MS_SAMPLE:
	case SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2:
	default:
		return SYSCALL_INVALID_ID;
	}
}
