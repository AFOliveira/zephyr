/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * cm-umode syscall ABI (vendored subset).
 *
 * Authoritative source:
 *   et-platform/et-common-libs/include/isa/etsoc/syscall.h
 *
 * Only the U-mode syscall ID range (0..127) and the KERNEL_RETURN_* codes are
 * carried in this header.  The inline syscall() helper matches the ET-SoC1
 * register contract: a0 = id, a1..a3 = args, return value in a0.
 */

#ifndef ZEPHYR_AIFOUNDRY_UMODE_ABI_H_
#define ZEPHYR_AIFOUNDRY_UMODE_ABI_H_

#ifdef __cplusplus
extern "C" {
#endif

/* SYSCALL IDs for syscalls allowed from U-Mode (range 0..127). */
#define SYSCALL_UMODE_THRESHOLD             0
#define SYSCALL_CACHE_OPS_EVICT_SW          (SYSCALL_UMODE_THRESHOLD + 1)
#define SYSCALL_CACHE_OPS_FLUSH_SW          (SYSCALL_UMODE_THRESHOLD + 2)
#define SYSCALL_CACHE_OPS_LOCK_SW           (SYSCALL_UMODE_THRESHOLD + 3)
#define SYSCALL_CACHE_OPS_UNLOCK_SW         (SYSCALL_UMODE_THRESHOLD + 4)
#define SYSCALL_CACHE_OPS_INVALIDATE        (SYSCALL_UMODE_THRESHOLD + 5)
#define SYSCALL_CACHE_OPS_EVICT_L1          (SYSCALL_UMODE_THRESHOLD + 6)
#define SYSCALL_SHIRE_CACHE_BANK_OP         (SYSCALL_UMODE_THRESHOLD + 7)
#define SYSCALL_RETURN_FROM_KERNEL          (SYSCALL_UMODE_THRESHOLD + 8)
#define SYSCALL_PMC_SC_SAMPLE               (SYSCALL_UMODE_THRESHOLD + 9)
#define SYSCALL_PMC_MS_SAMPLE               (SYSCALL_UMODE_THRESHOLD + 10)
#define SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2 (SYSCALL_UMODE_THRESHOLD + 11)
#define SYSCALL_UMODE_THRESHOLD_LIMIT       127

/* SYSCALL error codes */
#define SYSCALL_SUCCESS    0
#define SYSCALL_INVALID_ID -1

/* Kernel return types (kept in sync with FW). */
#define KERNEL_RETURN_SUCCESS      0
#define KERNEL_RETURN_SELF_ABORT   1
#define KERNEL_RETURN_SYSTEM_ABORT 2
#define KERNEL_RETURN_EXCEPTION    3

#ifndef __ASSEMBLER__

#include <stdint.h>

/*
 * U-mode -> M-mode syscall helper.
 *
 * Matches the cm-umode ABI verbatim so that U-mode kernels compiled against
 * the et-platform headers can run against this dispatcher unchanged.
 */
static inline __attribute__((always_inline)) int64_t syscall(
	uint64_t syscall_id, uint64_t arg1, uint64_t arg2, uint64_t arg3)
{
	register uint64_t a0 __asm__("a0") = syscall_id;
	register uint64_t a1 __asm__("a1") = arg1;
	register uint64_t a2 __asm__("a2") = arg2;
	register uint64_t a3 __asm__("a3") = arg3;

	__asm__ volatile("ecall"
			 : "+r"(a0)
			 : "r"(a1), "r"(a2), "r"(a3)
			 : "memory");

	return (int64_t)a0;
}

/*
 * M-mode-side entry points exposed to the sample / Zephyr app code.
 */

/**
 * Enter U-mode at @p entry with stack pointer @p stack, passing @p arg in a0.
 *
 * Returns the kernel return code supplied by U-mode via
 * SYSCALL_RETURN_FROM_KERNEL.  The M-mode caller's context is saved on entry
 * and restored on return.
 *
 * Single-hart assumption: there is exactly one active launcher at a time.
 */
int64_t aifoundry_launch_umode(void *entry, void *stack, uint64_t arg);

/**
 * Open MPROT so that U-mode has full access to the memory map.
 *
 * On Erbium, the MACHINE_NEIGH.mprot register reset value (mprot_en = 0)
 * already disables MPROT enforcement, so this is effectively a no-op for
 * the single-minion emulator slice.  Kept as a hook for future work.
 */
void erbium_mprot_allow_all(void);

/**
 * Lock MPROT for M-mode-only accesses (placeholder, currently no-op).
 */
void erbium_mprot_lock_m(void);

#endif /* !__ASSEMBLER__ */

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_AIFOUNDRY_UMODE_ABI_H_ */
