/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thin forwarder: all cm-umode syscall IDs, return codes and the inline
 * syscall() helper come from et-common-libs directly.  We just add the
 * Zephyr-M-side entry points the sample and the arch wrapper need.
 */

#ifndef ZEPHYR_AIFOUNDRY_UMODE_ABI_H_
#define ZEPHYR_AIFOUNDRY_UMODE_ABI_H_

#include <isa/common/syscall.h>      /* SYSCALL_* IDs, KERNEL_RETURN_*, syscall() */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __ASSEMBLER__

#include <stdint.h>

/**
 * Enter U-mode at @p entry with stack pointer @p stack, passing @p arg in a0.
 *
 * Returns the kernel return code supplied by U-mode via
 * SYSCALL_RETURN_FROM_KERNEL.  The M-mode caller's context is saved on
 * entry and restored on return.  Single-hart assumption.
 */
int64_t aifoundry_launch_umode(void *entry, void *stack, uint64_t arg);

/**
 * Open MPROT so that U-mode has full access to the memory map.
 * (No-op in the current slice; MPROT enforcement is off via reset value.)
 */
void erbium_mprot_allow_all(void);

/**
 * Lock MPROT for M-mode-only accesses.  Placeholder; no-op today.
 */
void erbium_mprot_lock_m(void);

#endif /* !__ASSEMBLER__ */

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_AIFOUNDRY_UMODE_ABI_H_ */
