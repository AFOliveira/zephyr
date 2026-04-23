/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Syscall-sweep U-mode kernel.
 *
 * Invokes every U-mode-range cm-umode syscall ID (1..11) in order, stashing
 * each ecall return value in a shared memory region ("syscall_results") that
 * the M-mode launcher reads after RETURN_FROM_KERNEL.
 *
 * Layout of syscall_results[]:
 *     [0] SYSCALL_CACHE_OPS_EVICT_SW           (id 1)
 *     [1] SYSCALL_CACHE_OPS_FLUSH_SW           (id 2)
 *     [2] SYSCALL_CACHE_OPS_LOCK_SW            (id 3)
 *     [3] SYSCALL_CACHE_OPS_UNLOCK_SW          (id 4)
 *     [4] SYSCALL_CACHE_OPS_INVALIDATE         (id 5)
 *     [5] SYSCALL_CACHE_OPS_EVICT_L1           (id 6)
 *     [6] SYSCALL_SHIRE_CACHE_BANK_OP          (id 7)
 *     [7] SYSCALL_PMC_SC_SAMPLE                (id 9)
 *     [8] SYSCALL_PMC_MS_SAMPLE                (id 10)
 *     [9] SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2  (id 11)
 * (ID 8 = RETURN_FROM_KERNEL is how we exit; its return isn't captured.)
 */

	.section .text, "ax", @progbits
	.balign 4
	.globl _umode_entry
	.type _umode_entry, @function

/* Helper: ecall with (id in a0, a1, a2, a3) and store a0 to memory.
 *
 * Macro args:
 *   id      syscall id
 *   arg1    scalar (passed in a1)
 *   arg2    scalar (passed in a2)
 *   arg3    scalar (passed in a3)
 *   offset  byte offset into syscall_results where a0 is stored
 */
.macro SYSCALL_AND_STORE id, arg1, arg2, arg3, offset
	li   a0, \id
	li   a1, \arg1
	li   a2, \arg2
	li   a3, \arg3
	ecall
	sd   a0, \offset(s0)
.endm

_umode_entry:
	/* Keep the results pointer in s0 across all ecalls. */
	la   s0, syscall_results

	/* id 1: EVICT_SW   — a1 = (dst=to_L2<<58) = 0x04 << 56 */
	li   t0, 0x4
	slli a1, t0, 56
	li   a0, 1
	li   a2, 0
	li   a3, 0
	ecall
	sd   a0, 0(s0)

	/* id 2: FLUSH_SW — same encoding */
	li   t0, 0x4
	slli a1, t0, 56
	li   a0, 2
	li   a2, 0
	li   a3, 0
	ecall
	sd   a0, 8(s0)

	/* id 3: LOCK_SW — way=0 paddr=0x40000000 (MRAM base, 64B-aligned) */
	li   a1, 0x40000000
	li   a0, 3
	li   a2, 0
	li   a3, 0
	ecall
	sd   a0, 16(s0)

	/* id 4: UNLOCK_SW — way=0 set=0 */
	li   a1, 0
	li   a0, 4
	li   a2, 0
	li   a3, 0
	ecall
	sd   a0, 24(s0)

	/* id 5: INVALIDATE — a1=1 (icache invalidate) */
	SYSCALL_AND_STORE 5, 1, 0, 0, 32

	/* id 6: EVICT_L1 — use_tmask=0, dest_level=to_L2=1 */
	SYSCALL_AND_STORE 6, 0, 1, 0, 40

	/* id 7: SHIRE_CACHE_BANK_OP — shire=0 bank=0 op=0 (unused on Erbium) */
	SYSCALL_AND_STORE 7, 0, 0, 0, 48

	/* id 9: PMC_SC_SAMPLE   — zeros */
	SYSCALL_AND_STORE 9, 0, 0, 0, 56

	/* id 10: PMC_MS_SAMPLE  — zeros */
	SYSCALL_AND_STORE 10, 0, 0, 0, 64

	/* id 11: EVICT_WHOLE_L1_L2 — no args */
	SYSCALL_AND_STORE 11, 0, 0, 0, 72

	/* id 8: RETURN_FROM_KERNEL — a1=0 (success). Does not resume here. */
	li   a0, 8
	li   a1, 0
	li   a2, 0
	li   a3, 0
	ecall

1:	j    1b

	.size _umode_entry, . - _umode_entry

/* Results area — populated by U-mode, read by M-mode launcher. */
	.section .data, "aw", @progbits
	.balign 8
	.globl syscall_results
	.type  syscall_results, @object
	.size  syscall_results, 80
syscall_results:
	.zero 80
