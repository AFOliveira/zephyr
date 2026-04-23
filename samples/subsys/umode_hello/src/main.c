/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Phase 1 sanity check for Zephyr running as a cm-umode kernel on
 * ET-SoC1 + full sweep over every U-mode cm-umode syscall id
 * (1..11).  For each id we invoke the matching cacheops-umode.h
 * wrapper when one exists, otherwise a raw syscall() helper.  Every
 * return value is emitted through the .noinit __umode_printk_ring so
 * the sys_emu launcher can dump and decode it.  The sweep ends with
 * SYSCALL_RETURN_FROM_KERNEL (id 8).
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>

#include <isa/common/syscall.h>
#include <isa/common/cacheops-umode.h>

/* Match the Erbium-side sweep so return values are directly
 * comparable.  Our Zephyr image is linked to load base 0x8006335000. */
#define SWEEP_LOCK_PADDR  0x8006335000ULL

int main(void)
{
	int64_t rc;

	printk("SWEEP: start (cm-umode U-mode syscall sweep)\n");

	/* id 1: SYSCALL_CACHE_OPS_EVICT_SW */
	rc = cache_ops_priv_evict_sw(/*use_tmask=*/0, /*dst=*/to_L2,
				     /*way=*/0, /*set=*/0,
				     /*num_lines=*/0);
	printk("SWEEP:  1 EVICT_SW              -> %lld\n", (long long)rc);

	/* id 2: SYSCALL_CACHE_OPS_FLUSH_SW */
	rc = cache_ops_priv_flush_sw(/*use_tmask=*/0, /*dst=*/to_L2,
				     /*way=*/0, /*set=*/0,
				     /*num_lines=*/0);
	printk("SWEEP:  2 FLUSH_SW              -> %lld\n", (long long)rc);

	/* id 3: SYSCALL_CACHE_OPS_LOCK_SW */
	rc = cache_ops_priv_l1_cache_lock_sw(/*way=*/0,
					     /*phy_addr=*/SWEEP_LOCK_PADDR);
	printk("SWEEP:  3 LOCK_SW               -> %lld\n", (long long)rc);

	/* id 4: SYSCALL_CACHE_OPS_UNLOCK_SW */
	rc = cache_ops_priv_l1_cache_unlock_sw(/*way=*/0, /*set=*/0);
	printk("SWEEP:  4 UNLOCK_SW             -> %lld\n", (long long)rc);

	/* id 5: SYSCALL_CACHE_OPS_INVALIDATE */
	rc = cache_ops_priv_cache_invalidate(/*inval_instr_cache=*/1,
					     /*inval_TLBs_and_PTW=*/0);
	printk("SWEEP:  5 INVALIDATE            -> %lld\n", (long long)rc);

	/* id 6: SYSCALL_CACHE_OPS_EVICT_L1 */
	rc = cache_ops_priv_evict_l1(/*use_tmask=*/0, /*dest_level=*/to_L2);
	printk("SWEEP:  6 EVICT_L1              -> %lld\n", (long long)rc);

	/* id 7: SYSCALL_SHIRE_CACHE_BANK_OP (no wrapper) */
	rc = syscall(SYSCALL_SHIRE_CACHE_BANK_OP, 0, 0, 0);
	printk("SWEEP:  7 SHIRE_CACHE_BANK_OP   -> %lld\n", (long long)rc);

	/* id 9: SYSCALL_PMC_SC_SAMPLE (no wrapper) */
	rc = syscall(SYSCALL_PMC_SC_SAMPLE, 0, 0, 0);
	printk("SWEEP:  9 PMC_SC_SAMPLE         -> %lld\n", (long long)rc);

	/* id 10: SYSCALL_PMC_MS_SAMPLE (no wrapper) */
	rc = syscall(SYSCALL_PMC_MS_SAMPLE, 0, 0, 0);
	printk("SWEEP: 10 PMC_MS_SAMPLE         -> %lld\n", (long long)rc);

	/* id 11: SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2 */
	rc = cache_ops_priv_evict_whole_l1_l2();
	printk("SWEEP: 11 EVICT_WHOLE_L1_L2     -> %lld\n", (long long)rc);

	printk("SWEEP: done\n");

	/* id 8: SYSCALL_RETURN_FROM_KERNEL - last, does not return. */
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);

	for (;;) {
	}
	return 0;
}
