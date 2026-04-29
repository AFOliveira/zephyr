/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode IRQ lock stubs.  We run with interrupts effectively disabled
 * (we're a cm-umode kernel, the host firmware decides what delegates
 * land here) so lock/unlock are no-ops.  The upstream
 * arch_irq_lock/unlock inline implementations in arch.h bounce to these
 * z_soc_* hooks when CONFIG_RISCV_SOC_HAS_CUSTOM_IRQ_LOCK_OPS is set.
 */

#include <zephyr/kernel.h>

unsigned int z_soc_irq_lock(void)
{
	return 0;
}

void z_soc_irq_unlock(unsigned int key)
{
	ARG_UNUSED(key);
}

bool z_soc_irq_unlocked(unsigned int key)
{
	ARG_UNUSED(key);
	return true;
}
