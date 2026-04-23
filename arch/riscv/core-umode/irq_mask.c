/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode IRQ lock/unlock.  We cannot touch mstatus.MIE directly from
 * U-mode, so we bounce to MachineMinion via the OSKERN cm-umode syscalls
 * (et-common-libs/include/isa/common/syscall.h).
 *
 * The upstream arch_irq_lock/unlock inline implementations in arch.h
 * delegate to z_soc_irq_lock / z_soc_irq_unlock when
 * CONFIG_RISCV_SOC_HAS_CUSTOM_IRQ_LOCK_OPS is set.
 */

#include <zephyr/kernel.h>
#include <isa/common/syscall.h>

unsigned int z_soc_irq_lock(void)
{
	return (unsigned int)syscall(SYSCALL_OSKERN_IRQ_MASK_GLOBAL, 0, 0, 0);
}

void z_soc_irq_unlock(unsigned int key)
{
	(void)syscall(SYSCALL_OSKERN_IRQ_UNMASK_GLOBAL, (uint64_t)key, 0, 0);
}

bool z_soc_irq_unlocked(unsigned int key)
{
	return key != 0U;
}
