/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode "reboot" = SYSCALL_RETURN_FROM_KERNEL.  There is no way for a
 * compute kernel to actually reset the SoC by itself; just return to the
 * host firmware.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>
#include <isa/common/syscall.h>   /* cm-umode ABI */

void sys_arch_reboot(int type)
{
	ARG_UNUSED(type);

	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);

	for (;;) {
	}
}
