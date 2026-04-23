/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thread-arch stubs for Zephyr-in-U-mode.  With MULTITHREADING=n and
 * ARCH_SWITCH_TO_MAIN_NO_MULTITHREADING undefined, the generic kernel
 * init calls bg_thread_main() (and thereby main()) directly.  We just
 * need to supply any residual symbols the kernel references
 * unconditionally.
 */

#include <zephyr/kernel.h>

int arch_coprocessors_disable(struct k_thread *thread)
{
	ARG_UNUSED(thread);
	return -ENOTSUP;
}
