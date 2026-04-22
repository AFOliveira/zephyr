/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal C-domain prep for Zephyr-in-U-mode.  Models prep_c.c but drops
 * everything that touches M-mode CSRs or privileged hooks.  BSS zeroing
 * and data relocation use the same arch_bss_zero()/arch_data_copy()
 * weak helpers that the upstream path uses.
 */

#include <stddef.h>
#include <zephyr/toolchain.h>
#include <zephyr/kernel_structs.h>
#include <zephyr/arch/common/xip.h>
#include <zephyr/arch/common/init.h>

FUNC_NORETURN void z_cstart(void);

FUNC_NORETURN void z_prep_c(void)
{
	arch_bss_zero();
	arch_data_copy();
	z_cstart();
	CODE_UNREACHABLE;
}
