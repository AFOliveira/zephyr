/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Backing storage for the cm-umode kernel-runtime ABI globals.  entry.S
 * stores a0 (kernel_args) and a1 (kernel_environment_t *) here on U-mode
 * entry; the SoC runtime.c reads them when the app asks for kernel_args
 * via the AIFoundry HAL.
 *
 * Both are forced into .data (with a non-zero default) so that the
 * arch_bss_zero() call in z_prep_c — which runs AFTER entry.S has
 * already populated them — does not silently zero them back out.
 */

#include <stdint.h>

void *cm_umode_kernel_args = (void *)~(uintptr_t)0;
void *cm_umode_kernel_env  = (void *)~(uintptr_t)0;
