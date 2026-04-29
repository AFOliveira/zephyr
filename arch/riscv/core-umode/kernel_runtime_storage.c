/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Storage for the cm-umode launch ABI. WorkerMinion enters U-mode with
 * a0 = kernel_args and a1 = kernel_environment_t *. Zephyr's C entry does
 * not accept those arguments, so entry.S saves them here before clearing
 * the normal C argument registers.
 */

#include <stdint.h>

void *cm_umode_kernel_args = (void *)~(uintptr_t)0;
void *cm_umode_kernel_env = (void *)~(uintptr_t)0;
