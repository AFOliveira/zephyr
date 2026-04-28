/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable kernel-runtime API for AIFoundry boards.
 *
 * Same surface compiles on:
 *   - etsoc1_minion_umode (cm-umode under MachineMinion + WorkerMinion)
 *   - erbium_minion       (Zephyr-M, native UART/console)
 *
 * Implementations live in each SoC's own dir
 * (soc/aifoundry/<soc>/runtime.c).  Build picks the right one; this
 * header carries no #ifdef.  Application code includes only this
 * file and never has to know which target it's on.
 *
 * Surface:
 *
 *   aifoundry_log(fmt, ...)
 *       Log a printf-formatted line.  printf-format-checked.
 *
 *   aifoundry_kernel_args()
 *       Returns a `void *` pointing at the host-supplied kernel_args
 *       blob, or NULL on M-mode (no host).
 *
 *   aifoundry_publish_results(buf, size)
 *       Copy `size` bytes from `buf` into the host-allocated result
 *       buffer (whose device address is the first uint64 of
 *       kernel_args).  Silent no-op on M-mode.
 *
 *   aifoundry_kernel_exit(rc)
 *       End the kernel with status code `rc`.  Doesn't return on
 *       either target (U-mode ecalls RETURN_FROM_KERNEL; M-mode spins
 *       in cpu_idle so Zephyr can complete teardown but main never
 *       comes back).
 */

#ifndef ZEPHYR_AIFOUNDRY_RUNTIME_H_
#define ZEPHYR_AIFOUNDRY_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void  aifoundry_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void *aifoundry_kernel_args(void);
void  aifoundry_publish_results(const void *src, size_t size);
__attribute__((noreturn))
void  aifoundry_kernel_exit(int rc);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_AIFOUNDRY_RUNTIME_H_ */
