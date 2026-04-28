/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode (cm-umode) implementation of the portable AIFoundry kernel
 * runtime declared in <zephyr/aifoundry/runtime.h>.
 *
 * Logging routes via Trace_Format_String_V → per-hart et-trace ring
 * (CM_UMODE_TRACE_CB).  Kernel exit ecalls SYSCALL_RETURN_FROM_KERNEL
 * after evicting L1+L2 to DRAM so the host's memcpyDeviceToHost picks
 * up our writes.
 *
 * Primitives come straight from vidas's per-platform headers:
 *   <erbium-soc1sim/isa/syscall.h>  — syscall(), SYSCALL_*, KERNEL_RETURN_*
 *   <trace/trace_umode_cb.h>        — CM_UMODE_TRACE_CB, GET_CB_INDEX
 *   <et-trace/encoder.h>            — Trace_Format_String_V
 */

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

#include <zephyr/aifoundry/runtime.h>
#include <erbium-soc1sim/isa/syscall.h>
#include <trace/trace_umode_cb.h>
#include <et-trace/encoder.h>

extern void *cm_umode_kernel_args;

static inline unsigned int umode_hartid(void)
{
	uint64_t v;
	__asm__ volatile("csrr %0, 0xcd0" : "=r"(v));
	return (unsigned int)v;
}

void aifoundry_log(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	Trace_Format_String_V(TRACE_EVENT_STRING_CRITICAL,
		&CM_UMODE_TRACE_CB[GET_CB_INDEX(umode_hartid())].cb,
		fmt, ap);
	va_end(ap);
}

void *aifoundry_kernel_args(void)
{
	return cm_umode_kernel_args;
}

void aifoundry_publish_results(const void *src, size_t size)
{
	void *args = aifoundry_kernel_args();
	if (args == NULL) {
		return;
	}
	uint64_t out_dev = *(volatile uint64_t *)args;
	if (out_dev == 0) {
		return;
	}
	const uint8_t *s = (const uint8_t *)src;
	volatile uint8_t *d = (volatile uint8_t *)(uintptr_t)out_dev;
	for (size_t i = 0; i < size; i++) {
		d[i] = s[i];
	}
}

void aifoundry_kernel_exit(int rc)
{
	(void)syscall(SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2, 0, 0, 0);
	int code = (rc == 0) ? KERNEL_RETURN_SUCCESS : KERNEL_RETURN_SELF_ABORT;
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, code, 0);
	for (;;) {
		__asm__ volatile("nop");
	}
}
