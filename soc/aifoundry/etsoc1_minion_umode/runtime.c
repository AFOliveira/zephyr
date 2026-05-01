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
#include <zephyr/init.h>
#include <zephyr/sys/printk-hooks.h>
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

/*
 * U-mode software clock.
 *
 * Real hardware counters (rdcycle/rdtime/rdinstret) trap on ET-SoC1
 * U-mode because the firmware sets mcounteren=0.  Our compromise is
 * a software counter advanced inside the delay primitives — sketches
 * that don't call delay() observe millis() == 0 forever.  Documented
 * in the HAL header.  Callers needing finer grain should batch into
 * larger delays.
 *
 * The "1000 spin iterations per microsecond" calibration is a coarse
 * guess.  ET-SoC1 minions clock around ~1 GHz with rv64imafc; a simple
 * volatile-asm loop dispatches on the order of 1 cycle per nop on the
 * scalar pipe but issue-rate / fetch effects bring real wall-clock
 * iteration time closer to single-digit nanoseconds.  We tune by
 * picking an iteration count that produces *visible* delay (Arduino
 * sketches use delay(1000) typically).  Refining the constant is a
 * follow-up; a dedicated cycle source would be the right v2 story.
 */
#define UMODE_NOPS_PER_US 1000U

static volatile uint64_t umode_us_counter;

static inline void umode_busy_us(uint32_t us)
{
	for (uint32_t outer = 0; outer < us; outer++) {
		for (volatile uint32_t inner = 0; inner < UMODE_NOPS_PER_US; inner++) {
			__asm__ volatile("nop");
		}
		umode_us_counter++;
	}
}

void aifoundry_delay_ms(uint32_t ms)
{
	umode_busy_us(ms * 1000U);
}

void aifoundry_delay_us(uint32_t us)
{
	umode_busy_us(us);
}

uint32_t aifoundry_uptime_ms(void)
{
	return (uint32_t)(umode_us_counter / 1000U);
}

uint32_t aifoundry_uptime_us(void)
{
	return (uint32_t)umode_us_counter;
}

#ifdef CONFIG_PRINTK
/*
 * printk → aifoundry_log shim.
 *
 * When the application enables CONFIG_PRINTK on this board (e.g. the
 * Arduino-flavoured emlearn sample uses ZephyrSerialStub which calls
 * printk), every byte of formatted output would otherwise need to
 * route through Zephyr's uart_console driver onto a chosen UART
 * device.  On U-mode that path is fragile (early callers can
 * deref wild format-string args via picolibc's vfprintf, observed
 * as a load access fault during boot on silicon).
 *
 * Installing this hook at PRE_KERNEL_1 makes printk emit one trace
 * ring entry per character — exactly the same end channel as
 * aifoundry_log itself, just one byte at a time.  Samples that don't
 * enable CONFIG_PRINTK (e.g. umode_emlearn) skip this glue entirely.
 */
static int aifoundry_printk_hook(int c)
{
	aifoundry_log("%c", c);
	return c;
}

static int aifoundry_install_printk_hook(void)
{
	__printk_hook_install(aifoundry_printk_hook);
	return 0;
}

SYS_INIT(aifoundry_install_printk_hook, PRE_KERNEL_1, 0);
#endif /* CONFIG_PRINTK */

void aifoundry_kernel_exit(int rc)
{
	(void)syscall(SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2, 0, 0, 0);
	int code = (rc == 0) ? KERNEL_RETURN_SUCCESS : KERNEL_RETURN_SELF_ABORT;
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, code, 0);
	for (;;) {
		__asm__ volatile("nop");
	}
}
