/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bootstrap printk backend for Zephyr-in-U-mode (legacy fallback).
 *
 * Apps that want logging on U-mode should prefer aifoundry_log() from
 * <zephyr/aifoundry/runtime.h>, which on this SoC writes to the et-trace
 * ring directly via Trace_Format_String_V (no Zephyr printk machinery,
 * smaller code path, host-readable via basic_launcher's dumpTracesToFile
 * + et_dev_trace_decoder).
 *
 * If Zephyr code unavoidably reaches `printk(...)` (e.g. boot banner
 * or assertion paths), the bytes land in this small in-DRAM ring so
 * they at least don't disappear. */

#include <stdint.h>
#include <stddef.h>
#include <zephyr/toolchain.h>

#ifndef CONFIG_UMODE_PRINTK_BUFFER_SIZE
#define CONFIG_UMODE_PRINTK_BUFFER_SIZE 2048
#endif

struct umode_printk_ring {
	uint32_t write_index;
	uint32_t length;
	uint8_t  data[CONFIG_UMODE_PRINTK_BUFFER_SIZE];
};

__attribute__((used))
struct umode_printk_ring __umode_printk_ring;

int arch_printk_char_out(int c)
{
	uint32_t idx = __umode_printk_ring.write_index % CONFIG_UMODE_PRINTK_BUFFER_SIZE;

	__umode_printk_ring.data[idx] = (uint8_t)c;
	idx = (idx + 1) % CONFIG_UMODE_PRINTK_BUFFER_SIZE;
	__umode_printk_ring.write_index = idx;
	if (__umode_printk_ring.length < CONFIG_UMODE_PRINTK_BUFFER_SIZE) {
		__umode_printk_ring.length++;
	}
	return c;
}
