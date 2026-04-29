/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bootstrap printk backend for Zephyr-in-U-mode. The UART is owned by
 * firmware, so route complete printk lines through Vidas' et_printf()
 * path into the per-hart ET trace ring.
 */

#include <stdint.h>

#include <etsoc/common/utils.h>

#ifndef CONFIG_UMODE_PRINTK_BUFFER_SIZE
#define CONFIG_UMODE_PRINTK_BUFFER_SIZE 2048
#endif

static char printk_line[CONFIG_UMODE_PRINTK_BUFFER_SIZE];
static uint32_t printk_line_len;

static void flush_printk_line(void)
{
	if (printk_line_len == 0U) {
		return;
	}

	printk_line[printk_line_len] = '\0';
	et_printf("%s", printk_line);
	printk_line_len = 0U;
}

int arch_printk_char_out(int c)
{
	if (c == '\n') {
		if (printk_line_len < (CONFIG_UMODE_PRINTK_BUFFER_SIZE - 1U)) {
			printk_line[printk_line_len++] = (char)c;
		}
		flush_printk_line();
		return c;
	}

	if (printk_line_len >= (CONFIG_UMODE_PRINTK_BUFFER_SIZE - 1U)) {
		flush_printk_line();
	}

	printk_line[printk_line_len++] = (char)c;
	return c;
}
