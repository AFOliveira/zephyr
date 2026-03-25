/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

int main(void)
{
#ifdef CONFIG_SOC_ERBIUM_MINION
	/* Signal end of test to cosim/VCS */
	__asm__ volatile(
		"fence\n"
		"lui a7, 0x1FEED\n"
		"csrw 0x8d0, a7\n"
	);
#endif

	return 0;
}
