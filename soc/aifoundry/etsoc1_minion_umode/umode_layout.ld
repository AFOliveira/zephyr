/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Erbium/soc1sim layout sections and symbols for the Zephyr U-mode
 * image.
 *
 * soc1sim_linker.ld makes the RAM MEMORY region come from region0.
 * This file adds the soc1sim stack/heap symbols after Zephyr's own
 * sections have been placed.
 */

NUM_HARTS = CONFIG_AIFOUNDRY_ETSOC1_UMODE_NUM_HARTS;
STACK_SIZE = CONFIG_AIFOUNDRY_ETSOC1_UMODE_STACK_SIZE;

PROVIDE(region1_base = 0);
PROVIDE(region1_size = 0);
PROVIDE(region2_base = 0);
PROVIDE(region2_size = 0);
PROVIDE(region3_base = 0);
PROVIDE(region3_size = 0);
PROVIDE(region4_base = 0);
PROVIDE(region4_size = 0);
PROVIDE(region5_base = 0);
PROVIDE(region5_size = 0);
PROVIDE(region6_base = 0);
PROVIDE(region6_size = 0);
PROVIDE(region7_base = 0);
PROVIDE(region7_size = 0);

SECTION_PROLOGUE(.soc1sim_stack, (NOLOAD),)
{
	. = ALIGN(4K);
	PROVIDE(__stack_start = .);
	. = . + (NUM_HARTS * STACK_SIZE);
	PROVIDE(__stack_base = .);
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, RAMABLE_REGION)

SECTION_PROLOGUE(.heap0, (NOLOAD),)
{
	. = ALIGN(16);
	PROVIDE(heap0_start = .);
	. = ORIGIN(RAM) + LENGTH(RAM);
	PROVIDE(heap0_end = .);
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, RAMABLE_REGION)

heap0_size = heap0_end - heap0_start;

PROVIDE(heap1_start = region1_base);
PROVIDE(heap1_size = region1_size);
PROVIDE(heap2_start = region2_base);
PROVIDE(heap2_size = region2_size);
PROVIDE(heap3_start = region3_base);
PROVIDE(heap3_size = region3_size);
PROVIDE(heap4_start = region4_base);
PROVIDE(heap4_size = region4_size);
PROVIDE(heap5_start = region5_base);
PROVIDE(heap5_size = region5_size);
PROVIDE(heap6_start = region6_base);
PROVIDE(heap6_size = region6_size);
PROVIDE(heap7_start = region7_base);
PROVIDE(heap7_size = region7_size);

ASSERT(heap0_start <= heap0_end,
       "Zephyr image overflows ET-SoC1 U-mode region")
ASSERT(((NUM_HARTS * STACK_SIZE) & 0xFFF) == 0,
       "ET-SoC1 runtime requires total stack size to be a multiple of 4 KiB")
ASSERT((__stack_start & 0xFFF) == 0,
       "ET-SoC1 runtime requires __stack_start to be 4 KiB aligned")
