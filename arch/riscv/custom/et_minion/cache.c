/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * ET-Minion cache management driver
 *
 * Phase 1: Use standard RISC-V fence instructions
 * Phase 2: Can be enhanced with ET-Minion specific cache instructions
 */

#include <zephyr/kernel.h>
#include <zephyr/arch/cpu.h>
#include <zephyr/cache.h>

/* ET-Minion cache parameters (from ET PRM):
 * - L1 D-Cache: 4KB, 4-way set associative, 64-byte lines
 * - L1 I-Cache: Implementation-specific
 * - Non-coherent between harts
 */

void arch_dcache_enable(void)
{
	/* ET-Minion cache enabled by default */
}

void arch_dcache_disable(void)
{
	/* ET-Minion cache always enabled, flush everything */
	arch_dcache_flush_all();
}

int arch_dcache_flush_all(void)
{
	/* Fence with I/O for full D-cache writeback */
	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

int arch_dcache_invd_all(void)
{
	/* Invalidate requires fence + wfi pattern for ET-Minion */
	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

int arch_dcache_flush_and_invd_all(void)
{
	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

int arch_dcache_flush_range(void *addr, size_t size)
{
	ARG_UNUSED(addr);
	ARG_UNUSED(size);

	/* ET-Minion Phase 1: Use full fence (no range-specific instruction) */
	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

int arch_dcache_invd_range(void *addr, size_t size)
{
	ARG_UNUSED(addr);
	ARG_UNUSED(size);

	/* ET-Minion Phase 1: Use full fence */
	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

int arch_dcache_flush_and_invd_range(void *addr, size_t size)
{
	ARG_UNUSED(addr);
	ARG_UNUSED(size);

	__asm__ volatile("fence iorw, iorw" ::: "memory");
	return 0;
}

void arch_icache_enable(void)
{
	/* ET-Minion I-cache enabled by default */
}

void arch_icache_disable(void)
{
	/* ET-Minion I-cache always enabled, flush everything */
	arch_icache_invd_all();
}

int arch_icache_flush_all(void)
{
	/* I-cache is write-through, only invalidate needed */
	return arch_icache_invd_all();
}

int arch_icache_invd_all(void)
{
	/* fence.i invalidates instruction cache */
	__asm__ volatile("fence.i" ::: "memory");
	return 0;
}

int arch_icache_flush_and_invd_all(void)
{
	return arch_icache_invd_all();
}

int arch_icache_flush_range(void *addr, size_t size)
{
	ARG_UNUSED(addr);
	ARG_UNUSED(size);

	/* I-cache is write-through */
	return arch_icache_invd_range(addr, size);
}

int arch_icache_invd_range(void *addr, size_t size)
{
	ARG_UNUSED(addr);
	ARG_UNUSED(size);

	/* ET-Minion Phase 1: Use full fence.i (no range-specific instruction) */
	__asm__ volatile("fence.i" ::: "memory");
	return 0;
}

int arch_icache_flush_and_invd_range(void *addr, size_t size)
{
	return arch_icache_invd_range(addr, size);
}

/* Called during early kernel init */
void arch_cache_init(void)
{
	/* ET-Minion caches enabled by default, nothing to initialize */
}

size_t arch_dcache_line_size_get(void)
{
	return CONFIG_DCACHE_LINE_SIZE;
}

size_t arch_icache_line_size_get(void)
{
	return CONFIG_ICACHE_LINE_SIZE;
}
