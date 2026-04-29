/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr-aware Erbium/soc1sim linker wrapper.
 *
 * The upstream soc1sim linker script cannot be used directly for
 * Zephyr: Zephyr needs its generated init/devices/iterable sections.
 * This wrapper keeps Zephyr's RISC-V section script, but makes its RAM
 * region derive from the same region0 + reserved-window contract used
 * by erbium-soc1sim.
 */

PROVIDE(region0_base = CONFIG_AIFOUNDRY_ETSOC1_UMODE_REGION0_BASE);
PROVIDE(region0_size = CONFIG_AIFOUNDRY_ETSOC1_UMODE_REGION0_SIZE);
PROVIDE(MMODE_RESERVED_SIZE = CONFIG_AIFOUNDRY_ETSOC1_UMODE_MMODE_RESERVED_SIZE);

/*
 * The generic RISC-V linker script derives RAM from CONFIG_SRAM_*.
 * Override those macros locally so MEMORY { RAM } follows region0.
 */
#undef CONFIG_SRAM_BASE_ADDRESS
#undef CONFIG_SRAM_SIZE
#define CONFIG_SRAM_BASE_ADDRESS (region0_base + MMODE_RESERVED_SIZE)
#define CONFIG_SRAM_SIZE ((region0_size - MMODE_RESERVED_SIZE) / 1024)

#include <zephyr/arch/riscv/common/linker.ld>
