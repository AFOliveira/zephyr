/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Erbium (ET-Minion) specific linker script
 *
 * ET-Minion requires mtvec/stvec at 4KB boundaries (ET PRM section 1.3, line 348).
 * We place boot code first, then align exceptions to the next 4KB boundary.
 * This wastes ~4KB but keeps boot at 0x40000000 (simple, predictable).
 */

#include <zephyr/devicetree.h>
#include <zephyr/linker/sections.h>
#include <zephyr/linker/devicetree_regions.h>
#include <zephyr/linker/linker-defs.h>
#include <zephyr/linker/linker-tool.h>

#define ROMABLE_REGION              RAM
#define RAMABLE_REGION              RAM

#define _EXCEPTION_SECTION_NAME     exceptions
#define _RESET_SECTION_NAME         reset

#define ROM_BASE CONFIG_SRAM_BASE_ADDRESS
#define ROM_SIZE (KB(CONFIG_SRAM_SIZE))
#define RAM_BASE CONFIG_SRAM_BASE_ADDRESS
#define RAM_SIZE KB(CONFIG_SRAM_SIZE)

#define MPU_MIN_SIZE_ALIGN
#define MPU_ALIGN(region_size) . = ALIGN(4)

#include <zephyr/linker/linker-devnull.h>

MEMORY
{
    RAM (rwx) : ORIGIN = RAM_BASE, LENGTH = RAM_SIZE
    LINKER_DT_REGIONS()
    IDT_LIST  (wx)      : ORIGIN = 0xFFFFF000, LENGTH = 4K
}

ENTRY(CONFIG_KERNEL_ENTRY)

SECTIONS
    {

#include <zephyr/linker/rel-sections.ld>
#include <zephyr/linker/llext-sections.ld>

    SECTION_PROLOGUE(.plt,,)
	{
		*(.plt)
	}

    SECTION_PROLOGUE(.iplt,,)
	{
		*(.iplt)
	}

    GROUP_START(ROMABLE_REGION)
    __rom_region_start = ROM_BASE;

    SECTION_PROLOGUE(rom_start,,)
    {
		. = ALIGN(16);
#include <snippets-rom-start.ld>
    } GROUP_LINK_IN(ROMABLE_REGION)

    SECTION_PROLOGUE(_RESET_SECTION_NAME,,)
    {
		KEEP(*(.reset.*))
    } GROUP_LINK_IN(ROMABLE_REGION)

    /* ET-Minion: Pad to 4KB boundary for trap vector requirement */
    SECTION_PROLOGUE(erbium_trap_padding,,)
    {
		. = ALIGN(4096);
    } GROUP_LINK_IN(ROMABLE_REGION)

    SECTION_PROLOGUE(_EXCEPTION_SECTION_NAME,,)
    {
		 KEEP(*(".exception.entry.*"))
		 *(".exception.other.*")
    } GROUP_LINK_IN(ROMABLE_REGION)

    SECTION_PROLOGUE(_TEXT_SECTION_NAME,,)
	{
		. = ALIGN(4);
		KEEP(*(.openocd_debug))
		KEEP(*(".openocd_debug.*"))

		__text_region_start = .;

		*(.text)
		*(".text.*")
		*(.gnu.linkonce.t.*)
#include <zephyr/linker/kobject-text.ld>
	} GROUP_LINK_IN(ROMABLE_REGION)

    __text_region_end = .;

	__rodata_region_start = .;
#include <zephyr/linker/common-rom.ld>
#include <snippets-rom-sections.ld>
#include <zephyr/linker/thread-local-storage.ld>

    SECTION_PROLOGUE(_RODATA_SECTION_NAME,,)
	{
		 . = ALIGN(4);
		 *(.srodata)
		 *(".srodata.*")
		 *(.rodata)
		 *(".rodata.*")
		 *(.gnu.linkonce.r.*)
	} GROUP_LINK_IN(ROMABLE_REGION)

#include <zephyr/linker/cplusplus-rom.ld>

	__rodata_region_end = .;

#ifndef CONFIG_XIP
#ifdef CONFIG_RISCV_PMP
	SECTION_PROLOGUE(rom_mpu_padding,,)
	{
		MPU_ALIGN(__rodata_region_end - __rom_region_start);
	} GROUP_LINK_IN(ROMABLE_REGION)
#endif
	__rom_region_end = .;
	__rom_region_size = __rom_region_end - __rom_region_start;
#endif
    GROUP_END(ROMABLE_REGION)

    GROUP_START(RAMABLE_REGION)

	. = RAM_BASE;
	_image_ram_start = .;

#ifdef CONFIG_USERSPACE
#include <app_smem.ld>

	_app_smem_size = _app_smem_end - _app_smem_start;
	_app_smem_rom_start = LOADADDR(_APP_SMEM_SECTION_NAME);
#endif

    SECTION_DATA_PROLOGUE(_DATA_SECTION_NAME,,)
	{
		 . = ALIGN(4);
		 __kernel_ram_start = .;
		 __data_region_start = .;
		 __data_start = .;

		 *(.data)
		 *(".data.*")

#ifdef CONFIG_RISCV_GP
		 . = ALIGN(8);
		 PROVIDE (__global_pointer$ = . + 0x800);
#endif

		 *(.sdata .sdata.* .gnu.linkonce.s.*)

#include <snippets-rwdata.ld>

		 __data_end = .;

	}  GROUP_DATA_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)
	__data_size = __data_end - __data_start;
	__data_load_start = LOADADDR(_DATA_SECTION_NAME);

	__data_region_load_start = LOADADDR(_DATA_SECTION_NAME);

#include <zephyr/linker/common-ram.ld>
#include <zephyr/linker/kobject-data.ld>
#include <zephyr/linker/cplusplus-ram.ld>
#include <snippets-data-sections.ld>

    __data_region_end = .;

    SECTION_DATA_PROLOGUE(_BSS_SECTION_NAME,(NOLOAD),)
	{
        MPU_MIN_SIZE_ALIGN
        /*
         * For performance, BSS section is assumed to be 4 byte aligned and
         * a multiple of 4 bytes
         */
        . = ALIGN(4);
	__bss_start = .;
	__kernel_ram_start = .;
        *(.sbss)
        *(".sbss.*")
	*(.bss)
	*(".bss.*")
	COMMON_SYMBOLS
        /*
         * As memory is cleared in words only, it is simpler to ensure the BSS
         * section ends on a 4 byte boundary. This wastes a maximum of 3 bytes.
		 */
	__bss_end = ALIGN(4);
	}  GROUP_DATA_LINK_IN(RAMABLE_REGION, RAMABLE_REGION)

#include <zephyr/linker/common-noinit.ld>

	__kernel_ram_end = .;
	__kernel_ram_size = __kernel_ram_end - __kernel_ram_start;

#include <snippets-sections.ld>

#define LAST_RAM_ALIGN MPU_MIN_SIZE_ALIGN

#include <zephyr/linker/ram-end.ld>

    GROUP_END(RAMABLE_REGION)

#include <zephyr/linker/intlist.ld>
#include <zephyr/linker/debug-sections.ld>

    SECTION_PROLOGUE(.riscv.attributes, 0,)
	{
	KEEP(*(.riscv.attributes))
	KEEP(*(.gnu.attributes))
	}

    /DISCARD/ : { *(.note.GNU-stack) }

}
