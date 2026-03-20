# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Use the standard RISC-V linker script; trap alignment is handled by
# CONFIG_RISCV_TRAP_HANDLER_ALIGNMENT in arch/riscv/core/isr.S
set(SOC_LINKER_SCRIPT ${ZEPHYR_BASE}/include/zephyr/arch/riscv/common/linker.ld CACHE INTERNAL "")

zephyr_sources(soc.c)
