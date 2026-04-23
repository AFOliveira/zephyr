# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Use the standard RISC-V linker script.  Under CONFIG_RISCV_RUN_IN_UMODE
# the privileged reset vector / vector table sections are simply empty,
# so we do not need a custom linker script for the MVP.
set(SOC_LINKER_SCRIPT ${ZEPHYR_BASE}/include/zephyr/arch/riscv/common/linker.ld CACHE INTERNAL "")
