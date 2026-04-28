# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Use the standard RISC-V linker script.  Under CONFIG_RISCV_RUN_IN_UMODE
# the privileged reset vector / vector table sections are simply empty,
# so we do not need a custom linker script for the MVP.
set(SOC_LINKER_SCRIPT ${ZEPHYR_BASE}/include/zephyr/arch/riscv/common/linker.ld CACHE INTERNAL "")

# Portable AIFoundry kernel-runtime — U-mode flavor.  Header in
# <zephyr/aifoundry/runtime.h> declares the surface; this .c provides
# the cm-umode-backed implementation.  Include paths and the
# `asm=__asm__` shim are inherited from arch/riscv/core/CMakeLists.txt.
zephyr_sources(runtime.c)
