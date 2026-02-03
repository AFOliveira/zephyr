# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Use custom linker script for ET-Minion 4KB trap vector alignment requirement
set(SOC_LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/linker.ld CACHE INTERNAL "")

# SMP IPI driver for ET-Minion
zephyr_library_sources_ifdef(CONFIG_SMP ipi.c)
