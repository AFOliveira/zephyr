# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Use a Zephyr-aware wrapper around the Erbium/soc1sim region contract.
# The wrapper keeps Zephyr's generated sections but derives RAM placement
# from region0_base/region0_size/MMODE_RESERVED_SIZE.
set(SOC_LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/soc1sim_linker.ld CACHE INTERNAL "")
zephyr_linker_sources(SECTIONS umode_layout.ld)
