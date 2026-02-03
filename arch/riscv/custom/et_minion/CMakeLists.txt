# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_RISCV_SOC_HAS_ET_MINION)
  zephyr_library_sources(cache.c)
endif()
