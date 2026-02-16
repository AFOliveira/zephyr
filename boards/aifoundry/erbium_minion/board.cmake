# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

set(SUPPORTED_EMU_PLATFORMS et-emu)
set(ETEMU_SHIRES 0x1)
set(ETEMU_RESET_TARGET main)
set(ETEMU_ENABLE_MINIONS true)
include(${ZEPHYR_BASE}/boards/common/etemu.board.cmake)
