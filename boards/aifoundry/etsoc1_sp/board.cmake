# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

set(SUPPORTED_EMU_PLATFORMS et-emu)

set(ETEMU_SHIRES 0x400000000)
set(ETEMU_RESET_TARGET sp)

include(${ZEPHYR_BASE}/boards/common/etemu.board.cmake)
