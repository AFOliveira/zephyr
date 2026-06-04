# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

set(SUPPORTED_EMU_PLATFORMS et-emu)
# erbium_minion Zephyr links at 0x40000200; boot via erbium_emu (-reset_pc from ELF entry).
# sys_emu (ET-SoC1) needs -sp_reset_pc and shires with the IO-shire bit — see runtime docs.
set(ETEMU_RESET_TARGET main)
set(ETEMU_ENABLE_MINIONS true)
include(${ZEPHYR_BASE}/boards/common/etemu.board.cmake)

if(DEFINED ENV{SYS_EMU})
  board_runner_args(etemu --sys-emu=$ENV{SYS_EMU})
elseif(DEFINED ENV{ERBIUM_EMU})
  board_runner_args(etemu --sys-emu=$ENV{ERBIUM_EMU})
else()
  board_runner_args(etemu --sys-emu=erbium_emu)
endif()
