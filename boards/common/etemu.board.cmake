# SPDX-License-Identifier: Apache-2.0

board_set_sim_runner_ifnset(etemu)

if(DEFINED ETEMU_SYS_EMU)
  board_runner_args(etemu --sys-emu=${ETEMU_SYS_EMU})
endif()

if(DEFINED ETEMU_BOOTROM)
  board_runner_args(etemu --bootrom=${ETEMU_BOOTROM})
endif()

if(DEFINED ETEMU_SHIRES)
  board_runner_args(etemu --shires=${ETEMU_SHIRES})
endif()

if(DEFINED ETEMU_RESET_TARGET)
  board_runner_args(etemu --reset-target=${ETEMU_RESET_TARGET})
endif()

if(DEFINED ENV{SYS_EMU})
  board_runner_args(etemu --sys-emu=$ENV{SYS_EMU})
endif()

if(DEFINED ENV{BOOTROM_TRAMPOLINE_TO_BL2_ELF})
  board_runner_args(etemu --bootrom=$ENV{BOOTROM_TRAMPOLINE_TO_BL2_ELF})
endif()

board_finalize_runner_args(etemu)
