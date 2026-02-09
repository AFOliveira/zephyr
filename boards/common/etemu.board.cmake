# SPDX-License-Identifier: Apache-2.0

board_set_sim_runner_ifnset(etemu)

if(DEFINED ETEMU_SHIRES)
  board_runner_args(etemu --shires=${ETEMU_SHIRES})
endif()

if(DEFINED ETEMU_RESET_TARGET)
  board_runner_args(etemu --reset-target=${ETEMU_RESET_TARGET})
endif()

if(DEFINED ETEMU_ENABLE_MINIONS)
  board_runner_args(etemu --enable-minions)
endif()

if(DEFINED ENV{SYS_EMU})
  board_runner_args(etemu --sys-emu=$ENV{SYS_EMU})
endif()

board_finalize_runner_args(etemu)
