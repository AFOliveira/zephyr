# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Erbium erbium_emu emulator support

zephyr_get(ERBIUM_EMU_PATH)

find_program(ERBIUM_EMU erbium_emu
  HINTS
    ${ERBIUM_EMU_PATH}
    $ENV{ERBIUM_EMU_PATH}
)

if(NOT ERBIUM_EMU)
  message(WARNING "erbium_emu not found. Set ERBIUM_EMU_PATH to enable emulation.")
endif()

set(ERBIUM_EMU_FLAGS
  -elf_load ${APPLICATION_BINARY_DIR}/zephyr/${KERNEL_ELF_NAME}
  -reset_pc 0x40000000
  -minions 0x1
  -single_thread
  -max_cycles 20000000000
)

# Allow extra flags from environment
set(env_erbium_emu $ENV{ERBIUM_EMU_EXTRA_FLAGS})
separate_arguments(env_erbium_emu)
list(APPEND ERBIUM_EMU_EXTRA_FLAGS ${env_erbium_emu})

add_custom_target(run_erbium-emu
  COMMAND
  ${ERBIUM_EMU}
  ${ERBIUM_EMU_FLAGS}
  ${ERBIUM_EMU_EXTRA_FLAGS}
  DEPENDS ${logical_target_for_zephyr_elf}
  WORKING_DIRECTORY ${APPLICATION_BINARY_DIR}
  COMMENT "[erbium_emu] Running on Erbium Minion emulator"
  USES_TERMINAL
)
