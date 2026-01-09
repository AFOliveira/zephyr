# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# ET-SOC1 et-emu (sys_emu) emulator support

zephyr_get(ET_EMU_PATH)

find_program(ET_EMU sys_emu
  HINTS
    ${ET_EMU_PATH}
    $ENV{ET_EMU_PATH}
    $ENV{SYSEMU_PATH}
)

if(NOT ET_EMU)
  message(WARNING "sys_emu not found. Set ET_EMU_PATH to enable emulation.")
endif()

set(ET_EMU_RUNNER ${ZEPHYR_BASE}/scripts/et_emu_runner.py)

# Default flags for SP emulation
set(ET_EMU_FLAGS
  -elf_load ${APPLICATION_BINARY_DIR}/zephyr/${KERNEL_ELF_NAME}
  -shires 0x400000000
  -mins_dis
  -sp_reset_pc 0x40400000
  -max_cycles 20000000000
)

# Allow extra flags from environment
set(env_et_emu $ENV{ET_EMU_EXTRA_FLAGS})
separate_arguments(env_et_emu)
list(APPEND ET_EMU_EXTRA_FLAGS ${env_et_emu})

add_custom_target(run_et_emu
  COMMAND
  ${Python3_EXECUTABLE} ${ET_EMU_RUNNER}
  ${ET_EMU}
  ${ET_EMU_FLAGS}
  ${ET_EMU_EXTRA_FLAGS}
  DEPENDS ${logical_target_for_zephyr_elf}
  WORKING_DIRECTORY ${APPLICATION_BINARY_DIR}
  COMMENT "[sys_emu] Running on ET-SOC1 Service Processor emulator"
  USES_TERMINAL
)
