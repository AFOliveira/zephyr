# Copyright (c) 2025 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# ET-SOC1 et-emu (sys_emu) emulator support

zephyr_get(ET_EMU_PATH)

if(DEFINED ENV{SYS_EMU})
  if(EXISTS "$ENV{SYS_EMU}" AND NOT IS_DIRECTORY "$ENV{SYS_EMU}")
    set(ET_EMU "$ENV{SYS_EMU}")
  endif()
endif()

if(NOT ET_EMU)
  find_program(ET_EMU sys_emu
    HINTS
      ${ET_EMU_PATH}
      $ENV{ET_EMU_PATH}
      $ENV{SYSEMU_PATH}
      $ENV{SYS_EMU}
  )
endif()

if(NOT ET_EMU)
  message(WARNING "sys_emu not found. Set ET_EMU_PATH to enable emulation.")
endif()

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

add_custom_target(run_et-emu
  COMMAND
  ${ET_EMU}
  ${ET_EMU_FLAGS}
  ${ET_EMU_EXTRA_FLAGS}
  DEPENDS ${logical_target_for_zephyr_elf}
  WORKING_DIRECTORY ${APPLICATION_BINARY_DIR}
  COMMENT "[sys_emu] Running on ET-SOC1 Service Processor emulator"
  USES_TERMINAL
)
