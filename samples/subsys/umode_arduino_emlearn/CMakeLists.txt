# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0
#
# Arduino-flavoured emlearn iris-tree demo.  Builds the upstream
# ArduinoCore-zephyr module (cores/arduino/main.cpp -> setup/loop) plus
# our sketch.  The module's CMakeLists.txt does most of the heavy
# lifting (zephyr_sources for main.cpp, zephyrSerial.cpp, etc) when
# CONFIG_ARDUINO_API=y.  We just contribute the sketch and pull in
# the variant overlay so `Serial` resolves to the per-board UART
# (real Shakti UART on erbium_minion, virtual trace UART on
# etsoc1_minion_umode).

cmake_minimum_required(VERSION 3.20.0)

# The variant overlay (set per-board under
# modules/lib/Arduino-Zephyr-API/variants/<board>/<board>.overlay)
# ships the `zephyr,user::serials` property the upstream ZephyrSerial
# class consumes.  The board's own .overlay (in this sample's boards/)
# is layered on top via Zephyr's standard board-overlay loader.
cmake_path(SET ZephyrBase $ENV{ZEPHYR_BASE})
set(VARIANT_OVERLAY ${ZephyrBase}/modules/lib/Arduino-Zephyr-API/variants/${BOARD}/${BOARD}.overlay)
if(EXISTS ${VARIANT_OVERLAY})
    list(APPEND EXTRA_DTC_OVERLAY_FILE ${VARIANT_OVERLAY})
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(umode_arduino_emlearn)

target_sources(app PRIVATE
    src/sketch.cpp
)
target_include_directories(app PRIVATE src)

# Static initializers / global ctors are not run on the U-mode build
# (no MULTITHREADING + no kernel main running .init_array).  The sketch
# is written to avoid them, but the upstream ZephyrSerial declares
# `Serial` as a file-scope global with a non-trivial ctor — Zephyr's
# C++ glue invokes those during pre-kernel/pre-main automatically.
#
# We deliberately do NOT pass --emit-relocs.  The ET-SoC1 launcher's
# loadCode path runs a relocateELF() pass over every SHT_RELA section it
# finds with a hard-coded BASE_OFFSET=0x1000 that is wrong for our
# segment layout (LOAD starts at file offset 0xb0, not 0x1000); when
# --emit-relocs preserves R_RISCV_64 entries against initlevel /
# device_area, that loader pass scribbles over rodata-adjacent bytes and
# leaves dev->ops.init pointers garbage.  The linker has already applied
# every needed relocation; keeping the .rela* sections out of the ELF
# means the launcher has nothing to (mis)patch and the bytes reach the
# device exactly as linked.
#
# Force the LOAD segment to start at file offset 0x1000.  The launcher's
# elf-loader writes file bytes verbatim at `deviceBuffer + segment_offset`,
# but the firmware expects the kernel image at `deviceBuffer + 0x1000`
# (matches the BASE_OFFSET=0x1000 hard-coded in relocateELF()).  When the
# segment lives at file offset 0xb0 (the default for a 2-PHDR ELF), the
# kernel is loaded 0xf50 bytes too low: every absolute data read returns
# bytes from 0xb0 EARLIER than the linker placed them, and only the
# trivial baseline that performs no static-data reads happens to survive.
# Bumping the segment offset to 0x1000 through max-page-size pads the
# file just enough that segment_offset == BASE_OFFSET, and absolute reads
# resolve to the correct bytes.
