# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0
#
# M3 batch runner for gp-sdk test-compute-kernels on ET-SoC1 Zephyr-U.
# The specific kernel to load is selected at configure time:
#
#   west build -b etsoc1_minion_umode samples/subsys/umode_kernel_runner \
#     -- -DGPSDK_KERNEL_SRC=/abs/path/to/test-compute-kernels/src/abort/abort.c \
#        -DGPSDK_KERNEL_NAME=abort

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(aifoundry_umode_kernel_runner)

if(NOT DEFINED GPSDK_KERNEL_SRC)
    message(FATAL_ERROR "Pass -DGPSDK_KERNEL_SRC=<abs path to kernel .c>")
endif()
if(NOT DEFINED GPSDK_KERNEL_NAME)
    get_filename_component(GPSDK_KERNEL_NAME ${GPSDK_KERNEL_SRC} NAME_WE)
endif()

# et-trace stub first on the include path so trace_umode_cb.h's include of
# <et-trace/encoder.h> resolves to our local no-op instead of the real
# esperantoTrace lib (which we don't link).
zephyr_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stubs)

# The kernel may include "common.h", "markers.h", "lfsr.h", "sync_minions.h"
# from its own directory; add that on the include path too.
get_filename_component(GPSDK_KERNEL_DIR ${GPSDK_KERNEL_SRC} DIRECTORY)
zephyr_include_directories(${GPSDK_KERNEL_DIR})
zephyr_include_directories(${GPSDK_KERNEL_DIR}/include)
# test-compute-kernels shared headers
zephyr_include_directories(/home/afonso/et-platform/test-compute-kernels/src/shared/include)

# cm-umode non-trace sources — provide et_abort, et_printf, memcpy/memset,
# libc stubs.  Deliberately skip src/trace/trace_umode.c (needs real
# esperantoTrace) and src/drivers/etsoc/pmu/pmu.c (unused by our kernels).
set(CM_UMODE_ROOT /home/afonso/et-platform/et-common-libs)
target_sources(app PRIVATE
    src/main.c
    ${CM_UMODE_ROOT}/src/common/etsoc/utils.c
    ${CM_UMODE_ROOT}/src/common/printf.c
    ${CM_UMODE_ROOT}/src/common/printf_dummy.c
    # Skip cm-umode's libc_stub/{stdlib,string}.c — they collide with
    # Zephyr's minimal libc (`abort`, `memcpy`, ...).  Zephyr already
    # provides all of those.
    ${GPSDK_KERNEL_SRC}
)

target_compile_definitions(app PRIVATE
    GPSDK_KERNEL_NAME="${GPSDK_KERNEL_NAME}"
)

# Teach the assembler the custom Minion CSR names used by cm-umode inline asm.
# (csrr %0, hartid is in isa/common/hart.h; Zephyr's bundled assembler doesn't
# know the ET-Minion symbolic CSR table.)
# Teach the assembler more ET-Minion custom CSR names used by kernels.
zephyr_compile_options(
    -Wa,--defsym,hartid=0xcd0
    -Wa,--defsym,tensor_error=0xcd1
    -Wa,--defsym,cache_invalidate=0x7d0
    -Wa,--defsym,mcache_control=0x7c0
    -Wa,--defsym,fcc=0x821
)
