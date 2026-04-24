# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0
#
# emlearn-style inference demo on Zephyr.  Runs a small hand-written
# decision-tree classifier (representative of emlearn's C output) over
# a handful of test vectors, prints each prediction, then exits via
# SYSCALL_RETURN_FROM_KERNEL.  No OSKERN syscalls needed — the only
# ecall made is the final kernel-exit.

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(umode_emlearn)

target_sources(app PRIVATE
    src/main.c
    src/iris_classifier.c
)

# cm-umode headers supply the `syscall()` helper and SYSCALL_RETURN_FROM_KERNEL.
# Only needed on the U-mode build (ET-SoC1); Erbium Zephyr-M doesn't ecall.
if(CONFIG_RISCV_RUN_IN_UMODE)
    if(NOT DEFINED CM_UMODE_ROOT)
        set(CM_UMODE_ROOT /home/afonso/et-platform/et-common-libs)
    endif()
    target_include_directories(app PRIVATE ${CM_UMODE_ROOT}/include)
    target_compile_definitions(app PRIVATE asm=__asm__)
endif()
