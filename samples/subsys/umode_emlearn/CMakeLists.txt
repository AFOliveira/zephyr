# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0
#
# emlearn-style inference demo on Zephyr.  Runs generated decision-tree
# classifiers over canned test vectors and exits via the cm-umode kernel
# return syscall on ET-SoC1.

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(umode_emlearn)

target_sources(app PRIVATE
    src/main.c
)
# The generated classifiers are header-only.
target_include_directories(app PRIVATE src)

zephyr_ld_options(-Wl,--emit-relocs)
