# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(neighborhood_probe)

target_sources(app PRIVATE src/main.c)

zephyr_ld_options(-Wl,--emit-relocs)
