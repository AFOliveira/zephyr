# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(isa_migration_smoke)

target_sources(app PRIVATE
	src/main.c
	src/aif_encode_verify.c
	src/aif_compile_only_emitters.c
)

# Retain compile-only raw emitters in the final ELF (not called from main()).
zephyr_ld_options(
	-Wl,-u,aif_co_emit_flq2
	-Wl,-u,aif_co_emit_fsq2
	-Wl,-u,aif_co_emit_faddi_pi
	-Wl,-u,aif_co_emit_fandi_pi
	-Wl,-u,aif_co_emit_fcmov_ps
	-Wl,-u,aif_co_emit_fcmovm_ps
	-Wl,-u,aif_co_emit_packb
	-Wl,-u,aif_co_emit_bitmixb
	-Wl,-u,aif_co_emit_summit
)
