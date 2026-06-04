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

if(CONFIG_AIF_ET_ISA_MIGRATION_PATCHED_ASM)
	get_filename_component(IMPL_ROOT "${ZEPHYR_BASE}/../.." ABSOLUTE)
	set(AIF_PATCHED_AS_DEFAULT
	    "${IMPL_ROOT}/agentws/jobs/isa-toolchain/workspace/aiflow/repos/binutils-gdb/build-riscv64-unknown-elf/gas/as-new")
	set(AIF_PATCHED_OD_DEFAULT
	    "${IMPL_ROOT}/agentws/jobs/isa-toolchain/workspace/aiflow/repos/binutils-gdb/build-riscv64-unknown-elf/binutils/objdump")
	if(DEFINED ENV{AIF_PATCHED_AS})
		set(AIF_PATCHED_AS "$ENV{AIF_PATCHED_AS}")
	else()
		set(AIF_PATCHED_AS "${AIF_PATCHED_AS_DEFAULT}")
	endif()
	if(DEFINED ENV{AIF_PATCHED_OBJDUMP})
		set(AIF_PATCHED_OBJDUMP "$ENV{AIF_PATCHED_OBJDUMP}")
	else()
		set(AIF_PATCHED_OBJDUMP "${AIF_PATCHED_OD_DEFAULT}")
	endif()

	set(AIF_MN_ASM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/aif_mnemonic_emitters.S)
	set(AIF_MN_OBJ ${CMAKE_CURRENT_BINARY_DIR}/aif_mnemonic_emitters.o)
	set(AIF_MN_VERIFY_LOG ${CMAKE_CURRENT_BINARY_DIR}/aif_patched_binutils_verify.log)

	add_custom_command(
		OUTPUT ${AIF_MN_OBJ}
		COMMAND ${AIF_PATCHED_AS} -march=rv64i_xaifet -mabi=lp64f -o ${AIF_MN_OBJ}
			${AIF_MN_ASM_SRC}
		DEPENDS ${AIF_MN_ASM_SRC}
		COMMENT "Assembling aif_mnemonic_emitters.S with patched Binutils"
		VERBATIM
	)

	add_custom_command(
		OUTPUT ${AIF_MN_VERIFY_LOG}
		COMMAND ${CMAKE_COMMAND} -E env
			AIF_PATCHED_AS=${AIF_PATCHED_AS}
			AIF_PATCHED_OBJDUMP=${AIF_PATCHED_OBJDUMP}
			${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_patched_binutils.sh
			${AIF_MN_OBJ} ${AIF_MN_VERIFY_LOG}
		DEPENDS ${AIF_MN_OBJ}
			${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_patched_binutils.sh
		COMMENT "Verifying migration mnemonics with patched objdump"
		VERBATIM
	)

	add_custom_target(aif_patched_binutils_verify DEPENDS ${AIF_MN_VERIFY_LOG})
	add_dependencies(app aif_patched_binutils_verify)

	target_sources(app PRIVATE ${AIF_MN_OBJ})

	zephyr_ld_options(
		-Wl,-u,aif_mn_emit_flq2
		-Wl,-u,aif_mn_emit_fsq2
		-Wl,-u,aif_mn_emit_faddi_pi
		-Wl,-u,aif_mn_emit_fandi_pi
		-Wl,-u,aif_mn_emit_fcmovm_ps
		-Wl,-u,aif_mn_emit_fcmov_ps
		-Wl,-u,aif_mn_emit_packb
		-Wl,-u,aif_mn_emit_bitmixb
		-Wl,-u,aif_mn_emit_summit
	)
endif()
