# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

zephyr_sources(
	umode_launch.S
	umode_return.S
	umode_syscall.c
	umode_mprot.c
	umode_fault.c
)

zephyr_include_directories(.)

# cm-umode (et-common-libs) headers for the M-mode dispatcher: use the
# canonical isa/erbium/cacheops.h direct-CSR helpers so we don't maintain a
# parallel set of inline CSR writes here.  Same trick as on the ET-SoC1
# Zephyr-U branch: alias bare `asm` to __asm__ so the cm-umode register-hint
# inlines parse under Zephyr's strict C mode.
set(CM_UMODE_ROOT /home/afonso/et-platform/et-common-libs)
zephyr_include_directories(${CM_UMODE_ROOT}/include)
zephyr_compile_definitions(asm=__asm__)
