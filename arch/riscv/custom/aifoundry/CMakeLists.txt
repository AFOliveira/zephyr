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
