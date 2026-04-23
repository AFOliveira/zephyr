# SPDX-License-Identifier: Apache-2.0

zephyr_library()

if(CONFIG_RISCV_RUN_IN_UMODE)

# U-mode build: use the minimal crt0-style sources under arch/riscv/core-umode/.
# None of the standard M-mode core sources (reset.S, prep_c.c, thread.c,
# switch.S, irq_manage.c, fatal.c, cpu_idle.c, reboot.c) are compiled because
# they touch privileged CSRs that are not accessible from U-mode.
zephyr_library_sources(
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/entry.S
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/prep_c.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/thread.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/cpu_idle.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/irq_mask.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/fatal.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/reboot.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/printk_stub.c
)

# arch_irq_enable/disable + upcall entry point live here.  Only pulled in
# when MULTITHREADING is on; otherwise a U-mode kernel without IRQs
# doesn't need them (and would force the whole sw_isr_table cost).
if(CONFIG_MULTITHREADING)
  zephyr_library_sources(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core-umode/irq_entry.c
    # Cooperative context-switch primitive.  z_riscv_switch is pure
    # register save/restore (no CSRs) so it's safe from U-mode.
    switch.S
  )
endif()

# cm-umode integration (ET-SoC1): headers only for phase 1.  `asm` is not a
# reserved keyword under Zephyr's strict C mode, so alias it to __asm__ for
# cm-umode headers that use it (e.g. isa/common/syscall.h).  utils.c and the
# other cm-umode .c sources are skipped here because they transitively pull
# in <et-trace/encoder.h>, which requires the external esperantoTrace package
# we deliberately aren't linking in phase 1.
set(CM_UMODE_ROOT /home/afonso/et-platform/et-common-libs)
zephyr_include_directories(${CM_UMODE_ROOT}/include)
zephyr_compile_definitions(asm=__asm__)

else()

zephyr_library_sources(
  cpu_idle.c
  fatal.c
  irq_manage.c
  prep_c.c
  reboot.c
  reset.S
  switch.S
  thread.c
)

if((CONFIG_MP_MAX_NUM_CPUS GREATER 1) OR (CONFIG_SMP))
  zephyr_library_sources(smp.c)
endif()

if(CONFIG_SMP)
  zephyr_library_sources(ipi.c)

  zephyr_library_sources_ifdef(CONFIG_RISCV_SMP_IPI_CLINT ipi_clint.c)
endif()

zephyr_library_sources_ifdef(CONFIG_FPU_SHARING fpu.c fpu.S)
zephyr_library_sources_ifdef(CONFIG_DEBUG_COREDUMP coredump.c)
zephyr_library_sources_ifdef(CONFIG_IRQ_OFFLOAD irq_offload.c)
zephyr_library_sources_ifdef(CONFIG_USE_ISR_WRAPPER isr.S)
zephyr_library_sources_ifdef(CONFIG_RISCV_PMP pmp.c pmp.S)
zephyr_linker_sources_ifdef(CONFIG_RISCV_PMP ROM_SECTIONS pmp.ld)
zephyr_library_sources_ifdef(CONFIG_THREAD_LOCAL_STORAGE tls.c)
zephyr_library_sources_ifdef(CONFIG_USERSPACE userspace.S)
zephyr_library_sources_ifdef(CONFIG_SEMIHOST semihost.c)
zephyr_library_sources_ifdef(CONFIG_ARCH_STACKWALK stacktrace.c)
zephyr_linker_sources(ROM_START SORT_KEY 0x0vectors vector_table.ld)
zephyr_library_sources_ifdef(CONFIG_LLEXT elf.c)

if(CONFIG_GEN_SW_ISR_TABLE)
  if(CONFIG_DYNAMIC_INTERRUPTS)
    zephyr_linker_sources(RWDATA swi_tables.ld)
  else()
    zephyr_linker_sources(RODATA swi_tables.ld)
  endif()
endif()

endif() # CONFIG_RISCV_RUN_IN_UMODE
