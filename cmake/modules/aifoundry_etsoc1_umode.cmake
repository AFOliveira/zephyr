# Copyright (c) 2026 AIFoundry
# SPDX-License-Identifier: Apache-2.0

# Locate the ET Platform support used by the ET-SoC1 U-mode backend.
#
# Preferred input:
#   -DET_PLATFORM_ROOT=/path/to/et-platform
#
# The ET subtrees are derived from that top-level checkout:
#   ${ET_PLATFORM_ROOT}/et-common-libs
#   ${ET_PLATFORM_ROOT}/et-trace
#
# The ET_PLATFORM_ROOT environment variable is also honored. If no root is
# set, fall back to sibling checkouts next to this Zephyr tree.

function(aifoundry_find_existing_dir out_var)
  foreach(candidate IN LISTS ARGN)
    if(candidate AND IS_DIRECTORY "${candidate}")
      set(${out_var} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

function(aifoundry_require_existing_file path description)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "Could not find ${description}: ${path}. Set ET_PLATFORM_ROOT to "
      "the top-level et-platform directory.")
  endif()
endfunction()

function(aifoundry_configure_etsoc1_umode)
  get_filename_component(_zephyr_parent "${ZEPHYR_BASE}/.." ABSOLUTE)

  set(ET_PLATFORM_ROOT "$ENV{ET_PLATFORM_ROOT}" CACHE PATH
      "Root of an ET Platform checkout")

  if(NOT ET_PLATFORM_ROOT)
    aifoundry_find_existing_dir(ET_PLATFORM_ROOT
      "${_zephyr_parent}/et-platform-vidas"
      "${_zephyr_parent}/et-platform")
    set(ET_PLATFORM_ROOT "${ET_PLATFORM_ROOT}" CACHE PATH
        "Root of an ET Platform checkout" FORCE)
  endif()

  if(NOT ET_PLATFORM_ROOT)
    message(FATAL_ERROR
      "Could not find an ET Platform checkout. Set ET_PLATFORM_ROOT to "
      "the top-level et-platform directory.")
  endif()

  set(ET_COMMON_LIBS_ROOT "${ET_PLATFORM_ROOT}/et-common-libs"
      CACHE INTERNAL "Derived ET Platform et-common-libs tree" FORCE)
  set(ET_TRACE_ROOT "${ET_PLATFORM_ROOT}/et-trace"
      CACHE INTERNAL "Derived ET Platform et-trace tree" FORCE)

  aifoundry_require_existing_file(
    "${ET_COMMON_LIBS_ROOT}/include/erbium-soc1sim/isa/syscall.h"
    "erbium-soc1sim U-mode headers")
  aifoundry_require_existing_file(
    "${ET_COMMON_LIBS_ROOT}/include/etsoc/common/utils.h"
    "cm-umode utility headers")
  aifoundry_require_existing_file(
    "${ET_COMMON_LIBS_ROOT}/src/trace/trace_umode.c"
    "cm-umode trace source")
  aifoundry_require_existing_file(
    "${ET_TRACE_ROOT}/include/et-trace/encoder.h"
    "et-trace headers")

  # The soc1sim backend's public include contract is <erbium/isa/...>.
  # In an installed ET Platform this is already true. In a source tree
  # the headers live under include/erbium-soc1sim/isa, so stage that
  # source layout into the same public include shape inside the Zephyr
  # build directory.
  set(_soc1sim_staged_include
      "${CMAKE_BINARY_DIR}/aifoundry/erbium-soc1sim-staged-include")
  file(REMOVE_RECURSE "${_soc1sim_staged_include}")
  file(MAKE_DIRECTORY "${_soc1sim_staged_include}/erbium")
  file(COPY "${ET_COMMON_LIBS_ROOT}/include/erbium-soc1sim/isa"
       DESTINATION "${_soc1sim_staged_include}/erbium")

  if(NOT TARGET aifoundry_erbium_soc1sim_interface)
    add_library(aifoundry_erbium_soc1sim_interface INTERFACE)
    target_include_directories(aifoundry_erbium_soc1sim_interface INTERFACE
      "${_soc1sim_staged_include}")
  endif()

  if(NOT TARGET et-common-libs::erbium-soc1sim)
    add_library(et-common-libs::erbium-soc1sim ALIAS
      aifoundry_erbium_soc1sim_interface)
  endif()

  # Keep app and Zephyr sources on the public ET include surfaces.  The
  # explicit include directories are still needed because Zephyr sources do not
  # all consume target usage requirements from external interface targets.
  zephyr_link_libraries(et-common-libs::erbium-soc1sim)
  zephyr_include_directories(
    "${_soc1sim_staged_include}"
    "${ET_COMMON_LIBS_ROOT}/include"
    "${ET_TRACE_ROOT}/include")

  # ET headers use the GNU spelling in a few inline asm helpers. Zephyr
  # builds with strict C modes where `asm` is not a keyword.
  zephyr_compile_definitions(asm=__asm__)

  # Vidas' ET headers use symbolic custom CSR names in inline asm. Zephyr's
  # RISC-V SDK does not know those non-standard names, so teach GAS the
  # constants while still compiling the vendor headers unchanged.
  zephyr_compile_options(
    -Wa,-defsym,hartid=0xcd0
    -Wa,-defsym,tensor_wait=0x830)

  if(NOT TARGET aifoundry_et_cm_umode)
    zephyr_library_named(aifoundry_et_cm_umode)
    zephyr_library_sources(
      "${ET_COMMON_LIBS_ROOT}/src/common/printf.c"
      "${ET_COMMON_LIBS_ROOT}/src/common/printf_dummy.c"
      "${ET_COMMON_LIBS_ROOT}/src/etsoc/common/utils.c"
      "${ET_COMMON_LIBS_ROOT}/src/etsoc/drivers/pmu/pmu.c"
      "${ET_COMMON_LIBS_ROOT}/src/trace/trace_umode.c")
    zephyr_library_include_directories(
      "${ET_COMMON_LIBS_ROOT}/include"
      "${ET_TRACE_ROOT}/include")
  endif()

  set(AIFOUNDRY_ERBIUM_SOC1SIM_INCLUDE "${_soc1sim_staged_include}"
      CACHE PATH "Staged public include path for erbium-soc1sim" FORCE)
  set(AIFOUNDRY_CM_UMODE_LIBRARY aifoundry_et_cm_umode
      CACHE INTERNAL "Zephyr-built ET cm-umode support library" FORCE)
endfunction()
