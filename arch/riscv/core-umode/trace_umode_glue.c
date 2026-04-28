/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Single-translation-unit instantiation of the et-trace encoder for
 * Zephyr-U.  The encoder.h is a header-only library; defining
 * ET_TRACE_ENCODER_IMPL exactly once before the include emits its
 * implementation into this object file.
 *
 * Customisation macros below adapt et-trace to a U-mode-only build:
 *   - memcpy / strlen / vsnprintf come from picolibc (the standard
 *     Zephyr libc on this build).
 *   - ET_TRACE_GET_TIMESTAMP returns 0 because the standard RISC-V
 *     `rdcycle` CSR (mcycle) is gated by mcounteren on this silicon
 *     and traps with cause-2 when read from U-mode (verified
 *     empirically).  Trace records will therefore have all-zero
 *     timestamps; et_dev_trace_decoder still prints the strings.
 *   - PMC counter accessors are stubbed out for the same reason
 *     (no privileged PMC reads from U-mode); the et_trace_pmc_*
 *     helpers won't produce useful data, but et_printf doesn't need
 *     them.
 */

#include <string.h>
#include <stdio.h>
#include <stdint.h>

static inline uint64_t et_trace_zero_timestamp(void) { return 0; }

#define ET_TRACE_READ_MEM(dst, src, size)             memcpy((dst), (src), (size))
#define ET_TRACE_WRITE_MEM(dst, src, size)            memcpy((dst), (src), (size))
#define ET_TRACE_STRLEN(str)                          strlen((str))
#define ET_TRACE_VSNPRINTF(buf, count, fmt, va)       vsnprintf((buf), (count), (fmt), (va))
#define ET_TRACE_GET_TIMESTAMP()                      et_trace_zero_timestamp()

/* Tell encoder.h to NOT emit hpmcounter reads (they would compile to
 * unprivileged hpmcounter CSR reads which trap in U-mode). */
#define ET_TRACE_GET_HPM_COUNTER(id)                  ((uint64_t)0)

#define ET_TRACE_ENCODER_IMPL
#include <et-trace/encoder.h>
