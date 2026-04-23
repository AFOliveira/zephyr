/*
 * Stub for et-trace/encoder.h.
 *
 * Satisfies transitive includes of cm-umode without linking the real
 * esperantoTrace package.  The Trace_Format_String / Trace_Insert_Event
 * encoders become no-ops, so et_printf and trace events silently drop.
 * For real trace output, replace this stub with the real package and
 * link esperantoTrace::et_trace.
 */

#ifndef _ET_TRACE_STUB_ENCODER_H_
#define _ET_TRACE_STUB_ENCODER_H_

#ifdef __cplusplus
extern "C" {
#endif

struct trace_control_block_t {
    char opaque[1];
};

/* Trace event ids — values don't matter; the stubbed macros ignore them. */
#define TRACE_EVENT_STRING_CRITICAL   0
#define TRACE_EVENT_STRING_ERROR      0
#define TRACE_EVENT_STRING_WARNING    0
#define TRACE_EVENT_STRING_INFO       0
#define TRACE_EVENT_STRING_DEBUG      0
#define TRACE_EVENT_STRING_VERBOSE    0
#define TRACE_EVENT_TIMESTAMP         0
#define TRACE_EVENT_HART_ID           0

/* Encoder API — variadic no-ops that swallow their args. */
#define Trace_Format_String(level, cb, fmt, ...)          ((void)0)
#define Trace_Insert_Event(cb, evt, ...)                  ((void)0)
#define Trace_Init(cb, base, size, version)               ((void)0)
#define Trace_Flush(cb)                                   ((void)0)

#ifdef __cplusplus
}
#endif

#endif /* _ET_TRACE_STUB_ENCODER_H_ */
