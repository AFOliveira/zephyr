# umode_emlearn_multi — single ELF, multiple programs, parallel across shires

Status: **POC skeleton, not yet wired up.**

## Goal

Run **N different programs concurrently** on N different ET-SoC1 shires
from a **single `kernelLaunch`** with a multi-bit `shire_mask`. This
sidesteps the single-stream serialization we hit in `quad_shire_launcher`
(four launches all queue on `defaultStreams_[0]` and run sequentially).

Empirical anchor: a single `basic_launcher --shire_mask=0xF` of the same
ELF on 4 shires runs in **12.25 s** vs **12.17 s** for a single-shire
launch — proving the firmware *does* parallelize within a single launch
across disjoint shires. We just need each shire to do *different* work.

## Architecture

```
single ELF, single kernelLaunch(shire_mask=0xF)
   │
   └── 4 lead harts (one per shire) wake up at _start
        │
        ├── entry.S filters non-zero local-hart-ids
        ├── entry.S reads shire_id = (hartid >> 6) & 0x1F
        ├── entry.S sets sp = z_main_stack + (shire_id+1)*MAIN_STACK_SIZE
        └── entry.S dispatches by shire_id:
              shire 0 → main_a()  → iris_tree_predict
              shire 1 → main_b()  → iris_rf_predict
              shire 2 → main_c()  → wine_rf_predict
              shire 3 → main_d()  → iris_tree_predict (2nd copy)
        │
        └── each shire writes its 5 predictions into its own
            60-byte slot of a 240-byte host result buffer:
              shire 0 → out[0..59]
              shire 1 → out[60..119]
              shire 2 → out[120..179]
              shire 3 → out[180..239]
```

## What changes vs the single-program sample

| File / area | Change |
|---|---|
| `arch/riscv/core-umode/entry.S` | Add per-shire stack carving + shire-id dispatch table |
| `prj.conf` | Bump `CONFIG_MAIN_STACK_SIZE` to 4×4096 = 16384 so all 4 shires have stack room in a single `z_main_stack[]` |
| `src/main.c` | Define 4 entry symbols (`main_a..main_d`) with distinct programs; standard `main()` is unused |
| Result-buffer protocol | Each program writes its 5 predictions into its own 60-byte sub-slot of a shared 240-byte buffer |
| Launcher (server-side) | `mallocDevice(240)` instead of 60; read all 4 slots and label them |

## Build / run plan

1. **Phase 1 — skeleton (this branch)**: directory + 4 stub `main_X()` functions that just write a sentinel value. No real classifier work yet.
2. **Phase 2 — entry.S patch**: per-shire stack + dispatch. Verify all 4 sentinels appear in the result buffer, none crash.
3. **Phase 3 — wire real classifiers**: replace sentinels with `iris_tree_predict` / `iris_rf_predict` / `wine_rf_predict`.
4. **Phase 4 — launcher tweak**: write `single_launch_multi_program.cpp` (or extend `basic_launcher` with a `--multi-program` flag and 240-byte result buffer). Likely lives next to `dual_shire_launcher.cpp` and `quad_shire_launcher.cpp` on the server.
5. **Phase 5 — wall-time verification**: re-run the heavy busy-loop test. Expect ~12 s (single-launch parallelism), not 48 s (serialized streams).

## Key technical risks

1. **Per-shire stack overlap**: if entry.S doesn't carve correctly, lead harts on different shires write to overlapping stack regions → corruption. Fix: validated formula `sp = &z_main_stack[(shire_id+1) * MAIN_STACK_SIZE]`. With `MAIN_STACK_SIZE=4096` and 4 shires, total `z_main_stack[]` size = 16384 bytes; fits in our SRAM budget easily.
2. **Shared globals other than stack**: anything in `.data`/`.bss` that all four programs touch (e.g., `cm_umode_kernel_args` itself, `__umode_printk_ring`, the et-trace state if it has any). The args global is read-only after `_start`, OK. The trace ring is per-hart-indexed by global hartid, OK. Watch for: any new globals introduced by the multi-program code itself.
3. **Compiler may emit shared rodata between identical `main_X` bodies**: harmless (read-only), no race.
4. **Firmware shire-mask broadcast vs. single-shire launches**: empirically tested OK — `shire_mask=0xF` parallelizes a single ELF across 4 shires.

## Open questions

- Do we want **the C-level dispatcher** (`main()` reads shire-id and switches) or **the asm-level dispatcher** (entry.S jumps to `main_a/b/c/d`)? Both work; asm is one less call frame but harder to read. Default: C-level dispatch in main.c, entry.S only does the per-shire stack setup.
- Where to put per-shire stack carving — in our own `entry.S` patch, or via Kconfig knobs that the existing `entry.S` consumes? The latter is cleaner long-term. For POC: just patch entry.S directly.
- Should the result-buffer protocol be hardcoded (60 bytes per slot, 4 slots) or parameterized via `kernel_args` itself (host writes "expected slot stride" into params)? Hardcoded for POC; parameterize later.

## Out of scope for this POC

- Heterogeneous launches *across* devices (multi-card)
- More than one hart per shire participating in a program (single lead hart per shire, others idle-exit as today)
- Cross-shire synchronization between the 4 programs (they're independent; if they need to coordinate, that's FCC primitives, separate work)

## Files in this directory

- `PLAN.md` (this file) — architecture + checklist
- `CMakeLists.txt`, `sample.yaml`, `prj.conf`, `boards/erbium_minion.conf` — copies from `umode_emlearn`, will be tweaked in phases
- `src/main.c` — to be written: 4 stub `main_X()` functions
- `src/iris_tree.h`, `src/iris_rf.h`, `src/wine_rf.h`, `src/test_vectors_gen.h` — classifier headers (copies for now; can be deduplicated later by sharing across samples)
