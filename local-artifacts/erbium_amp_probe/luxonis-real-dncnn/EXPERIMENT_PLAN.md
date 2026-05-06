# DnCNN3 Audited 100-Experiment Plan

Goal: drive the audited (silicon-vs-ORT) GOPS of the verified Luxonis DnCNN3
port from the current best (1.63 GOPS at 64×64×16h, allclose) up toward and
past the unaudited proxy headline (2.41 GOPS), and beyond if possible.

Every experiment MUST:
- Run on `esperanto-soc6`, shire 0, the verified port (`dncnn3_luxonis_real_vpu.c`).
- Use the VPU (packed-single FP32 dot products are already in the inner loops).
- Pass `allclose(silicon, ORT)` with `max_abs ≤ 1e-5`. Bit-exact preferred but
  not required (FP reorder makes that tile-size dependent).
- Be tabulated in `audit_master.tsv` with timing + correctness + flags.

## Fixed regime for the 100-experiment loop

- Tile: **64×64**. (Largest tile size that round-trips cleanly through the
  launcher; smallest where per-pass overhead is amortized; matches proxy headline.)
- Harts: **16**, mask `0xffff` (8 minions × 2 threads). Even harts use VPU; odd
  harts are scalar siblings sharing the minion pipeline.
- Passes: **8** (matches proxy benchmark amortization).
- Static blobs: input/ORT-reference/weights baked into ELF.
- Avoid `-Ofast` whenever it triggers the libetrt host-side segfault.

## Search space (the knobs)

### A — Compile flags (cheap, 1-line edit per variant)
GCC 12 / RV64IMFC / lp64f. Any subset combinable.

| Group | Flags |
|---|---|
| base level | `-O2`, `-O3`, `-Ofast` |
| loop | `-funroll-loops`, `-fpredictive-commoning`, `-fmodulo-sched`, `-fmove-loop-stores`, `-fno-tree-loop-distribute-patterns`, `-fno-tree-loop-vectorize`, `-fno-schedule-insns`, `-fno-schedule-insns2` |
| inlining | `-fno-ipa-cp -fno-ipa-sra`, `-finline-limit=N` (N ∈ {200, 600, 2000}) |
| codegen | `-frename-registers`, `-falign-functions={32,64,128}`, `-falign-loops={32,64,128}` |
| FP | (only with `-Ofast`) `-fno-fast-math` then re-add `-funsafe-math-optimizations` |

### B — Code-level VPU macros (require porting from proxy `dncnn3_vpu_fp_argbuf.c`)
Each adds a new code path to `dncnn3_luxonis_real_vpu.c` guarded by `#ifdef`:

| Macro | What it does | Status |
|---|---|---|
| `DNCNN_VPU_OC2` | compute 2 output channels per inner loop iter (reuses each loaded input row) | not in verified port |
| `DNCNN_VPU_ACCUM_3X3` | fuse the 3×3 accumulator into one VPU pass; needs `f0..f4` clobbers | not in verified port |
| `DNCNN_VPU_PREFETCH_WPACK` | issue `prefetch.va` on next layer's weight block | not in verified port |
| `DNCNN_VPU_SHARED_WPACK` | one packed-weight blob shared across passes/layers via a host-loaded layout | not in verified port |
| `DNCNN_VPU_SKIP_READ_EVICT` | drop read-side cache evict (only valid when no cross-hart sharing) | not in verified port |
| `DNCNN_VPU_BOUNDARY_ONLY_EVICT` | evict only halo rows, not own range (saves DRAM writes) | not in verified port |
| `DNCNN_VPU_PREFETCH_READ_WINDOW` | prefetch upcoming activation rows including halo | not in verified port |

### C — Cache maintenance strategies
- Full-region evict (current default).
- Boundary-only evict (proxy macro).
- Skip read evict (works only at 1 hart).
- Output-last-only evict.
- Per-layer alternation.

### D — Hart geometry
- 1 / 2 / 4 / 8 (even-only mask 0x55) / 16 (full).
- Row-stripe vs. column-stripe partition.
- Halo-doubling overlap.

### E — Algorithmic (highest leverage, biggest code change)
- Winograd F(2,3) 3×3 conv (theoretical 2.25× FLOP reduction for 3×3, but more memory).
- Implicit GEMM (treat conv as matmul, reuse OC2 logic).
- Layer fusion: Conv+ReLU+Conv+ReLU as a single VPU loop spanning two layers.
- NHWC blocking: NHWC16 (channel groups of 16) to align with 16-float packed-single VPU.

### F — Quantization (deferred, separate experiment regime)
- INT8 packed-integer VPU (the KB's flagged "next NPU-like work").

## 100-experiment schedule

### Batch 1 — Compile-flag exploration (10 exp, exp 1-10)
At 64×64×16h, vary just compile flags. No code changes.

| # | Variant | Flags |
|---|---|---|
| 1 | base_o2 | `-O2` |
| 2 | base_o3 | `-O3` |
| 3 | o3_unroll | `-O3 -funroll-loops` |
| 4 | o3_unroll_noipa | `-O3 -funroll-loops -fno-ipa-cp -fno-ipa-sra` |
| 5 | o3_unroll_align64 | `-O3 -funroll-loops -falign-functions=64 -falign-loops=64` |
| 6 | o3_unroll_align128 | `-O3 -funroll-loops -falign-functions=128 -falign-loops=128` |
| 7 | o3_unroll_predcomm | `-O3 -funroll-loops -fpredictive-commoning` |
| 8 | o3_unroll_modsched | `-O3 -funroll-loops -fmodulo-sched` |
| 9 | o3_unroll_movestores | `-O3 -funroll-loops -fmove-loop-stores` |
|10 | o3_unroll_inline2k | `-O3 -funroll-loops -finline-limit=2000` |

Plus pilots from the in-flight job: ofast_rename, ofast_rename_unroll (already done at 64×64×16h, may segfault).

**Triage rule after batch 1**: pick top 3 by GOPS (allclose=true required).

### Batch 2 — Combine top compile flags (10 exp, exp 11-20)
Pairwise and 3-way of top flags from batch 1.

**Triage rule after batch 2**: pick top 1 compile-flag baseline as the carrier
for all subsequent experiments.

### Batch 3 — Port OC2 macro (10 exp, exp 21-30)
Add `DNCNN_VPU_OC2=1` paths to verified port (one-time code change). Sweep
top compile flags × OC2 on/off. This is the biggest single code change.

**Triage rule after batch 3**: confirm OC2 helps and quantify the gain.

### Batch 4 — Port PREFETCH macros (10 exp, exp 31-40)
Add `DNCNN_VPU_PREFETCH_WPACK` and `DNCNN_VPU_PREFETCH_READ_WINDOW`. Sweep
across last batch's winners.

### Batch 5 — Port ACCUM_3X3 (10 exp, exp 41-50)
Add fused `DNCNN_VPU_ACCUM_3X3` path. This is also a meaningful code change.

### Batch 6 — Cache strategy sweep (10 exp, exp 51-60)
`DNCNN_VPU_BOUNDARY_ONLY_EVICT`, output-last-only, etc. — sweep across the
top-3 from batch 5.

### Batch 7 — Hart geometry (10 exp, exp 61-70)
Try 8-hart (even-only mask 0x55) vs 16-hart with current best macros. Verify
the cross-hart correctness at each. Note: thread1 (odd harts) is scalar.

### Batch 8 — Pass-count sweep (10 exp, exp 71-80)
At top variant, vary `DNCNN_PASSES ∈ {1,2,4,8,16,32,64}`. Reports steady-state
vs. amortized GOPS. Confirms we're not overhead-bound.

### Batch 9 — Algorithm-level (10 exp, exp 81-90)
- Layer fusion experiment (Conv+ReLU collapsed).
- NHWC16 blocking experiment.
- Implicit GEMM rewrite of conv_hidden_fp.

### Batch 10 — Push the best (10 exp, exp 91-100)
Combinatorial fine-tuning of the top-3 from batches 8–9: alignment, inline
limit, evict cadence. Only knob-tweaks, no new code paths.

## Audit gate

Each variant's silicon dump produces a 16-uint32 summary at offset 0x1000 with:
`output_hash`, `reference_hash` (kernel-computed against the static ORT blob
linked into the ELF), `max_abs`, `mean_abs`, `done_count`, `magic`, `ops`.

A variant counts as a valid experiment iff:
- `magic == DNCNN_MAGIC` (kernel ran)
- `done_count == active_harts` (all harts completed)
- `max_abs <= 1e-5` (allclose to ORT)

Bit-exact (`output_hash == reference_hash`) is recorded but not required.

## Master TSV columns

`exp_id, batch, variant, tile, harts, passes, flags, macros, wait_s, gops,
ops, allclose, bit_exact, max_abs, mean_abs, output_hash, reference_hash,
status, dump, log, build_at, run_at`

Stored at `audit_master.tsv` under this directory. Append-only.
