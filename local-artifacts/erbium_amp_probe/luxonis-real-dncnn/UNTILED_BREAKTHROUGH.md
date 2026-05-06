# Untiled 240×320 DnCNN3 — 2.86× speedup over tiled

## TL;DR

Single-launch full-image kernel. **29.57 s vs 84.5 s** on `esperanto-soc4`, shire 0,
16 harts, audit-clean (`max_abs = 8.941e-07 ≤ 1e-5` vs full ORT FP32).

| pipeline | wait_s | inf/s | full-image GOPS | max_abs vs ORT |
|---|---:|---:|---:|---:|
| tiled (tg02 best, h=24) | 84.59 | 0.012 | 3.090 | 8.941e-07 |
| **untiled u04 (champion)** | **29.57** | **0.034** | **3.452** | **8.941e-07** |

Same numerical result (8.941e-07 — it's the FP32 round-off floor of the model);
just produced 2.86× faster.

## What changed

1. `region0_size` extended `16 MB → 64 MB` via `-Wl,--defsym=region0_size=0x04000000`.
   This lets `.bss` host the two activation banks (`static_act0`, `static_act1`,
   each 19.66 MB at full 240×320×64ch FP32). Verified via `riscv-elf-size`:
   bss = 39.4 MB, total elf-loaded region usage ~38 MB inside the 64 MB window.
2. Single ELF, single launcher invocation. No per-tile flock + dump-after overhead.
3. No halo redundancy. Tiled at h=20/t=64 over-computes by ~1.6× (boundary tiles
   double-count their halo row strip with adjacent tiles); untiled does each
   pixel exactly once.

Build:
```bash
cd luxonis-real-dncnn
LABEL=u04_no_prefetch \
EXTRA_FLAGS="-UDNCNN_VPU_PREFETCH_READ_WINDOW" \
bash build_untiled.sh
```

Run on soc4:
```bash
"$LAUNCH" --elf-load u04_no_prefetch.elf --shire 0 \
  --file_load 0x0,zero64k.bin --dump_after dump.bin --timeout 240
# Output at offset 0x300000 of dump.bin (240*320 FP32 = 307200 bytes)
```

## Ablation findings

| variant | flags | wait_s | GOPS | Δ vs u01 |
|---|---|---:|---:|---:|
| u01 baseline | `BOUNDARY_ONLY_EVICT` + `PREFETCH_READ_WINDOW` + `FUSED9TAP` | 29.95 | 3.41 | — |
| u02 no_boundary | (no BOUNDARY_ONLY_EVICT) | 32.13 | 3.18 | **+7.3% slower** |
| u03 skip_read | `SKIP_READ_EVICT` instead of BOUNDARY | 30.52 | 3.35 | +1.9% slower |
| **u04 no_prefetch** | drop `PREFETCH_READ_WINDOW` | **29.57** | **3.45** | **−1.3% faster** |

Notably, `PREFETCH_READ_WINDOW` *helps* tiled (cold cache restart per tile) but
*hurts* untiled (steady-state activation stream — prefetch competes with current
loads). This is a flag that needs to flip when going from tiled to untiled.

## PMC profile (untiled u04)

| metric | tiled tg01 | tiled tg02 | **untiled u04** |
|---|---:|---:|---:|
| cycles/op (max T0) | 0.281 | 0.255 | **0.235** |
| L2 reads/kop | 55.9 | 56.0 | (counter wrap, see note) |
| DRAM reads BW | 0.004 GB/s | 0.003 | ~0.11 GB/s |

Cycles/op dropped 16% vs tg01 because activations stay warm in L2 across all
20 layers — no per-tile cold-cache restart. Still compute-bound (fmadd.ps
issue/latency stalls); same architectural bottleneck identified in the tiled
PMC analysis. DRAM remained idle even with 4× larger working set.

(The L2 syscall counters wrap on a 30-second window; need 64-bit-safe
delta or per-second sampling to read them cleanly. The cycles+DRAM signals
are reliable.)

## What's left on the table

| approach | est. speedup | effort | risk |
|---|---|---|---|
| **DONE: untile** | **2.86×** | done | done |
| multi-shire fanout (4–32 shires) | 4–32× | medium | needs launcher work |
| better VPU asm scheduling | ~2× | medium | requires asm hand-tune |
| TFMA matrix engine (FP32) | 2–10× | high | new toolchain (etsoc/isa headers, scratchpad management) |
| TFMA + FP16 | additional 2× | high | re-baseline audit gate (~1e-3 vs FP16-ORT) |

Untiled at 30 s/image is still ~30× away from realtime (33 ms = 30 fps). The
clean wins now are multi-shire and better asm; TFMA is the long-pole architectural
pivot.

## Files

- `build_untiled.sh` — builder, `region0_size` extension is the critical change
- `/tmp/dncnn_untiled/u0*.elf` — built ELFs (3.3 MB each)
- `/tmp/untiled_ablation/` — per-variant run dumps
- `/tmp/untiled_pmc/u04/` — PMC capture for the champion
- `/home/afonso/etsoc1-luxonis-experiments/results/audit_master_untiled.tsv` — TSV summary
