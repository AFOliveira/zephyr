# PMC bottleneck analysis — 3 best tiled-v2 variants on ET-SoC1

Captured 2026-05-05 on `esperanto-soc4`, shire 0, 16 harts (full SMT). Each variant
ran one *interior* tile (`tile_06`, away from image edges) bracketed by
`pmc_snapshot_16h.elf` for free-running PMC + syscall-mediated SC/MS counters.

## What was actually measured

`tile_06` because it's the maximum-input interior tile in all three geometries:
output 64×64, input 104×104 (h=20) or 112×112 (h=24). Boundary tiles do less work
and would skew the per-cycle metrics. Also cleanly comparable: tg01 and tg09 share
the *exact* same input shape; only the kernel macro differs.

Ops are computed from tile geometry — `(H*W) * (CH*K*K + 18*CH*CH*K*K + CH*K*K) * 2`
with `CH=64, K=3`. The kernel summary lives at `heap0_end-64MB`, which the raw
launcher's `--dump_after` doesn't capture.

Reliable PMC channels (verified against `optimization-kb/current_best_pmc_results.tsv`):

| Source | Counter | Reliable? |
|---|---|---|
| HPM[0] cycles | T0 of even minions (m0,m2,m4,m6) | ✓ |
| HPM[0] cycles | T0 of odd minions (m1,m3,m5,m7) | ✗ underflows |
| HPM[1..5] | per-hart events (icache, instret) | ✗ unreliable, same in focused20 |
| Syscall PMCs | SC (L2) reads/writes/cycles | ✓ |
| Syscall PMCs | MS (DRAM) reads/writes/cycles | ✓ |

## Numbers

| metric | tg01 h20 base | tg02 h24 halo | tg09 skip_read |
|---|---:|---:|---:|
| input shape | 104×104 | 112×112 | 104×104 |
| wait per tile (s) | 4.825 | 5.061 | 4.562 |
| **tile GOPS** | **2.980** | **3.295** | **3.152** |
| ops (MAC×2) | 14.38 G | 16.68 G | 14.38 G |
| max T0 cycles | 4.033 G | 4.250 G | 3.878 G |
| implied freq (MHz) | 836 | 840 | 850 |
| **cycles / op** | **0.281** | **0.255** | **0.270** |
| **L2 reads / kop** | **55.90** | 55.99 | 56.27 |
| L2 writes / kop | 1.15 | 1.15 | 1.56 |
| L2 read BW (GB/s, 64B lines) | 10.66 | 11.81 | 11.35 |
| L2 write BW (GB/s) | 0.22 | 0.24 | 0.31 |
| DRAM reads / kop | 0.0201 | 0.0148 | 0.0169 |
| DRAM read BW (GB/s) | 0.0038 | 0.0031 | 0.0034 |
| DRAM write BW (GB/s) | 0.0040 | 0.0036 | 0.0039 |

## Interpretation

### 1. **DRAM is a non-issue.** ~3–4 MB/s sustained.
Per-tile data set: 1× weights (≈4.5 MB packed FP32) + 1× input strip (≈2.7 MB) +
2× activation banks (≈2.7 MB each). All fits in the 16 MB L2 per shire after
the first load. DRAM PMCs confirm: only ~280 K total read+write transactions
across the whole 5 s tile, ≈ 18 MB total — basically just the cold miss to fill
L2. No DRAM streaming during compute. **Don't optimize DRAM access.**

### 2. **L2 carries the steady-state traffic, but isn't the bottleneck.**
~56 L2 reads per 1000 ops, sustained at ~11 GB/s. The shire L2 peak is in the
30+ GB/s range, so we're at roughly 30–40% of L2 read capacity. Plenty of
headroom — we're not L2-bandwidth-bound.

The skip_read variant *increases* L2 read count slightly (56.27 vs 55.90) and
write count (+37%) yet runs faster — confirming the win is the elimination of
the `evict` instruction overhead in the minion pipeline, **not** memory traffic.

### 3. **Compute is the bottleneck — fmadd.ps issue/latency stalls.**
Theoretical peak per minion = 1 fmadd.ps / cycle = 8 ops / cycle. With 8 minions
at 833 MHz = **53 Gops/s peak per shire**. We're getting 3 Gops/s. **6% of peak.**

cycles/op = 0.255–0.281 vs ideal 1/64 = 0.0156. Gap = 16–18×. Since L2 traffic
is comfortable and DRAM is idle, the stalls live in the minion pipeline:
fmadd.ps result-to-source dependencies blocking issue.

### 4. **Why h=24 (tg02) wins despite 16% more compute.**
- L2 reads/kop is *flat* across tile sizes (≈56) — same memory access pattern
- cycles/op drops from 0.281 → 0.255 (9% better) because:
  - the larger inner-loop trip count amortizes loop epilogue / startup spill
  - more useful work per FENCE/evict barrier
- Net: more ops, same memory pressure, fewer fixed-cost stalls per op.

### 5. **Why skip_read (tg09) wins by ~5%.**
- Identical geometry to tg01
- cycles/op 0.270 vs 0.281 (4% better) — direct savings from removing
  per-row read-side evict instructions. Each evict is ~few cycles; over 18
  hidden layers × 64 channels × 104 rows that adds up.
- Memory traffic is *not* lower (slightly higher, in fact). The win is purely
  pipeline.

### 6. **What this rules out.**
- "More cache" / "more L2" wouldn't help — we're not L2-bound.
- "Better DRAM prefetch" wouldn't help — DRAM isn't being used.
- "Less halo" wouldn't help — confirmed h=20 is the minimum that audits, and
  bigger halo (h=24) actually wins because compute amortizes loop overhead.

### 7. **What would actually help (data-driven).**
1. **More accumulators in the inner asm to hide fmadd.ps latency.** FUSED9TAP_OC4
   tried this but regressed — likely register-pressure spill. Need a careful
   schedule: 4 OCs × 9 taps with explicit register allocation, possibly fewer
   active output rows per pass.
2. **Software-pipeline weight loads against pending fmadd.ps.** The `flq2`
   load issues into the same pipeline as `fmadd.ps`; if it can issue while
   the previous fmadd is in-flight latency, we close part of the 17× gap.
3. **Channel-dim tiling for L1D residency.** 104 × 64 ch × 4 B = 26 KB per row,
   far over L1D (4 KB). If we process channels in groups of 8 (2 KB), each
   chunk fits in L1D and fmadd.ps reads from L1, not L2. Cuts L2 traffic to
   maybe 20 GB/s headroom but more importantly cuts fmadd.ps load latency
   (L1 hit ~3 cycles vs L2 hit ~10–15 cycles).
4. **T1 (odd hart) doing weight-prefetch.** SMT_HETERO_T0_ONLY refuted T1
   "parking", but T1 *not running fmadd.ps* could still issue prefetches that
   warm L1D for T0. PREFETCH_READ_WINDOW already does this somewhat.

### 8. **Reality check on full-image numbers.**
Per-tile interior GOPS (3.0–3.3) is higher than the *full-image* GOPS in
`audit_master_tiled_v2.tsv` (2.9–3.1) because:
- 4 corner tiles at 84×84 input (h=20) do less work and finish faster
- 12 edge tiles at 84×104 are partial
- Aggregate inf/s factors in launcher overhead per tile (~50 ms wrapper cost
  flock→stage→dump, ~1 s total per 20-tile inference)

Single-tile interior measurement is the apples-to-apples view of the kernel
itself. Full-image GOPS is what end-users see for real 240×320 inference.

## Files

- `runs/<label>_t06/` — per-variant captures (snap before/after, kernel dump,
  launch logs, manifest)
- `pmc_results.tsv` — machine-readable summary
- `pmc_capture_one_tile.sh` — staged on soc4, takes one flock for all 3 launches
- `parse_pmc_tiled.py` — host-side parser (delta + geometry-derived ops)
