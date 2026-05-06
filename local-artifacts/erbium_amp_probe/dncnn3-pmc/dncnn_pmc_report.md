# DnCNN PMC profiling report

Date: 2026-05-01
Board host: `esperanto-soc6`
Remote artifact root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`
Local artifact root: `local-artifacts/erbium_amp_probe/dncnn3-pmc`

## Method

Each exact DnCNN ELF from the previous optimization sweep was profiled with:

1. `pmc_snapshot_16h.elf` before the DnCNN run.
2. The unmodified DnCNN ELF.
3. `pmc_snapshot_16h.elf` after the DnCNN run.

This gives reliable cross-run deltas for:

- kernel wait time
- Shire Cache PMC cycle/read/write counters
- Memshire PMC cycle/read/write counters
- `hpmcounter3` cycle counter as a rough corroborating signal

Direct `hpmcounter4..8` deltas are not treated as reliable in this
cross-kernel snapshot method. They are accessible from U-mode, but for exact
old ELFs we would need in-kernel instrumentation to sample them cleanly inside
the measured region.

## 16-hart comparison

| Variant | Wait s | GOPS | SC reads | SC writes | MS reads | MS writes |
|---|---:|---:|---:|---:|---:|---:|
| baseline_o2_full_evict | 2.38513 | 0.19783 | 82,800,973 | 1,694,553 | 238,927 | 345,212 |
| window_o3 | 0.73536 | 0.64167 | 82,883,394 | 1,735,672 | 229,326 | 336,925 |
| window_no_prebar_o3 | 0.73871 | 0.63876 | 82,883,274 | 1,735,588 | 229,330 | 336,831 |
| nhwc_no_prebar_o3 | 0.75188 | 0.62757 | 81,077,364 | 3,955,841 | 229,352 | 336,977 |
| nhwc_interior_direct_o3 | 0.44763 | 1.05413 | 13,989,807 | 2,075,342 | 228,954 | 336,562 |
| nhwc_all_channels_o3_rejected | 0.50779 | 0.92925 | 40,667,875 | 4,923,503 | 229,049 | 336,657 |
| best | 0.44768 | 1.05402 | 13,988,556 | 2,074,941 | 228,812 | 336,446 |
| boundary_only_cacheops | 0.46296 | 1.01923 | 13,460,393 | 1,983,125 | 228,985 | 336,570 |

## Findings

- The best and `nhwc_interior_direct_o3` variants are effectively tied.
  `nhwc_interior_direct_o3` won this rerun by a tiny margin.
- The big win came from the interior-direct/NHWC path, not from the later
  cache-maintenance tweak. It cuts SC reads from roughly 83M to roughly 14M at
  16 harts.
- Memshire traffic is almost flat across variants, around 229K reads and 336K
  writes. This benchmark is not DRAM/mesh bandwidth bound.
- The rejected all-channel split is correctly rejected: it is slower than best
  and uses about 3x the SC reads and more than 2x the SC writes.
- Boundary-only activation cache maintenance validated functionally, but it was
  slower than best despite slightly lower SC traffic. The extra small cache ops
  and reduced locality did not pay off for this 64x64 case.

## Best next direction

Do not chase Memshire traffic first. The current limiter is scalar compute plus
Shire Cache/L1 traffic around activation materialization.

The next serious optimization should be a tiled in-kernel implementation that
keeps a row band or channel block hot across hidden layers, with halo handling
done deliberately. For this small 64x64 benchmark, avoid excessive redundant
halo compute; for larger images, layer-fused row-band tiling becomes more
attractive.

For a model-level improvement, prefer architectures with fewer dense
`3x3 x 16 x 16` hidden convolutions: grouped/depthwise separable convolutions or
lower channel counts should map much better to this scalar path. For a hardware
path, the real jump is tensorizing the dense hidden layers on hart 0 of each
minion and using hart 1 for orchestration/prefetch, but that needs tensor-event
PMCs to tune properly.

## Key artifacts

- `parsed/dncnn_pmc_results.tsv`: full parsed PMC table.
- `parsed/dncnn_pmc_reliable_summary.tsv`: stable comparison columns only.
- `parsed/dncnn_pmc_per_hart.tsv`: raw direct HPM deltas per active hart.
- `pmc_snapshot.c`: U-mode snapshot source.
- `dncnn3_bench_boundary_*.elf`: boundary-only cache-maintenance experiment.
- `artifact_manifest_local.tsv` and `artifact_manifest_remote.tsv`: hashes for
  reproducibility.
