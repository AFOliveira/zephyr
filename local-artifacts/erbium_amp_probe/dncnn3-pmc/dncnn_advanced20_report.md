# DnCNN Advanced Optimization Sweep

Date: 2026-05-01
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`
Local root: `local-artifacts/erbium_amp_probe/dncnn3-pmc`

## What Was Tried

The sweep used 20 separate 16-hart test plans:

1. scalar control
2. scalar final-output-evict/barrier only on last pass
3. scalar hidden-loop unroll
4. scalar hidden-loop unroll plus final-output-last-only
5. scalar boundary-only cache maintenance
6. scalar boundary-only cache maintenance plus hidden-loop unroll
7. scalar read-window prefetch
8. scalar read-window prefetch plus hidden-loop unroll
9. scalar private-scratch fused layers
10. scalar private-scratch fused layers plus hidden-loop unroll
11. scalar private-scratch fused layers plus pass barrier
12. scalar private-scratch fused layers plus pass barrier and hidden-loop unroll
13. scalar 8-channel model shape
14. scalar 4-channel model shape
15. VPU FP32 DnCNN-shaped path, 1 pass
16. VPU FP32 DnCNN-shaped path, 8 passes
17. VPU FP32, 8 passes, final-output-last-only
18. VPU FP32, 8 passes, shared packed weights
19. VPU FP32, 8 passes, private-scratch fused layers
20. VPU FP32, 8 passes, private-scratch fused layers plus shared packed weights

All 20 built and completed on silicon.  The same-int8 scalar variants were
accepted only with output sum `514785`.  The VPU FP32 variants use a different
numeric path and were accepted by magic/done/mask plus slot/output checksum
agreement.

## 20-Plan Results

| # | Variant | Valid | Wait s | GOPS | Passes | Output | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | adv20_01_scalar_control | 1 | 0.467043 | 1.010312 | 8 | 514785 | same int8 output |
| 2 | adv20_02_scalar_lastonly | 1 | 0.462395 | 1.020468 | 8 | 514785 | same int8 output |
| 3 | adv20_03_scalar_unroll | 1 | 0.452853 | 1.041970 | 8 | 514785 | same int8 output |
| 4 | adv20_04_scalar_unroll_lastonly | 1 | 0.456881 | 1.032784 | 8 | 514785 | same int8 output |
| 5 | adv20_05_scalar_boundary | 1 | 0.463084 | 1.018949 | 8 | 514785 | same int8 output |
| 6 | adv20_06_scalar_boundary_unroll | 1 | 0.443350 | 1.064304 | 8 | 514785 | same int8 output |
| 7 | adv20_07_scalar_prefetch | 1 | 0.468485 | 1.007202 | 8 | 514785 | same int8 output |
| 8 | adv20_08_scalar_prefetch_unroll | 1 | 0.451286 | 1.045588 | 8 | 514785 | same int8 output |
| 9 | adv20_09_scalar_private_fused | 1 | 0.885934 | 0.532612 | 8 | 514785 | same int8 output |
| 10 | adv20_10_scalar_private_fused_unroll | 1 | 0.831732 | 0.567321 | 8 | 514785 | same int8 output |
| 11 | adv20_11_scalar_private_fused_barrier | 1 | 0.886619 | 0.532201 | 8 | 514785 | same int8 output |
| 12 | adv20_12_scalar_private_fused_barrier_unroll | 1 | 0.837622 | 0.563332 | 8 | 514785 | same int8 output |
| 13 | adv20_13_scalar_ch8 | 1 | 0.134837 | 0.909864 | 8 | 524288 | smaller scalar model shape |
| 14 | adv20_14_scalar_ch4 | 1 | 0.040165 | 0.822363 | 8 | 524288 | smaller scalar model shape |
| 15 | adv20_15_vpu_fp_p1 | 1 | 0.038180 | 1.544835 | 1 | 514073 | VPU FP32 path |
| 16 | adv20_16_vpu_fp_p8 | 1 | 0.300984 | 1.567722 | 8 | 514073 | VPU FP32 path |
| 17 | adv20_17_vpu_fp_p8_lastonly | 1 | 0.299743 | 1.574213 | 8 | 514073 | VPU FP32 path |
| 18 | adv20_18_vpu_fp_p8_sharedw | 1 | 0.296009 | 1.594070 | 8 | 514073 | VPU FP32 path |
| 19 | adv20_19_vpu_fp_p8_private | 1 | 0.635012 | 0.743071 | 8 | 514073 | VPU FP32 path |
| 20 | adv20_20_vpu_fp_p8_private_sharedw | 1 | 0.622392 | 0.758138 | 8 | 514073 | VPU FP32 path |

## Scaling Check

| Variant | Harts | Valid | Wait s | GOPS | Output |
|---|---:|---:|---:|---:|---:|
| scale_scalar_boundary_unroll | 1 | 1 | 4.651260 | 0.101448 | 514785 |
| scale_scalar_boundary_unroll | 2 | 1 | 3.365730 | 0.140195 | 514785 |
| scale_scalar_boundary_unroll | 4 | 1 | 1.668040 | 0.282882 | 514785 |
| scale_scalar_boundary_unroll | 8 | 1 | 0.862339 | 0.547185 | 514785 |
| scale_scalar_boundary_unroll | 16 | 1 | 0.443444 | 1.064078 | 514785 |
| scale_vpu_fp_p8_sharedw | 1 | 1 | 3.752710 | 0.125738 | 514073 |
| scale_vpu_fp_p8_sharedw | 2 | 1 | 2.137790 | 0.220723 | 514073 |
| scale_vpu_fp_p8_sharedw | 4 | 1 | 1.096050 | 0.430509 | 514073 |
| scale_vpu_fp_p8_sharedw | 8 | 1 | 0.565877 | 0.833855 | 514073 |
| scale_vpu_fp_p8_sharedw | 16 | 1 | 0.296077 | 1.593704 | 514073 |

## Conclusions

- Best same-output int8 scalar result: `adv20_06_scalar_boundary_unroll`,
  `0.443350 s`, `1.064304 GOPS`.
- Best DnCNN-shaped VPU result: `adv20_18_vpu_fp_p8_sharedw`,
  `0.296009 s`, `1.594070 GOPS`.
- VPU FP32 is about `1.50x` faster than the best same-output scalar int8 run
  in this 16-hart sweep.
- Private-scratch fused scalar was correct, but slower.  The extra halo compute
  and larger private footprint cost more than the removed barriers/cacheops for
  this 64x64 case.
- Shared packed weights helped the VPU path.  Private-scratch fused VPU was
  correct, but slower for the same reason as scalar private fusion.
- Prefetch did not help.  The best scalar improvement came from combining the
  boundary-only cache-maintenance idea with the hidden-loop unroll.

## Artifacts

- `dncnn3_bench_argbuf.c`: scalar source with private scratch, prefetch, and
  output-last-only variants.
- `dncnn3_vpu_fp_argbuf.c`: VPU FP32 DnCNN-shaped source using
  `flq2`, `fsq2`, and `fmadd.ps`.
- `build_dncnn_advanced20.sh`: builds the 20-plan matrix.
- `run_dncnn_advanced20.sh`: runs the 20-plan matrix on `esperanto-soc6`.
- `advanced20_results.tsv`: parsed 20-plan results.
- `advanced20_scaling.tsv`: parsed scaling sweep for scalar winner and VPU winner.
- `artifact_manifest_advanced20_local.tsv`: local source/ELF/result hashes.
- `artifact_manifest_advanced20_remote.tsv`: remote ELF/log/dump/result hashes.
