# DnCNN VPU v2 Optimization Report

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`
Local root: `local-artifacts/erbium_amp_probe/dncnn3-pmc`

## Goal

Continue from the 20-plan DnCNN sweep by reducing VPU overhead in the FP32
DnCNN-shaped path.  The v1 VPU kernel used packed-single VPU instructions, but
spilled and scalar-summed once per 16-channel dot.  That meant nine spills and
horizontal sums per 3x3 output point.

## Techniques Tried

1. `DNCNN_VPU_ACCUM_3X3`
   - Accumulates the full 3x3x16 point in one VPU accumulator.
   - Reduces spill/sum work from nine per output point to one.

2. `DNCNN_VPU_OC2`
   - Computes two hidden-layer output channels at a time.
   - Reuses each loaded input vector for two output-channel weight vectors.

3. `DNCNN_VPU_OC2 + DNCNN_VPU_ACCUM_3X3`
   - Uses OC2 in hidden layers and the fused 3x3 accumulator in the final
     single-output layer.

4. Explicit FP register clobbers in inline asm
   - The VPU asm now declares `f0`/`f1`/`f2` and, for OC2, `f3`/`f4` clobbers.
   - Without those clobbers, the compiler could keep FP temporaries in registers
     the asm overwrote.  A first run exposed this as checksum drift in some
     multi-pass `ACCUM_3X3` variants.

Tensor-reduce was not used for horizontal summation because the local ET docs
include a TensorReduce/VPU deadlock warning (`RTLMIN-6181`).  These kernels
stay on the already validated packed-single surface: `flq2`, `fsq2`,
`fbcx.ps`, and `fmadd.ps`.

## Silicon Results

All variants below ran on silicon and validated with magic, done count, active
mask, and checksum equality.  All final clobber-fixed variants produced the same
FP32 VPU checksum: `514073`.

| Variant | Wait s | GOPS | Harts | Notes |
|---|---:|---:|---:|---|
| vpuv2_00_baseline_sharedw | 0.295969 | 1.594286 | 16 | v1-equivalent baseline |
| vpuv2_01_acc3x3_sharedw | 0.222907 | 2.116843 | 16 | fused 3x3 |
| vpuv2_02_acc3x3_sharedw_lastonly | 0.222903 | 2.116881 | 16 | fused 3x3, final output evict only on last pass |
| vpuv2_03_acc3x3_noshared | 0.230112 | 2.050563 | 16 | fused 3x3, per-hart packed weights |
| vpuv2_04_acc3x3_p1_sharedw | 0.028990 | 2.034577 | 16 | one pass |
| vpuv2_05_acc3x3_private_sharedw | 0.468731 | 1.006674 | 16 | private fused scratch, slower |
| vpuv2_06_oc2_sharedw | 0.201169 | 2.345586 | 16 | best overall |
| vpuv2_07_oc2_sharedw_lastonly | 0.205879 | 2.291925 | 16 | last-only output evict, slower |
| vpuv2_08_oc2_accfinal_sharedw | 0.205417 | 2.297080 | 16 | OC2 plus fused final, slower at 16 harts |
| vpuv2_09_oc2_accfinal_lastonly | 0.203335 | 2.320600 | 16 | OC2 plus fused final/last-only, still slower |

## Scaling

| Variant | 1 hart | 2 harts | 4 harts | 8 harts | 16 harts |
|---|---:|---:|---:|---:|---:|
| acc3x3 sharedw GOPS | 0.226243 | 0.349047 | 0.636692 | 1.191167 | 2.117717 |
| oc2 sharedw GOPS | 0.244402 | 0.380014 | 0.737298 | 1.349129 | 2.343408 |
| oc2 accfinal GOPS | 0.248246 | 0.389702 | 0.727693 | 1.328365 | 2.298904 |

`oc2 accfinal` helps slightly at 1-2 harts but loses at 4, 8, and 16 harts.
The best 16-hart result remains plain `DNCNN_VPU_OC2` with shared packed
weights.

## Comparison To Previous Bests

- Previous best exact int8 scalar: `1.064304 GOPS`.
- Previous best v1 VPU FP32: `1.594070 GOPS`.
- New best v2 VPU FP32: `2.345586 GOPS`.

That is about `2.20x` over the best exact scalar path and `1.47x` over the
previous VPU path.  This is still the FP32 VPU benchmark path, not an exact int8
numeric replacement for the scalar kernel.

## PMC Probe

I also ran the existing before/kernel/after PMC snapshot flow on three kernels:
the v1-equivalent VPU baseline, fused 3x3, and the OC2 winner.

| Variant | GOPS | HPM max cycles/op | SC reads/kop | SC writes/kop | MS reads/kop | MS writes/kop |
|---|---:|---:|---:|---:|---:|---:|
| vpuv2_00_baseline_sharedw | 1.593839 | 1.149237338 | 66.563990 | 4.903365 | 0.563992 | 0.751858 |
| vpuv2_01_acc3x3_sharedw | 2.115847 | 1.038085403 | 69.673971 | 7.903324 | 0.484664 | 0.712713 |
| vpuv2_06_oc2_sharedw | 2.346123 | 1.009591291 | 53.060390 | 6.202944 | 0.484594 | 0.712469 |

The useful signal is that OC2 reduces shire-cache read traffic per kilo-op
substantially versus the baseline, while also lowering max HPM cycles per op.
Some direct HPM instruction/L2-miss deltas wrapped or reset across the separate
snapshot launches, so those fields are preserved in `vpuv2_pmc_results.tsv` but
not treated as valid comparative data here.

## Artifacts

- `dncnn3_vpu_fp_argbuf.c`
- `build_dncnn_vpu_v2.sh`
- `run_dncnn_vpu_v2.sh`
- `parse_dncnn_vpu_v2.py`
- `vpuv2_results.tsv`
- `run_dncnn_vpu_v2_pmc.sh`
- `parse_dncnn_vpu_v2_pmc.py`
- `vpuv2_pmc_results.tsv`
- `artifact_manifest_vpu_v2_local.tsv`
- `artifact_manifest_vpu_v2_remote.tsv`
- `artifact_manifest_vpu_v2_pmc_remote.tsv`
