# DnCNN V3 100-Variant Refinement

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`
Local root: `local-artifacts/erbium_amp_probe/dncnn3-pmc`

## Starting Point

The previous best DnCNN-shaped VPU FP32 kernel was:

- `vpuv2_06_oc2_sharedw`
- `0.201169 s`
- `2.345586 GOPS`
- checksum `514073`

## First 10 Distinct Paths

I first ran ten distinct OC2/shared-weight variants:

- base OC2
- boundary-only activation cache maintenance
- boundary-only plus skipped final barrier
- skipped final barrier
- skipped all-hart weight-pack evict
- boundary-only plus skipped all-hart weight-pack evict
- activation read prefetch
- boundary-only plus activation read prefetch
- weight-pack prefetch
- `-Ofast`

Best of this 10-path filter:

- `v3x_09_oc2_prefetch_wpack`
- `0.202034 s`
- `2.335544 GOPS`

That did not beat the v2 best, but it showed that the useful search area was
still OC2/shared weights with weight-pack handling and compiler/code-layout
variation.  Boundary-only float activation cache maintenance and read-window
prefetch were both slower.

## 100-Variant Sweep

The 100-run matrix crossed 10 macro choices with 10 compiler/code-layout
profiles.

Macro choices:

- base
- weight-pack prefetch
- skip all-hart weight-pack evict
- skip all-hart weight-pack evict plus weight-pack prefetch
- skip final barrier
- skip final barrier plus weight-pack prefetch
- skip final barrier plus skip all-hart weight-pack evict
- output-last-only plus weight-pack prefetch
- OC2 hidden layers plus fused final layer
- OC2 hidden layers plus fused final layer plus weight-pack prefetch

Compiler/code-layout profiles:

- `-O3`
- `-O3 -fno-schedule-insns -fno-schedule-insns2`
- `-O3 -funroll-loops`
- `-O3 -fno-unroll-loops`
- `-O3 -falign-functions=32 -falign-loops=32`
- `-O3 -falign-functions=64 -falign-loops=64`
- `-O3 -fno-ipa-cp -fno-ipa-sra`
- `-O3 -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize`
- `-Ofast`
- `-O2`

All 100 variants built and ran on silicon.  All 100 were valid with checksum
`514073`.

## Top Results

| Rank | Variant | Wait s | GOPS |
|---:|---|---:|---:|
| 1 | `v3100_o08_ofast_m09_accfinal_prefw` | 0.195860 | 2.409166 |
| 2 | `v3100_o08_ofast_m07_lastonly_prefw` | 0.197410 | 2.390250 |
| 3 | `v3100_o08_ofast_m08_accfinal` | 0.198121 | 2.381672 |
| 4 | `v3100_o08_ofast_m01_prefw` | 0.200676 | 2.351348 |
| 5 | `v3100_o08_ofast_m04_skipfinal` | 0.201400 | 2.342896 |

The clear winner is:

- `-Ofast`
- `DNCNN_VPU_OC2`
- `DNCNN_VPU_SHARED_WPACK`
- `DNCNN_VPU_ACCUM_3X3`
- `DNCNN_VPU_PREFETCH_WPACK`

In this winning build, OC2 handles the hidden layers and the fused 3x3
accumulator handles the final layer.  Weight-pack prefetch also helps.

## Improvement

- Previous exact int8 scalar best: `1.064304 GOPS`
- Previous v1 VPU best: `1.594070 GOPS`
- Previous v2 OC2 best: `2.345586 GOPS`
- New v3 100-sweep best: `2.409166 GOPS`

The new best is about:

- `2.26x` over the best exact int8 scalar path
- `1.51x` over the first VPU path
- `1.03x` over the previous OC2 VPU best

This remains the FP32 VPU benchmark path, not an exact int8 numeric replacement
for the scalar DnCNN kernel.

## Artifacts

- `build_dncnn_v3_explore.sh`
- `run_dncnn_v3_explore.sh`
- `parse_dncnn_v3.py`
- `v3x_results.tsv`
- `build_dncnn_v3_100.sh`
- `run_dncnn_v3_100.sh`
- `parse_dncnn_v3_100.py`
- `v3_100_results.tsv`
- `artifact_manifest_v3_100_remote.tsv`
