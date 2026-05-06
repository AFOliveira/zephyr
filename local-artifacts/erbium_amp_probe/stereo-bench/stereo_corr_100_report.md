# Foundation Stereo-Shaped VPU 100-Variant Refinement

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/stereo-bench`
Local root: `local-artifacts/erbium_amp_probe/stereo-bench`

## Benchmark Shape

This benchmark is not a full Foundation Stereo model runner.  It isolates a
stereo cost-volume kernel:

- left/right feature maps: 80x48
- 32 feature channels
- 32 disparities
- 64 passes per launch

The total measured work per run is `503316480` operations.

## First 10 Distinct Paths

The first filter tested:

- scalar-disparity VPU baseline
- two-disparity VPU path
- right-feature prefetch
- left-row prefetch
- skipped read evict
- output-last-only cache maintenance
- skipped read evict plus output-last-only
- right prefetch plus skipped read evict
- unroll
- `-Ofast`

Best of the 10-path filter:

- `s10_09_disp2_ofast`
- `0.225329 s`
- `2.233696 GOPS`

## 100-Variant Sweep

The 100-run matrix crossed 10 macro choices with 10 compiler/code-layout
profiles, all fixed to the same 80x48x32x32 correlation workload.

Macro choices:

- base two-disparity VPU
- right-feature prefetch
- left-row prefetch
- skip read evict
- output-last-only cache maintenance
- skip read evict plus output-last-only
- right prefetch plus skip read evict
- left-row prefetch plus skip read evict
- right prefetch plus output-last-only
- left-row prefetch plus output-last-only

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

All 100 variants built and ran on silicon.  All 100 were valid with output sum
`11732640`.

## Top Results

| Rank | Variant | Wait s | GOPS |
|---:|---|---:|---:|
| 1 | `s100_o02_o3_unroll_m09_prefl_lastout` | 0.222081 | 2.266364 |
| 2 | `s100_o02_o3_unroll_m05_skipread_lastout` | 0.223946 | 2.247490 |
| 3 | `s100_o02_o3_unroll_m07_prefl_skipread` | 0.224449 | 2.242454 |
| 4 | `s100_o08_ofast_m04_lastout` | 0.224568 | 2.241265 |
| 5 | `s100_o08_ofast_m05_skipread_lastout` | 0.224778 | 2.239171 |

The current best Foundation-Stereo-shaped path is:

- `-O3 -funroll-loops`
- `STEREO_DISP2`
- `STEREO_PREFETCH_LEFT_ROWS`
- `STEREO_EVICT_OUTPUT_LAST_ONLY`

## Takeaways

- Computing two disparities per VPU pass is the important structural win.
- Left-row prefetch plus output-last-only wins in the 100-way sweep.
- Right-feature prefetch is consistently poor for this access pattern.
- `-O3 -funroll-loops` beats the best `-Ofast` variant.

## Artifacts

- `stereo_corr_vpu_argbuf.c`
- `build_stereo_10.sh`
- `run_stereo_10.sh`
- `stereo_10_results.tsv`
- `build_stereo_100.sh`
- `run_stereo_100.sh`
- `parse_stereo.py`
- `stereo_100_results.tsv`
- `artifact_manifest_stereo_100_remote.tsv`
- `artifact_manifest_stereo_100_local.tsv`
