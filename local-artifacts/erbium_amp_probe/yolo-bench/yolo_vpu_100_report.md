# YOLO-Shaped VPU 100-Variant Refinement

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/yolo-bench`
Local root: `local-artifacts/erbium_amp_probe/yolo-bench`

## Benchmark Shape

This benchmark is not a full YOLO parser.  It isolates a YOLO-like detector hot
path:

- 80x80 feature map
- 16 channels
- 4 blocks of 3x3 convolution plus 1x1 pointwise convolution
- 16-channel 1x1 detection head
- 4 passes per launch

The total measured work per comparable run is `537395200` operations.

## First 10 Distinct Paths

The first filter tested:

- scalar-output-channel VPU baseline
- OC2 two-output-channel VPU path
- OC2 plus weight prefetch
- OC2 plus boundary-only activation cache maintenance
- OC2 plus skipped activation read evict
- OC2 plus output-last-only cache maintenance
- OC2 plus skipped final barrier
- 3-block and 5-block shape checks
- OC2 plus `-Ofast`

Best of the 10-path filter:

- `y10_09_oc2_ofast`
- `0.241274 s`
- `2.227323 GOPS`

OC2 was the main improvement.  Weight prefetch did not help once isolated in
the 100-run matrix.

## 100-Variant Sweep

The 100-run matrix crossed 10 macro choices with 10 compiler/code-layout
profiles, all fixed to the comparable 4-block YOLO-shaped workload.

Macro choices:

- base OC2
- weight prefetch
- boundary-only activation cache maintenance
- skip activation read evict
- skip final barrier
- output-last-only cache maintenance
- boundary-only plus skip final barrier
- boundary-only plus weight prefetch
- skip activation read evict plus skip final barrier
- output-last-only plus skip final barrier

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
`13107200`.

## Top Results

| Rank | Variant | Wait s | GOPS |
|---:|---|---:|---:|
| 1 | `y100_o08_ofast_m03_skipread` | 0.238041 | 2.257574 |
| 2 | `y100_o08_ofast_m08_skipread_skipfinal` | 0.238575 | 2.252521 |
| 3 | `y100_o08_ofast_m00_base` | 0.240033 | 2.238839 |
| 4 | `y100_o08_ofast_m01_prefw` | 0.241284 | 2.227231 |
| 5 | `y100_o08_ofast_m07_boundary_prefw` | 0.241367 | 2.226465 |

The current best YOLO-shaped path is:

- `-Ofast`
- `YOLO_VPU_OC2`
- `YOLO_SKIP_READ_EVICT`

## Takeaways

- For this shape, avoiding the activation read evict wins slightly.
- Skipping the final barrier does not improve the top single-shire result.
- Output-last-only cache maintenance is consistently slower.
- `-Ofast` matters more than the cache-maintenance micro-variants.

## Artifacts

- `yolo_vpu_argbuf.c`
- `build_yolo_10.sh`
- `run_yolo_10.sh`
- `yolo_10_results.tsv`
- `build_yolo_100.sh`
- `run_yolo_100.sh`
- `parse_yolo.py`
- `yolo_100_results.tsv`
- `artifact_manifest_yolo_100_remote.tsv`
- `artifact_manifest_yolo_100_local.tsv`
