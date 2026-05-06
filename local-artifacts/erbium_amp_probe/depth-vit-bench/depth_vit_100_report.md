# Depth Anything V2-Shaped VPU 100-Variant Refinement

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/depth-vit-bench`
Local root: `local-artifacts/erbium_amp_probe/depth-vit-bench`

## Benchmark Shape

This benchmark is not a full Depth Anything V2 model runner.  It isolates the
ViT-style compute path:

- 192 tokens
- 64-wide embedding
- token dot-product score matrix
- attention-value matmul
- 64 -> 128 -> 64 MLP projection
- 32 passes per launch

The total measured work per run is `503316480` operations.

## First 10 Distinct Paths

The first filter tested:

- scalar-output-channel VPU baseline
- OC2 two-output-channel VPU path
- static prefetch
- per-stage B-matrix prefetch
- A-row prefetch
- skipped read evict
- output-last-only cache maintenance
- skipped read evict plus output-last-only
- unroll
- `-Ofast`

Best of the 10-path filter:

- `d10_03_oc2_prefb`
- `0.217563 s`
- `2.313429 GOPS`

## 100-Variant Sweep

The 100-run matrix crossed 10 macro choices with 10 compiler/code-layout
profiles, all fixed to the same 192-token transformer workload.

Macro choices:

- base OC2
- static prefetch
- B-matrix prefetch
- A-row prefetch
- A-row plus B-matrix prefetch
- B-matrix plus static prefetch
- skip read evict
- B-matrix prefetch plus skip read evict
- output-last-only cache maintenance
- B-matrix prefetch plus output-last-only

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
`1572864`.

## Top Results

| Rank | Variant | Wait s | GOPS |
|---:|---|---:|---:|
| 1 | `d100_o02_o3_unroll_m07_prefb_skipread` | 0.214204 | 2.349706 |
| 2 | `d100_o02_o3_unroll_m04_prefa_prefb` | 0.214367 | 2.347920 |
| 3 | `d100_o02_o3_unroll_m02_prefb` | 0.214445 | 2.347066 |
| 4 | `d100_o02_o3_unroll_m05_prefb_static` | 0.214553 | 2.345884 |
| 5 | `d100_o04_o3_align32_m02_prefb` | 0.216435 | 2.325486 |

The current best Depth/ViT-shaped path is:

- `-O3 -funroll-loops`
- `DEPTH_VPU_OC2`
- `DEPTH_PREFETCH_B`
- `DEPTH_SKIP_READ_EVICT`

## Takeaways

- OC2 helps the matmul path substantially.
- B-matrix prefetch is consistently useful for this transformer shape.
- `-O3 -funroll-loops` beats `-Ofast` here.
- Output-last-only cache maintenance is slower for this workload.

## Artifacts

- `depth_vit_vpu_argbuf.c`
- `build_depth_10.sh`
- `run_depth_10.sh`
- `depth_10_results.tsv`
- `build_depth_100.sh`
- `run_depth_100.sh`
- `parse_depth.py`
- `depth_100_results.tsv`
- `artifact_manifest_depth_100_remote.tsv`
- `artifact_manifest_depth_100_local.tsv`
