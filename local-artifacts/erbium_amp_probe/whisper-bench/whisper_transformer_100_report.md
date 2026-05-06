# Whisper Tiny En-Shaped VPU 100-Variant Refinement

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-bench`
Local root: `local-artifacts/erbium_amp_probe/whisper-bench`

## Benchmark Shape

This benchmark is not a full Whisper Tiny En model runner.  It isolates the
audio-transformer compute path:

- 256 tokens
- 64-wide embedding
- token dot-product score matrix
- attention-value matmul
- 64 -> 256 -> 64 MLP projection
- 16 passes per launch

The total measured work per run is `536870912` operations.

## First 10 Distinct Paths

The first filter tested:

- scalar-output-channel VPU baseline
- OC2 two-output-channel VPU path
- static prefetch
- per-stage B-matrix prefetch
- A-row prefetch
- skipped read evict
- output-last-only cache maintenance
- B-matrix prefetch plus skipped read evict
- unroll
- `-Ofast`

Best of the 10-path filter:

- `w10_07_oc2_prefb_skipread`
- `0.234311 s`
- `2.291275 GOPS`

## 100-Variant Sweep

The 100-run matrix crossed 10 macro choices with 10 compiler/code-layout
profiles, all fixed to the same 256-token transformer workload.

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
`2097152`.

## Top Results

| Rank | Variant | Wait s | GOPS |
|---:|---|---:|---:|
| 1 | `w100_o02_o3_unroll_m04_prefa_prefb` | 0.229464 | 2.339674 |
| 2 | `w100_o02_o3_unroll_m07_prefb_skipread` | 0.229825 | 2.335999 |
| 3 | `w100_o02_o3_unroll_m02_prefb` | 0.230089 | 2.333318 |
| 4 | `w100_o02_o3_unroll_m05_prefb_static` | 0.230287 | 2.331312 |
| 5 | `w100_o02_o3_unroll_m09_prefb_lastout` | 0.230562 | 2.328532 |

The current best Whisper-shaped path is:

- `-O3 -funroll-loops`
- `DEPTH_VPU_OC2`
- `DEPTH_PREFETCH_A_ROWS`
- `DEPTH_PREFETCH_B`

## Takeaways

- This behaves like the Depth/ViT path: OC2 plus B-matrix prefetch is the core
  win.
- A-row prefetch helps more here than it did in the 192-token Depth shape.
- `-O3 -funroll-loops` beats `-Ofast`.

## Artifacts

- `whisper_transformer_vpu_argbuf.c`
- `build_whisper_10.sh`
- `run_whisper_10.sh`
- `whisper_10_results.tsv`
- `build_whisper_100.sh`
- `run_whisper_100.sh`
- `parse_whisper.py`
- `whisper_100_results.tsv`
- `artifact_manifest_whisper_100_remote.tsv`
- `artifact_manifest_whisper_100_local.tsv`
