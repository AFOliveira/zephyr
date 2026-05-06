# Erbium/ET-SoC1 Model Optimization Knowledge Base

Date: 2026-05-02

Board validation host: `esperanto-soc6`

Remote artifact root:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe`

Local artifact root:
`local-artifacts/erbium_amp_probe`

## Current Hardware Interpretation

The docs in `/home/afonso/docsET` describe the compute surface as ET-Minion
cores with a VPU/tensor extension, not as a separate opaque NPU block.  The
current working strategy is therefore:

- Use the VPU packed-single FP32 path for reliable high-throughput kernels.
- Treat "NPU" work as VPU/tensor work unless a separate programmable NPU
  interface is found.
- Keep explicit cache maintenance around shared buffers because the L1D cache
  is documented as non-coherent.
- Avoid `TensorReduce` on ET-SoC1 for now because the local docs mention a
  `TensorReduce`/VPU deadlock erratum.
- Be careful with tensor-only paths across Erbium versus ET-SoC1:
  `/home/afonso/docsET/doc/minion.md` says Erbium does not support
  `TensorLoadL2Scp`, while ET-SoC1 docs describe cooperative tensor load as a
  feature.

Useful doc anchors:

- `/home/afonso/docsET/doc/index.md`: non-coherent L1D, VPU lane count, packed
  single/integer/tensor extension summary.
- `/home/afonso/docsET/doc/minion.md`: Erbium ISA deltas and tensor caveats.
- `/home/afonso/docsET/reference_manual (2).txt`: ET-Minion/VPU hardware
  summary and ET-SoC1 errata notes.

## Cross-Check Status

Only DnCNN3 currently has a real Luxonis ONNX to Erbium package checker.

| Model | Current status | Cross-check level |
|---|---|---|
| DnCNN3 | Real Luxonis ONNX weights and topology packaged into Erbium C/ELF for small tiles | ONNXRuntime checked; weights bit-exact; silicon tiles checked |
| Depth Anything V2 | Real ONNX inventoried; Erbium work is a Depth/ViT-shaped hot-path benchmark | Output checksum only for benchmark shape |
| Whisper Tiny En | Real encoder/decoder ONNX inventoried; Erbium work is a transformer-shaped hot-path benchmark | Output checksum only for benchmark shape |
| Foundation Stereo | Real ONNX inventoried; Erbium work is a stereo cost-volume hot-path benchmark | Output checksum only for benchmark shape |
| YOLO | YOLO-shaped detector hot-path benchmark; no Luxonis YOLO ONNX artifact was part of this local set | Output checksum only for benchmark shape |

DnCNN package checker:

- Script: `local-artifacts/erbium_amp_probe/luxonis-real-dncnn/verify_dncnn3_package.py`
- Report: `local-artifacts/erbium_amp_probe/luxonis-real-dncnn/dncnn3_package_verify_report.json`
- Latest result: pass.

DnCNN checker details:

- Packed Erbium weights match ONNX initializers bit-exactly.
- Host references match ONNXRuntime for `1x1`, `4x4`, `16x16`, `64x64`, and
  full `240x320`.
- Silicon `1x1` is bit-exact.
- Silicon `4x4` is allclose with max absolute error `4.73e-7`.

For VPU paths, bit-exact output should not be required unless the operation
order is intentionally preserved.  VPU/vectorized accumulation changes FP
operation order, so the correctness gate should be strict `allclose` plus
checksums/hashes, with bit-exact reserved for scalar or same-order paths.

## ONNX Inventory

Raw inventory:

- `local-artifacts/erbium_amp_probe/optimization-kb/onnx_inventory.py`
- `local-artifacts/erbium_amp_probe/optimization-kb/onnx_inventory.json`

Summary:

| Model | Nodes | Initializer MiB | Dominant ops |
|---|---:|---:|---|
| DnCNN3 240x320 | 40 | 2.5 | Conv, Relu, Sub |
| DnCNN3 480x640 | 40 | 2.5 | Conv, Relu, Sub |
| Depth Anything V2 252x336 | 1108 | 94.3 | Constant, Add, Gather, MatMul, Mul, LayerNormalization, Softmax |
| Depth Anything V2 420x560 | 1108 | 94.3 | Constant, Add, Gather, MatMul, Mul, LayerNormalization, Softmax |
| Whisper Tiny En encoder | 257 | 35.8 | MatMul, Add, Transpose, LayerNormalization, Softmax, Conv |
| Whisper Tiny En decoder | 476 | 184.1 | Slice, MatMul, Add, Softmax, LayerNormalization |
| Foundation Stereo 640x416 | 37817 | 712.0 | Constant, Mul, Expand, Unsqueeze, Where, Concat |
| Foundation Stereo 1280x800 | 58447 | 712.0 | Constant, Shape, Unsqueeze, Mul, Cast, Gather, Add |

## Best Silicon Results

Raw per-model 100-run reports:

- DnCNN: `local-artifacts/erbium_amp_probe/dncnn3-pmc/dncnn_v3_100_report.md`
- Depth/ViT: `local-artifacts/erbium_amp_probe/depth-vit-bench/depth_vit_100_report.md`
- Whisper: `local-artifacts/erbium_amp_probe/whisper-bench/whisper_transformer_100_report.md`
- Stereo: `local-artifacts/erbium_amp_probe/stereo-bench/stereo_corr_100_report.md`
- YOLO: `local-artifacts/erbium_amp_probe/yolo-bench/yolo_vpu_100_report.md`

| Model path | Best variant | Wait s | GOPS | What made it better |
|---|---|---:|---:|---|
| DnCNN-shaped | `v3100_o08_ofast_m09_accfinal_prefw` | 0.195860 | 2.409166 | `-Ofast`, OC2 hidden layers, shared packed weights, fused final 3x3 accumulator, weight-pack prefetch |
| Depth/ViT-shaped | `d100_o02_o3_unroll_m07_prefb_skipread` | 0.214204 | 2.349706 | `-O3 -funroll-loops`, OC2 matmul, B-matrix prefetch, skipped read evict |
| Whisper-shaped | `w100_o02_o3_unroll_m04_prefa_prefb` | 0.229464 | 2.339674 | `-O3 -funroll-loops`, OC2 matmul, A-row prefetch, B-matrix prefetch |
| Foundation-Stereo-shaped | `s100_o02_o3_unroll_m09_prefl_lastout` | 0.222081 | 2.266364 | `-O3 -funroll-loops`, two-disparity VPU path, left-row prefetch, output-last-only cache maintenance |
| YOLO-shaped | `y100_o08_ofast_m03_skipread` | 0.238041 | 2.257574 | `-Ofast`, OC2 convolution path, skipped activation read evict |

## Best-Variant PMC Suite

Raw PMC suite artifacts:

- Runner: `local-artifacts/erbium_amp_probe/optimization-kb/run_best_pmc_suite.sh`
- Parser: `local-artifacts/erbium_amp_probe/optimization-kb/parse_best_pmc_suite.py`
- Raw parsed TSV: `local-artifacts/erbium_amp_probe/optimization-kb/best_pmc_results.tsv`
- Current-best runner: `local-artifacts/erbium_amp_probe/optimization-kb/run_current_best_pmc_suite.sh`
- Current-best parser: `local-artifacts/erbium_amp_probe/optimization-kb/parse_current_best_pmc_suite.py`
- Current-best raw parsed TSV: `local-artifacts/erbium_amp_probe/optimization-kb/current_best_pmc_results.tsv`

The original 100-sweep-winner suite ran on `esperanto-soc6` at stamp
`20260502-065820`.

Use these fields as the reliable comparison set:

- `wait_s`
- `gops`
- `hpm_cycles_max`
- `cycles_per_op_max`
- SC syscall PMCs: `sc_reads_per_kop`, `sc_writes_per_kop`
- MS syscall PMCs: `ms_reads_per_kop`, `ms_writes_per_kop`

Do not use the raw direct HPM instruction/L2-miss deltas from this separate
snapshot flow as comparative truth when they wrap to huge values.  Preserve them
in raw TSV, but treat SC/MS syscall PMCs and HPM cycle max as the actionable
signals.

Current-best suite, after the second focused 20-variant pass, ran at stamp
`20260502-072025`.

| Model | GOPS | Cycles/op max | SC reads/kop | SC writes/kop | MS reads/kop | MS writes/kop |
|---|---:|---:|---:|---:|---:|---:|
| DnCNN | 2.414553 | 1.000548556 | 53.212151 | 6.205281 | 0.559016 | 0.749796 |
| Depth/ViT | 2.353409 | 0.960076855 | 52.871503 | 2.888181 | 0.454336 | 0.668126 |
| Whisper | 2.347028 | 0.938125381 | 52.303055 | 2.575299 | 0.426017 | 0.626421 |
| Stereo | 2.311591 | 0.968622889 | 46.788444 | 6.978518 | 0.454307 | 0.668077 |
| YOLO | 2.304489 | 0.925763877 | 57.372100 | 6.303984 | 0.425545 | 0.625767 |

## Focused 20-Variant Follow-Up

Raw artifacts:

- Builder: `local-artifacts/erbium_amp_probe/optimization-kb/build_focused20.sh`
- Runner: `local-artifacts/erbium_amp_probe/optimization-kb/run_focused20.sh`
- Parser: `local-artifacts/erbium_amp_probe/optimization-kb/parse_focused20.py`
- Parsed TSV: `local-artifacts/erbium_amp_probe/optimization-kb/focused20_results.tsv`
- Remote dumps/logs: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/optimization-kb/focused20`

The focused pass tested 20 combinations that were not all represented in the
original 10x10 sweeps: combined compiler flags, combined prefetch/cache policies,
and current-best macro families.

New best results:

| Model | Previous best GOPS | Focused best GOPS | New best variant | Exact improvement |
|---|---:|---:|---|---|
| DnCNN | 2.409166 | 2.410717 | `f20_dncnn_ofast_noipa_accfinal_prefw` | `-Ofast` plus `-fno-ipa-cp -fno-ipa-sra` gave a tiny win over plain `-Ofast` |
| Depth/ViT | 2.349706 | 2.345206 | unchanged | focused variants did not beat the 100-sweep winner |
| Whisper | 2.339674 | 2.348434 | `f20_whisper_o3_unroll_prefa_prefb_skipread` | adding skipped read evict to A-row plus B-matrix prefetch helped |
| Stereo | 2.266364 | 2.316250 | `f20_stereo_ofast_unroll_prefl_lastout` | combining `-Ofast`, unroll, left-row prefetch, and output-last-only cache maintenance helped substantially |
| YOLO | 2.257574 | 2.303708 | `f20_yolo_ofast_unroll_skipread` | combining `-Ofast`, unroll, and skipped activation read evict helped substantially |

## Second Focused 20-Variant Follow-Up

Raw artifacts:

- Builder: `local-artifacts/erbium_amp_probe/optimization-kb/build_focused20b.sh`
- Runner: `local-artifacts/erbium_amp_probe/optimization-kb/run_focused20b.sh`
- Parsed TSV: `local-artifacts/erbium_amp_probe/optimization-kb/focused20b_results.tsv`
- Remote dumps/logs: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/optimization-kb/focused20b`

New best results after this pass:

| Model | Previous focused best GOPS | Second focused best GOPS | Current best variant | Exact improvement |
|---|---:|---:|---|---|
| DnCNN | 2.410717 | 2.414022 | `f20b_dncnn_ofast_rename` | `-Ofast -frename-registers` beat no-IPA and alignment variants |
| Depth/ViT | 2.349706 | 2.352518 | `f20b_depth_o3_unroll_noipa_prefb_skipread` | adding no-IPA to the original `-O3 -funroll-loops` winner helped |
| Whisper | 2.348434 | 2.342532 | unchanged | second focused pass did not beat `f20_whisper_o3_unroll_prefa_prefb_skipread` |
| Stereo | 2.316250 | 2.299992 | unchanged | second focused pass did not beat `f20_stereo_ofast_unroll_prefl_lastout` |
| YOLO | 2.303708 | 2.305952 | `f20b_yolo_ofast_unroll_align32_skipread` | adding 32-byte function/loop alignment to `-Ofast -funroll-loops` helped slightly |

## Approach Patterns That Worked

### OC2

Computing two output channels, two disparities, or otherwise two independent
accumulators per loaded input vector is the most repeatable VPU optimization.
It reduces activation or feature reloads per operation.

Worked in:

- DnCNN hidden layers
- YOLO-shaped convolution blocks
- Depth/Whisper transformer matmuls
- Stereo two-disparity correlation

### Weight or B-Matrix Prefetch

Prefetch helps when the next reused operand is stable and accessed repeatedly.

Worked in:

- DnCNN packed-weight prefetch
- Depth/Whisper B-matrix prefetch

Did not consistently work in:

- YOLO weight prefetch, where `-Ofast` plus skip-read dominated
- Stereo right-feature prefetch, where left-row prefetch was better

### Fused Accumulation

Reducing spill/sum frequency matters.  DnCNN improved when the final layer used
a fused 3x3 accumulator path.

Worked in:

- DnCNN `DNCNN_VPU_ACCUM_3X3`

Risk:

- More inline asm clobbers are required.  `f0`..`f4` clobbers were necessary to
  avoid compiler FP register corruption.

### Compiler Profile

The best compiler setting is model-shape dependent.

- DnCNN and YOLO liked `-Ofast`.
- Depth, Whisper, and Stereo liked `-O3 -funroll-loops`.
- 32/64-byte alignment variants were sometimes close but not winners.

### Cache Maintenance

Cache maintenance must be workload-specific.

- Skipping activation/read evict helped YOLO.
- Output-last-only helped Stereo.
- Output-last-only slowed Depth/Whisper.
- Boundary-only activation maintenance was not a DnCNN winner.

## Current Recommended Next Steps

1. Keep DnCNN as the first real full-model path because the ONNX graph is small,
   regular, and already has a working checker.
2. Convert the manual DnCNN package into generated code from ONNX initializers
   and graph metadata.
3. Add ONNXRuntime cross-checkers for each future generated package before
   silicon runs.
4. For Depth/Whisper, prioritize a general matmul/softmax/layernorm runtime
   over hand-porting full graphs.
5. For Foundation Stereo, start with cost-volume/correlation and shape handling
   extraction; the full graph is too large for manual direct porting.
6. For YOLO, obtain a specific ONNX source model before claiming exact model
   parity; the current result is a detector-shaped kernel benchmark only.
7. Investigate a tensor path only after a small smoke-test matrix proves no
   tensor errata path is hit on the exact silicon/firmware combination.

## External Optimization Leads

Sources checked after the local search space stopped producing obvious next
micro-variants:

- Depth Anything V2 official repo/site:
  `https://github.com/DepthAnything/Depth-Anything-V2` and
  `https://depth-anything-v2.github.io/`
- Qualcomm Depth Anything V2 model card:
  `https://huggingface.co/qualcomm/Depth-Anything-V2`
- Qualcomm DnCNN model card:
  `https://huggingface.co/qualcomm/DnCNN`
- whisper.cpp:
  `https://github.com/ggml-org/whisper.cpp`
- NVIDIA displacement-invariant stereo cost computation:
  `https://research.nvidia.com/publication/2022-01_displacement-invariant-cost-computation-efficient-stereo-matching`
- SCV-Stereo sparse cost volume:
  `https://arxiv.org/abs/2107.08187`
- Ultralytics YOLO optimization overview:
  `https://www.ultralytics.com/blog/how-to-make-yolo-models-fast-on-your-favouruite-chip`
- OpenVINO YOLOv8 optimization tutorial:
  `https://docs.openvino.ai/2023.3/notebooks/230-yolov8-object-detection-with-output.html`

Actionable leads for this hardware:

1. Quantization is the next high-value path for DnCNN, YOLO, and Whisper.
   Whisper.cpp and the Qualcomm/Luxonis-style model cards point toward
   ahead-of-time low-bit or INT8 weight formats.  On Erbium this should be a
   packed-integer VPU experiment, not just scalar int8.
2. YOLO should use a real source model and then test layer fusion before
   micro-optimizing: conv/bn/activation fusion, then INT8 calibration, then OC2
   VPU kernels.
3. Depth/Whisper should share a generated matmul kernel library.  The ONNX
   inventories show both are dominated by MatMul/Add/Transpose/LayerNorm/Softmax,
   so optimizing one general transformer block buys coverage for both.
4. Foundation Stereo should not start from the full 4D volume.  External stereo
   work suggests sparse cost volumes or displacement-invariant cost computation
   to avoid exploding memory bandwidth.  This matches our PMCs: stereo has low
   SC reads/kop but high writes/kop, so reducing volume materialization is more
   promising than another compiler sweep.
5. NPU framing should be translated to Erbium VPU/tensor primitives.  The next
   hardware experiment should be a packed-integer VPU dot-product path and, only
   after smoke tests, a tensor path for int8/fp16 matrix blocks.
