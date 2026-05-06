# Erbium Model-Shape Sweep Summary

Date: 2026-05-02
Board host: `esperanto-soc6`
Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe`
Local root: `local-artifacts/erbium_amp_probe`

## Scope

Each model class used the same silicon flow:

1. Build standalone U-mode Erbium ELFs locally.
2. Stage under the allowed `esperanto-soc6` artifact tree.
3. Run through `erbium_soc1sim_argbuf` on shire 0.
4. Dump the 16 MiB result buffer.
5. Parse the summary marker from the dump.

All runs used 16 harts and completed with `active_mask=0xffff`.

These are kernel/model-shape benchmarks, not full framework-level imported
models, except that DnCNN started from the existing DnCNN-shaped benchmark
harness.  They are meant to identify the best Erbium execution style for each
model family before wiring in real model weights/importers.

## Best Silicon Results

| Model class | Best variant | Wait s | GOPS | Notes |
|---|---|---:|---:|---|
| DnCNN3 | `v3100_o08_ofast_m09_accfinal_prefw` | 0.195860 | 2.409166 | 64x64x16, 3x3 conv stack, FP32 VPU path |
| YOLO | `y100_o08_ofast_m03_skipread` | 0.238041 | 2.257574 | 80x80, 3x3 + 1x1 detector-block shape |
| Depth Anything V2 | `d100_o02_o3_unroll_m07_prefb_skipread` | 0.214204 | 2.349706 | 192-token ViT/attention/MLP shape |
| Foundation Stereo | `s100_o02_o3_unroll_m09_prefl_lastout` | 0.222081 | 2.266364 | 80x48x32 feature cost-volume correlation |
| Whisper Tiny En | `w100_o02_o3_unroll_m04_prefa_prefb` | 0.229464 | 2.339674 | 256-token transformer/attention/MLP shape |

## Patterns

- OC2/two-output style is consistently important for convolution, matmul, and
  correlation shapes.
- `-O3 -funroll-loops` is best for transformer and stereo shapes.
- `-Ofast` is best for the DnCNN and YOLO convolution-heavy shapes.
- B-matrix prefetch helps transformer shapes.
- Right-feature prefetch hurts the stereo correlation shape; left-row prefetch
  plus output-last-only is best there.
- Output-last-only cache maintenance is not universally good: it helps stereo,
  but hurts YOLO and transformer shapes.

## Artifact Reports

- `dncnn3-pmc/dncnn_v3_100_report.md`
- `yolo-bench/yolo_vpu_100_report.md`
- `depth-vit-bench/depth_vit_100_report.md`
- `stereo-bench/stereo_corr_100_report.md`
- `whisper-bench/whisper_transformer_100_report.md`
