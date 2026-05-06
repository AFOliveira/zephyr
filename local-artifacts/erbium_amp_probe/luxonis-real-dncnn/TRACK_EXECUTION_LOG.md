# Luxonis DnCNN3 Track Execution Log

Date: 2026-05-02

Scope:

- Track A: tiled FP32 full-resolution DnCNN3.
- Track B: INT8 ONNX quantization/audit groundwork.
- Silicon target is `esperanto-soc6` only. Remote staging path remains
  `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe`.
- No reboot, reset, bus reset, push, commit, tag, or release operation was run.

## Track A Status

Completed locally:

- Extended `dncnn3_luxonis_real_vpu.c` for generic tile image sizes through
  `IMG_W` / `IMG_H`.
- Added generic tile blob linkage through `DNCNN_TILE_BLOBS`:
  `_binary_dncnn_tile_input_bin_start` and `_binary_dncnn_tile_ref_bin_start`.
- Moved the static final output buffer to the dumpable arg-buffer region at
  `base + OUTPUT_OFFSET`, so the tiled runner can extract image crops from
  `dump_after`.
- Added `tools/run_tiled_inference.py`.
- Added fast-failing noninteractive SSH preflight and `--build-only`.

Reference geometry:

- Existing A1 geometry file:
  `tiled-fp32/audit_a1_geometry.json`
- Image: 240x320.
- Tile output stride: 64x64.
- Tiles: 20.
- Halo: 20.
- Halo 20/24/32 were bit-exact against full ORT; halo 16 was not.
- Largest tile is tile 6, 104x104 input, not tile 4.
- Restitching the saved A1 ORT tile crops locally is bit-exact against
  `dncnn3_luxonis_240x320_ort_output_f32.bin` (`max_abs = 0`), so the stitch
  geometry itself is not the risk.

Build-only artifacts:

- A2 largest single tile:
  `tiled-fp32/builds/a2_tile104_f9t_pfw_align128_20260502-225735/a2_tile104_f9t_pfw_align128_t06_104x104.elf`
- A3 full 20-tile ELF set:
  `tiled-fp32/builds/a3_tiled64_halo20_f9t_pfw_align128_20260502-225743/`

Build stack:

- `DNCNN_VPU_FUSED9TAP=1`
- `DNCNN_VPU_PREFETCH_READ_WINDOW=1`
- `ACTIVE_HARTS=16`
- `DNCNN_PASSES=1`
- `-O3 -funroll-loops -falign-functions=128 -falign-loops=128`
- `-fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize`

Silicon status:

- Not completed.
- `esperanto-soc6` SSH is blocked by Tailscale SSH reauthentication.
- The runner now fails before building/staging if SSH auth is unavailable.
- Failed preflight command:

```sh
python3 local-artifacts/erbium_amp_probe/luxonis-real-dncnn/tools/run_tiled_inference.py --variant a2_tile104_f9t_pfw_align128 --tile-id 6 --keep-builds --ssh-preflight-timeout 20
```

Observed failure:

```text
Tailscale SSH requires an additional check.
Connection to 100.121.90.41 port 22 timed out
```

Resume commands after SSH is authenticated:

```sh
python3 local-artifacts/erbium_amp_probe/luxonis-real-dncnn/tools/run_tiled_inference.py --variant a2_tile104_f9t_pfw_align128 --tile-id 6 --keep-builds
python3 local-artifacts/erbium_amp_probe/luxonis-real-dncnn/tools/run_tiled_inference.py --variant a3_tiled64_halo20_f9t_pfw_align128 --keep-builds
```

Expected outputs after A3 run:

- Per-tile logs/summaries/crops under `tiled-fp32/runs/<variant_timestamp>/`.
- `stitched_output.bin`.
- `full_audit_report.json`.
- Rows appended to `audit_master_tiled.tsv`.

## Track B Status

Completed locally:

- Added `tools/quantize_dncnn.py`.
- Quantizes the Luxonis 240x320 DnCNN3 ONNX with static calibration.
- Runs ORT on the quantized model.
- Saves:
  - quantized ONNX,
  - `dncnn3_luxonis_240x320_int8_ort_output_f32.bin`,
  - `int8_scales.json`,
  - `quantize_report.json`.
- Appends rows to `audit_master_int8.tsv`.
- Added `tools/export_int8_package.py` to turn the QDQ ONNX into an explicit
  20-layer INT8 parameter package for a future C/Tensor kernel.

Best local INT8 results so far:

| Variant | Format | Per-channel | Activations | Samples | PSNR vs FP32 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| `dncnn3-int8-s10` | QDQ | yes | QUInt8 | 10 | 40.823520 dB | 0.055608034 |
| `dncnn3-int8` | QDQ | yes | QUInt8 | 100 | 40.578305 dB | 0.040394977 |
| `dncnn3-int8-signedact-s100` | QDQ | yes | QInt8 | 100 | 39.013303 dB | 0.055895925 |
| `dncnn3-int8-pertensor-s100` | QDQ | no | QUInt8 | 100 | 38.371341 dB | 0.067885041 |
| `dncnn3-int8-qop-pertensor-s10` | QOperator | no | QUInt8 | 10 | 38.345394 dB | 0.057664201 |

Interpretation:

- The Track B PSNR gate is passed locally by every successful variant.
- Per-channel weight quantization is materially better than per-tensor.
- QUInt8 activations are better than QInt8 activations for this model/input.
- The 10-sample QDQ variant has the best PSNR, but the 100-sample QDQ variant
  has the lowest max absolute error and should be the safer default for a
  silicon kernel target.

Failed local quantization:

- QOperator + per-channel failed in ONNXRuntime bias requantization with a
  scale broadcasting error between `(576,)` weights and `(64,)` scales.
- QOperator + per-tensor works but loses about 2.2 dB versus QDQ per-channel.

Primary INT8 artifacts:

- `dncnn3-int8/dncnn3-240x320-int8.onnx`
- `dncnn3-int8/int8_scales.json`
- `dncnn3-int8/quantize_report.json`
- `dncnn3-int8-package/dncnn3_int8_package.npz`
- `dncnn3-int8-package/dncnn3_int8_manifest.json`
- `audit_master_int8.tsv`

INT8 package export:

- Layers: 20 Conv layers.
- Quantized weights: `int8`.
- Quantized biases: `int32`.
- Worst bias scale error versus `input_scale * weight_scale`:
  `6.38929707752478e-12`.
- This confirms the exported package has the expected QLinearConv-style bias
  scaling and is suitable as the source of truth for a hand-written integer
  kernel.

## Tensor/NPU Notes

Relevant hardware path found:

- Erbium docs describe Tensor support for `IMA8A32`, `QUANT`, and `REDUCE`.
- Vidas/ET headers expose tensor wrappers in
  `/home/afonso/et-platform-vidas/et-common-libs/include/erbium/isa/tensors.h`.
- `tensor_fma(... opcode=6 ...)` is used/commented as `TensorIMA8A32` in
  ET platform code.
- Quant transforms exist for `INT32_TO_FP32`, `FP32_TO_INT32`, `SATINT8`,
  `SATUINT8`, and `PACK_128B`.

Next implementation step:

- Build an INT8 silicon microkernel around Tensor IMA8A32 rather than scalar
  byte loops.
- Keep the first silicon target to one audited tile before attempting full
  240x320 INT8.
- Use the QDQ per-channel 100-sample quantization as the reference package
  unless the kernel implementation requires a per-tensor QOperator layout.
