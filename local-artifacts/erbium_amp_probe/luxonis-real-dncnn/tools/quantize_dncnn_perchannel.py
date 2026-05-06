#!/usr/bin/env python3
"""Per-channel INT8 PTQ calibration for DnCNN3.

For each Conv layer:
  - Per-OC weight scale: w_scale[oc] = max(|W[oc, :, :, :]|) / 127
  - Per-tensor activation scale: a_scale[L] = max(|A^L|) / 127
    (computed from ORT intermediate outputs on the calibration set)
  - Quantize: w_int8[oc, *, *, *] = round(W[oc, *, *, *] / w_scale[oc]).clip(-128, 127)

Strategy: keep first conv (1→64) and final conv (64→1) in FP32 — they're tiny
and accuracy-sensitive. Quantize only the 18 hidden convs.

Output blobs:
  dncnn3_int8_hidden_weights_int8.bin   — 18 × 64 × 3 × 3 × 64 int8 (in [L, OC, ky, kx, IC] layout)
  dncnn3_int8_hidden_w_scales_fp32.bin  — 18 × 64 fp32 (per-OC weight scales)
  dncnn3_int8_act_scales_fp32.bin       — 19 fp32 (per-tensor activation scales: a_in for hidden0, hidden1, ..., final)
"""
from __future__ import annotations
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper as nh
from pathlib import Path

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")
ONNX_PATH = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/dncnn3-240x320.onnx/dncnn3-240x320.onnx")

# 1. Load ONNX + initializers
m = onnx.load(str(ONNX_PATH))
inits = {init.name: nh.to_array(init) for init in m.graph.initializer}

# 2. Expose intermediate ReLU outputs (= activation tensors for next conv)
target_outputs = []
for n in m.graph.node:
    if n.op_type == 'Relu':
        target_outputs.append(n.output[0])
for name in target_outputs:
    vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    m.graph.output.append(vi)

# 3. Run calibration (use the test image + 99 augmented variants for richer dist)
img = np.fromfile(DIR / "dncnn3_luxonis_240x320_input_f32.bin", dtype=np.float32).reshape(1, 1, 240, 320)
sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])

CALIB_N = 32  # smaller calibration since we have one good real image
np.random.seed(0xCA1BCAFE)

# Track per-tensor activation max-abs across calibration
act_max = {name: 0.0 for name in target_outputs}

for n in range(CALIB_N):
    if n == 0:
        x = img
    else:
        # Augment: noise + brightness shifts around the test image
        sigma = np.random.uniform(0.0, 0.15)
        gain = np.random.uniform(0.7, 1.3)
        bias = np.random.uniform(-0.05, 0.05)
        x = (img * gain + bias + np.random.randn(*img.shape).astype(np.float32) * sigma).clip(0, 1).astype(np.float32)
    outs = sess.run(None, {'image': x})
    out_dict = dict(zip([o.name for o in sess.get_outputs()], outs))
    for name in target_outputs:
        a = out_dict[name]
        m_abs = float(np.abs(a).max())
        if m_abs > act_max[name]:
            act_max[name] = m_abs

# 4. Per-OC weight scales for each Conv (model.0, model.2, ..., model.38)
conv_names = sorted([k for k in inits if k.endswith('.weight')], key=lambda s: int(s.split('.')[1]))
print(f'conv weights: {len(conv_names)}')
# 0=first, 2..36=hidden (18), 38=final
hidden_layer_idx = list(range(2, 38, 2))   # model.2, model.4, ..., model.36
assert len(hidden_layer_idx) == 18

# Activation tensor for each layer's INPUT:
# - Conv (model.0) input: image (already [0, 1])
# - Conv (model.2) input: ReLU(model.1) output
# - Conv (model.4) input: ReLU(model.3) output
# - ...
# - Conv (model.38) input: ReLU(model.37) output

# Map: hidden L=0 (model.2) input scale = a_max of model.1 ReLU output
relu_for_conv = {idx: f'/model/model.{idx-1}/Relu_output_0' for idx in hidden_layer_idx + [38]}
relu_for_conv[2] = '/model/model.1/Relu_output_0'

# 19 activation scales (input to each of: 18 hidden + final = 19 convs after the first)
a_scales = []
for idx in hidden_layer_idx + [38]:
    relu_name = relu_for_conv[idx]
    a_max = act_max[relu_name]
    a_scale = a_max / 127.0
    a_scales.append(a_scale)
a_scales = np.array(a_scales, dtype=np.float32)
print(f'a_scales: {a_scales}')

# Per-OC weight scales for the 18 hidden convs
WH_int8 = np.zeros((18, 64, 3, 3, 64), dtype=np.int8)
WH_scales = np.zeros((18, 64), dtype=np.float32)
for L, idx in enumerate(hidden_layer_idx):
    W = inits[f'model.{idx}.weight']  # [OC=64, IC=64, ky=3, kx=3]
    # Reorder to [OC, ky, kx, IC] for our layout
    W_reorder = W.transpose(0, 2, 3, 1).copy()    # [OC, ky, kx, IC]
    for oc in range(64):
        s = float(np.abs(W_reorder[oc]).max() / 127.0)
        if s == 0: s = 1e-30
        WH_scales[L, oc] = s
        WH_int8[L, oc] = np.clip(np.round(W_reorder[oc] / s), -128, 127).astype(np.int8)
    # Sanity: dequant error
    deq = WH_int8[L].astype(np.float32) * WH_scales[L][:, None, None, None]
    err = float(np.abs(deq - W_reorder).max())
    print(f'  L={L} (model.{idx}): max_dequant_err={err:.4e}')

WH_int8.tofile(DIR / "dncnn3_int8_hidden_weights_int8.bin")
WH_scales.tofile(DIR / "dncnn3_int8_hidden_w_scales_fp32.bin")
a_scales.tofile(DIR / "dncnn3_int8_act_scales_fp32.bin")
print(f'\nWritten:')
print(f'  WH_int8 {WH_int8.shape} = {WH_int8.nbytes:,} B')
print(f'  WH_scales {WH_scales.shape} = {WH_scales.nbytes:,} B')
print(f'  a_scales {a_scales.shape} = {a_scales.nbytes:,} B')
