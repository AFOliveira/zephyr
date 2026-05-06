#!/usr/bin/env python3
"""Generate 3 distinct input images + ORT references for the streaming demo.

img_0: original Luxonis test image (unchanged from baseline)
img_1: original + sigma=0.05 Gaussian noise, clipped to [0, 1]
img_2: original + sigma=0.10 Gaussian noise, clipped to [0, 1]

Each image is run through the FP32 ONNX DnCNN3 model; the output is saved
as the per-image reference. The streaming kernel's INT8 output for each
image will be audited against its own reference at the 5e-2 gate.
"""
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")
ONNX_PATH = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/dncnn3-240x320.onnx/dncnn3-240x320.onnx")

img0 = np.fromfile(DIR / "dncnn3_luxonis_240x320_input_f32.bin", dtype=np.float32).reshape(240, 320)

np.random.seed(0xDA1A)
img1 = (img0 + np.random.randn(240, 320).astype(np.float32) * 0.05).clip(0, 1).astype(np.float32)
np.random.seed(0xDA2A)
img2 = (img0 + np.random.randn(240, 320).astype(np.float32) * 0.10).clip(0, 1).astype(np.float32)

inputs = {0: img0, 1: img1, 2: img2}

sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

for i, img in inputs.items():
    img.tofile(DIR / f"dncnn3_stream_input_{i}_f32.bin")
    out = sess.run([out_name], {in_name: img.reshape(1, 1, 240, 320)})[0].reshape(240, 320)
    out.tofile(DIR / f"dncnn3_stream_ort_output_{i}_f32.bin")
    print(f"img {i}: input range [{img.min():.4f}, {img.max():.4f}]  output range [{out.min():.4f}, {out.max():.4f}]")
print(f"\n3 inputs + 3 ORT references written to {DIR}/dncnn3_stream_*_f32.bin")
