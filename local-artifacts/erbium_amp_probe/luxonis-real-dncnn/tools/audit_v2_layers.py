#!/usr/bin/env python3
"""Read static_act0/act1 buffers from a v2 dump.bin and audit each layer's
INT8 activation (after dequant) against the ORT FP32 reference at the
corresponding ReLU output.

Static buffer offsets (from `riscv64-unknown-elf-nm int8_tfma_v2.elf`):
  static_act0 at base + 0x616b00
  static_act1 at base + 0x166b00
where base = heap0_end - 64MB.

After 18 hidden layers (even count), the FINAL hidden output sits in act0.
With ping-pong:
  L=0 dst=act1, L=1 dst=act0, ..., L=17 dst=act0.

So the buffer state at end-of-kernel:
  act0 holds layer 17's int8 output  (post-relu, requantized)
  act1 holds layer 16's int8 output

This script just diffs act0 (layer 17 output) vs ORT layer-17 ReLU output.
"""
import sys
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper as nh
from pathlib import Path

DIR = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn")
ONNX_PATH = Path("/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/dncnn3-240x320.onnx/dncnn3-240x320.onnx")

if len(sys.argv) < 2:
    print("usage: audit_v2_layers.py <dump.bin>")
    sys.exit(1)
DUMP = Path(sys.argv[1])
data = DUMP.read_bytes()

# Static buffer offsets in dump (16MB heap region starting at base = heap_end - 64MB)
ACT0_OFF = 0x616b00
ACT1_OFF = 0x166b00
H, W, CH = 240, 320, 64

act0 = np.frombuffer(data[ACT0_OFF:ACT0_OFF + H*W*CH], dtype=np.int8).reshape(H, W, CH)
act1 = np.frombuffer(data[ACT1_OFF:ACT1_OFF + H*W*CH], dtype=np.int8).reshape(H, W, CH)

print(f"act0 (= layer-17 output): range [{act0.min()}, {act0.max()}], unique={len(np.unique(act0))}")
print(f"act1 (= layer-16 output): range [{act1.min()}, {act1.max()}], unique={len(np.unique(act1))}")

# Load activation scales
a_scales = np.fromfile(DIR / "dncnn3_int8_act_scales_fp32.bin", dtype=np.float32)
print(f"a_scales[17] = {a_scales[17]:.6f}, [18] = {a_scales[18]:.6f}")

# Compute ORT intermediate ReLU outputs on the same input
m = onnx.load(str(ONNX_PATH))
relu_outputs = [n.output[0] for n in m.graph.node if n.op_type == 'Relu']
for name in relu_outputs:
    vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    m.graph.output.append(vi)
sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
img = np.fromfile(DIR / "dncnn3_luxonis_240x320_input_f32.bin", dtype=np.float32).reshape(1, 1, 240, 320)
outs = sess.run(None, {'image': img})
out_dict = dict(zip([o.name for o in sess.get_outputs()], outs))

# Layer 17's ReLU output is the LAST relu (model.37/Relu_output_0)
relu_layer17 = relu_outputs[-1]   # last relu = layer 17 output
relu_layer16 = relu_outputs[-2]
print(f"using relu_layer17 = {relu_layer17}")
print(f"using relu_layer16 = {relu_layer16}")

ref17 = out_dict[relu_layer17][0, :, :, :].transpose(1, 2, 0)   # NCHW → NHWC: [240, 320, 64]
ref16 = out_dict[relu_layer16][0, :, :, :].transpose(1, 2, 0)

# Dequantize act0/act1: dequant = int8 * a_scale[L+1]
dq_layer17 = act0.astype(np.float32) * a_scales[18]
dq_layer16 = act1.astype(np.float32) * a_scales[17]

# Compare
def stat(a, b, label):
    diff = a - b
    print(f"{label}: max_abs={np.abs(diff).max():.4e}  mean_abs={np.abs(diff).mean():.4e}  ref_range=[{b.min():.3f}, {b.max():.3f}]  kern_range=[{a.min():.3f}, {a.max():.3f}]")

stat(dq_layer17, ref17, "layer 17 (act0)")
stat(dq_layer16, ref16, "layer 16 (act1)")

# Sample a few values
print("\nSample [120, 160, 0..7]:")
print(f"  ref17[120,160,:8]    = {ref17[120, 160, :8]}")
print(f"  dq_layer17[120,160,:8] = {dq_layer17[120, 160, :8]}")

# Per-row max_abs to see if some rows are good and others bad
print("\nLayer 17 max_abs per row (should be uniformly small if all minions OK):")
for h in range(8):
    r0, r1 = h * 30, (h + 1) * 30
    diff = dq_layer17[r0:r1] - ref17[r0:r1]
    print(f"  hart {h} rows [{r0},{r1}): max_abs={np.abs(diff).max():.4e}")
