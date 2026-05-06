#!/usr/bin/env python3
"""Generate a self-contained HTML page visualizing DnCNN3 silicon results.

Pulls each (input, silicon-output, ORT-reference) triple from the streaming
runs we did, encodes the panels as base64 PNGs, and emits a single HTML file
with all images inline + an interactive slider to compare input ↔ output."""
import os, base64, io
import numpy as np
from PIL import Image

DIR = "/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn"
RUNS_DIR = "/tmp/dncnn_tfma"
H, W = 240, 320

def fp32_to_png_b64(arr, vmin=0.0, vmax=1.0, cmap=None):
    """Convert HxW fp32 array to a base64 PNG. cmap=None → grayscale.
    cmap='diff' → blue-white-red, vmin/vmax symmetric."""
    if cmap == 'diff':
        # symmetric blue-white-red
        v = np.clip(arr / max(abs(vmin), abs(vmax)), -1, 1)
        # build RGB
        r = np.where(v > 0, 255, 255 + v * 255).astype(np.uint8)
        g = np.where(np.abs(v) < 0.05, 255, (255 - np.abs(v) * 255)).astype(np.uint8)
        b = np.where(v < 0, 255, 255 - v * 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(rgb)
    else:
        v = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        img = Image.fromarray((v * 255).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def latest_run(prefix):
    runs = sorted([d for d in os.listdir(RUNS_DIR) if d.startswith(prefix)])
    return os.path.join(RUNS_DIR, runs[-1]) if runs else None

# Test cases: (label, input_bin, ref_bin, run_dir_prefix, description)
cases = [
    ('luxonis_clean',  f'{DIR}/dncnn3_stream_input_0_f32.bin',
                        f'{DIR}/dncnn3_stream_ort_output_0_f32.bin',
                        'stream_img0-', 'Luxonis test pattern (nearly noise-free)'),
    ('luxonis_noisy',  f'{DIR}/dncnn3_stream_input_2_f32.bin',
                        f'{DIR}/dncnn3_stream_ort_output_2_f32.bin',
                        'stream_img2-', 'Luxonis pattern + σ=0.10 Gaussian noise'),
    ('chessboard',     f'{DIR}/dncnn3_other_input_0_f32.bin',
                        f'{DIR}/dncnn3_other_ort_0_f32.bin',
                        'other_img0-', '20-px chessboard + σ=0.10 noise'),
    ('gradient',       f'{DIR}/dncnn3_other_input_1_f32.bin',
                        f'{DIR}/dncnn3_other_ort_1_f32.bin',
                        'other_img1-', 'Radial gradient + σ=0.10 noise'),
    ('circles',        f'{DIR}/dncnn3_other_input_2_f32.bin',
                        f'{DIR}/dncnn3_other_ort_2_f32.bin',
                        'other_img2-', '5 disks of varying brightness + σ=0.10 noise'),
    ('horiz_lines',    f'{DIR}/dncnn3_other_input_3_f32.bin',
                        f'{DIR}/dncnn3_other_ort_3_f32.bin',
                        'other_img3-', '4-px horizontal stripes + σ=0.10 noise'),
]

panels_html = []
for label, in_bin, ref_bin, run_prefix, desc in cases:
    run_dir = latest_run(run_prefix)
    if run_dir is None:
        print(f"  SKIP {label}: no run found for prefix {run_prefix}")
        continue
    inp = np.fromfile(in_bin, dtype=np.float32).reshape(H, W)
    ref = np.fromfile(ref_bin, dtype=np.float32).reshape(H, W)
    out = np.frombuffer(open(run_dir + "/dump.bin", "rb").read()[0x300000:0x300000 + H*W*4],
                       dtype=np.float32).reshape(H, W)
    diff = out - ref

    # extract wait_s
    log = open(run_dir + "/run.log").read()
    import re
    m = re.search(r"Kernel wait seconds: ([0-9.]+)", log)
    wait_s = m.group(1) if m else "?"

    inp_b64  = fp32_to_png_b64(inp, 0, 1)
    out_b64  = fp32_to_png_b64(out, 0, 1)
    ref_b64  = fp32_to_png_b64(ref, 0, 1)
    diff_b64 = fp32_to_png_b64(diff, vmin=-0.05, vmax=0.05, cmap='diff')

    max_abs = np.abs(diff).max()
    noise_in = (inp - ref).std()
    noise_removed = (inp - out).std()

    panels_html.append(f"""
    <div class="case">
      <h3>{label} — {desc}</h3>
      <div class="metrics">
        <span>kernel time: <b>{wait_s} s</b></span>
        <span>noise σ (in): <b>{noise_in:.3f}</b></span>
        <span>noise removed: <b>{noise_removed:.3f}</b></span>
        <span>max_abs vs ORT: <b>{max_abs:.3e}</b></span>
        <span class="{'pass' if max_abs < 5e-2 else 'fail'}">{'PASS 5e-2' if max_abs < 5e-2 else 'FAIL'}</span>
      </div>
      <div class="grid">
        <figure>
          <img src="data:image/png;base64,{inp_b64}" alt="input">
          <figcaption>INPUT (noisy)</figcaption>
        </figure>
        <figure class="slider-pane">
          <div class="slider-frame">
            <img class="img-bottom" src="data:image/png;base64,{inp_b64}" alt="input">
            <img class="img-top" src="data:image/png;base64,{out_b64}" alt="output" style="--clip:50%;">
            <div class="slider-handle"></div>
          </div>
          <input type="range" min="0" max="100" value="50" class="slider"
                 oninput="this.parentNode.querySelector('.img-top').style.setProperty('--clip', this.value + '%'); this.parentNode.querySelector('.slider-handle').style.left = this.value + '%';">
          <figcaption>← input | OUTPUT silicon →</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{out_b64}" alt="output">
          <figcaption>OUTPUT silicon (denoised)</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{ref_b64}" alt="ort_ref">
          <figcaption>ORT FP32 reference</figcaption>
        </figure>
        <figure>
          <img src="data:image/png;base64,{diff_b64}" alt="diff">
          <figcaption>silicon − ORT (±0.05 scale)</figcaption>
        </figure>
      </div>
    </div>
""")
    print(f"  {label}: wait={wait_s}s  max_abs={max_abs:.3e}  PASS={max_abs<5e-2}")

html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DnCNN3 on Erbium ETSOC1 silicon — visualization</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1500px; margin: 1em auto; padding: 0 1em; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }
  .summary { background: #f5f5f0; border: 1px solid #ddd; padding: 1em 1.5em;
             border-radius: 6px; margin-bottom: 2em; }
  .summary table { border-collapse: collapse; }
  .summary td { padding: .25em 1em .25em 0; }
  .summary td:first-child { font-weight: 600; color: #555; }
  .case { border: 1px solid #ddd; border-radius: 6px; padding: 1em 1.5em;
          margin-bottom: 1.5em; background: #fafafa; }
  .case h3 { margin: 0 0 .4em 0; }
  .metrics { display: flex; gap: 1.5em; font-size: 0.9em; margin-bottom: .8em;
             color: #555; flex-wrap: wrap; }
  .metrics b { color: #111; }
  .pass { color: #2a7a2a; font-weight: 600; }
  .fail { color: #b00; font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: .8em; }
  figure { margin: 0; text-align: center; }
  figure img { width: 100%; height: auto; image-rendering: pixelated;
                border: 1px solid #aaa; display: block; }
  figcaption { font-size: .8em; color: #555; margin-top: .25em; }
  .slider-pane { position: relative; }
  .slider-frame { position: relative; width: 100%; aspect-ratio: 320/240;
                  overflow: hidden; border: 1px solid #aaa; }
  .slider-frame img { position: absolute; top: 0; left: 0;
                      width: 100%; height: 100%; }
  .img-top { clip-path: inset(0 calc(100% - var(--clip, 50%)) 0 0); }
  .slider-handle { position: absolute; top: 0; bottom: 0; width: 2px;
                   background: #ff0; box-shadow: 0 0 4px rgba(0,0,0,.5);
                   pointer-events: none; left: 50%; transform: translateX(-1px); }
  .slider { width: 100%; margin-top: .25em; }
  .legend { font-size: .85em; color: #666; margin-bottom: 1em; }
</style>
</head><body>
<h1>DnCNN3 on Erbium ETSOC1 silicon — visualization</h1>

<div class="summary">
<table>
<tr><td>Hardware</td><td>1 shire of Erbium ETSOC1 (8 minions × T0)</td></tr>
<tr><td>Kernel</td><td>dncnn3_tfma_int8_v2 (TFMA INT8 + VPU FP32 marshalling, padded NHWC)</td></tr>
<tr><td>Per-image kernel time</td><td>2.43 s mean (σ &lt; 2%, n=5+ runs, bit-exact)</td></tr>
<tr><td>Per-image steady-state (3-image loop)</td><td>2.26 s</td></tr>
<tr><td>Speedup vs v1 scalar (106.7s)</td><td>43.8× (best single-shot) / 47.2× (steady)</td></tr>
<tr><td>Audit gate</td><td>max_abs ≤ 5e-2 vs FP32 ORT</td></tr>
<tr><td>Quantization</td><td>Per-channel symmetric INT8 PTQ on 18 hidden layers; FP32 conv_first / conv_final</td></tr>
</table>
</div>

<p class="legend">
Each case shows: the noisy <b>input</b>, an interactive <b>side-by-side slider</b> (drag the
yellow handle to sweep input vs silicon output), the <b>silicon output</b>, the <b>ORT FP32
reference</b>, and the <b>silicon - ORT diff</b> at ±0.05 scale (red = silicon higher,
blue = silicon lower; gray means no difference).
</p>
""" + "\n".join(panels_html) + """
</body></html>
"""

out_path = "/tmp/dncnn_visualization.html"
with open(out_path, "w") as f:
    f.write(html)
print(f"\nWrote: {out_path}")
print(f"Open in a browser: file://{out_path}")
