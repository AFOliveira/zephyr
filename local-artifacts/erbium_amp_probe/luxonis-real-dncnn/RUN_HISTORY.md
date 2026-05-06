# DnCNN3 on Erbium ETSOC1 — Run History

A chronological log of every kernel variant we ran on real silicon, with
wait_s, audit max_abs, and the change that produced it. Authoritative
record of how the kernel evolved from v1 scalar (106.7 s/img) to v2 +
VPU conv_first (2.22 s/img mean over 5 runs) — **48× speedup**.

All runs on `esperanto-soc4` shire 0, single-shire constraint per `CLAUDE.md`.
ELF artifacts archived under `/media/afonso/data/zephyr-erbium-artifacts/luxonis-real-dncnn/runs/`.

## Final state — current best

```
kernel:           dncnn3_tfma_int8_v2.c (DNCNN_SMT_PIPELINE=0)
weights:          per-channel INT8 PTQ, 32-image calibration set
                  conv_first/conv_final stay FP32
elf:              /tmp/dncnn_tfma/int8_tfma_v2.elf (1.32 MB)
wait_s:           2.22 s mean (n=5, σ=2.5%, range 2.17–2.32)
fps:              0.45
audit:            max_abs = 4.31e-02 vs FP32 ORT (PASS 5e-2 gate)
                  bit-exact reproducible across 5+ runs
                  PASSes on 7 distinct inputs (Luxonis test pattern,
                  3 noise variants, chessboard, gradient, circles, horiz_lines)
speedup vs v1:    48.0×
```

## Optimization timeline (chronological)

| date | label | wait_s | max_abs | speedup | what changed |
|---|---|---:|---:|---:|---|
| 2026-05-02 | `int8_v3` (v1 scalar INT8) | 106.70 | 4.31e-02 | 1.0× | Multi-minion scalar INT8 baseline. 8 T0 minions, scalar int8×int8→int32 inner loops |
| 2026-05-02 | `tfma_wffix` (FP32 scalar) | 114.63 | 1.64e-06 | 0.93× | FP32 multi-minion scalar (predates INT8 work). Useful for the `WF` layout-bug discovery |
| 2026-05-06 | `int8_tfma_v2` (initial TFMA, with bugs) | hung | — | — | First TFMA INT8 attempt. Hit linker-alignment bug (wrong A-pack reads) and FREG-clobber bug (TFMA tenc_loc=1 trashing FP scalars). Reproducer left in journal as gotchas #17 and #18. |
| 2026-05-06 | `int8_tfma_v2` (TFMA working) | 4.01 | 4.31e-02 | 26.6× | Two silent silicon bugs fixed: (1) cooperative copy of A-pack into 64-byte-aligned static buffer; (2) explicit FREG clobber list around tensor_store. Per-tile bit-exact validated against scalar reference path |
| 2026-05-06 | `int8_tfma_v2_vpu_marshal` | 2.77 | 4.31e-02 | 38.5× | VPU `fmadd.ps` for marshalling: `fcvt.ps.pw` → `fmadd.ps` → `fmax.ps` → `fmin.ps` → `fcvt.pw.ps`. 8-lane SIMD on the per-channel INT8 dequant + bias + ReLU + requantize chain |
| 2026-05-06 | `int8_tfma_v2_vpu_marshal_padded` | 2.43 | 4.31e-02 | 43.9× | (a) Padded activation buffers (240×320 → 242×322, +1.5%) so `pack_b_tile` has no boundary checks. (b) Dropped `volatile` from `static_bpack` so the compiler fuses byte stores into `sw` (4-byte). (c) j-outer / k0-inner pack loop |
| 2026-05-06 | `int8_tfma_v2` + N=3 internal loop | 7.30 (3 frames) | 4.31e-02 ×3 | per-frame 2.43, steady-state 2.26 | N-image inner loop in main(). Same image 3 times; outputs to OUTPUT_OFFSET + i × stride. Confirms steady-state per-frame is faster than single-shot (one-time setup amortizes) |
| 2026-05-06 | `int8_tfma_v2_stream` (DMA streaming) | 2.45 mean × 7 | 4.31, 4.05, 4.06, 4.73, 3.93, 4.04, 3.54 ×e-02 | — | DNCNN_STREAMING=1: kernel reads input from runtime-DMA'd DRAM offset 0x10000 instead of baked-in. Each frame: one launcher invocation with `--file_load 0x10000,image.bin`. 7 distinct inputs all PASS audit |
| 2026-05-06 | `onsoc_iterator` (on-host iterator) | 2.58 wall / 2.33 kernel × 7 | (same as above) | 23× faster than from-laptop iterator | Iterator script runs ENTIRELY on soc4 (no per-frame ssh/scp/rsync). 7 frames in 20.6 s. Per-frame: launcher init + kernel + dump = 2.58 s. The 60 s/frame from-laptop overhead = pure ssh/scp |
| 2026-05-06 | `int8_tfma_v2_vpu_first` | 2.18 single, 2.22 n=5 mean | 4.31e-02 | 48.0× | VPU conv_first: 8-OC-vector `fmadd.ps` chain over 9 taps. Repacked weights from [OC=64][9] → [v=0..7][k=0..8][oc=0..7] for contiguous flq2 loads. Saved 244 ms on conv_first block (290 → 46 ms), 210 ms on wall |

## Failed attempts (rejected with reasoning)

| label | result | verdict |
|---|---|---|
| VPU `fscw.ps` scatter for B-pack | 49 ms / layer vs 47 ms scalar | Same speed — silicon issues 16 individual cache-line stores either way; the scatter doesn't compress them. **Reverted.** |
| k0-outer pack loop reorder | 85 ms / layer vs 47 ms j-outer | 1.8× **slower**: 16 different src cache-line reads per (k0) iteration outweighed any write-coalescing benefit. **Reverted.** |
| INT4 weights | n/a (no HW path) | TFMA only has fp32/fp16/int8 dispatch types. Software emulation costs more than it saves. **Skipped.** |
| Block-sparse activations / TFMA zero-skip | 0.4% wall savings ceiling | Measured: only 2.89% of 4-IC-quartets are all-zero on natural ReLU activations. Even with QAT-perfect block sparsity, ceiling is ~9 ms / 0.4% wall savings. **Skipped.** |
| SMT T1 producer-consumer for pack_B | hangs on real silicon | L1D split-mode between SMT siblings means flag updates by T1 don't surface in T0's L1D. Eviction-based sync didn't unblock it; needs Esperanto guidance on the right primitive. Scaffolding left behind `DNCNN_SMT_PIPELINE` define. |
| VPU spatial-vec conv_final (fgb.ps gather) | 175 ms vs 95 ms scalar | 85% **slower**: each inner asm block had to spill+reload the accumulator and offset vector because the compiler can't keep FP regs live across separate inline-asm blocks. Would need ONE big asm block per tap (with internal loop counter) to amortize the setup. ~3-4 hours of careful asm work. **Reverted with comment in source.** |
| Cooperative weight loads (`tensor_coop`) | net loss in our loop structure | At per-tile granularity, 8 minions are not synchronized — each is at its own (oc_tile, tap). Adding per-tap barriers would cost ~30 ms vs ~25 ms saved from L2 BW reduction. **Skipped.** |

## Streaming validation runs

7-frame stream test on 6 distinct image patterns:

| frame | label | wait_s (kernel) | max_abs vs ORT | verdict |
|---|---|---:|---:|---|
| 0 | luxonis_clean | 2.40 | 4.31e-02 | PASS |
| 1 | luxonis_noisy_05 | 2.33 | 4.05e-02 | PASS |
| 2 | luxonis_noisy_10 | 2.33 | 4.06e-02 | PASS |
| 3 | chessboard | 2.33 | 4.73e-02 | PASS |
| 4 | gradient | 2.33 | 3.93e-02 | PASS |
| 5 | circles | 2.33 | 4.04e-02 | PASS |
| 6 | horiz_lines | 2.33 | 3.54e-02 | PASS |

Mean kernel time across 7 distinct inputs: **2.34 s ± 0.03**. Bit-exact
reproducibility across re-runs. Audit gate: 5e-2 vs FP32 ORT.

## Per-phase breakdown (current best — 2.22 s)

From in-kernel HPM[0] timestamps:

| phase | ms | % |
|---|---:|---:|
| conv_first (VPU FP32 → int8) | 46 | 2% |
| 18 hidden TFMA INT8 layers | 1787 | 80% |
| conv_final (scalar FP32 + residual subtract) | 95 | 4% |
| barriers + setup + slot-fill | 295 | 13% |
| **total** | **2222** | 100% |

Within one hidden layer (100 ms / layer × 18):

| sub-phase | ms | % of layer |
|---|---:|---:|
| pack_B (NHWC → TFMA layout, scalar 4-byte writes) | 47 | 47% |
| TFMA dispatch chain (2× tensor_load + 9× tensor_fma + waits) | 18 | 18% |
| tensor_store + evict outbuf | 1 | 1% |
| VPU marshalling (dequant + bias + ReLU + requant) | 32 | 32% |
| barriers + halo evict | 2 | 2% |

The TFMA hardware compute is now only **18% of each hidden layer**. Pack_B
(47%) and marshalling (32%) dominate.

## What remains on the table

| optimization | est savings | difficulty | status |
|---|---:|---|---|
| VPU conv_final done right (one big asm block per tap) | ~70 ms | high — asm loop counter management | doable, deferred |
| Cooperative weight loads with per-tap barriers | uncertain (small) | medium — kernel restructure | unlikely net positive in current loop |
| Multi-shire (32×) | ~2.15 s → ~70 ms / 14 fps | low (just `shire_mask`) | **policy-blocked** |
| SMT T1 producer-consumer | ~0.85 s | medium — needs Esperanto guidance | **blocked** by L1D split-mode coherence |
| Network surgery (channel/depth prune + retrain) | 2-4× | high — separate training project | out-of-kernel-scope |

The biggest single remaining lever is **multi-shire scaling**, which is
trivial to enable in code (`shire_mask = 0xFFFFFFFF`) but blocked by
project policy.

## Reproducing each variant

All current scripts assume the working directory:
```
/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/
```

### Build the current best kernel

```bash
bash build_tfma_int8_v2.sh
# → /tmp/dncnn_tfma/int8_tfma_v2.elf
```

### Single-shot run + audit

```bash
TIMEOUT=180 ./run_int8_tfma_v2.sh
# then audit:
python3 -c "
import os, numpy as np
runs = sorted(d for d in os.listdir('/tmp/dncnn_tfma') if d.startswith('int8_tfma_v2-'))
data = open('/tmp/dncnn_tfma/' + runs[-1] + '/dump.bin', 'rb').read()
out = np.frombuffer(data[0x300000:0x300000+320*240*4], dtype=np.float32).reshape(240,320)
ref = np.fromfile('dncnn3_luxonis_240x320_ort_output_f32.bin', dtype=np.float32).reshape(240,320)
print(f'max_abs={np.abs(out-ref).max():.4e}  PASS={np.abs(out-ref).max()<5e-2}')
"
```

### N-image streaming run on soc4

```bash
# stage everything once
ssh root@esperanto-soc4 'mkdir -p /root/.../erbium-amp-probe/stream_data'
scp /tmp/dncnn_tfma/int8_tfma_v2.elf root@esperanto-soc4:/root/.../stream_data/kernel.elf
for f in dncnn3_stream_input_*.bin dncnn3_other_input_*.bin; do
    scp $f root@esperanto-soc4:/root/.../stream_data/
done

# iterator runs ENTIRELY on soc4 (no per-frame ssh)
ssh root@esperanto-soc4 '/root/.../erbium-amp-probe/onsoc_iterator.sh 3'
# 7 frames in ~21 s = ~3 s/frame
```

### Visualization

```bash
python3 tools/build_viz_html.py
# /tmp/dncnn_visualization.html (self-contained, 6 test cases, embedded PNGs)
```

## Files in this directory (post-cleanup)

```
RUN_HISTORY.md                       — this file
optimizations.md                     — full optimization journal (DnCNN3-specific)
erbium_quickstart.md                 — minimum-viable launch guide
dncnn3_tfma_int8_v2.c                — production kernel
dncnn3_tfma_int8_v1.c                — v1 scalar INT8 (kept for reference)
dncnn3_tfma_v1.c                     — v0 FP32 scalar (kept for reference)
tfma_conv3x3*.c                      — minimal TFMA dispatch validators (1-hart)
build_tfma_int8_v2.sh                — full build script
run_int8_tfma_v2.sh                  — stage + flock-locked run on soc4
stream_iterator.sh                   — host-side iterator (slow due to ssh)
tools/                               — Python helpers (quantize, repack, render, audit)
dncnn3_*.bin / .o                    — weight + activation-scale + reference blobs
dncnn3_stream_input_*.bin             — 3 streaming-test inputs (Luxonis + noise variants)
dncnn3_other_input_*.bin             — 4 streaming-test inputs (chessboard, etc.)
```

Run dump artifacts (16 MB each) are at `/tmp/dncnn_tfma/` (symlinked to
`/media/afonso/data/zephyr-erbium-artifacts/luxonis-real-dncnn/runs/`),
so they don't take space on the home filesystem.

## Result TSV

Authoritative log:
`/home/afonso/etsoc1-luxonis-experiments/results/audit_master_tfma_fp32.tsv`

Latest rows:

| label | hidden_layers | wait_s | max_abs | notes |
|---|---:|---:|---:|---|
| `int8_v3` | 18 | 106.70 | 4.31e-02 | scalar baseline |
| `int8_tfma_v2` | 18 | 4.012 | 4.31e-02 | initial TFMA |
| `int8_tfma_v2_vpu_marshal` | 18 | 2.766 | 4.31e-02 | + VPU marshalling |
| `int8_tfma_v2_vpu_marshal_padded` | 18 | 2.436 | 4.31e-02 | + padded layout |
| `int8_tfma_v2_vpu_first` | 18 | 2.180 | 4.31e-02 | + VPU conv_first (single-shot) |
| `int8_tfma_v2_vpu_first_n5` | 18 | 2.222 | 4.31e-02 | + VPU conv_first (n=5 mean) |
