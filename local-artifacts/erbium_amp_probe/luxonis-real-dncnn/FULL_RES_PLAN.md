# DnCNN3 Full-Resolution Plan: Tiled FP32 (Track A) + INT8 Quantization (Track B)

Two independent tracks, both targeting **end-to-end audited 240×320 Luxonis DnCNN3
inference on `esperanto-soc6`**. They can run in parallel; the deliverables stack.

Both reuse the existing audit framework (`experiment_runner.py`,
`audit_master.tsv`, etc.) — every variant goes through silicon, gets an ORT
cross-check, and lands a row in the master TSV. The only thing different is the
spec generator and (per-track) the kernel C source.

---

## Invariant rules (apply to all batches in either track)

1. Run on `esperanto-soc6`, shire 0, root@ login. No reboots, no other shires.
2. Stage under `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/`.
3. Every variant must pass an explicit audit before being recorded as "ok":
   - **FP32 (Track A):** `max_abs(silicon, ORT FP32) ≤ 1e-5` (same gate as the 100-experiment loop)
   - **INT8 (Track B):** `max_abs_dequant(silicon, ORT INT8) ≤ 2 LSB` AND `psnr(silicon_dequant, ORT FP32) ≥ 35 dB`
4. Append every variant — pass *or* fail — to the appropriate master TSV with
   provenance (flags, macros, geometry, hashes, dump path). Failed variants are
   data too.
5. Build local, stage to remote in batches, run sequentially (the launcher is
   single-process), pull dumps back, audit on the dev box.
6. Don't touch `dncnn3_luxonis_real_vpu.c` knobs we already established as
   winners (FUSED9TAP=1, PREFETCH_READ_WINDOW=1, `-O3 -funroll-loops
   -falign-functions=128 -falign-loops=128 -fno-tree-loop-distribute-patterns
   -fno-tree-loop-vectorize`, ACTIVE_HARTS=16, PASSES=1) without an experiment
   that justifies the change.

---

# Track A — Tiled FP32 240×320 inference

## Goal

Run the actual Luxonis DnCNN3 model at its native 240×320 resolution on silicon,
producing bit-exact-allclose output vs ORT, **without quality loss**.

## Why it works

- Each conv layer has receptive field 3×3 (kernel 3). Stacked 20 layers → total RF = 41×41 (each layer adds ±1).
- A 64×64 *output* tile depends on a 104×104 *input* tile (32-pixel halo each side, 20 layers × 1 + 12 from the residual subtract).  Choose 32 as the "safe" halo number for clean math.
- Memory per tile: 104×104×64ch×4 × 2 = **5.5 MB activation**, fits the 16 MB U-mode region with 10 MB headroom.
- 240×320 = 4×5 = **20 tiles** at 64×64 stride (modulo right/bottom boundary tiles which are smaller).
- Each tile reuses the kernel we already audited at 2.55 GOPS — no change to the inner loop.

## Phases (each phase has its own batch structure)

### Phase A1 — Reference and design (no silicon yet)
**A1.1** Compute the exact halo size needed for DnCNN3 (residual subtract is point-wise, doesn't add to RF). Confirm with a Python reference: `tiled_dncnn(input, tile_h=64, tile_w=64, halo=H)` produces bit-equivalent output to `ort.run(input)` for some H.
**A1.2** Generate the tile geometry table: tile (i, j) → (input_y0, input_y1, input_x0, input_x1, output_y0, output_y1, output_x0, output_x1). Edge tiles are smaller. Save as JSON.
**A1.3** Pre-bake per-tile input bins and per-tile ORT-output bins from the full 240×320 inputs/outputs. Save under `tiled-fp32/tiles/`.

**Audit gate A1:** Python `tiled_dncnn` + `ort.run(full)` agree to bit-exact at the full-image hash, for at least one (input, halo) pair. If not bit-exact, halo is too small — increase H and retry.

### Phase A2 — Single-tile silicon validation
**A2.1** Build verified-port at 104×104 (or whatever halo geometry) with FUSED9TAP. Need to add `DNCNN_TILE104_BLOBS` (or generic `DNCNN_TILE_W/H`) macros to support arbitrary tile sizes — same pattern as the existing TILE1/4/16/64 blocks.
**A2.2** Stage one tile's input + ORT-output as static blobs, run on silicon, audit.

**Audit gate A2:** Single tile passes the standard FP32 audit (`max_abs ≤ 1e-5`).

### Phase A3 — Multi-tile orchestration
**A3.1** Write a host-side runner that for each of the 20 tiles: builds an ELF with the right input/ORT-output blobs, scps, runs, dumps, captures the output region. (Or builds **one** ELF with a tile-id arg and 20 input blobs linked, dispatched from the host.)
**A3.2** Stitch the 20 silicon outputs into a full 240×320 image.
**A3.3** Compare stitched image vs full-resolution ORT output.

**Audit gate A3:** Stitched image `max_abs(stitched, ORT_full) ≤ 1e-5`.

### Phase A4 — Iteration loop (10 batches × 10 experiments = 100)
Same structure as the 100-experiment loop we just finished. Spec generator
appends to `audit_master_tiled.tsv`. After each batch, leaderboard sort by
`gops_full_image = ops_full / wait_s_total_across_tiles`.

| Batch | Focus |
|---|---|
| 1 | tile geometry sweep (32×32, 48×48, 64×64, 80×80, 96×96 at output; halo varies) |
| 2 | halo size sweep at the winning tile size (verify smaller halo doesn't break audit) |
| 3 | per-tile compile-flag confirmation (winners likely transfer from existing best) |
| 4 | per-tile cache strategy (own evict scope is now smaller — boundary-only might help) |
| 5 | tile dispatch overhead (build single multi-tile ELF vs N single-tile ELFs) |
| 6 | tile prefetch (start loading tile N+1 input while N is computing) |
| 7 | parallel-shire dispatch — if the launcher supports `--shire 1` simultaneously, run 2 tiles per launch on different shires |
| 8 | reduced PASSES per tile (1 is best at 64×64 from 100-exp loop; reconfirm at tile boundaries) |
| 9 | hart count sweep (8 vs 16, since multi-tile changes the per-hart work) |
| 10 | validation (10 full-image runs, noise floor) |

**Stopping condition for Track A:** stitched 240×320 silicon output passes
`allclose(silicon, ORT_full, atol=1e-5)`. **Success metric:** full-image GOPS
(throughput accounting tile-overhead) and full-image wait_s.

### Track A success criteria

- ≥ 1 audited 240×320 inference on silicon
- Full-image GOPS ≥ **2.0 GOPS** (allowing some per-tile overhead)
- Total wall time ≤ **35 s** (single-shire, 16 harts) for one inference

---

# Track B — INT8 quantization

## Goal

Replace the FP32 packed-single VPU compute with **packed-integer VPU**, getting
~4× memory savings (so 240×320 fits without tiling) and ideally 2–4× silicon
throughput. Audit against the original FP32 ORT model with PSNR ≥ 35 dB.

## Why it could be a big win

- Activation memory: 64×64×64ch × **1 byte** (INT8) × 2 buffers = 524 KB per tile or 9.8 MB for full 240×320 → fits 16 MB with margin.
- Weights: 2.5 MiB FP32 → 0.6 MiB INT8.
- VPU packed-integer can do more lanes per fmadd than packed-single (assuming hardware supports it — verify from docs).
- Same audit chain works, just with a different reference model (the INT8-quantized ONNX).

## Risks and validation discipline

- INT8 introduces quantization error. We need TWO reference points:
  1. **INT8 ORT reference**: deterministic — silicon should match this bit-exact-allclose (max_abs ≤ 2 LSB, allowing for accumulator-order differences).
  2. **FP32 ORT reference**: fuzzy — silicon-dequantized output should match this within a PSNR threshold (35 dB is conservative for denoising).
- Need calibration data: a small set of representative inputs to determine per-tensor scales.
- Per-channel vs per-tensor quantization affects accuracy and kernel complexity; start per-tensor (simpler) then iterate.

## Phases

### Phase B1 — Quantize the ONNX model and build references (no silicon yet)
**B1.1** Use `onnxruntime.quantization.quantize_static()` with a calibration dataset
of ~100 representative noisy images (Luxonis sample set, or augmented). Output:
`dncnn3-240x320-int8.onnx`.
**B1.2** Inspect the quantized model: extract per-tensor scales/zero_points for each conv weight + activation. Save as JSON.
**B1.3** Run the INT8 ORT model on the same inputs we used for FP32:
`dncnn3_luxonis_64x64_int8_input.bin`, `dncnn3_luxonis_64x64_int8_ort_output.bin`.
**B1.4** Compute PSNR(INT8 dequantized output, FP32 ORT output) on each tile size.

**Audit gate B1:** PSNR(INT8 ORT, FP32 ORT) ≥ 35 dB on 64×64 tile and full image. If lower, recalibrate (more samples / per-channel weights).

### Phase B2 — INT8 kernel reference implementation
**B2.1** Identify Erbium VPU packed-integer instructions (`fmadd.pi`/`fmadd.qb`/whatever the doc calls them; check `/home/afonso/docsET/`).
**B2.2** Write a new C path `conv_hidden_int8` using the packed-integer VPU
instructions. Accumulate in INT32 (the packed-integer VPU's typical accumulator
width). Apply per-tensor dequant scale at end of layer to produce next-layer INT8.
**B2.3** Write a host-side reference in Python that mimics the kernel's dequant
order — to produce the exact `int8_silicon_target_output.bin` for cross-check.

**Audit gate B2:** Host reference matches INT8 ORT to ≤ 2 LSB max-abs.

### Phase B3 — Single-tile INT8 silicon validation
**B3.1** Build a `dncnn3_luxonis_int8_64x64.elf` linking int8 input + int8 ORT output + int8 weights. Run on silicon. Audit.

**Audit gate B3:** silicon INT8 output matches int8 ORT ≤ 2 LSB max-abs AND PSNR vs FP32 ORT ≥ 35 dB.

### Phase B4 — Iteration loop (10 batches × 10 = 100 experiments)

| Batch | Focus |
|---|---|
| 1 | INT8 fused-9-tap asm baseline (port FUSED9TAP for packed-int) |
| 2 | per-tensor vs per-channel scales (PSNR + GOPS tradeoff) |
| 3 | symmetric vs asymmetric quant |
| 4 | accumulator width: INT32 vs INT16 (faster but rolls over more often) |
| 5 | dequant placement: per-layer vs at-end-of-network (latter saves intermediate storage but loses some accuracy) |
| 6 | calibration size: 10, 100, 1000 samples |
| 7 | mixed precision: keep first/last layer FP32 (typical PTQ recipe) |
| 8 | full 240×320 INT8 (does it fit at 9.8 MB activations?) |
| 9 | combine with Track A tiling — INT8 + tiling, pick the smaller halo |
| 10 | validation runs at the best variant |

**Stopping condition for Track B:** silicon INT8 output passes the dual-gate
audit on at least one full inference. **Success metric:** wall time per
inference vs Track A's, AND audited PSNR ≥ 35 dB.

### Track B success criteria

- ≥ 1 audited INT8 inference on silicon (any tile size)
- ≥ 1 audited INT8 inference at full 240×320 (would fit per memory math)
- Full-image GOPS (counting INT8 ops) ≥ **5 GOPS**
- PSNR(silicon_dequant, ORT_FP32) ≥ **35 dB** on full image

---

# Cross-track integration

After both tracks have a working audited variant, run a comparison batch:

| Variant | Tile/Geom | GOPS | PSNR vs FP32 ORT | Wall-time |
|---|---|---:|---:|---:|
| Track A best (FP32 tiled) | 240×320 (20 tiles 64×64) | ? | ∞ (bit-exact) | ? |
| Track B best (INT8 single-shot) | 240×320 (no tile) | ? | ? | ? |
| Track A+B (INT8 tiled) | 240×320 (4 INT8 tiles 128×160) | ? | ? | ? |

The "right" answer depends on the use case — denoising quality might mandate
Track A; throughput might mandate Track B; embedded power might mandate the hybrid.

---

# Iteration framework

## Reusable: `experiment_runner.py`

Already handles build → stage → silicon run → dump → audit → master.tsv. Two
small additions needed:

1. Per-track audit gate selector (`--audit fp32_allclose | int8_lsb_psnr`)
2. Tile-aware spec for Track A (`spec.tile_geometry = "tiled_64_halo_32"` etc.)

## Per-track master TSV

- `audit_master_tiled.tsv` (Track A)
- `audit_master_int8.tsv` (Track B)

Same column structure as `audit_master.tsv`, plus track-specific:
- Track A: `n_tiles`, `tile_h`, `tile_w`, `halo`, `total_wait_s_all_tiles`, `gops_full_image`
- Track B: `quant_scheme`, `bits`, `psnr_vs_fp32`, `max_abs_lsb`, `dequant_placement`

## Daily iteration cadence

For each track, per day:
1. **Triage** (10 min): read last batch's leaderboard, pick top 3 variants
2. **Spec next batch** (10 min): write batchN.json with 10 variants doubling down
3. **Execute** (5–15 min wall, mostly silicon): `python experiment_runner.py --batch batchN.json`
4. **Audit + record** (automatic via the runner)
5. **Decision point**: continue, pivot, or freeze

If a batch produces no improvement for **two consecutive batches**, that track
has hit its local optimum — pivot to the other track (or to the cross-track
combo) rather than burning more silicon time.

## When to stop a track entirely

- Track A: stop after the first `allclose` 240×320 stitched output passes audit AND no further GOPS gains for 20 experiments. Promote the best variant to a release artifact.
- Track B: stop after the first INT8 inference passes both gates AND no PSNR gains for 20 experiments. Promote the best variant.

## What to do at "freeze"

1. Tag the best variant's source state in git (`feat/etsoc1-dncnn-tiled-fp32-v1` etc.)
2. Update `optimization_knowledge_base.md` with the new headline numbers and the macro/flag set
3. Add the variant to `current_best_pmc_results.tsv`
4. Move on to the next model in the inventory (Depth, Whisper, Stereo, YOLO) using the same playbook

---

# Concrete kickoff — exactly what to run first

```bash
cd /home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn

# Track A, Phase A1.1: build the Python tiled reference
python3 tools/build_tiled_reference.py \
   --onnx /path/to/dncnn3-240x320.onnx \
   --tile_h 64 --tile_w 64 --halo 32 \
   --out audit_a1_geometry.json \
   --verify_against_full_ort

# Track B, Phase B1.1: quantize the ONNX
python3 tools/quantize_dncnn.py \
   --onnx /path/to/dncnn3-240x320.onnx \
   --calibration_dir tools/calibration_inputs/ \
   --out dncnn3-240x320-int8.onnx \
   --report quantize_report.json

# Both can run concurrently — they don't share kernel modifications
```

Each subsequent batch follows the same `experiment_runner.py` invocation as the
100-experiment loop, just with different spec JSONs.
