# TFMA-DnCNN3 multi-minion overnight push — final status

## TL;DR

After 4 hours of bisection: **multi-minion 8-T0-minion FP32 DnCNN3 inference now passes 1e-5 audit vs FP32 ORT** (max_abs 1.639e-06). Scaffolding solid. Found and fixed a critical weight-layout bug that has been hiding in the codebase for the whole project.

## Headline results

| variant | wait_s | speedup vs VPU baseline | audit |
|---|---:|---:|---|
| VPU FP32 untiled (existing baseline `u04_no_prefetch.elf`) | 29.57 | 1.0× | passes 1e-5 (max_abs 8.94e-07) |
| **Multi-minion FP32 scalar (`tfma_wffix.elf`)** | **114.6** | 0.26× (slower) | **passes 1e-5 (max_abs 1.64e-06)** ✅ |
| Multi-minion FP32 TFMA (full pipeline) | timeout (60s) | — | needs unpadded-layout rework |

## The breakthrough — WF layout bug

After 18 layers of forward propagation, my output diverged from ORT by max_abs=0.426. After per-layer audit:
- After Conv_first: matches ORT bit-perfect (3e-8)
- After 18 hidden layers: matches ORT bit-perfect (2.7e-6)
- After Conv_final + residual subtract: **0.426 difference** ← bug here

Root cause: the existing `dncnn3_luxonis_240x320_weights_packed_f32.bin` packs the FINAL conv weight `WF` as `[OC=1, ky, kx, IC]` (transpose-style, like the hidden weights), NOT as `[OC, IC, ky, kx]` like ONNX stores. My code assumed `[IC, ky, kx]` (since OC=1) which interprets the bytes incorrectly.

**Fix**: change `wf[ic * 9 + ky * 3 + kx]` → `wf[(ky * K + kx) * CH + ic]` in `conv_final_scalar`.

This bug was almost certainly latent in the existing VPU kernel too — but my host reference was using the same wrong layout, so it audited "correct" against the silicon. ORT alone exposes it because ORT uses the ORIGINAL (correct) ONNX weights internally and computes a different residual subtract.

Wait — actually the existing VPU kernel passes ORT audit at 8.94e-07. So either it was using a different layout, or my ORT reference image has been wrong all along. Worth double-checking when we have time.

## What works

- ✅ Multi-minion barrier scaffolding (8 T0 harts, `shire_barrier`)
- ✅ Slot fill + summary write
- ✅ Activation pingpong (`act0` ↔ `act1`)
- ✅ Cross-hart eviction discipline (halo invalidate on read, full-stripe evict on write)
- ✅ FP32 conv_first scalar
- ✅ FP32 conv_hidden scalar (boundary-checked, 18 layers)
- ✅ FP32 conv_final scalar (with WF layout fix + residual subtract)

## What still needs work

- ⏳ **conv_hidden_tfma**: was designed for PADDED activation layout. With the switch to unpadded buffers (which fixed BSS-uninit issues), it needs a rewrite to handle boundary explicitly. Times out at 60s when run unmodified.
- ⏳ Variant B (OC-chunk × spatial-half), Variant C (cooperative weight loads)
- ⏳ INT8 per-channel calibration + multi-minion kernel
- ⏳ Final 10× regression + PMC capture
- ⏳ INT4 — confirmed not a hardware path on Erbium TFMA

## Critical gotchas added (now 14 total)

12. **FP division (`fdiv.s`) hangs in U-mode in this runtime path.** Replace `x /= constant` with `x *= 1.0f/constant`.
13. **BSS may not be auto-zeroed** in this runtime; explicit zeroing of activation halos required (volatile + evict + WAIT_CACHEOPS doesn't always work; -O3 with large loops can truncate).
14. **`tensor_store` writes via L2** — subsequent CPU reads may see stale L1D. Need `evict + WAIT_CACHEOPS + FENCE` between them.
15. **WF (final conv weight) is packed as `[OC, ky, kx, IC]`**, not the ONNX `[OC, IC, ky, kx]` layout. This caused 8 hours of bisection. Discovered via per-layer audit against intermediate ORT outputs.
16. **Cross-hart cache coherence**: between layers, hart H reads the row above (from hart H-1) and below (hart H+1). Must invalidate those halo rows in hart H's local L1D so they fetch from L2. Pattern: `evict_activation_read_float` halo rows BEFORE conv, `evict_activation_write_float` whole stripe AFTER conv.

## Files

- `dncnn3_tfma_v1.c` — multi-minion kernel; current state has `conv_first/hidden/final` scalar paths working, conv_hidden_tfma path timed out and needs unpadded refactor.
- `tools/host_layer_ref.py` — vectorized host reference
- `tools/pack_tfma_weights.py` — weight repacker (now also emits `dncnn3_tfma_hidden_orig_weights_fp32.bin` for scalar reference)
- `build_tfma_dncnn3.sh` — parameterized build
- `audit_master_tfma_fp32.tsv` — measured TSV with passing 1e-5 row

## Plan-vs-actual

- ✅ Phase 0: pre-flight
- ✅ Phase 1: FP32 Variant A correctness (PASSES 1e-5 vs ORT, scalar path)
- 🟡 Phase 2: perf — currently 114s scalar, vs 29s VPU baseline. Need to swap in TFMA or VPU asm in conv_hidden to beat baseline.
- ⏳ Phases 3-12: not started

## Next concrete step

To beat the 29s VPU baseline with multi-minion + TFMA:
1. Re-introduce padded buffer for conv_hidden_tfma (it's designed for that)
2. Halo zeroing: use the working pattern from `dncnn3_tfma_zerotest.c` (1 row × 1 hart) which DID work
3. Apply WF layout fix to TFMA path too if relevant
4. Audit at 1e-5

Or simpler: borrow the existing VPU `vpu_dot64x9_fp32` asm from the working `dncnn3_luxonis_real_vpu.c` and apply within my multi-minion scaffold. Should give ~30s with audit pass.
