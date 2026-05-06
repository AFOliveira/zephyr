# Whisper Tiny En on ET-SoC1 — Direction-Finding + Layer Optimization Plan

The plan is split into three stages, in priority order:

1. **Phase 0 — Direction-finding** (5 cheap experiments that ELIMINATE bad paths). This is where the user explicitly wants to invest most thinking; getting Phase 0 right makes Phases 1-5 either easy or impossible.
2. **Phases 1–3 — Op library + end-to-end** (build the kernels Phase 0 chose, validate per-layer, stitch into encoder then decoder).
3. **Phase 4 — Per-layer optimization** (the 100-experiment-per-op loops, where the GOPS gains live).

Whisper Tiny En facts:

| Stage | Nodes | Weights | Dominant ops | Runs per inference |
|---|---:|---:|---|---|
| Encoder | 257 | 35.8 MiB | MatMul ×72, Add ×59, Transpose ×41, LayerNorm ×9, Erf ×6, Conv ×2, Softmax ×4 | 1× per audio clip |
| Decoder | 476 | 184 MiB | MatMul ×121, Slice ×152, Add ×69, Softmax ×48, LayerNorm ×13 | N× (one per output token, N≈30–100 for 30 s audio) |

U-mode region: **16 MB**. Encoder weights overflow by 2.2×; decoder by 11.5×. **Memory is the headline problem** and Phase 0 has to solve it before any kernel work matters.

---

# Phase 0 — Direction-finding experiments (1–2 weeks total)

Five cheap, parallel A/B tests. None require committing to a kernel implementation. The output is a **single decision matrix** that determines what Phases 1–5 even look like.

## 0.1 — Memory feasibility scan (host-only, no silicon)

**Question:** what's the smallest representation of each layer that fits in 16 MB U-mode region?

**Method:** Python script reads each `.onnx` initializer, computes per-layer footprint at FP32 / FP16 / INT8 / INT4 / INT8-per-channel. Outputs a table: layer-by-layer size and a running cumulative max.

**Decision triggers:**
- If FP16 fits the encoder → no quantization needed for encoder, just convert. Skip 0.2 for encoder.
- If INT8 doesn't fit the decoder either → must stream weights from DRAM (not just fit-and-execute). Phase 0.5 becomes load-bearing.
- If INT4 fits the decoder → consider INT4 weights / INT8 activations as a path.

**Cost:** ~1 hour to write, runs in seconds.

## 0.2 — Quantization quality sweep (host-only)

**Question:** what's the lowest precision we can get away with for an audited Whisper inference?

**Method:**
- Sample audio clips (5–10 short LibriSpeech-style samples, or synthetic from Whisper's own outputs).
- Run baseline FP32 ORT → produce reference transcriptions and per-layer activations.
- Quantize the ONNX at FP16, INT8 (per-tensor), INT8 (per-channel), INT4 weights / INT8 activations.
- For each: re-run ORT, measure (a) per-layer activation max-abs vs FP32, (b) WER on the test transcriptions, (c) PSNR on the encoder output (k_cache_cross / v_cache_cross).

**Decision matrix:**

| Variant | Encoder size | Decoder size | Both fit 16 MB? | WER δ | Decision |
|---|---:|---:|---|---:|---|
| FP32 | 36 MB | 184 MB | no / no | 0 | only with streaming |
| FP16 | 18 MB | 92 MB | no / no | ~0 | only with streaming |
| INT8 per-channel | 9 MB | 46 MB | yes / no | ? | encoder fits, decoder streams |
| INT8 per-tensor | 9 MB | 46 MB | yes / no | ? worse | maybe encoder, decoder streams |
| INT4 wgt / INT8 act | 4.5 MB | 23 MB | yes / partial | ? | needs WER < threshold |

**Triggering rule:**
- If WER δ at INT8 ≤ 5 % absolute → INT8 is the path.
- If even INT4 gives WER δ ≤ 10 % → use INT4 weights / INT8 activations on decoder, INT8 everything on encoder.
- If quantization devastates WER even at INT8 → fallback to FP16 + streaming for both stages (slower but accurate).

**Cost:** 1–2 days. Pure host work.

## 0.3 — Silicon matmul throughput on Whisper-shaped work (silicon, ≤ 1 day)

**Question:** what's our actual matmul GOPS on the shapes Whisper uses (which differ from DnCNN's 64×64×64 conv)?

**Why it matters:** the `whisper_transformer_vpu_argbuf.c` proxy gives 2.347 GOPS, but it's a TOK=256, DIM=64, HIDDEN=256 shape — much smaller than real Whisper Tiny matmuls. Real shapes:
- Encoder attention: `(1500, 384) × (384, 1500)` = 1.7 GFLOPs per matmul × 24 attention matmuls per encoder layer × 4 layers = 163 GFLOPs / encoder
- Decoder cross-attention: `(1, 384) × (384, 1500)` = 1.2 MFLOPs (tiny)
- Decoder vocab projection: `(1, 384) × (384, 51864)` = 39 MFLOPs

**Method:** sweep matmul kernel at 5–10 representative shapes from Whisper's actual inventory. Use FUSED9TAP-style fused asm but for matmul (single-row × column-block). Audit each against host NumPy.

**Decision triggers:**
- If small-shape matmul (decoder steps) is dominated by per-call overhead → must batch decoder steps.
- If large-shape matmul scales linearly to proxy's 2.35 GOPS → kernel work is straightforward.
- If memory-bound (more than half time in load) → INT8 mandatory for decoder vocab projection.

**Cost:** 4–6 hours implementation, 1 hour silicon. Reuses the existing experiment_runner.py.

## 0.4 — Audit-gate decision (host-only)

**Question:** how do we decide a Whisper inference "passed audit"?

**Three candidate gates** (use ALL THREE in parallel during dev, settle on one for production):

| Gate | What it measures | Cost to evaluate | When it fails |
|---|---|---|---|
| Per-tensor allclose | every intermediate vs ORT | high (must dump every layer) | quantization wobble, kernel bug |
| End-of-encoder hash | k_cache + v_cache hash vs ORT | low | kernel bug only, masks small quant error |
| WER on transcription | end-to-end transcription vs ground truth | medium (need decoder loop) | semantic regression |

**Recommendation:** during build-up, dump per-layer activations and check tensor-by-tensor (catches kernel bugs early). At "freeze", switch to WER on a held-out audio set (proves the system works).

**Decision triggers:**
- Per-tensor max_abs threshold: 1e-3 for FP16, 2 LSB for INT8, 4 LSB for INT4.
- WER acceptance threshold: ≤ 1.5× the FP32 baseline WER. Below that we're competitive.

**Cost:** 1 day to instrument the audit harness.

## 0.5 — DRAM weight-streaming bandwidth (silicon, ½ day)

**Question:** can we stream weights from DRAM fast enough that compute isn't memory-bound?

**Why it matters:** decoder weights are 184 MiB FP32 / 46 MiB INT8 / 23 MiB INT4. Even INT4 doesn't fit U-mode region. Streaming is mandatory for decoder. Can the silicon stream at the rate compute needs?

**Method:**
- Tiny kernel that does NOTHING but `prefetch.va` weights from a DRAM buffer at a configurable rate, while a parallel hart does fmadd.ps on the same data in cache.
- Vary: prefetch lead-time, prefetch lookback, simultaneous compute load.
- Measure: when does compute stall waiting for weights?

**Decision triggers:**
- If we can sustain ≥ 1 GB/s prefetch + concurrent 2 GOPS compute → streaming is viable.
- If prefetch + compute throughput < 1 GB/s → must INT8 or INT4 to halve/quarter the bandwidth pressure.
- If the launcher's --file_load reload between tokens is a viable alternative → consider per-token weight reload (slower but simpler).

**Cost:** 6 hours implementation, 2 hours silicon.

## Phase 0 output: the decision matrix

After the 5 experiments, fill in this single table; everything downstream derives from it:

```
Encoder precision: ___ (FP32 / FP16 / INT8 / INT4)
Encoder weight strategy: ___ (fit / stream)
Decoder precision: ___ (FP32 / FP16 / INT8 / INT4)
Decoder weight strategy: ___ (fit / stream / reload)
Audit gate (production): ___ (per-tensor / hash / WER)
Audit gate (dev): per-tensor allclose
Order: ___ (encoder-first / decoder-first / parallel)
```

**The right default given what we know about ET-SoC1:** encoder INT8, weights fit + stream the larger MatMul weight blocks; decoder INT8, weights stream layer-by-layer; per-tensor dev / WER prod; encoder-first.

But run Phase 0 to confirm — don't trust this guess.

---

# Phase 1 — Build the op library (after Phase 0 commits to a precision)

Each op gets its own micro-iteration loop. Build → audit per-op vs ORT (or vs Python NumPy reference for non-ONNX ops) → tune. Don't optimize yet — just make each op produce correct output at the chosen precision.

| Op | Code path | Validation reference | Hardest part |
|---|---|---|---|
| **MatMul** | extend existing FUSED9TAP-style fused asm | NumPy `@` on int8 inputs with FP32 dequant | non-square matmuls, small-K cases |
| **LayerNorm** | new — fused asm: mean → var → normalize → scale + bias | NumPy/PyTorch LayerNorm | numerical stability of variance |
| **Softmax** | new — fused asm: max → exp → sum → divide | scipy.special.softmax | exp() approximation, num stability |
| **Erf** (GELU) | new — polynomial approximation in fmadd.ps | NumPy erf | accuracy of polynomial expansion |
| **Conv1D** (encoder mel→hidden) | adapt FUSED9TAP for stride/dilation | NumPy 1D conv | kernel size 3, stride 2, padding |
| **Add / Mul / Sub / Div** | trivial elementwise asm | NumPy | nothing |
| **Transpose / Reshape** | host-side memcpy with permutation | NumPy | layout matching downstream consumer |
| **Slice / Concat / Split** | host-side pointer arithmetic + memcpy | NumPy | KV cache index handling |
| **Gather** | scalar indexed load | NumPy | embedding table layout |

**Per-op iteration loop** (mirror the 100-experiment DnCNN loop but smaller):

For each op:
1. Write reference C kernel + Python NumPy ground truth.
2. Build at one shape, run on silicon, audit (max_abs vs reference).
3. Sweep 5–10 compile flag variants. Triage.
4. Add fused asm if op is hot. Triage.
5. Freeze at first variant within 5 % of the per-op theoretical peak.

**Cost per op:** ½–2 days. Total Phase 1: ~2–3 weeks.

---

# Phase 2 — Encoder end-to-end (smaller, simpler — do first)

Once each op passes its per-op audit, stitch them into the encoder graph.

## 2.1 — Reference graph executor (Python)

Write `tools/whisper_encoder_pyref.py` that walks the ONNX nodes in topological order and dispatches each to a NumPy implementation. Validates against ORT for one input clip. **Gate:** max_abs ≤ 1e-5 vs ORT FP32 (per-tensor).

## 2.2 — Silicon graph executor (C, no optimization)

Same node walk in C, dispatching each conv/matmul/etc. to its silicon kernel. Write per-tensor dumps to a buffer the host can read after kernel completion. **Gate:** silicon allclose to Python ref at every intermediate, max_abs ≤ chosen-precision threshold.

## 2.3 — Audit harness

`tools/audit_whisper_encoder.py` reads silicon dumps, compares each tensor to ORT FP32 with the Phase 0.4 gate. Fails fast on the first divergence — that's where the bug is.

## 2.4 — Iteration loop (10 batches × 10 experiments)

| Batch | Focus |
|---|---|
| 1 | end-to-end runtime measurement; identify which layers eat the most time |
| 2 | swap top-3 hot layers to fused-asm kernels |
| 3 | activation memory layout — NHWC vs NCHW for matmul vs conv |
| 4 | weight pre-arrangement (pack/transpose at static link time so kernel is straight-line) |
| 5 | bench_barrier reduction (skip per-op barrier, hold one per layer instead) |
| 6 | KV cache write coalescing (writing 4×6×64×1500 cache must be cache-line aligned) |
| 7 | overlap fetch + compute (start loading next layer's weights while current runs) |
| 8 | eliminate redundant transposes by adjusting downstream kernel input order |
| 9 | hart partitioning: row-stripe vs column-stripe for big matmuls |
| 10 | validation: 10 runs of best variant, noise floor + WER check |

**Encoder success criteria:**
- WER δ vs ORT FP32 ≤ 1.5×
- Wall time ≤ 5 seconds per 30-second audio clip (1 shire, 16 harts)
- Audit passes per Phase 0.4 gate

---

# Phase 3 — Decoder per-token (after encoder works)

The decoder is autoregressive and uses growing KV caches. Different challenges from encoder.

## 3.1 — KV cache plumbing

K/V caches grow by 1 token per decode step. Need either:
- **Static allocation** at max length (224 tokens for Whisper Tiny En = 4×6×64×224 + 4×6×224×64 = 691 KB self KV cache; cross KV cache from encoder is fixed 4.6 MB).
- **Append-on-step**, treating cache as a grown buffer.

Static allocation is simpler — pre-allocate 224-token K/V buffers, mask out unused positions in attention. Pay a small compute tax for masked-out positions in early steps.

## 3.2 — One-token decode (audit gate)

Run one decode step: `(x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self)` → `(logits, k_cache_self', v_cache_self')`. Audit logits + updated K/V vs ORT.

## 3.3 — Multi-token loop

Host orchestrates: kernel runs decode_step(token=N), reads logits, applies argmax, feeds back as token=N+1. Loop until end-of-sequence token.

## 3.4 — Iteration loop (10 batches × 10)

| Batch | Focus |
|---|---|
| 1 | one-step latency baseline; profile where time goes (attention vs FFN vs vocab projection) |
| 2 | vocab-projection optimization — biggest matmul in decoder (384×51864) |
| 3 | KV-cache attention — `Slice`-heavy; can we fuse Slice + MatMul + Softmax? |
| 4 | mask the K/V cache padding cleanly (avoid filling with -inf in softmax) |
| 5 | per-layer weight reload from DRAM (since decoder weights don't fit) — minimize reload count |
| 6 | overlap weight reload with compute (double-buffer the layer weights) |
| 7 | INT4 weights for FFN if Phase 0.2 said it's safe (FFN is the largest weight chunk per layer) |
| 8 | early-exit on confident tokens (skip remaining layers if argmax is already winner) — risky for accuracy |
| 9 | speculative decoding — run two tokens ahead, validate the second against full path |
| 10 | end-to-end transcription test on 5 audio clips, WER report |

**Decoder success criteria:**
- WER δ vs ORT FP32 ≤ 1.5× (same as encoder)
- Per-token latency ≤ 100 ms (so 30 tokens / clip = ≤ 3 s)
- End-to-end transcription latency (encoder + decoder) ≤ 8 s per 30-s clip

---

# Phase 4 — Per-layer optimization (the GOPS deep-dive)

This is the layer-by-layer optimization the user explicitly asked about. Once Phases 2 and 3 produce a *correct* end-to-end encoder + decoder, run a per-layer 100-experiment loop on the 4 dominant op kinds.

## 4.A — MatMul optimization (200 experiments)

Whisper has 72 + 121 = **193 matmuls** across encoder + decoder. They cluster into 4 shape families:
- `(seq, dim) × (dim, dim)` — KQV projections, attention output projection, FFN up/down
- `(seq, dim) × (dim, seq)` — attention QK^T (scaled dot product)
- `(1, dim) × (dim, vocab)` — decoder vocab projection (logits)
- `(seq, dim) × (dim, hidden)` — FFN intermediate

| Batch | Focus | Best knobs to try |
|---|---|---|
| 1 | tile size sweep at the dominant shape | M-tile, N-tile, K-tile each ∈ {16, 32, 64, 128} |
| 2 | **fused multi-K** — accumulate K across multiple inner-loop chunks before storing | matches the FUSED9TAP win pattern from DnCNN |
| 3 | OC2 vs OC4 vs OC8 for non-square matmuls | the DnCNN audit showed OC4 regressed; verify for non-square |
| 4 | weight prefetch lead-time | sweep prefetch distance ∈ {1, 2, 4, 8} chunks ahead |
| 5 | activation prefetch | the read-window pattern that won DnCNN |
| 6 | INT8 packed-integer fmadd if Phase 0 chose INT8 | new asm variants |
| 7 | per-matmul scheduling — dispatch K-loop differently for small vs large K | small-K: cache the full B; large-K: stream B |
| 8 | hart partition: split M (output rows) vs split N (output cols) vs split K (parallel reduce) | K-split needs a final reduce barrier |
| 9 | cache strategy: keep B static across multiple A presentations (works for KQV with shared input) | reuse pattern from DnCNN OC2 logic |
| 10 | combined best of all above | 10 candidate combinations |

**Direction-finding for each batch:** start at the proxy's 2.347 GOPS as baseline. Goal: get to 3+ GOPS audited (assuming Phase 0 chose INT8 packed-integer for the headline gain).

## 4.B — LayerNorm optimization (50 experiments)

LayerNorm runs 9 + 13 = 22 times per inference. Per-call cost is small but call count is high.

| Batch | Focus |
|---|---|
| 1 | naive 3-pass (mean → var → normalize) vs 2-pass (Welford / Chan-style online) |
| 2 | vector reduction strategies for mean: lane-shuffle vs scalar |
| 3 | fused with subsequent op (LayerNorm → MatMul, LayerNorm → Add+Erf+...) |
| 4 | precision tradeoff: keep variance in FP32 even when activations are INT8 |
| 5 | hart partitioning for the per-token normalize | split tokens across harts; per-token is independent |

## 4.C — Softmax optimization (60 experiments — decoder-heavy)

48 softmaxes per decode step (in the K/V attention). This adds up.

| Batch | Focus |
|---|---|
| 1 | exp() approximation: lookup table vs polynomial vs `exp2(x · log2(e))` decomposition |
| 2 | numerical stability: subtract max before exp |
| 3 | fused with attention scale and mask |
| 4 | online softmax (avoid the full pass) — Flash-Attention-style |
| 5 | INT8 softmax variant if precision allows |
| 6 | hart partitioning across the seq dim of attention scores |

## 4.D — Erf / GELU optimization (30 experiments)

Six instances per encoder layer. Polynomial approximation is the standard.

| Batch | Focus |
|---|---|
| 1 | polynomial degree sweep (2, 3, 5) — accuracy vs speed |
| 2 | fused with bias-add and dropout (no-op in inference) |
| 3 | tanh-based vs erf-based GELU formulation |

---

# Phase 5 — End-to-end optimization (50 experiments)

After per-layer is tuned, optimize the layer-to-layer flow.

| Batch | Focus |
|---|---|
| 1 | overlap encoder-output read with decoder-input setup |
| 2 | KV cache placement: scratchpad vs DRAM vs L2 |
| 3 | speculative load: while decoder layer N runs, prefetch layer N+1 weights |
| 4 | shire allocation: encoder on shire 0, decoder on shire 1 (if achievable) |
| 5 | quantization-aware retraining (host-side, retrain a tiny adapter to recover WER) — only if WER δ > 1.5× |

---

# How to PICK A DIRECTION at each phase boundary

The plan above looks linear, but every phase boundary is a fork. Here's how to decide:

**After Phase 0:** look at the decision matrix. If encoder fits in U-mode region at the chosen precision but decoder doesn't, the next 2 weeks of work go into weight-streaming infra (Phase 0.5 follow-on), not into op kernels. Decide based on what's actually broken.

**After Phase 1:** if any op's per-op audit fails, debug it; do NOT proceed to Phase 2 until every op passes. A broken Softmax kernel will silently corrupt the entire decoder.

**After Phase 2 (encoder works):** look at where the time goes. If matmul is 70 % of encoder time, Phase 4.A is mandatory. If LayerNorm is 30 % (high call count), Phase 4.B matters more than people expect.

**After Phase 3 (decoder works):** the per-token latency budget is tight. If decoder dominates total latency (likely, since it runs N times), invest in 4.A vocab-projection + 4.C softmax. If encoder dominates (rare), the 4.A matmul work transfers down.

**After each Phase 4 sub-phase:** apply the same rules as the DnCNN 100-experiment loop — pivot if 2 batches show no improvement, freeze when no gain for 20 experiments.

---

# Iteration framework reuse

Same orchestrator pattern as the DnCNN 100-experiment loop:
- `experiment_runner.py` extended to accept a `--model whisper_encoder | whisper_decoder` flag
- Per-track master TSV (`audit_master_whisper_encoder.tsv`, `audit_master_whisper_decoder.tsv`)
- Spec JSON per batch
- Triage → spec → execute → audit per cycle
- Pivot rule: 2 consecutive flat batches → switch focus
- Freeze rule: 20 experiments without improvement → tag and move on

---

# What to NOT do

These are tempting but should be skipped or deferred:

1. **Don't write the decoder kernel before the encoder works end-to-end.** Encoder is smaller, simpler, and validates the op library. Decoder reuses everything.
2. **Don't optimize a layer before its naive version passes audit.** A fast wrong answer is worse than a slow right answer; quantization makes it especially easy to hide bugs in slop.
3. **Don't quantize without measuring WER.** Per-tensor allclose can drift quietly while still passing tolerance — the integrated transcription either works or it doesn't.
4. **Don't try Flash Attention or speculative decoding before the basic decoder works.** Those are end-game optimizations, not foundation work.
5. **Don't port Conv2D for the encoder's mel preprocessor.** It's 2 layers, shape (80, 3000) → (384, 1500). Just precompute on the host and pass the (384, 1500) tensor as input. Saves kernel work.
6. **Don't try to fit both encoder + decoder in U-mode region simultaneously.** Run encoder, dump K/V, swap kernels (different ELF), run decoder. The launcher already supports per-call ELF load.

---

# Estimated total cost

| Phase | Effort | Silicon time |
|---|---|---|
| 0 — direction-finding | 1–2 weeks (mostly host-side) | ~½ day |
| 1 — op library | 2–3 weeks | ~2 days |
| 2 — encoder e2e + audit | 1 week | ~1 day |
| 3 — decoder e2e + audit | 1–2 weeks | ~2 days |
| 4 — per-layer optimization | 3–4 weeks | ~5 days |
| 5 — end-to-end optimization | 1 week | ~1 day |
| **total** | **~10 weeks of focused work** | **~12 silicon days** |

After Whisper, the kernel library + audit harness applies directly to Depth Anything V2 (transformer-shaped, similar ops) and to YOLO once a real ONNX is sourced. Foundation Stereo's full graph stays out of scope.

---

# Verification — the audit discipline (deep dive)

Verification on Whisper is harder than on DnCNN3 because the model is a graph
of 257 + 476 = 733 nodes producing intermediate tensors of varying shape,
running across two stages, with KV state that mutates each decode step. A
single end-to-end "max_abs ≤ 1e-5" check is insufficient — it tells you nothing
about *where* a divergence started.

The verification harness has to operate at **seven distinct levels**, each
catching a different class of bug. Skipping any one of them lets bugs hide.

## V1 — Per-operator kernel verification (Phase 1)

**Granularity:** one op kernel against a NumPy / PyTorch reference for the same shape.
**When:** before the op is allowed into the graph executor.
**Reference:** synthetic random inputs (fixed seed) feeding both the silicon kernel and a `numpy_reference.py` that computes the same op the same way.
**Audit gate:** max_abs ≤ chosen-precision threshold (1e-5 FP32 / 1e-3 FP16 / 2 LSB INT8 / 4 LSB INT4).

**Trap to avoid:** generating the random inputs uniformly in [-1, 1]. Real Whisper activations are tightly distributed (LayerNorm-ed) but with occasional outliers in attention scores. Generate test inputs with the **distribution that real model intermediates have** — easiest way is to dump real activations from one ORT run and use *those* as test inputs. This catches numerical-stability bugs (e.g., softmax on very large attention scores) that random-uniform inputs hide.

**Cost per op:** ½ day (write reference + harness + 5 shape variants).
**Output:** `verify_ops/op_<name>_audit.tsv` with one row per (shape, precision) tested.

## V2 — Single-layer numerical verification (between Phase 1 and 2)

**Granularity:** one transformer layer (LayerNorm → attention → LayerNorm → FFN) against ORT.
**When:** after stitching ops into a complete layer, before stitching layers into the encoder.
**Reference:** ORT executing just that layer (use `onnx.utils.extract_model` to slice out a single-layer subgraph).
**Audit gate:** max_abs ≤ chosen-precision threshold per intermediate tensor.

**Why this catches what V1 doesn't:** V1 verifies an op in isolation. V2 verifies that op-to-op data layout/transpose/dtype handoffs are correct. The most common Whisper port bug is a transpose-axis mismatch between MatMul output and the next LayerNorm's expected input shape — V1 won't catch it because each op is correct individually.

**Cost:** 1 day to write the slice-and-compare harness. Reusable for every layer of every model.

## V3 — Per-tensor graph verification (Phase 2)

**Granularity:** every intermediate tensor in the encoder graph.
**When:** during Phase 2.2 (silicon graph executor build-up).
**Method:**
1. Modify the silicon graph executor to dump each intermediate tensor to a known offset in the device buffer.
2. Run the same input through ORT FP32, dump every intermediate in matching layout.
3. Compare tensor-by-tensor.
**Audit gate:** max_abs per tensor ≤ chosen-precision threshold. Report the FIRST tensor that diverges, not just the final one.

**Trap to avoid:** the silicon dump buffer is 16 MB. The encoder produces ~4.6 MB of total intermediates per inference; dumping all of them is borderline. Solution: dump every-Nth tensor in a sweep, or selectively dump only the layer-boundary tensors and rely on V2 for intra-layer checks.

**Cost:** 2 days to instrument both executors and write the diff harness.

## V4 — Stage-output hash verification (Phase 2 → 3 handoff)

**Granularity:** encoder produces `k_cache_cross + v_cache_cross`. Hash both.
**When:** after Phase 2 freezes, every encoder change re-runs this gate.
**Method:** silicon kernel computes its own hash of the K/V output and compares to a hash of the linked ORT-reference K/V (same pattern as DnCNN's `output_hash == reference_hash`). Recorded in the slot summary.
**Audit gate:** allclose at chosen-precision threshold; bit-exact preferred but unlikely under FP-reorder.

**Why this matters even with V3:** V3 is dev-time, expensive. V4 is the production audit gate that runs every time an encoder variant is built, with zero per-tensor dumping overhead. Same role as DnCNN's `verify_dncnn3_package.py`.

## V5 — Per-token decoder verification (Phase 3)

**Granularity:** one decode step. Inputs are (token, KV caches). Outputs are (logits, updated KV caches).
**When:** during Phase 3.2.
**Method:** drive the silicon decoder with the same (token, KV caches) that ORT uses; compare logits vector + updated K/V slabs.
**Audit gate:** max_abs(logits) ≤ chosen-precision threshold. Plus check that the **argmax position** of the logits matches ORT — even if the values diverge slightly, if the argmax (selected token) matches, the transcription is unaffected.

**Why argmax matters more than max_abs:** Whisper greedy-decode picks the token with highest logit. If the silicon's max-abs error is 0.01 but the top-1 logit is 5.3 vs 5.4 ORT, the same token is still selected and the transcription is identical. **The audit gate that matters for STT is "does the selected token match ORT".**

**Cost:** 1 day. Reuse V3 harness with one decoder step instead of full encoder.

## V6 — Multi-token transcription verification (Phase 3.3)

**Granularity:** full decoder loop on a real audio clip. Compare transcription text.
**When:** before Phase 3 freeze.
**Method:** run silicon decoder loop on encoder output, collect tokens until end-of-sequence, decode tokens to text via the Whisper tokenizer (host-side). Compare to ORT's transcription on the same input.
**Audit gate:**
- **Best:** transcription text identical (string-equal). This is the most rigorous gate but fragile to small per-token divergences.
- **Acceptable:** WER between silicon and ORT transcriptions on a 50–100 clip set ≤ 1.5× the ORT vs ground-truth WER.

**Trap to avoid:** picking a too-easy or too-hard test set. LibriSpeech test-clean is the standard; pick 50 random clips of 5–15 s each. Running 50 transcriptions takes silicon time (~5 minutes per clip × 50 = 4 hours), so do this only at freeze, not per batch.

## V7 — Cross-precision PSNR verification (only if quantizing)

**Granularity:** silicon-dequant output vs FP32-ORT output.
**When:** during all of Phase 0.2 and Phase 4 if INT8 / INT4 chosen.
**Method:** silicon produces INT8 output → host dequantizes per the kernel's scale → compare to FP32 ORT output via PSNR.
**Audit gate:** PSNR ≥ 35 dB on the encoder K/V output and on per-token logits.

**Why PSNR not max_abs:** quantization is allowed to introduce error; max_abs measures worst-case-pixel divergence and gets dominated by outlier activations. PSNR captures average-case quality and correlates better with WER.

## How the seven levels stack

```
audio input
    │
    ▼ V7 (cross-precision PSNR if INT8) ◀──┐
encoder graph                             │
    │                                     │
    ▼ V3 (per-tensor) ──┐                 │
    ▼ V2 (per-layer) ───┤                 │
    ▼ V1 (per-op) ──────┘                 │
    │                                     │
    ▼ V4 (encoder hash gate)              │
k_cache, v_cache                          │
    │                                     │
    ▼ V5 (per-token) ◀──── decoder kernel │
    ▼ V6 (transcription)                  │
    │                                     │
    ▼ ────────────────────────────────────┘
final transcription
```

**Run cadence:**
- V1 — every op kernel build, automated in the build script
- V2 — every layer build
- V3 — first build of every executor variant (expensive, dev-only)
- V4 — every encoder variant in the iteration loop
- V5 — every decoder variant in the iteration loop
- V6 — at every phase freeze, with WER on 50 clips
- V7 — every variant if quantizing

## Verification infrastructure to build (concrete)

### V1: `tools/verify_op.py`
- args: `--op layernorm --shape "1,1500,384" --precision int8 --test_inputs distribution.json`
- runs: synthetic input → silicon kernel + numpy ref → max_abs report
- output: append to `verify_ops/op_<name>_<precision>_audit.tsv`

### V2: `tools/verify_layer.py`
- args: `--layer encoder.layer.0 --onnx whisper_tiny_en_encoder.onnx --precision fp16`
- uses `onnx.utils.extract_model` to slice subgraph → ORT vs silicon
- output: per-intermediate max_abs

### V3: `tools/verify_full_graph.py`
- args: `--stage encoder --dump_every_tensor`
- runs ORT + silicon, compares all intermediates
- output: first divergence + heat-map of error per layer

### V4: encoder kernel computes its own hash (already the DnCNN pattern)
### V5: same harness as V3 but for one decoder step
### V6: `tools/verify_transcription.py`
- runs silicon end-to-end on N audio clips → reports per-clip WER vs ORT and vs ground-truth
- output: `verify_transcription_<date>.json` with WER table

### V7: built into V4/V5 — additional PSNR column

## Audit master TSVs (one per stage)

`audit_master_whisper_encoder.tsv` columns:
```
exp_id, batch, variant, precision, weight_strategy,
v1_pass, v2_pass, v3_first_diverge_layer, v4_max_abs, v4_hash_match,
v7_psnr,
encoder_wait_s, encoder_gops,
flags, macros, dump, log, build_at, run_at
```

`audit_master_whisper_decoder.tsv` adds:
```
v5_argmax_match_rate, v5_logit_max_abs, v6_wer_silicon, v6_wer_ort,
per_token_latency_s, total_decode_wait_s
```

## What "verified" means at each phase boundary

| Phase | Verification gate to clear before moving on |
|---|---|
| 0 | decision matrix filled in; v0 thresholds set per precision |
| 1 | every op passes V1 at all shapes used in the encoder/decoder |
| 2 | encoder passes V2 (per-layer) and V3 (per-tensor) at the chosen precision |
| 2 freeze | V4 (encoder hash) reproducibly passes 10 runs |
| 3 | decoder passes V5 (per-token) for at least 30 consecutive tokens |
| 3 freeze | V6 (transcription) on 50 clips, silicon WER ≤ 1.5× ORT WER |
| 4 | per-op-batch winners maintain V1/V4/V5; no regression in V6 |
| 5 | V6 on 100 clips reproducible across 5 runs |

## What goes wrong without this discipline

- **No V1 + jumping to V3:** silicon shows wrong output at layer 7. You don't know if it's the LayerNorm kernel, the MatMul kernel, the Softmax kernel, or the layout converter between them. Spend 3 days bisecting that you wouldn't have spent if V1 caught it on day 1.
- **No V5 + relying on V6 only:** transcription is wrong. You can't tell if decoder kernel is broken, the KV cache is being corrupted, or the tokenizer is feeding the wrong vocab. Spend a week reproducing.
- **No V7 + quantizing:** silicon allclose at INT8 LSB threshold passes V4, but the rounding error compounds across 4 attention layers and the final transcription is garbage. WER δ shows up only at V6, after a day of silicon iteration.
- **No argmax-match check (part of V5):** logits diverge by 0.05 max-abs. V5 passes max_abs gate. But the divergence flips the top-1 token at 3 of 30 positions. Transcription has 3 wrong words. WER tanks. Catch this with the argmax-match counter, not max_abs.

The DnCNN3 100-experiment loop succeeded because every variant passed a single, machine-checkable audit gate (`output_hash == reference_hash` or `max_abs ≤ 1e-5`). For Whisper, no single gate is sufficient — but the seven-level structure above is a generalization of the same pattern, applied at the right granularity for each phase.
