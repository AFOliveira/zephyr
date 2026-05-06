# Whisper Full-Silicon Optimization Plan

Date: 2026-05-06

Scope: ET-SoC1 silicon on `esperanto-soc6`, staged only under
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2`.

Current audited baseline:

- Path: log-mel features -> encoder cross-cache -> decoder token IDs on
  ET-SoC1.
- Correctness: silicon token IDs exactly match the host ONNXRuntime token IDs.
- Generated tokens: 23.
- Encoder wait: 148.186 s.
- Decoder wait: 79.7042 s.
- Total kernel wait: 227.8902 s.
- Full generated-token rate: 0.100926 token/s.
- Decoder-only generated-token rate: 0.288567 token/s.
- Encoder PMCs: hpm3 88908887555, hpm4 64252377643,
  hpm5 64568437733, hpm6 17623484414, hpm7 17936796095,
  hpm8 213.
- Decoder PMCs: hpm3 47821019324, hpm4 8048391777, hpm5 1244,
  hpm6 910760327, hpm7 1500998336, hpm8 204.

Important hardware/software constraints:

- One shire gives 8 minions and 16 harts.
- The practical scheduler split is even harts / thread 0 for VPU or tensor
  compute and odd harts / thread 1 for helper work.
- Existing `txfma` and gp-sdk generated matmul examples show tensor compute
  issued from thread 0. Treat thread 1 as helper/prefetch/scalar work unless a
  specific tensor test proves otherwise.
- L1D is non-coherent. Cross-hart records and partials need 64-byte padding and
  explicit cache operations.
- ET tensor functionality includes FMA32, FMA16A32, IMA8A32, QUANT, REDUCE, and
  tensor load/store/wait. Opcode mapping for non-FP32 TFMA must be verified
  with a standalone audited kernel before use in Whisper.

## Optimization Inventory

### 1. Replace scalar decoder matvecs with multi-hart VPU FP32

Why: the current decoder is mostly scalar. It spends 79.7042 s for only
23 generated tokens, and hpm5 is nearly idle, which means thread 1 is not
helping materially yet.

Targets:

- Self-attention QKV: `384 x 1152` per layer per token.
- Self-attention output: `384 x 384`.
- Cross-attention query/output: `384 x 384`.
- MLP FC1/FC2: `384 x 1536` and `1536 x 384`.
- Final vocab projection: `384 x 51864`, fused with argmax.

First implementation:

- Keep the existing raw-INT8 weights and FP32 output math for audit stability.
- Partition output columns across the 8 thread-0 harts.
- Use thread-1 harts for layernorm, softmax prep, cache prefetch, and local
  reductions.
- Preserve exact greedy tokens as the gate.

Expected risk: low to medium. The VPU FP32 logits, QKV, MLP, score, and value
microbenchmarks already pass against ONNXRuntime tensors.

### 2. Convert the encoder matvec loops to resident VPU FP32 matmul tiles

Why: encoder wait is the largest single bucket at 148.186 s. The current
encoder does row-partitioned scalar loops over raw-INT8 weights.

Targets:

- Conv2 and input projection where profitable.
- Encoder QKV for all 4 blocks.
- Encoder attention score/value.
- Encoder MLP FC1/FC2.
- Final encoder post-layernorm and cross-attention K/V cache projection.

Known prior performance:

- `1500x1536x384`, 16 harts: 1.660121 s, 1.066 GOPS, audit pass.
- `1500x384x1152`, 16 harts: 1.710595 s, 0.776 GOPS, audit pass.
- `1500x64x1500`, 16 harts: 0.947814 s, 0.304 GOPS, audit pass.
- `1500x1504x64`, 16 harts: 0.270395 s, 1.068 GOPS, audit pass.

First implementation:

- Start with encoder MLP FC1/FC2 because those are the best-audited VPU shapes.
- Keep the host ONNX cross-cache oracle.
- Replace one stage at a time inside `whisper_resident_encoder_argbuf.c`.
- After each replacement, compare cross K/V and final generated tokens.

Expected risk: medium. Matmul kernels are audited, but resident integration has
cache, layout, and staging risk.

### 3. Use TFMA for the large matmuls

Why: TFMA should be the right hardware path for matrix/tensor blocks. The gp-sdk
`txfma` test and generated matmul code already show the flow:
`tensor_load`, `tensor_wait`, `tensor_fma`, `tensor_store`, then coalescing
buffer drain.

Candidate kernels:

- Encoder QKV: `1500 x 384 x 1152`.
- Encoder MLP FC1/FC2.
- Decoder vocab projection / argmax.
- Decoder MLP FC1/FC2.

Implementation steps:

1. Build an audited standalone TFMA FP32 kernel for one already-audited shape.
2. Compare against the existing VPU FP32 kernel and ONNXRuntime output.
3. Add double-buffered tensor loads.
4. Add thread-1 activation/weight/code prefetch using the gp-sdk generated
   matmul split as the template.
5. Integrate only after the standalone kernel beats or matches VPU FP32.

Expected risk: medium to high. The hardware path is promising, but tile layout,
TenB packing, tensor waits, and cache draining need exact handling.

### 4. Move from raw-INT8-weight FP32 math to real INT8 tensor compute

Why: the current native executor stores raw INT8 weights but still dequantizes
inside scalar FP32 multiply loops. That saves memory, not compute. IMA8A32 is
the path that can change throughput.

Plan:

- Keep FP16/FP32 activations first and use INT8 weights only where the
  instruction path allows it.
- Then test INT8 activations on isolated layers.
- Keep cross-attention K/V FP16 until token stability is proven; previous audits
  showed FP16 cross-cache passes with small error.
- Use TensorQuant for int32->fp32/dequant, ReLU/GELU-adjacent transforms where
  possible, saturation, and packing.

Audit gates:

- Layer output vs ONNXRuntime or raw-INT8 host oracle within the layer-specific
  tolerance.
- Full token sequence exact match.
- If activation quantization is enabled, also track confidence margin between
  top-1 and top-2 logits.

Expected risk: high. This is the biggest potential win, but it changes the
math and may affect token choices.

### 5. Put thread 1 to work deliberately

Thread 1 should not be treated as another main tensor compute lane yet. Use it
for:

- Weight prefetch for the next tile.
- Activation prefetch for the next tile.
- Instruction/code prefetch for hot kernels.
- LayerNorm partial sums and variance reductions.
- Softmax max/sum/exponent passes.
- GELU/transcendental work, if VPU/trans cannot cover it cheaply.
- Cache eviction/drain work.
- Argmax local reductions.

Implementation rules:

- Use `0x5555` as the thread-0 compute mask and `0xaaaa` as the thread-1 helper
  mask for one shire.
- Keep thread-0 and thread-1 mailboxes on separate 64-byte cache lines.
- Use explicit barriers and cache operations at each ownership handoff.
- Avoid TensorReduce until a dedicated test proves it is safe in the intended
  t0/t1 schedule, because the ET docs list an ET-SoC1 TensorReduce/VPU
  deadlock erratum.

### 6. Fuse graph boundaries

Priority fusions:

- LayerNorm + QKV input consumption, to avoid writing the normalized row to
  DRAM before matmul.
- Bias add inside matmul tile writeback.
- MLP FC1 + GELU + FC2 tile streaming where memory permits.
- Attention score scale + softmax in one score buffer pass.
- Final LayerNorm + vocab projection + argmax.
- Cross-cache projection + FP16 pack in the encoder.

Expected risk: medium. Fusions reduce memory traffic but make audit failures
harder to localize, so they should happen after the unfused kernel passes.

### 7. Fix attention score layout

Why: the audited VPU score shape `1500x64x1500` only reaches about 0.304 GOPS,
much weaker than MLP/value shapes.

Attempts:

- Column blocking instead of only row blocking.
- Store K as head-major/position-contiguous for the exact access pattern.
- Prepack K/V once during encoder cross-cache generation so decoder
  cross-attention reads contiguous head tiles.
- Use TFMA for score matmul if its layout handles this shape better.
- For decoder cross-attention, keep K/V resident in FP16 and tile over source
  positions.

Expected risk: medium. This is likely required for good encoder performance.

### 8. Reduce cache operations and DRAM traffic

Current encoder flushes large regions after each stage for correctness. That is
safe but expensive.

Attempts:

- Replace whole-hidden/qkv evicts with per-partition evicts.
- Keep single-hart-owned row ranges until the next true cross-hart consumer.
- Use scratchpad for tile-local qkv, scores, and MLP intermediates.
- Align all hot buffers and manifests to 64 bytes.
- Track hpm6 L2 misses and hpm7 icache requests per variant.

Expected risk: medium. This can produce silent stale-data bugs, so every change
needs a token/cross-cache audit.

### 9. Collapse encoder and decoder into one resident kernel

Why: kernel launch overhead is not the main bottleneck anymore, but two kernels
still force extra staging and dump/reload of encoder cross-cache state.

Plan:

- Keep the two-kernel path as the correctness baseline.
- Add a single resident kernel mode that runs encoder, then decoder steps.
- Emit the same final status block and tokens as the current two-kernel path.
- Only use the single-kernel mode for performance once it exactly matches the
  two-kernel path.

Expected risk: low to medium after VPU/TFMA kernels are stable.

### 10. Reuse gp-sdk dnn-library/autogen where practical

The generated gp-sdk matmul code already solves several things we need:

- Thread-0 compute.
- Thread-1 activation prefetch.
- Thread-1 weight prefetch.
- Instruction prefetch.
- Tensor load/FMA/store sequencing.
- Cooperative masks and tile layouts.

Possible routes:

- Adapt the generated matmul kernel style into the current C runner.
- Generate fixed Whisper shapes and call them from the resident executor.
- Link a small C++ gp-sdk kernel if the build path is not too heavy.

Expected risk: medium. This may be faster than hand-writing TFMA, but build and
ABI integration can cost time.

## Execution Order

1. Add timers/PMCs around encoder sub-stages and decoder sub-stages, not only
   around whole kernels.
2. Multi-hart decoder VPU FP32 for final vocab projection and argmax.
3. Multi-hart decoder VPU FP32 for MLP FC1/FC2.
4. Thread-1 helper path for decoder LayerNorm/softmax/prefetch.
5. Encoder resident VPU FP32 for MLP FC1/FC2.
6. Encoder resident VPU FP32 for QKV and cross-cache projections.
7. Attention score layout sweep.
8. Standalone TFMA FP32 kernel for one MLP shape.
9. TFMA resident integration if it beats VPU.
10. IMA8A32/INT8 standalone kernel.
11. INT8 resident integration if token audit remains stable.
12. Single resident encoder+decoder kernel.

## Metrics To Record For Every Run

- Exact source/kernel variant and build defines.
- Generated tokens and text.
- Token exact-match vs ONNXRuntime.
- Layer or cross-cache max_abs/mean_abs when applicable.
- Encoder wait, decoder wait, total wait.
- Generated token/s total and decoder-only token/s.
- hpmcounter3..8 for whole kernel and, after instrumentation, per-stage.
- Remote work directory and local report directory.

## First Concrete Bet

The best first bet is not INT8. It is replacing the scalar decoder and encoder
matvecs with the VPU FP32 matmul path we already audited. That keeps the exact
model behavior while attacking the largest obvious waste: scalar multiply loops
over matrix shapes that the VPU already handles.

The second bet is TFMA FP32 for MLP/QKV shapes. The third bet is IMA8A32 INT8
compute after correctness and opcode calibration are proven.
