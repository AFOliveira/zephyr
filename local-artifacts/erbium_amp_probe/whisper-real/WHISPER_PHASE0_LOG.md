# Whisper Tiny En Phase 0 Log

Board host: `esperanto-soc6`

Remote root:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real`

## Artifacts

- `phase0/whisper_memory_scan.md`
- `phase0/whisper_memory_scan.json`
- `phase0/whisper_matmul_inventory.md`
- `phase0/whisper_matmul_inventory.json`
- `phase0/whisper_silicon_phase0.tsv`
- `phase0/whisper_real_audit.tsv`
- `whisper_encoder_mlp_tile_vpu_argbuf.c`
- `whisper_decoder_vocab_tile_vpu_argbuf.c`
- `whisper_mlp0_real_audit_vpu_argbuf.c`

## Phase 0.1 Memory Scan

| Model | FP32 weights MiB | FP16 | INT8 | INT4 | I/O FP32 MiB | I/O FP16 | I/O INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder | 35.818 | 17.909 | 8.955 | 4.477 | 18.494 | 9.247 | 4.623 |
| Decoder | 184.062 | 92.031 | 46.015 | 23.008 | 23.026 | 11.513 | 5.757 |

Implication: encoder INT8 weights fit in the 16 MiB region, but FP32 encoder
I/O does not.  The cross-attention K/V cache has to be stored as FP16/INT8 or
streamed.  Decoder weights do not fit even at INT4, so full decoder work needs
streaming, column tiling, or host-assisted paging.

## Phase 0.3 MatMul Inventory

- Encoder MatMul work: `38,596,608,000` ops per encoder pass.
- Decoder MatMul work: `66,938,880` ops per token.
- Decoder vocab projection alone is `39,831,552` ops per token and has a
  `[384, 51864]` weight matrix.

## First Silicon Probes

| Variant | Shape | Wait s | Ops | GOPS | Status |
|---|---|---:|---:|---:|---|
| `whisper_encoder_mlp_tile_p4_o3_unroll_prefab` | `M=128,K=384,N=1536`, 4 passes | 1.026240 | 1,308,622,848 | 1.275163 | pass |
| `whisper_decoder_vocab_tile_vpu_argbuf_p4_o3_unroll_prefab` | `M=32,K=384,N=4096`, 4 passes | 0.799069 | 811,597,824 | 1.015679 | pass |

Both silicon probes reported:

- `active_harts=16`
- `done_count=16`
- `active_mask=0xffff`
- `stream_error=false`

## First Real ONNX Tensor Audit

Audited ONNX node:

`/encoder/blocks.0/mlp/0/MatMul`

Export source:

- Activation input: `/encoder/blocks.0/mlp_ln/LayerNormalization_output_0`
- Weight initializer: `onnx::MatMul_780`
- ORT output: `/encoder/blocks.0/mlp/0/MatMul_output_0`

The export script adds those intermediate tensors as temporary ONNX outputs,
runs ONNXRuntime on a deterministic audio input, and saves:

- `phase0/mlp0_tile/mlp0_act_128x384.bin`
- `phase0/mlp0_tile/mlp0_weight_t_1536x384.bin`
- `phase0/mlp0_tile/mlp0_ref_128x1536.bin`

Silicon audit:

| Variant | Shape | Wait s | Ops | GOPS | Max Abs vs ORT | Mean Abs | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `whisper_mlp0_real_audit_vpu_argbuf` | `M=128,K=384,N=1536` | 0.206848 | 150,994,944 | 0.729998 | 1.9073486328125e-06 | 1.3637198037486087e-07 | pass |

All 16 per-hart slots completed:

- `active_mask=0xffff`
- `done_count=16`
- row stripes covered `0..128`

The launcher still logged `Error on kernel launch: 2` / `Stream error`, and
the summary block at `0x1000` was zero.  The audit therefore uses the same
fallback rule as the DnCNN tiled path: per-hart slots must be complete and the
dumped output tensor must match the ORT reference.  Under that rule this is the
first real Whisper ONNX numerical audit on silicon.

## Current Read

The existing `whisper-bench` 100-run proxy remains useful as a tuning baseline,
but it is too small to prove the real model path.  The real-shape probes are
slower than the small proxy, which points to memory traffic and access pattern
costs at real Whisper dimensions.  The real ONNX-tensor MLP audit now proves
one actual Whisper intermediate against ORT.  The next useful loop is to repeat
this real-tensor audit for more op shapes: encoder MLP second projection,
attention score MatMul, attention-value MatMul, then Conv and LayerNorm.

## Block 0 End-to-End-in-Pieces Audit

Completed a real ONNXRuntime tensor audit for every MatMul piece in Whisper
Tiny En encoder block 0:

- fused QKV projection
- attention score for heads 0..5
- attention value for heads 0..5
- attention output projection
- MLP first projection
- MLP second projection

Each piece was executed on ET-SoC1 silicon through `esperanto-soc6`, stitched
from tiled outputs, and compared with the ORT tensor for the same ONNX node.
All 16 block-0 MatMul pieces pass `allclose(rtol=1e-4, atol=1e-4)`.

Summary artifact:

- `phase0/WHISPER_BLOCK0_MATMUL_AUDIT.md`

Important caveat: this is not yet a native full Whisper graph executor.
Non-MatMul graph edges are still provided by host/ORT at piece boundaries.
It is, however, a real-tensor end-to-end-in-pieces audit for block 0's heavy
MatMul path.

## Optimization Pass

Ran 100 full-node silicon audit reports, including 84 optimization variants.
All optimization variants pass the ORT correctness gate.

Best current variants:

| Family | Best GOPS | Best tile rows | Notes |
|---|---:|---:|---|
| MLP2 | 1.275 | 768 | full-K VPU accumulator |
| MLP0 | 0.898 | 1024 | full-K VPU accumulator |
| QKV | 0.900 | 1280 | full-K VPU accumulator |
| attention output | 0.963 | 1024 | full-K VPU accumulator |
| attention score | 0.314 | 1280 | still the bottleneck |
| attention value | 1.284 | 512 | full-K VPU accumulator |

What helped:

- `VPU_ACCUM_FULL_K=1`: keep VPU lane accumulators live across the full K loop,
  then do one horizontal reduction per output.  This was the main improvement.
- 16 active harts.  8 active harts roughly halved MLP2 throughput.
- Larger row tiles, up to a shape-specific cache/memory pressure point.

What did not help:

- `VPU_DOT_COLS=4`; the four-output path was slower than the two-output path.
- DnCNN-style `-falign-*` and `-fno-tree-loop-*` compiler flags; they slowed
  this inline-VPU MatMul kernel.

Summary artifacts:

- `phase0/WHISPER_OPTIMIZATION_RESULTS.md`
- `phase0/whisper_real_full_node_audit.tsv`
