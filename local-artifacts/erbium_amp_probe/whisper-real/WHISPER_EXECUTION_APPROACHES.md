# Whisper End-to-End Execution Approaches

Goal: run Whisper Tiny En against the Luxonis ONNX reference with enough
evidence to call the silicon path correct before optimizing it.

## Approach 1: Host-Orchestrated Piecewise Audit

This is the active path.

The host runs ONNXRuntime once with extra intermediate outputs enabled.  For
each selected ONNX node, the host exports the real input tensor tile, real
initializer weights, and real ORT output tile.  Silicon runs that exact tile.
The host pulls the dump, stitches all tiles, and compares the stitched output
to the ORT tensor.

Correctness rule:

- each tile has all expected hart slots done
- stitched tensor matches ORT within tolerance
- once every op in a path passes, the path is considered piecewise equivalent

Why this first:

- fits the 16 MiB U-mode region
- gives exact ONNX/ORT audit data
- can find the first bad op quickly
- does not require implementing the full graph executor before learning

Limit:

- until we feed silicon outputs into later silicon ops, this is an audit path,
  not a standalone model runner

## Approach 2: Host-Scheduled Silicon Graph Executor

The host owns graph scheduling, weights, and large tensors.  Silicon owns hot
compute kernels.  The host loads one op tile at a time, retrieves output, and
feeds that output into the next op.

Correctness rule:

- encoder final `k_cache_cross` and `v_cache_cross` match ORT
- decoder logits for one token match ORT
- then autoregressive decode matches ORT/WER target

Why this next:

- this is the first real end-to-end-in-pieces runner
- it avoids fitting all Whisper weights in 16 MiB at once

Limit:

- launch and dump overhead are high until we batch multiple tiles/ops per launch

## Approach 3: Native Silicon Graph Executor

Implement a compact graph executor in C and stream weights/tensors through the
16 MiB region.  This is the eventual performance path.

Correctness rule:

- same as Approach 2, but with fewer host round trips

Why not first:

- encoder FP32 weights are 35.8 MiB and decoder FP32 weights are 184 MiB
- even decoder INT4 weights are about 23 MiB
- full decoder also needs KV cache management

## Current Execution Plan

1. Run full stitched audit of `/encoder/blocks.0/mlp/0/MatMul`.
2. Run `/encoder/blocks.0/mlp/2/MatMul`.
3. Run attention score/value MatMuls.
4. Add Conv and LayerNorm audits.
5. Once enough ops pass, build the host-scheduled encoder path and compare
   `k_cache_cross` / `v_cache_cross` to ORT.
6. Only after the encoder/decoder piecewise outputs match ORT do the 10-way
   optimization split and then the 100-iteration optimization loop.
