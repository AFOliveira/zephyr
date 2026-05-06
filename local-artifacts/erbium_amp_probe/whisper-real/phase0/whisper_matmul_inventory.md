# Whisper Tiny En MatMul Inventory

Shapes are from ONNX shape inference. Ops are one forward pass for the exported encoder, or one decoder token step for the exported decoder.

## encoder

- MatMul nodes: 72
- Total MatMul ops: 38,596,608,000

| Count | A shape | B shape | Output shape | Ops each | Ops total | Example nodes |
|---:|---|---|---|---:|---:|---|
| 4 | `[1, 1500, 384]` | `[384, 1536]` | `[1, 1500, 1536]` | 1,769,472,000 | 7,077,888,000 | `/encoder/blocks.0/mlp/0/MatMul, /encoder/blocks.1/mlp/0/MatMul, /encoder/blocks.2/mlp/0/MatMul, /encoder/blocks.3/mlp/0/MatMul` |
| 4 | `[1, 1500, 1536]` | `[1536, 384]` | `[1, 1500, 384]` | 1,769,472,000 | 7,077,888,000 | `/encoder/blocks.0/mlp/2/MatMul, /encoder/blocks.1/mlp/2/MatMul, /encoder/blocks.2/mlp/2/MatMul, /encoder/blocks.3/mlp/2/MatMul` |
| 4 | `[1, 6, 1500, 64]` | `[1, 6, 64, 1500]` | `[1, 6, 1500, 1500]` | 1,728,000,000 | 6,912,000,000 | `/encoder/blocks.0/attn/MatMul, /encoder/blocks.1/attn/MatMul, /encoder/blocks.2/attn/MatMul, /encoder/blocks.3/attn/MatMul` |
| 4 | `[1, 6, 1500, 1500]` | `[1, 6, 1500, 64]` | `[1, 6, 1500, 64]` | 1,728,000,000 | 6,912,000,000 | `/encoder/blocks.0/attn/MatMul_1, /encoder/blocks.1/attn/MatMul_1, /encoder/blocks.2/attn/MatMul_1, /encoder/blocks.3/attn/MatMul_1` |
| 4 | `[1, 1500, 384]` | `[384, 1152]` | `[1, 1500, 1152]` | 1,327,104,000 | 5,308,416,000 | `MatMul_from_/encoder/blocks.0/attn/query/MatMul_and_/encoder/blocks.0/attn/key/MatMul_and_/encoder/blocks.0/attn/value/MatMul, MatMul_from_/encoder/blocks.1/attn/query/MatMul_and_/encoder/blocks.1/attn/key/MatMul_and_/encoder/blocks.1/attn/value/MatMul, MatMul_from_/encoder/blocks.2/attn/query/MatMul_and_/encoder/blocks.2/attn/key/MatMul_and_/encoder/blocks.2/attn/value/MatMul, MatMul_from_/encoder/blocks.3/attn/query/MatMul_and_/encoder/blocks.3/attn/key/MatMul_and_/encoder/blocks.3/attn/value/MatMul` |
| 48 | `[1, 1500, 384]` | `[384, 64]` | `[1, 1500, 64]` | 73,728,000 | 3,538,944,000 | `/0/0/MatMul, /0/1/MatMul, /0/2/MatMul, /0/3/MatMul, /0/4/MatMul` |
| 4 | `[1, 1500, 384]` | `[384, 384]` | `[1, 1500, 384]` | 442,368,000 | 1,769,472,000 | `/encoder/blocks.0/attn/out/MatMul, /encoder/blocks.1/attn/out/MatMul, /encoder/blocks.2/attn/out/MatMul, /encoder/blocks.3/attn/out/MatMul` |

## decoder

- MatMul nodes: 121
- Total MatMul ops: 66,938,880

| Count | A shape | B shape | Output shape | Ops each | Ops total | Example nodes |
|---:|---|---|---|---:|---:|---|
| 1 | `[1, 1, 384]` | `[384, 51864]` | `[1, 1, 51864]` | 39,831,552 | 39,831,552 | `/logits/MatMul` |
| 4 | `[1, 1, 384]` | `[384, 1536]` | `[1, 1, 1536]` | 1,179,648 | 4,718,592 | `/0/mlp/0/MatMul, /1/mlp/0/MatMul, /2/mlp/0/MatMul, /3/mlp/0/MatMul` |
| 4 | `[1, 1, 1536]` | `[1536, 384]` | `[1, 1, 384]` | 1,179,648 | 4,718,592 | `/0/mlp/2/MatMul, /1/mlp/2/MatMul, /2/mlp/2/MatMul, /3/mlp/2/MatMul` |
| 24 | `[1, 1, 1, 64]` | `[1, 1, 64, 1500]` | `[1, 1, 1, 1500]` | 192,000 | 4,608,000 | `/0/cross_attn/MatMul, /0/cross_attn/MatMul_2, /0/cross_attn/MatMul_4, /0/cross_attn/MatMul_6, /0/cross_attn/MatMul_8` |
| 24 | `[1, 1, 1, 1500]` | `[1, 1, 1500, 64]` | `[1, 1, 1, 64]` | 192,000 | 4,608,000 | `/0/cross_attn/MatMul_1, /0/cross_attn/MatMul_3, /0/cross_attn/MatMul_5, /0/cross_attn/MatMul_7, /0/cross_attn/MatMul_9` |
| 4 | `[1, 1, 384]` | `[384, 1152]` | `[1, 1, 1152]` | 884,736 | 3,538,944 | `MatMul_from_/0/attn/query/MatMul_and_/0/attn/key/MatMul_and_/0/attn/value/MatMul, MatMul_from_/1/attn/query/MatMul_and_/1/attn/key/MatMul_and_/1/attn/value/MatMul, MatMul_from_/2/attn/query/MatMul_and_/2/attn/key/MatMul_and_/2/attn/value/MatMul, MatMul_from_/3/attn/query/MatMul_and_/3/attn/key/MatMul_and_/3/attn/value/MatMul` |
| 12 | `[1, 1, 384]` | `[384, 384]` | `[1, 1, 384]` | 294,912 | 3,538,944 | `/0/attn/out/MatMul, /0/cross_attn/query/MatMul, /0/cross_attn/out/MatMul, /1/attn/out/MatMul, /1/cross_attn/query/MatMul` |
| 24 | `[1, 1, 1, 64]` | `[1, 1, 64, 224]` | `[1, 1, 1, 224]` | 28,672 | 688,128 | `/0/attn/MatMul, /0/attn/MatMul_2, /0/attn/MatMul_4, /0/attn/MatMul_6, /0/attn/MatMul_8` |
| 24 | `[1, 1, 1, 224]` | `[1, 1, 224, 64]` | `[1, 1, 1, 64]` | 28,672 | 688,128 | `/0/attn/MatMul_1, /0/attn/MatMul_3, /0/attn/MatMul_5, /0/attn/MatMul_7, /0/attn/MatMul_9` |

## Silicon Targets

- Encoder first target: row-tiled MLP MatMul with `M=128, K=384, N=1536`; this mirrors the largest encoder MLP weight shape while staying under the 16 MiB arg buffer.
- Encoder attention target: row-tiled score MatMul with `batch=6, M=128, K=64, N=1500`; this tests the long-context attention access pattern.
- Decoder first target: vocab projection with `M=1, K=384, N=51864`; the full INT8 weight is still about 19 MiB, so it must be column-tiled or streamed.
- Existing `whisper-bench` proxy covers only `TOK=256, DIM=64, HIDDEN=256`; keep it as a tuning baseline, not a model-shape proof.
