# Whisper Tiny En Phase 0.1 Memory Scan

All sizes below are for static ONNX initializers unless otherwise noted.

| Model | Nodes | Initializers | FP32 weights MiB | FP16 | INT8 | INT4 | I/O FP32 MiB | I/O FP16 | I/O INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| encoder | 257 | 139 | 35.818 | 17.909 | 8.955 | 4.477 | 18.494 | 9.247 | 4.623 |
| decoder | 476 | 133 | 184.062 | 92.031 | 46.015 | 23.008 | 23.026 | 11.513 | 5.757 |

## Decision Notes

- Encoder static weights fit inside 16 MiB at INT8 and INT4, but not FP16/FP32.
- Encoder FP32 I/O does not fit inside 16 MiB because the cross-attention K/V cache outputs are about 17.6 MiB; FP16 or INT8 cache storage is needed.
- Decoder static weights do not fit inside 16 MiB even at INT4 in this exported graph.
- Decoder runtime I/O is also dominated by cross-attention and self-attention KV caches, so streaming, cache quantization, or host-assisted paging is required for a full decoder.
- The next silicon-relevant step is not full-model C yet; it is real-shape MatMul/streaming probes for the largest encoder and decoder weight blocks.

## encoder Op Counts

| Op | Count |
|---|---:|
| MatMul | 72 |
| Add | 59 |
| Transpose | 41 |
| Mul | 20 |
| Reshape | 16 |
| Concat | 10 |
| LayerNormalization | 9 |
| Unsqueeze | 8 |
| Div | 6 |
| Erf | 6 |
| Split | 4 |
| Softmax | 4 |
| Conv | 2 |

## encoder Largest Initializers

| Initializer | Shape | Type | MiB | INT8 MiB | INT4 MiB |
|---|---:|---|---:|---:|---:|
| `onnx::MatMul_780` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_783` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_812` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_815` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_844` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_847` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_876` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_879` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `encoder.positional_embedding` | `[1500, 384]` | FLOAT | 2.197 | 0.549 | 0.275 |
| `onnx::Conv_750` | `[384, 384, 3]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_648` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_655` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_662` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_669` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `onnx::MatMul_777` | `[384, 384]` | FLOAT | 0.562 | 0.141 | 0.070 |

## encoder Largest MatMul Weight Initializers

| Initializer | Shape | FP32 MiB | INT8 MiB | INT4 MiB |
|---|---:|---:|---:|---:|
| `onnx::MatMul_780` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_783` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_812` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_815` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_844` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_847` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_876` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_879` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `_v_648` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_655` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_662` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_669` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `onnx::MatMul_777` | `[384, 384]` | 0.562 | 0.141 | 0.070 |
| `onnx::MatMul_809` | `[384, 384]` | 0.562 | 0.141 | 0.070 |
| `onnx::MatMul_841` | `[384, 384]` | 0.562 | 0.141 | 0.070 |

## decoder Op Counts

| Op | Count |
|---|---:|
| Slice | 152 |
| MatMul | 121 |
| Add | 69 |
| Softmax | 48 |
| Reshape | 24 |
| Concat | 18 |
| LayerNormalization | 13 |
| Unsqueeze | 8 |
| Mul | 8 |
| Split | 4 |
| Div | 4 |
| Erf | 4 |
| Gather | 3 |

## decoder Largest Initializers

| Initializer | Shape | Type | MiB | INT8 MiB | INT4 MiB |
|---|---:|---|---:|---:|---:|
| `token_embedding.weight` | `[51864, 384]` | FLOAT | 75.973 | 18.993 | 9.497 |
| `onnx::MatMul_3194` | `[384, 51864]` | FLOAT | 75.973 | 18.993 | 9.497 |
| `onnx::MatMul_2580` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2583` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2783` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2786` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2986` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2989` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_3189` | `[384, 1536]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_3192` | `[1536, 384]` | FLOAT | 2.250 | 0.562 | 0.281 |
| `_v_2778` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_2785` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_2792` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `_v_2799` | `[384, 1152]` | FLOAT | 1.688 | 0.422 | 0.211 |
| `onnx::MatMul_2490` | `[384, 384]` | FLOAT | 0.562 | 0.141 | 0.070 |

## decoder Largest MatMul Weight Initializers

| Initializer | Shape | FP32 MiB | INT8 MiB | INT4 MiB |
|---|---:|---:|---:|---:|
| `onnx::MatMul_3194` | `[384, 51864]` | 75.973 | 18.993 | 9.497 |
| `onnx::MatMul_2580` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2583` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2783` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2786` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2986` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_2989` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_3189` | `[384, 1536]` | 2.250 | 0.562 | 0.281 |
| `onnx::MatMul_3192` | `[1536, 384]` | 2.250 | 0.562 | 0.281 |
| `_v_2778` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_2785` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_2792` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `_v_2799` | `[384, 1152]` | 1.688 | 0.422 | 0.211 |
| `onnx::MatMul_2490` | `[384, 384]` | 0.562 | 0.141 | 0.070 |
| `onnx::MatMul_2493` | `[384, 384]` | 0.562 | 0.141 | 0.070 |

