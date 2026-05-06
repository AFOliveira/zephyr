# Whisper Decoder Token Rate

This measures the decoder MatMul work for one token using real tensors exported from the Whisper Tiny En ONNX decoder.
Each row is run on ET-SoC1 silicon and compared back to the corresponding ONNX MatMul node output.

- Active harts: 16
- Decoder MatMul ops/token: 66,938,880
- Measured decoder MatMul wait/token: 1.129927 s
- Measured decoder MatMul token rate: 0.885013 token/s
- Effective decoder MatMul throughput: 0.059242 GOPS

| family | count | K | N | wait each (s) | wait/token (s) | max abs vs ONNX | report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logits | 1 | 384 | 51864 | 0.605366 | 0.605366 | 1.43051147e-05 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_logits_real_coltile8192_ah16_20260504-160136/full_audit_report.json` |
| mlp0 | 4 | 384 | 1536 | 0.018592 | 0.074369 | 3.81469727e-06 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_mlp0_ah16_20260504-160552/full_audit_report.json` |
| mlp2 | 4 | 1536 | 384 | 0.017343 | 0.069372 | 1.43051147e-06 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_mlp2_ah16_20260504-160620/full_audit_report.json` |
| self_qkv | 4 | 384 | 1152 | 0.013221 | 0.052886 | 3.57627869e-07 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_self_qkv_ah16_20260504-160640/full_audit_report.json` |
| dense384 | 12 | 384 | 384 | 0.007361 | 0.088332 | 1.78813934e-07 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_dense384_ah16_20260504-160707/full_audit_report.json` |
| cross_score | 24 | 64 | 1500 | 0.004483 | 0.107588 | 1.90734863e-06 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_cross_score_ah16_20260504-160729/full_audit_report.json` |
| cross_value | 24 | 1500 | 64 | 0.002901 | 0.069622 | 1.1920929e-07 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_cross_value_ah16_20260504-160950/full_audit_report.json` |
| self_score | 24 | 64 | 224 | 0.001891 | 0.045383 | 2.38418579e-07 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_self_score_ah16_20260504-161010/full_audit_report.json` |
| self_value | 24 | 224 | 64 | 0.000709 | 0.017011 | 0 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/decoder_token_self_value_ah16_20260504-161028/full_audit_report.json` |

Scope: this is an audited decoder-MatMul token-rate measurement, not a full native ONNX graph executor.
The measured silicon kernels are mathematically equivalent to the listed ONNX MatMul nodes within FP32 accumulation-order tolerance.
LayerNorm, softmax, GELU, cache update, token selection, and tokenizer work are not included in the token/s number above.
