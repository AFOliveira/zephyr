# Whisper Native-v1 Coverage Summary

Bundle: `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/native_v1_bundles/bundle_20260505-205553`

## ONNX Node Counts

| graph | op | count |
| --- | --- | ---: |
| encoder | Add | 59 |
| encoder | Concat | 10 |
| encoder | Conv | 2 |
| encoder | Div | 6 |
| encoder | Erf | 6 |
| encoder | LayerNormalization | 9 |
| encoder | MatMul | 72 |
| encoder | Mul | 20 |
| encoder | Reshape | 16 |
| encoder | Softmax | 4 |
| encoder | Split | 4 |
| encoder | Transpose | 41 |
| encoder | Unsqueeze | 8 |
| decoder | Add | 69 |
| decoder | Concat | 18 |
| decoder | Div | 4 |
| decoder | Erf | 4 |
| decoder | Gather | 3 |
| decoder | LayerNormalization | 13 |
| decoder | MatMul | 121 |
| decoder | Mul | 8 |
| decoder | Reshape | 24 |
| decoder | Slice | 152 |
| decoder | Softmax | 48 |
| decoder | Split | 4 |
| decoder | Unsqueeze | 8 |

## Audited Silicon Coverage

| area | pass | selected | audited | skipped | max abs | report |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| encoder_data_ops | True | 79 | 103 | 0 | 0 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/encoder_data_ops_audit_runs/run_20260505-163601/encoder_data_ops_audit_report.json` |
| encoder_layernorm | True | 9 | 27 |  | 1.52588e-05 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/encoder_layernorm_audit_runs/run_20260505-171859/encoder_layernorm_rows_audit_report.json` |
| encoder_scalar_ops | True | 95 | 95 |  | 0.000160992 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/remote_native_encoder_scalar_runs/run_20260505-222129/results_full_0_94_softmax3/encoder_scalar_remote_native_report.json` |
| decoder_data_ops | True | 209 | 217 | 0 | 0 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/decoder_data_ops_audit_runs/run_20260505-152431/decoder_data_ops_audit_report.json` |
| decoder_layernorm | True | 13 |  |  |  | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase0/decoder_layernorm_audit_runs/run_20260505-111622/decoder_layernorm_audit_report.json` |
| decoder_scalar_ops | True | 133 | 133 |  | 1.54972e-06 | `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/decoder_scalar_fullstep_runs/run_20260505-224625/decoder_scalar_ops_audit_report.json` |

## E2E Tail-Offload Status

- Passing rows: `6` / `6`
- Best host-orchestrated wall rate: `0.10833401508254874` token/s from `tail_parallel_tile10240`
- Best raw silicon tail rate: `20.791000859346305` token/s from `tail_sequential_tile8192`

## Remaining Gaps

- full resident graph executor still needs dynamic-memory paging/provenance validation
- dynamic INT8 ONNX reference exists, but native INT8 kernels are not yet audited layer-by-layer
