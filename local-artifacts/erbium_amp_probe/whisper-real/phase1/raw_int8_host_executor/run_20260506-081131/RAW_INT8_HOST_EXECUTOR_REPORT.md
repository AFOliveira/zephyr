# Whisper Raw INT8 Host Executor Report

- Result: `PASS`
- Bundle: `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/native_v1_bundles/bundle_20260505-205553`
- Resident manifest: `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/resident_layout_host/run_20260506-080629/resident_weight_manifest.json`
- Raw INT8 text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- FP32 text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- Dynamic INT8 text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- Token match vs FP32: `True`
- Token match vs dynamic INT8: `True`
- Decode wall seconds: `0.496175`
- Tokens/s wall: `46.354621`

## Model Patching

| Part | Replaced float initializers | INT8 bytes | Materialized FP32 bytes | Max reconstruction error |
|---|---:|---:|---:|---:|
| encoder | 135 | 9389572 | 37558288 | 0.062300086 |
| decoder | 85 | 48250627 | 193002508 | 0.0658226013 |
