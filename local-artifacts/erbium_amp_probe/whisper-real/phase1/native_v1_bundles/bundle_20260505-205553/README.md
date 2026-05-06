# Whisper native-v1 bundle

This bundle freezes the log-mel input, FP32 ONNX reference, dynamic INT8 ONNX reference, and raw INT8 weight files for the ET-SoC1 log-mel-to-token-ID executor.

- FP32 text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- Dynamic INT8 text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- INT8 token match: `True`
- Manifest: `bundle_manifest.json`
