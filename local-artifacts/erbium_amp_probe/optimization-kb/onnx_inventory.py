#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter

import onnx
from onnx import TensorProto


DEFAULT_MODELS = {
    "dncnn3_240x320": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/dncnn3-240x320.onnx/dncnn3-240x320.onnx",
    "dncnn3_480x640": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/dncnn3-480x640.onnx/dncnn3-480x640.onnx",
    "depth_anything_v2_252x336": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/depth-anything-v2-vits-252x336.onnx/depth-anything-v2-vits-252x336.onnx",
    "depth_anything_v2_420x560": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/depth-anything-v2-vits-420x560.onnx/depth-anything-v2-vits-420x560.onnx",
    "whisper_tiny_en_encoder": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/whisper_tiny_en_encoder_qaihub.onnx/whisper_tiny_en_encoder_qaihub.onnx",
    "whisper_tiny_en_decoder": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/extracted/whisper_tiny_en_decoder_qaihub.onnx/whisper_tiny_en_decoder_qaihub.onnx",
    "foundation_stereo_640x416_32": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/foundation_stereo_640x416_32.onnx",
    "foundation_stereo_1280x800_32": "local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx/foundation_stereo_1280x800_32.onnx",
}


TYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}


def tensor_shape(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def tensor_nbytes(tensor):
    if tensor.raw_data:
        return len(tensor.raw_data)
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return count * TYPE_BYTES.get(tensor.data_type, 0)


def inventory_one(name, path):
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    initializers = list(graph.initializer)
    initializer_bytes = sum(tensor_nbytes(t) for t in initializers)
    input_names = {i.name for i in initializers}

    return {
        "name": name,
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "opset_import": [
            {"domain": imp.domain or "ai.onnx", "version": imp.version}
            for imp in model.opset_import
        ],
        "nodes": len(graph.node),
        "initializers": len(initializers),
        "initializer_bytes": initializer_bytes,
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in graph.input
            if value.name not in input_names
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in graph.output
        ],
        "op_counts": dict(op_counts.most_common()),
        "top_ops": op_counts.most_common(20),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="local-artifacts/erbium_amp_probe/optimization-kb/onnx_inventory.json")
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    out = []
    for name, path in DEFAULT_MODELS.items():
        if args.skip_large and os.path.getsize(path) > 300 * 1024 * 1024:
            out.append({
                "name": name,
                "path": path,
                "file_size_bytes": os.path.getsize(path),
                "skipped": "larger than 300 MiB",
            })
            continue
        out.append(inventory_one(name, path))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
