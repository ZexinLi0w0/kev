"""Pull the raw TensorRT engine out of a Torch-TensorRT exported program (.ep) into a plain .engine file + .json.

Loading a .ep (torch.export.load) holds the zip entry, the pickled engine state and the deserialised engine at once --
~3x the engine size in host memory before the device copy. On an 8 GB Orin Nano (CPU and GPU share DRAM) that alone
fails for a 1.3 GB engine. A plain .engine file is deserialised directly by kev.trt_runtime with nothing else resident.
Run it where the matching TensorRT version is installed (it deserialises once to check).

    python scripts/trt_extract_engine.py engines/kev-0.8b-static-fp16/prefill.ep engines/.../score128.ep
"""
import json
import sys
from pathlib import Path

import torch
import torch_tensorrt  # noqa: F401  registers torch.classes.tensorrt.Engine


def extract(ep_path):
    ep = torch.export.load(str(ep_path))
    names = [k for k in ep.constants if k.endswith("_engine")]
    if len(names) != 1:
        raise ValueError(f"{ep_path}: expected one TensorRT engine, found {names} (the graph was partitioned)")
    fields, _ = ep.constants[names[0]].__getstate__()
    # Torch-TensorRT's serialisation layout: [abi, name, device, engine, input names, output names, ...]
    # the engine is stored base64-encoded (pickle-safe text); raw bytes are 3/4 of its length
    import base64
    blob = base64.b64decode(fields[3])
    inputs, outputs = fields[4].split("%"), fields[5].split("%")
    out = Path(ep_path).with_suffix(".engine")
    out.write_bytes(blob)
    import tensorrt as trt
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(blob)
    io = [(eng.get_tensor_name(i), str(eng.get_tensor_mode(eng.get_tensor_name(i))).split(".")[-1],
           list(eng.get_tensor_shape(eng.get_tensor_name(i))), str(eng.get_tensor_dtype(eng.get_tensor_name(i))).split(".")[-1])
          for i in range(eng.num_io_tensors)]
    meta = {"source": Path(ep_path).name, "tensorrt": trt.__version__, "bytes": len(blob), "input_order": inputs,
            "output_order": outputs, "io": io, "device_memory_bytes": int(getattr(eng, "device_memory_size_v2", eng.device_memory_size))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        extract(p)
