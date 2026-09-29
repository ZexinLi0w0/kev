"""A minimal TensorRT runtime for Kev's static engines on memory-tight Jetsons.

Torch-TensorRT's own runtime loads an engine from its exported-program archive: the zip entry, the base64 text of the
engine and the deserialised engine are all resident at once, ~3x the engine in host memory before the device copy. On an
8 GB Orin Nano, where CPU and GPU share DRAM, that fails for a 1 GB engine even though the engine itself fits. This
module deserialises a plain .engine file (scripts/trt_extract_engine.py) and gives every engine ONE shared activation
buffer (TensorRT user-managed device memory): Kev's prefill and score engines never run at the same time, so they do not
need an activation region each.

Same engines, same kernels, same numbers as the Torch-TensorRT runtime; only loading and memory ownership differ.
"""
import json
from pathlib import Path

import numpy as np
import torch

_TORCH = {"FLOAT": torch.float32, "HALF": torch.float16, "BF16": torch.bfloat16, "INT64": torch.int64, "INT32": torch.int32,
          "BOOL": torch.bool}


class SharedActivations:
    """One device buffer sized to the largest engine's activation needs, handed to every context."""

    def __init__(self):
        self.buf = None

    def ensure(self, nbytes, device="cuda"):
        if self.buf is None or self.buf.numel() < nbytes:
            self.buf = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
        return self.buf


class Engine:
    """Call like the Torch-TensorRT module it came from: positional inputs in the order of the original forward()."""

    def __init__(self, path, arg_names, shared, device="cuda"):
        import tensorrt as trt
        self.trt, self.device = trt, device
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.arg_names, self.outputs = list(arg_names), meta["output_order"]
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        blob = path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(blob)
        del blob                                                 # host copy released before the next engine loads
        if self.engine is None:
            raise RuntimeError(f"{path}: TensorRT could not deserialise it (built with another TensorRT version?)")
        strategy = getattr(trt, "ExecutionContextAllocationStrategy", None)
        self.ctx = (self.engine.create_execution_context(strategy.USER_MANAGED) if strategy
                    else self.engine.create_execution_context_without_device_memory())
        self.need = int(getattr(self.engine, "device_memory_size_v2", self.engine.device_memory_size))
        self.shared = shared
        shared.ensure(self.need, device)
        self.out = {}
        for name in self.outputs:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TORCH[str(self.engine.get_tensor_dtype(name)).split(".")[-1]]
            self.out[name] = torch.empty(shape, dtype=dtype, device=device)

    def __call__(self, *args):
        if len(args) != len(self.arg_names):
            raise TypeError(f"expected {len(self.arg_names)} inputs ({self.arg_names}), got {len(args)}")
        keep = []
        for name, t in zip(self.arg_names, args):
            want = _TORCH[str(self.engine.get_tensor_dtype(name)).split(".")[-1]]
            t = t.to(self.device, want).contiguous()
            keep.append(t)
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
        for name, t in self.out.items():
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
        buf = self.shared.ensure(self.need, self.device)
        if hasattr(self.ctx, "set_device_memory"):
            try:
                self.ctx.set_device_memory(int(buf.data_ptr()), int(buf.numel()))
            except TypeError:
                self.ctx.device_memory = int(buf.data_ptr())
        else:
            self.ctx.device_memory = int(buf.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        outs = tuple(self.out[n].clone() for n in self.outputs)   # clone: the next call reuses the output buffers
        return outs if len(outs) > 1 else outs[0]


PREFILL_ARGS = ["x", "n"]                                    # kev.trt_static.StaticPrefill.forward
SCORE_ARGS = ["x", "n", "conv", "recurrent", "kv"]           # kev.trt_static.StaticScore.forward


def load_static(engine_dir, buckets, device="cuda"):
    """prefill + score<B> engines from `engine_dir` (plain .engine files) sharing one activation buffer."""
    d, shared = Path(engine_dir), SharedActivations()
    pre = Engine(d / "prefill.engine", PREFILL_ARGS, shared, device)
    scores = {b: Engine(d / f"score{b}.engine", SCORE_ARGS, shared, device) for b in buckets if (d / f"score{b}.engine").exists()}
    return pre, scores, shared
