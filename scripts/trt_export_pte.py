"""Write a Kev ExecuTorch program whose methods are TensorRT engines -- the route suggested on jaredpalmer/kev#71:
Torch-TensorRT compiles, `torch_tensorrt.executorch` lowers to a .pte, and the program is served by #69's
`kev.executorch_model.ExecuTorchDecisionModel` through the ExecuTorch runtime.

The program is the dynamic one of `kev.trt_program` (ExecuTorch's examples/kev model with the TensorRT-lowerable delta
rule), so it has exactly #69's contract: `prefill(tokens)` -> conv, recurrent, kv and `score(tokens, decide, options,
conv, recurrent, kv)` -> logits / temperature, plus the constant methods (version, limits, pad / delimiter ids,
temperature, checkpoint id) that ExecuTorchDecisionModel checks on load.

    python scripts/trt_export_pte.py --run jaredpalmer/kev-0.8b --out programs/kev-0.8b-trt
    KEV_BACKEND=executorch KEV_PROGRAM=programs/kev-0.8b-trt/model.pte python -m kev.serve --run jaredpalmer/kev-0.8b

A .pte carries serialised TensorRT engines, which only load on the TensorRT version and GPU architecture they were built
with: export on the machine class you will serve on.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--max-prefix", type=int, default=384)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--max-branch", type=int, default=None)
    ap.add_argument("--executorch-dir", default=os.environ.get("EXECUTORCH_DIR"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch_tensorrt
    import torch_tensorrt.executorch as te
    import trt_probe as tp
    from torch.export import Dim
    # keep size-1 dimensions generic: without this torch.export specialises 0/1 and records Dim(min=1) as min=2, so the
    # engine would reject a request with one question -- and a .pte is served by #69's backend unchanged, with no
    # padding shim in front of it
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True

    device = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck, tok, model, prefill, score, consts = tp.build(a.run, device, a.max_prefix, a.max_context, a.executorch_dir,
                                                      max_branch=a.max_branch)
    mb = a.max_branch or (a.max_context - 1)
    with torch.no_grad():
        conv, rec, kv = prefill(torch.zeros(1, 3, dtype=torch.long, device=device))
    pdim, qd = Dim("prefix", min=1, max=a.max_prefix), Dim("questions", min=1, max=8)
    info = {"run": a.run, "tensorrt": __import__("tensorrt").__version__, "torch_tensorrt": torch_tensorrt.__version__,
            "limits": {"prefix": a.max_prefix, "context": a.max_context, "branch": mb, "questions": 8}}
    t = time.perf_counter()
    ep_pre = torch.export.export(prefill, (torch.zeros(1, 3, dtype=torch.long, device=device),), dynamic_shapes=({1: pdim},), strict=True)
    ep_sco = torch.export.export(score, (torch.zeros(2, 5, dtype=torch.long, device=device), torch.full((2,), 4, device=device),
                                         torch.tensor([[2, 3], [2, 3]], device=device), conv, rec, kv),
                                 dynamic_shapes=({0: qd, 1: Dim("branch", min=2, max=mb)}, {0: qd},
                                                 {0: qd, 1: Dim("options", min=1, max=255)}, None, None, {4: pdim}), strict=True)
    info["export_s"] = round(time.perf_counter() - t, 1)
    for name, ep in (("prefill", ep_pre), ("score", ep_sco)):
        info[f"{name}_input_ranges"] = {str(n): str(r) for n, r in ep.range_constraints.items()}
    kvs = list(kv.shape)
    pre_in = [torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(1, 270), max_shape=(1, a.max_prefix), dtype=torch.long)]
    sco_in = [torch_tensorrt.Input(min_shape=(1, 2), opt_shape=(5, 40), max_shape=(8, mb), dtype=torch.long),
              torch_tensorrt.Input(min_shape=(1,), opt_shape=(5,), max_shape=(8,), dtype=torch.long),
              torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(5, 3), max_shape=(8, 255), dtype=torch.long),
              torch_tensorrt.Input(shape=tuple(conv.shape), dtype=conv.dtype), torch_tensorrt.Input(shape=tuple(rec.shape), dtype=rec.dtype),
              torch_tensorrt.Input(min_shape=tuple(kvs[:4] + [1] + kvs[5:]), opt_shape=tuple(kvs[:4] + [270] + kvs[5:]),
                                   max_shape=tuple(kvs[:4] + [a.max_prefix] + kvs[5:]), dtype=kv.dtype)]
    t = time.perf_counter()
    gm_pre, info["prefill_compile_s"] = tp.compile_exported(ep_pre, pre_in, torch.float32)
    gm_sco, info["score_compile_s"] = tp.compile_exported(ep_sco, sco_in, torch.float32)
    # each method as an ExportedProgram that already holds its engine (save + load, without re-tracing)
    eps = {}
    for name, gm, ins in (("prefill", gm_pre, pre_in), ("score", gm_sco, sco_in)):
        f = out / f"{name}.ep"
        torch_tensorrt.save(gm, str(f), output_format="exported_program", arg_inputs=ins, retrace=False)
        eps[name] = torch.export.load(str(f))
    t = time.perf_counter()
    edge = te.export(eps, constant_methods={k: v for k, v in consts.items()})
    program = edge.to_executorch()
    with (out / "model.pte").open("wb") as f:
        program.write_to_file(f)
    program.write_tensor_data_to_file(str(out))
    info["lower_s"] = round(time.perf_counter() - t, 1)
    info["files"] = {p.name: p.stat().st_size for p in out.iterdir() if p.suffix in (".pte", ".ptd")}
    (out / "export.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
