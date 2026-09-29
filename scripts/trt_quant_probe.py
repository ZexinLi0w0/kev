"""Low-precision quantization of Kev's TensorRT program with NVIDIA ModelOpt, measured on the device it will run on.

Starts from the static mixed-fp16 program (`kev.trt_static`: fp16 GEMMs, fp32 DeltaNet recurrence / norms / pointer head,
the one that runs on both Jetsons) and quantizes the backbone's nn.Linear layers only -- the recurrence
(`kev.trt_ops`), the norms and the pointer head keep their precision. Calibration runs Kev's own serving path over real
records, so the activation ranges are the ones served. Each config is checked in two stages, each a result on its own:

  1. fake-quantized in PyTorch  -> parity vs fp32 (cheap filter; a config that fails here is not built)
  2. TensorRT engines           -> build outcome, parity through the engines, latency / memory on the README request

    python scripts/trt_quant_probe.py --configs INT8_WEIGHT_ONLY_CFG,INT8_SMOOTHQUANT_CFG,INT4_AWQ_CFG,FP8_DEFAULT_CFG \
        --reference ref.json --out quant.json

FP8 needs FP8 tensor cores (Ada sm_89 / Hopper sm_90): on Orin (Ampere sm_87) its build is expected to fail, and the
failure is recorded rather than assumed.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import trt_probe as tp  # noqa: E402


def calib_records(n):
    from kev.data import materialize
    from kev.suite import load_split
    import random
    recs = [materialize(r) for r in load_split(str(HERE.parent / "evals/v7/decision-v7"), "development")]
    random.Random(0).shuffle(recs)
    return recs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--configs", default="INT8_WEIGHT_ONLY_CFG,INT8_SMOOTHQUANT_CFG,INT4_AWQ_CFG,FP8_DEFAULT_CFG")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--max-prefix", type=int, default=384)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--workspace-mb", type=int, default=8192)
    ap.add_argument("--stages", default="fake,trt")
    ap.add_argument("--executorch-dir", default=os.environ.get("EXECUTORCH_DIR"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import modelopt.torch.quantization as mtq
    device = "cuda"
    ref = json.loads(Path(a.reference).read_text())
    recs = tp.records(str(HERE.parent / "evals/smoke-v1/development.jsonl"), ref["n"])
    out = {"run": a.run, "device": torch.cuda.get_device_name(0), "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
           "tensorrt": __import__("tensorrt").__version__, "modelopt": __import__("modelopt").__version__, "configs": {}}
    save = lambda: Path(a.out).write_text(json.dumps(out, indent=2))
    calib = calib_records(a.calib)

    for cfg_name in a.configs.split(","):
        st = out["configs"][cfg_name] = {}
        print(f"=== {cfg_name} ===", flush=True)
        try:
            ck, tok, model, _, _, consts = tp.build(a.run, device, a.max_prefix, 1024, a.executorch_dir)
            model.lm.layers.to(torch.float16); model.lm.norm.to(torch.float16)
            sp, ss = tp.build_static(model, model.lm, consts, a.max_prefix, [a.bucket], a.executorch_dir, device)
            m = tp.static_backend(sp, ss, model, consts, tok, ck, a.max_prefix, device, x_dtype=torch.float16)
            bb = sp.backbone

            def forward_loop(_):
                with torch.no_grad():
                    for r in calib:
                        try:
                            m.probs(m.encode(tok, r))
                        except ValueError:
                            pass                                    # rows longer than the bucket: not calibrated on
            t = time.perf_counter()
            mtq.quantize(bb, getattr(mtq, cfg_name), forward_loop)
            st["calibrate_s"] = round(time.perf_counter() - t, 1)
            st["quantized_linears"] = sum(1 for mod in bb.modules() if type(mod).__name__.startswith("Quant"))
            if "fake" in a.stages:
                st["fake_quant_parity"] = tp.parity(m, tok, recs, ref["probs"])
                print(f"[fake] {json.dumps(st['fake_quant_parity'])}", flush=True); save()
            if "trt" in a.stages:
                import torch_tensorrt
                from modelopt.torch.quantization.utils import export_torch_mode
                d = model.lm.config.hidden_size
                with torch.no_grad():
                    conv, rec, kv = sp(torch.zeros(1, a.max_prefix, d, device=device, dtype=torch.float16), torch.tensor([3], device=device))
                specs = {"prefill": (sp, (torch.zeros(1, a.max_prefix, d, device=device, dtype=torch.float16), torch.tensor([3], device=device))),
                         f"score{a.bucket}": (ss[a.bucket], (torch.zeros(8, a.bucket, d, device=device, dtype=torch.float16),
                                                               torch.tensor([3], device=device), conv, rec, kv))}
                built = {}
                for name, (mod, ex) in specs.items():
                    t = time.perf_counter()
                    try:
                        with torch.no_grad(), export_torch_mode():
                            ep = torch.export.export(mod, ex, strict=False)
                            built[name] = torch_tensorrt.dynamo.compile(
                                ep, arg_inputs=[torch_tensorrt.Input(shape=tuple(x.shape), dtype=x.dtype) for x in ex],
                                use_explicit_typing=True, disable_tf32=True, min_block_size=1, truncate_double=True,
                                workspace_size=a.workspace_mb * 2**20)
                        st.setdefault("build_s", {})[name] = round(time.perf_counter() - t, 1)
                        print(f"[trt] {name} built in {st['build_s'][name]}s", flush=True)
                    except Exception as e:
                        st.setdefault("build_failed", {})[name] = f"{type(e).__name__}: {str(e)[:600]}"
                        print(f"[trt] {name} FAILED {st['build_failed'][name][:300]}", flush=True)
                        break
                    save()
                if len(built) == len(specs):
                    torch.cuda.reset_peak_memory_stats()
                    q = tp.static_backend(built["prefill"], {a.bucket: built[f"score{a.bucket}"]}, model, consts, tok, ck,
                                          a.max_prefix, device, x_dtype=torch.float16)
                    st["trt_parity"] = tp.parity(q, tok, recs, ref["probs"])
                    st["timing_reference_request"] = tp.timing(q, tok, tp.reference_request(tok), device, a.iters, 3)
                    print(f"[trt] parity {json.dumps(st['trt_parity'])} | p50 {st['timing_reference_request']['p50_ms']} ms", flush=True)
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {str(e)[:600]}"; st["trace"] = traceback.format_exc()[-2000:]
            print(f"[error] {st['error'][:300]}", flush=True)
        save()
        import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
