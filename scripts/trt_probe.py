"""Explore a TensorRT-compiled Kev program on one device: parity against fp32 PyTorch, latency, memory, energy.

The program is ExecuTorch's `examples/kev` two-method model (`prefill`, `score`) with the TensorRT-lowerable delta rule
from `kev.trt_ops` (see `kev.trt_program`). It is served through `kev.executorch_model.ExecuTorchDecisionModel` itself --
the backend of jaredpalmer/kev#69 -- via a thin adapter that exposes torch / Torch-TensorRT callables with the
ExecuTorch Method interface. So encoding, the state/row split, question chunking, contract checks and the temperature
are exactly the backend's, and only the program differs.

    # server (reference, once): fp32 PyTorch DecisionModel probabilities for the parity set
    python scripts/trt_probe.py --make-reference ref.json --run jaredpalmer/kev-0.8b
    # device: eager program, then Torch-TensorRT compiled program, both checked against ref.json
    python scripts/trt_probe.py --reference ref.json --run jaredpalmer/kev-0.8b --out probe.json

Stages run in order and each records its own outcome, so a failure (export, TensorRT build, OOM) is a result in the JSON
rather than a lost run.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RT_JEV_BENCH = os.environ.get("RT_JEV_BENCH")          # optional: RT-jev's bench/ for the INA3221 power sampler
if RT_JEV_BENCH:
    sys.path.insert(0, RT_JEV_BENCH)


def records(path, n):
    from kev.api import SystemOneRequest, to_record
    out = []
    for line in Path(path).read_text().splitlines()[:n]:
        req = SystemOneRequest.model_validate(json.loads(line))
        out.append(to_record(req)[0])
    return out


def reference_request(tok):
    """kev README's benchmark shape: five questions, three options each, on a ~270-token state."""
    text = ("Order 4411 shipped from the Riverside depot on the 3rd and the customer reports the package was marked "
            "delivered but never arrived. Support already issued one partial refund. ") * 20
    ids = tok(text, add_special_tokens=False).input_ids[:270]
    qs = [{"instr": f"Which team should handle this, for routing purpose {i}?",
           "options": ["returns: exchanges and refunds", "shipping: delivery problems", "billing: charges"], "label": 0}
          for i in range(5)]
    return {"state": tok.decode(ids), "questions": qs}


class _Method:
    def __init__(self, fn, device):
        self.fn, self.device = fn, device

    def execute(self, args):
        with torch.no_grad():
            out = self.fn(*[a.to(self.device) if isinstance(a, torch.Tensor) else a for a in args])
        return list(out) if isinstance(out, (tuple, list)) else [out]


class _Const:
    def __init__(self, v):
        self.v = v

    def execute(self, args):
        return [self.v]


class TorchProgram:
    """ExecuTorch Program interface over callables: what ExecuTorchDecisionModel needs and nothing more."""

    def __init__(self, methods, constants, device):
        self._m = {k: _Method(f, device) for k, f in methods.items()}
        self._m.update({k: _Const(v) for k, v in constants.items()})
        self.method_names = list(self._m)

    def load_method(self, name):
        return self._m[name]


def build(run, device, max_prefix, max_context, executorch_dir, max_questions=8, max_options=255, load_device=None,
          max_branch=None):
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SPECIAL
    from kev.trt_program import load_upstream_model, make_backbone_class
    up = load_upstream_model(executorch_dir)
    TRTBackbone = make_backbone_class(up)
    ck = Checkpoint(run)
    # fp32, LoRA merged in fp32. load_device="cpu" does the merge in host memory and moves only the merged weights: on an
    # 8 GB Orin Nano a CUDA-side merge doubles residency and fails as an NVML assert in the caching allocator.
    tok, model = ck.load(load_device or device, LoadOptions(dtype=None, attn="sdpa"))
    if load_device and load_device != device:
        import gc
        model = model.to(device); model.device = device
        gc.collect()
    lm = model.lm.eval()
    prefill = up.Prefill(TRTBackbone(lm, max_prefix)).eval()
    # a question row is padded to max_branch inside the recurrence: default max_context - 1 (upstream's branch_dim), but a
    # program exported for long states should cap it, or every row of a 2,200-token-state program pads to ~2,400 tokens
    score = up.Score(TRTBackbone(lm, max_branch or (max_context - 1)), model.head).eval()
    consts = {"get_kev_version": 1, "get_max_prefix": max_prefix, "get_max_context": max_context,
              "get_max_questions": max_questions, "get_max_options": max_options, "get_pad_id": model.pad_id,
              "get_temperature": float(model.head.temperature),
              "get_checkpoint_id": "sha256:" + hashlib.sha256(ck.file("head.pt").read_bytes()).hexdigest()}
    for i, t in enumerate(SPECIAL):
        consts[f"get_special_{i}"] = tok.convert_tokens_to_ids(t)
    return ck, tok, model, prefill, score, consts


def build_light(run, device, max_prefix, max_context, embed_dtype=torch.float32, max_questions=8, max_options=255):
    """What serving a TensorRT program needs, without loading the backbone: the tokenizer, the token embedding (one tensor
    read from the base checkpoint's safetensors -- Kev's LoRA does not touch embeddings), the pointer head and its
    temperature (head.pt), and the contract constants. On an 8 GB Orin Nano, loading the whole fp32 model just to keep
    these leaves ~3 GB of host allocations behind (CPU and GPU share DRAM) and the engines no longer fit.
    Returns (ck, tok, model-like object with .head / .lm.embed_tokens / .pad_id, consts)."""
    import types
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from kev.checkpoint import Checkpoint
    from kev.model import SPECIAL, PointerHead, load_tokenizer, pad_id
    ck = Checkpoint(run)
    if ck.adapter_config().get("trainable_token_indices"):
        raise ValueError("this checkpoint trains token embeddings; the light loader would read the base ones")
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    base = Path(snapshot_download(ck.meta.base, revision=ck.meta.base_revision, allow_patterns=["*.safetensors", "*.json"]))
    weight = None
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            key = next((k for k in sf.keys() if k.endswith("embed_tokens.weight") and "visual" not in k), None)
            if key:
                weight = sf.get_tensor(key); break
    if weight is None:
        raise KeyError("no embed_tokens.weight in the base checkpoint")
    embed = torch.nn.Embedding.from_pretrained(weight.to(embed_dtype), freeze=True).to(device)
    del weight
    head = PointerHead(embed.embedding_dim, dp=ck.meta.head_dim)
    head.load_state_dict(ck.meta.head); head.eval().to(device)
    head.temperature = ck.meta.temperature
    model = types.SimpleNamespace(head=head, lm=types.SimpleNamespace(embed_tokens=embed), pad_id=pad_id(tok))
    consts = {"get_kev_version": 1, "get_max_prefix": max_prefix, "get_max_context": max_context,
              "get_max_questions": max_questions, "get_max_options": max_options, "get_pad_id": model.pad_id,
              "get_temperature": float(ck.meta.temperature),
              "get_checkpoint_id": "sha256:" + hashlib.sha256(ck.file("head.pt").read_bytes()).hexdigest()}
    for i, t in enumerate(SPECIAL):
        consts[f"get_special_{i}"] = tok.convert_tokens_to_ids(t)
    return ck, tok, model, consts


def build_static(model, lm, consts, max_prefix, buckets, executorch_dir, device):
    from kev.trt_program import load_upstream_model
    from kev.trt_static import StaticBackbone, StaticPrefill, StaticScore
    up = load_upstream_model(executorch_dir)
    bb = StaticBackbone(lm, up.l2norm, up.apply_rotary_pos_emb).eval().to(device)
    return StaticPrefill(bb).eval(), {b: StaticScore(bb).eval() for b in buckets}


def combined_backend(full_engine, bucket, model, consts, tok, ck, max_prefix, device):
    from kev.executorch_model import ExecuTorchDecisionModel
    from kev.trt_static import CombinedProgram
    prog = CombinedProgram(full_engine, bucket, model.head, consts, max_prefix, consts["get_max_questions"], model.pad_id,
                           device, embed=model.lm.embed_tokens)
    return ExecuTorchDecisionModel(prog, tok, ck.meta.temperature, consts["get_checkpoint_id"])


def build_combined(a, model, consts, device):
    """Build (or load) the single StaticFull engine for bucket = the one --score-buckets entry; returns (engine, bucket)."""
    import torch_tensorrt
    from kev.trt_program import load_upstream_model
    from kev.trt_static import StaticBackbone, StaticFull
    up = load_upstream_model(a.executorch_dir)
    bucket = int(a.score_buckets.split(",")[0])
    path = Path(a.engine_dir) / f"full{a.max_prefix}-{bucket}.ep" if a.engine_dir else None
    if path and path.exists():
        return torch_tensorrt.load(str(path)).module(), bucket, "cached"
    full = StaticFull(StaticBackbone(model.lm, up.l2norm, up.apply_rotary_pos_emb).eval().to(device)).eval()
    d = model.lm.config.hidden_size
    ex = (torch.zeros(1, a.max_prefix, d, device=device), torch.tensor([3], device=device), torch.zeros(8, bucket, d, device=device))
    ep = torch.export.export(full, ex, strict=True)
    gm, secs = compile_exported(ep, [torch_tensorrt.Input(shape=tuple(t.shape), dtype=t.dtype) for t in ex],
                                torch.float32, a.workspace_mb, False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch_tensorrt.save(gm, str(path), output_format="exported_program", arg_inputs=list(ex))
    return gm, bucket, secs


def static_backend(prefill_engine, score_engines, model, consts, tok, ck, max_prefix, device, x_dtype=None):
    from kev.executorch_model import ExecuTorchDecisionModel
    from kev.trt_static import StaticProgram
    prog = StaticProgram(prefill_engine, score_engines, model.head, consts, max_prefix, consts["get_max_questions"], model.pad_id,
                         device, embed=model.lm.embed_tokens)
    prog.x_dtype = x_dtype
    return ExecuTorchDecisionModel(prog, tok, ck.meta.temperature, consts["get_checkpoint_id"])


def min2(score_gm):
    """torch.export records Dim(min=1) as min=2 (0/1 specialisation), so a dynamic score engine rejects one question or
    one option: duplicate the single row / option and slice the result back."""
    def score(tokens, decide, options, conv, rec, kv):
        Q, K = tokens.shape[0], options.shape[1]
        if Q == 1:
            tokens, decide, options = tokens.repeat(2, 1), decide.repeat(2), options.repeat(2, 1)
        if K == 1:
            options = options.repeat(1, 2)
        return score_gm(tokens, decide, options, conv, rec, kv)[:Q, :K]
    return score


def backend(prefill_fn, score_fn, consts, tok, ck, device):
    from kev.executorch_model import ExecuTorchDecisionModel
    prog = TorchProgram({"prefill": prefill_fn, "score": score_fn}, consts, device)
    return ExecuTorchDecisionModel(prog, tok, ck.meta.temperature, consts["get_checkpoint_id"])


def parity(m, tok, recs, ref):
    deltas, flips, n, rejected = [], 0, 0, 0
    for rec, want in zip(recs, ref):
        try:                                   # ContextOverflow (a ValueError): the state or a row does not fit the program
            enc = m.encode(tok, rec)
            got = [p.float().cpu() for p in m.probs(enc)]
        except ValueError:
            rejected += 1
            continue
        for g, w in zip(got, want):
            w = torch.tensor(w)
            deltas.append(float((g - w).abs().max()))
            flips += int(g.argmax() != w.argmax())
            n += 1
    return {"questions": n, "records_rejected": rejected, "max_abs_dp": max(deltas) if deltas else None,
            "mean_abs_dp": statistics.fmean(deltas) if deltas else None, "argmax_flips": flips}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timing(m, tok, rec, device, iters, warmup):
    enc = m.encode(tok, rec)
    for _ in range(warmup):
        m.probs(enc)
    sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    lat = []
    sampler = None
    if RT_JEV_BENCH:
        from power import PowerSampler
        sampler = PowerSampler()
        sampler.__enter__()
    for _ in range(iters):
        t = time.perf_counter()
        m.probs(enc)
        sync(device)
        lat.append(time.perf_counter() - t)
    if sampler:
        sampler.__exit__(None, None, None)
    lat.sort()
    out = {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0), "questions": len(enc["decide_idx"]),
           "p50_ms": round(lat[len(lat) // 2] * 1000, 3), "p99_ms": round(lat[min(len(lat) - 1, int(0.99 * len(lat)))] * 1000, 3),
           "min_ms": round(lat[0] * 1000, 3), "max_ms": round(lat[-1] * 1000, 3)}
    out["jitter_p99_p50"] = round(out["p99_ms"] / out["p50_ms"], 3)
    if str(device).startswith("cuda"):
        out["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if sampler:
        s = sampler.summary()
        out["power"] = s
        for rail, v in s["rails"].items():
            v["j_per_decision"] = round(v["energy_j"] / iters, 4)
    return out


def compile_exported(ep, inputs, precision, workspace_mb=None, offload=False, explicit_typing=False):
    """Torch-TensorRT on an already exported program. `offload` moves the module's weights to host memory for the build:
    on an 8 GB Orin Nano the PyTorch copy of the fp32 weights, the engine's copy and the builder's scratch do not fit
    together, and TensorRT reports it as "Could not find any implementation for node {ForeignNode[...]}" after a
    failed '...-consts' region allocation."""
    import gc
    import torch_tensorrt
    kw = {"workspace_size": workspace_mb * 2**20} if workspace_mb else {}
    if offload:
        kw["offload_module_to_cpu"] = True
    # explicit typing (mixed fp16): TensorRT keeps every tensor in the dtype the graph gives it -- fp16 GEMMs, fp32
    # recurrence / norms / head -- instead of choosing precisions itself
    prec_kw = {"use_explicit_typing": True} if explicit_typing else {"enabled_precisions": {precision}}
    t = time.perf_counter()
    gm = torch_tensorrt.dynamo.compile(ep, arg_inputs=inputs, disable_tf32=True, min_block_size=1, truncate_double=True,
                                       use_python_runtime=False, **prec_kw, **kw)
    gc.collect(); torch.cuda.empty_cache()
    return gm, round(time.perf_counter() - t, 1)


def compile_trt(module, example, dynamic_shapes, inputs, precision, workspace_mb=None):
    """torch.export, then Torch-TensorRT. `workspace_mb` caps the builder workspace: on Jetson (shared DRAM) an
    uncapped build of the fused graph fails with TensorRT's "Could not find any implementation for node
    {ForeignNode[...]}", which never mentions memory."""
    import torch_tensorrt
    ep = torch.export.export(module, example, dynamic_shapes=dynamic_shapes, strict=True)
    t = time.perf_counter()
    kw = {"workspace_size": workspace_mb * 2**20} if workspace_mb else {}
    # TF32 off: TensorRT enables it for fp32 GEMMs by default, which moved the delta rule's state by 4e-4 on Orin
    # (1e-6 with it off). A decision model's output is the probability, so fp32 must mean fp32.
    gm = torch_tensorrt.dynamo.compile(ep, arg_inputs=inputs, enabled_precisions={precision}, disable_tf32=True,
                                       min_block_size=1, truncate_double=True, use_python_runtime=False, **kw)
    return gm, round(time.perf_counter() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--records", default=str(ROOT / "evals/smoke-v1/development.jsonl"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-prefix", type=int, default=384)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--max-branch", type=int, default=None, help="dynamic program: longest question row (default max-context - 1)")
    ap.add_argument("--executorch-dir", default=os.environ.get("EXECUTORCH_DIR"))
    ap.add_argument("--make-reference")
    ap.add_argument("--reference")
    ap.add_argument("--stages", default="eager,trt")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--workspace-mb", type=int, default=None)
    ap.add_argument("--engine-dir", default=None, help="save built engines here and reuse them on the next run")
    ap.add_argument("--offload", action="store_true", help="export everything first, then build with the module on CPU")
    ap.add_argument("--load-device", default=None, help="load and merge on this device first (cpu on 8 GB boards)")
    ap.add_argument("--program", default="dynamic", choices=["dynamic", "static", "combined"],
                    help="dynamic: one engine per method (tested on TensorRT 11.3); static: fixed-shape engines (TensorRT 10.3)")
    ap.add_argument("--build-only", action="store_true", help="build and cache the engines, measure nothing")
    ap.add_argument("--mixed-fp16", action="store_true",
                    help="static program: backbone GEMMs in fp16, recurrence/norm/head in fp32 (half the engine memory)")
    ap.add_argument("--score-buckets", default="64,256,1024", help="static program: branch-length buckets (the last must cover max_context - 1)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    device = "cuda"

    if a.make_reference:                                     # fp32 PyTorch DecisionModel (transformers) reference
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import MAX_BRANCH, MAX_STATE
        ck = Checkpoint(a.run)
        tok, model = ck.load(device, LoadOptions(dtype=None))
        recs = records(a.records, a.n)
        ref = []
        for rec in recs:
            enc = model.encode(tok, rec, max_state=a.max_prefix, max_branch=a.max_context)
            ref.append([p.float().cpu().tolist() for p in model.probs(enc)])
        Path(a.make_reference).write_text(json.dumps({"run": a.run, "records": a.records, "n": a.n, "probs": ref}))
        print(f"[reference] {len(ref)} records -> {a.make_reference}")
        return

    out = {"run": a.run, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "args": vars(a), "stages": {}}
    try:
        import torch_tensorrt, tensorrt
        out["torch_tensorrt"], out["tensorrt"] = torch_tensorrt.__version__, tensorrt.__version__
    except Exception as e:
        out["torch_tensorrt"] = f"unavailable: {e}"
    save = lambda: Path(a.out).write_text(json.dumps(out, indent=2))
    ref = json.loads(Path(a.reference).read_text())["probs"] if a.reference else None
    t = time.perf_counter()
    ck, tok, model, prefill, score, consts = build(a.run, device, a.max_prefix, a.max_context, a.executorch_dir,
                                                   load_device=a.load_device, max_branch=a.max_branch)
    out["load_s"] = round(time.perf_counter() - t, 1)
    recs = records(a.records, a.n)
    bench = reference_request(tok)
    print(f"[load] {a.run} in {out['load_s']}s", flush=True)

    buckets = [int(x) for x in a.score_buckets.split(",") if x.strip()]
    if a.program == "combined":                  # builds its own StaticFull; no separate prefill/score modules
        sprefill = sscores = None
        out["program"] = {"kind": a.program, "max_prefix": a.max_prefix, "score_buckets": buckets, "max_questions": 8}
    elif a.program == "static":
        sprefill, sscores = build_static(model, model.lm, consts, a.max_prefix, buckets, a.executorch_dir, device)
        out["program"] = {"kind": "static", "max_prefix": a.max_prefix, "score_buckets": buckets, "max_questions": 8}
    else:
        out["program"] = {"kind": "dynamic", "max_prefix": a.max_prefix, "max_context": a.max_context}

    if "eager" in a.stages:
        st = out["stages"]["eager"] = {}
        try:
            m = (static_backend(sprefill, sscores, model, consts, tok, ck, a.max_prefix, device) if a.program == "static"
                 else backend(prefill, score, consts, tok, ck, device))
            if ref: st["parity"] = parity(m, tok, recs, ref)
            st["timing_reference_request"] = timing(m, tok, bench, device, a.iters, a.warmup)
            print(f"[eager] {json.dumps(st)}", flush=True)
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {e}"; st["trace"] = traceback.format_exc()[-2000:]
            print(f"[eager] FAILED {st['error']}", flush=True)
        save()

    if "trt" in a.stages:
        import torch_tensorrt
        from torch.export import Dim
        st = out["stages"][f"trt_{a.precision}"] = {}
        prec = {"fp32": torch.float32, "fp16": torch.float16}[a.precision]
        if a.program == "combined":
            try:
                gm, bucket, st["compile_s"] = build_combined(a, model, consts, device)
                print(f"[trt] combined engine (prefix {a.max_prefix}, rows {bucket}): {st['compile_s']}", flush=True); save()
                if a.build_only:
                    print(f"[done] build-only", flush=True); return
                for layer in model.lm.layers:          # only the embedding and the head are still needed in PyTorch
                    layer.to("meta")
                torch.cuda.empty_cache()
                m = combined_backend(gm, bucket, model, consts, tok, ck, a.max_prefix, device)
                if ref: st["parity"] = parity(m, tok, recs, ref)
                st["timing_reference_request"] = timing(m, tok, bench, device, a.iters, a.warmup)
                print(f"[trt] {json.dumps({k: v for k, v in st.items() if k != 'trace'})}", flush=True)
            except Exception as e:
                st["error"] = f"{type(e).__name__}: {str(e)[:1500]}"; st["trace"] = traceback.format_exc()[-3000:]
                print(f"[trt] FAILED {st['error'][:400]}", flush=True)
            save(); print(f"[done] {a.out}", flush=True); return
        if a.program == "static":
            try:
                xdt = torch.float16 if a.mixed_fp16 else torch.float32
                if a.mixed_fp16:
                    model.lm.layers.to(torch.float16); model.lm.norm.to(torch.float16)
                with torch.no_grad():
                    conv, rec_state, kv = sprefill(torch.zeros(1, a.max_prefix, model.lm.config.hidden_size, device=device, dtype=xdt),
                                                   torch.tensor([3], device=device))       # prefill takes embeddings
                # export every method first (module on the GPU, where the example inputs live), then build.
                # Engines are cached in --engine-dir: a 1024-row score bucket takes ~20 min on an Orin AGX.
                d_model = model.lm.config.hidden_size
                edir = Path(a.engine_dir) if a.engine_dir else None
                if edir: edir.mkdir(parents=True, exist_ok=True)
                emb = lambda *shape: torch.zeros(*shape, d_model, device=device, dtype=xdt)
                specs = {"prefill": (sprefill, (emb(1, a.max_prefix), torch.tensor([3], device=device)))}
                for b in buckets:
                    specs[f"score{b}"] = (sscores[b], (emb(8, b), torch.tensor([3], device=device), conv, rec_state, kv))
                built, st["compile_s"], st["failed"] = {}, {}, {}
                # export every method that is not cached BEFORE building any: offload_module_to_cpu moves the shared
                # backbone to the CPU during a build, and a later export would then see CPU weights next to GPU inputs
                eps = {}
                for name, (mod, ex) in specs.items():
                    if not (edir and (edir / f"{name}.ep").exists()):
                        try:
                            eps[name] = torch.export.export(mod, ex, strict=True)
                        except Exception as e:
                            st["failed"][name] = f"export: {type(e).__name__}: {str(e)[:300]}"
                print(f"[trt] exported {len(eps)} methods ({len(specs) - len(eps)} cached or failed)", flush=True)
                if a.build_only:
                    # the engines take embeddings, so the ~1 GB fp32 table is dead weight during a build; on Jetson the
                    # CPU and GPU share DRAM, so moving it to the CPU would free nothing -- drop it
                    model.lm.embed_tokens.to("meta"); import gc; gc.collect(); torch.cuda.empty_cache()
                for name, (mod, ex) in specs.items():
                    path = edir / f"{name}.ep" if edir else None
                    if name in st["failed"]:
                        continue
                    try:
                        if path and path.exists():
                            if a.build_only:                       # do not hold a cached engine while building another
                                st["compile_s"][name] = "cached"; continue
                            built[name] = torch_tensorrt.load(str(path)).module(); st["compile_s"][name] = "cached"
                        else:
                            ins = [torch_tensorrt.Input(shape=tuple(t.shape), dtype=t.dtype) for t in ex]
                            built[name], st["compile_s"][name] = compile_exported(eps.pop(name), ins, prec, a.workspace_mb, a.offload,
                                                                                 explicit_typing=a.mixed_fp16)
                            if path:
                                torch_tensorrt.save(built[name], str(path), output_format="exported_program", arg_inputs=list(ex))
                        print(f"[trt] static {name}: {st['compile_s'][name]}{'s' if st['compile_s'][name] != 'cached' else ''}", flush=True)
                    except Exception as e:                      # one bucket failing must not lose the others
                        st["failed"][name] = f"{type(e).__name__}: {str(e)[:300]}"
                        print(f"[trt] static {name} FAILED {st['failed'][name][:200]}", flush=True)
                        import gc; gc.collect(); torch.cuda.empty_cache()
                    save()
                if a.build_only:
                    st["built"] = sorted(built); save(); print(f"[done] build-only: {sorted(built)}", flush=True); return
                if "prefill" not in built:
                    raise RuntimeError("prefill engine did not build; nothing to measure")
                pre_gm = built["prefill"]
                sc_gms = {b: built[f"score{b}"] for b in buckets if f"score{b}" in built}
                st["score_buckets_built"] = sorted(sc_gms)
                if not sc_gms:
                    raise RuntimeError("no score engine built; nothing to measure")
                if str(device).startswith("cuda"):
                    st["engines_resident_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
                m = static_backend(pre_gm, sc_gms, model, consts, tok, ck, a.max_prefix, device,
                                   x_dtype=torch.float16 if a.mixed_fp16 else None)
                if ref: st["parity"] = parity(m, tok, recs, ref)
                st["timing_reference_request"] = timing(m, tok, bench, device, a.iters, a.warmup)
                print(f"[trt] {json.dumps({k: v for k, v in st.items() if k != 'trace'})}", flush=True)
            except Exception as e:
                st["error"] = f"{type(e).__name__}: {str(e)[:1500]}"; st["trace"] = traceback.format_exc()[-3000:]
                print(f"[trt] FAILED {st['error'][:400]}", flush=True)
            save()
            print(f"[done] {a.out}", flush=True)
            return
        try:
            n_lin = sum(l.block_type == "linear_attention" for l in prefill.backbone.layers)
            with torch.no_grad():
                conv, rec_state, kv = prefill(torch.zeros(1, 3, dtype=torch.long, device=device))
            pdim = Dim("prefix", min=1, max=a.max_prefix)
            pre_inputs = [torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(1, 270), max_shape=(1, a.max_prefix), dtype=torch.long)]
            pre_gm, st["prefill_compile_s"] = compile_trt(
                prefill, (torch.zeros(1, 3, dtype=torch.long, device=device),), ({1: pdim},), pre_inputs, prec, a.workspace_mb)
            print(f"[trt] prefill compiled in {st['prefill_compile_s']}s", flush=True); save()
            mb = a.max_branch or (a.max_context - 1)
            qd, bd, od = Dim("questions", min=1, max=8), Dim("branch", min=2, max=mb), Dim("options", min=1, max=255)
            kvs = list(kv.shape)
            sc_inputs = [torch_tensorrt.Input(min_shape=(1, 2), opt_shape=(5, 40), max_shape=(8, mb), dtype=torch.long),
                 torch_tensorrt.Input(min_shape=(1,), opt_shape=(5,), max_shape=(8,), dtype=torch.long),
                 torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(5, 3), max_shape=(8, 255), dtype=torch.long),
                 torch_tensorrt.Input(shape=tuple(conv.shape), dtype=conv.dtype),
                 torch_tensorrt.Input(shape=tuple(rec_state.shape), dtype=rec_state.dtype),
                 torch_tensorrt.Input(min_shape=tuple(kvs[:4] + [1] + kvs[5:]), opt_shape=tuple(kvs[:4] + [270] + kvs[5:]),
                                      max_shape=tuple(kvs[:4] + [a.max_prefix] + kvs[5:]), dtype=kv.dtype)]
            sc_gm, st["score_compile_s"] = compile_trt(
                score, (torch.zeros(2, 5, dtype=torch.long, device=device), torch.full((2,), 4, device=device),
                        torch.tensor([[2, 3], [2, 3]], device=device), conv, rec_state, kv),
                ({0: qd, 1: bd}, {0: qd}, {0: qd, 1: od}, None, None, {4: pdim}), sc_inputs, prec, a.workspace_mb)
            print(f"[trt] score compiled in {st['score_compile_s']}s", flush=True); save()
            if a.engine_dir:                            # keep the dynamic engines for scripts/trt_serving_bench.py (best effort:
                edir = Path(a.engine_dir); edir.mkdir(parents=True, exist_ok=True)   # a retrace can merge two dynamic dims)
                for nm, g, ins in (("prefill", pre_gm, pre_inputs), ("score", sc_gm, sc_inputs)):
                    try:
                        torch_tensorrt.save(g, str(edir / f"{nm}.ep"), output_format="exported_program", arg_inputs=ins, retrace=False)
                    except Exception as e:
                        st.setdefault("save_failed", {})[nm] = f"{type(e).__name__}: {str(e)[:200]}"
            m = backend(pre_gm, min2(sc_gm), consts, tok, ck, device)
            if ref: st["parity"] = parity(m, tok, recs, ref)
            st["timing_reference_request"] = timing(m, tok, bench, device, a.iters, a.warmup)
            print(f"[trt] {json.dumps({k: v for k, v in st.items() if k != 'trace'})}", flush=True)
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {str(e)[:1500]}"; st["trace"] = traceback.format_exc()[-3000:]
            print(f"[trt] FAILED {st['error'][:400]}", flush=True)
        save()
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
