"""kev's README serving benchmark (scripts/serving_bench.py) run against a TensorRT program, or kev's own PyTorch backend.

The request shapes, the timing procedure and the metric are upstream's, imported from scripts/serving_bench.py (CASES,
request, latency) rather than re-written: model time per request as kev.serve.Server reports it (`latency_ms`), median of
`--reps` after two warm-up requests, for a new state per request and for the same state again (prefix-cache hit), on

    2 questions, short state | 6 questions, short state | 5 questions, 370-token state | 5 questions, 2,200-token state

and requests/s with p50 / p99 under 1 / 8 / 32 / 64 concurrent clients (upstream's throughput procedure, in-process
Server.probs, a first pass to meet every batch shape and a timed second pass), on the README's "6 questions, new short
state" traffic and on 2,200-token documents. Only the model behind the Server changes:

    --backend torch        kev's own serving path (Checkpoint.load, bf16 by default, as the README serves)
    --backend trt-dynamic  kev.trt_program engines (tested on TensorRT 11.3)
    --backend trt-static   kev.trt_static engines (TensorRT 10.3 / JetPack 6)

A shape a program was not exported for (a 2,200-token state against a 384-token prefill engine) is reported as
inadmissible with the server's own error, not skipped silently. Energy per request is recorded when RT_JEV_BENCH points
at a directory with power.py.
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
if os.environ.get("RT_JEV_BENCH"):
    sys.path.insert(0, os.environ["RT_JEV_BENCH"])

import serving_bench as sb                                  # upstream's CASES / request / latency  # noqa: E402


def power_sampler():
    if not os.environ.get("RT_JEV_BENCH"):
        return None
    from power import PowerSampler
    return PowerSampler()


def build_model(a, device):
    from kev.checkpoint import Checkpoint, LoadOptions
    if a.backend == "executorch":
        # #69's own path: Checkpoint.load(backend="executorch") -> ExecuTorchDecisionModel over the program
        ck = Checkpoint(a.run)
        tok, model = ck.load("cpu", LoadOptions(backend="executorch", program=a.program))
        return ck, tok, model, {"backend": "executorch", "program": Path(a.program).name,
                                "program_dtype": getattr(model, "dtype", None)}
    if a.backend == "torch":
        ck = Checkpoint(a.run)
        dtype = {"bf16": torch.bfloat16, "fp32": None, "fp16": torch.float16}[a.dtype]
        # merge=False loads the base straight into the serving dtype and keeps the LoRA unmerged: kev's default fp32 merge
        # needs 4 bytes per parameter at load, ~36 GB for Kev-9B, more than a 32 GB Jetson AGX has
        tok, model = ck.load(device, LoadOptions(dtype=dtype, merge=bool(a.merge)))
        return ck, tok, model, {"backend": "torch", "dtype": a.dtype, "lora_merged": bool(a.merge)}
    import trt_probe as tp
    import torch_tensorrt
    if a.backend in ("trt-static", "trt-combined"):
        # engines only need the embedding, the head and the contract: never load the backbone (8 GB boards)
        ck, tok, model, consts = tp.build_light(a.run, device, a.max_prefix, a.max_context,
                                                embed_dtype=torch.float16 if a.mixed_fp16 else torch.float32)
        prefill = score = None
    else:
        ck, tok, model, prefill, score, consts = tp.build(a.run, device, a.max_prefix, a.max_context, a.executorch_dir,
                                                          load_device=a.load_device, max_branch=a.max_branch)
    edir = Path(a.engine_dir)
    import gc

    def free_backbone():
        """The engines replace the PyTorch backbone; only the embedding (static/combined) and the head stay. Free it
        BEFORE loading engines: on Jetson CPU and GPU share DRAM, and each torch_tensorrt.load briefly holds the file's
        bytes next to the deserialised engine (five 2.6 GB engines plus the fp32 model were OOM-killed on a 32 GB AGX).
        The static and combined paths never load it (build_light)."""
        for layer in getattr(model.lm, "layers", []):
            layer.to("meta")
        gc.collect(); torch.cuda.empty_cache()

    def load(f):
        m = torch_tensorrt.load(str(f)).module(); gc.collect(); torch.cuda.empty_cache(); return m

    if a.backend == "trt-combined":
        bucket = int(a.score_buckets.split(",")[0])
        f = edir / f"full{a.max_prefix}-{bucket}.ep"
        if not f.exists():
            raise SystemExit(f"no combined engine {f}: build it with scripts/trt_probe.py --program combined --build-only")
        free_backbone()
        eng = load(f)
        m = tp.combined_backend(eng, bucket, model, consts, tok, ck, a.max_prefix, device)
        m.max_row = bucket
        return ck, tok, m, {"backend": "trt-combined", "engine": f.name, "max_prefix": a.max_prefix, "rows": bucket}
    if a.backend == "trt-static":
        # (with --mixed-fp16 build_light already stores the table in fp16: the engines take fp16 embeddings, so it is exact)
        buckets = [int(x) for x in a.score_buckets.split(",")]
        free_backbone()
        if a.raw_engines:
            # plain .engine files through kev.trt_runtime: no .ep archive in memory, one shared activation buffer
            from kev.trt_runtime import load_static
            pre, sc, _shared = load_static(edir, buckets, device)
            eng = {"prefill": pre, **{f"score{b}": e for b, e in sc.items()}}
            missing = [f"score{b}" for b in buckets if b not in sc]
        else:
            names = ["prefill"] + [f"score{b}" for b in buckets]
            missing = [n for n in names if not (edir / f"{n}.ep").exists()]
            if "prefill" in missing:
                raise SystemExit(f"no prefill engine in {edir}: build it first with scripts/trt_probe.py --program static --engine-dir")
            eng = {n: load(edir / f"{n}.ep") for n in names if n not in missing}
            sc = {b: eng[f"score{b}"] for b in buckets if f"score{b}" in eng}
        m = tp.static_backend(eng["prefill"], sc, model, consts, tok, ck, a.max_prefix, device,
                              x_dtype=torch.float16 if a.mixed_fp16 else None)
        m.max_row = max(sc)
        info = {"backend": "trt-static", "engines": sorted(eng), "missing_engines": missing, "max_prefix": a.max_prefix,
                "runtime": "kev.trt_runtime (raw .engine, shared activations)" if a.raw_engines else "Torch-TensorRT",
                "precision": "mixed fp16 (fp32 recurrence/norm/head)" if a.mixed_fp16 else "fp32"}
    else:
        if all((edir / f"{n}.ep").exists() for n in ("prefill", "score")):
            pre, sco = (torch_tensorrt.load(str(edir / f"{n}.ep")).module() for n in ("prefill", "score"))
            built = "loaded"
        else:                                          # no saved engines: build them here (same code as trt_probe)
            pre, sco, built = build_dynamic(a, prefill, score, device)
        m = tp.backend(pre, tp.min2(sco), consts, tok, ck, device)
        m.max_row = a.max_branch or (a.max_context - 1)
        info = {"backend": "trt-dynamic", "max_prefix": a.max_prefix, "max_context": a.max_context,
                "max_branch": a.max_branch, "engines": built}
    free_backbone()        # dynamic path (its engines were loaded or built above)
    return ck, tok, m, info


def build_dynamic(a, prefill, score, device):
    """The dynamic program's two engines, compiled exactly as scripts/trt_probe.py does (TF32 off, fp32)."""
    import torch_tensorrt
    import trt_probe as tp
    from torch.export import Dim
    mb = a.max_branch or (a.max_context - 1)
    with torch.no_grad():
        conv, rec, kv = prefill(torch.zeros(1, 3, dtype=torch.long, device=device))
    pdim, qd = Dim("prefix", min=1, max=a.max_prefix), Dim("questions", min=1, max=8)   # one Dim object per shared dimension
    pre, t1 = tp.compile_trt(prefill, (torch.zeros(1, 3, dtype=torch.long, device=device),), ({1: pdim},),
                             [torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(1, 270), max_shape=(1, a.max_prefix), dtype=torch.long)], torch.float32)
    kvs = list(kv.shape)
    ins = [torch_tensorrt.Input(min_shape=(1, 2), opt_shape=(5, 40), max_shape=(8, mb), dtype=torch.long),
           torch_tensorrt.Input(min_shape=(1,), opt_shape=(5,), max_shape=(8,), dtype=torch.long),
           torch_tensorrt.Input(min_shape=(1, 1), opt_shape=(5, 3), max_shape=(8, 255), dtype=torch.long),
           torch_tensorrt.Input(shape=tuple(conv.shape), dtype=conv.dtype), torch_tensorrt.Input(shape=tuple(rec.shape), dtype=rec.dtype),
           torch_tensorrt.Input(min_shape=tuple(kvs[:4] + [1] + kvs[5:]), opt_shape=tuple(kvs[:4] + [270] + kvs[5:]),
                                max_shape=tuple(kvs[:4] + [a.max_prefix] + kvs[5:]), dtype=kv.dtype)]
    sco, t2 = tp.compile_trt(score, (torch.zeros(2, 5, dtype=torch.long, device=device), torch.full((2,), 4, device=device),
                                     torch.tensor([[2, 3], [2, 3]], device=device), conv, rec, kv),
                             ({0: qd, 1: Dim("branch", min=2, max=mb)}, {0: qd}, {0: qd, 1: Dim("options", min=1, max=255)},
                              None, None, {4: pdim}), ins, torch.float32)
    return pre, sco, {"prefill_compile_s": t1, "score_compile_s": t2}


def latency_cases(server, reps):
    """upstream latency(), one case at a time so an inadmissible shape is recorded instead of aborting the run."""
    out, cases = {}, dict(sb.CASES)
    for case, spec in cases.items():
        sb.CASES = {case: spec}
        sampler = power_sampler()
        try:
            if sampler: sampler.__enter__()
            t0 = time.perf_counter()
            row = sb.latency(server, reps)[case]
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            if sampler:
                sampler.__exit__(None, None, None)
                s = sampler.summary()
                n = 1 + 2 * (reps + 2)                     # the requests latency() sends for one case
                row["power"] = {r: {"mean_w": v["mean_w"], "j_per_request": round(v["energy_j"] / n, 4)} for r, v in s["rails"].items()}
            out[case] = row
        except Exception as e:
            if sampler: sampler.__exit__(None, None, None)
            detail = getattr(e, "detail", None) or str(e)
            out[case] = {"inadmissible" if "422" in repr(e) or "exceed" in str(detail) or "at most" in str(detail) else "error": str(detail)[:300]}
        print(f"[latency] {case}: {json.dumps(out[case])[:400]}", flush=True)
    sb.CASES = cases
    return out


def throughput(server, suite, levels, only=None):
    """upstream throughput() (serving_bench.py), with records the program cannot admit filtered out and counted."""
    import random
    from concurrent.futures import ThreadPoolExecutor
    from kev.api import to_record
    from kev.data import materialize
    from kev.suite import load_split
    samples = {"6 questions, new short state": [to_record(sb.request("6 questions, short state", 1000 + i))[0] for i in range(256)],
               "decision-v7 development": random.Random(0).choices([materialize(r) for r in load_split(suite, "development")], k=256),
               "5 questions, 2,200-token state": [to_record(sb.request("5 questions, 2,200-token state", 1000 + i))[0] for i in range(64)]}

    max_row = getattr(server.model, "max_row", None)       # longest question row the engines were built for

    def admissible(rec):
        """Fits the program: the contract's state/context limits (encode) AND the engines' row cap, which the contract
        does not carry (a 128-row engine would otherwise receive decision-v7's ~700-token rows and fail inside TensorRT)."""
        from kev.model import rows_of
        try:
            enc = server.model.encode(server.tok, rec, max_state=sb_serve_max()[0], max_branch=sb_serve_max()[1])
        except ValueError:
            return False
        return max_row is None or max(len(r["ids"]) for r in rows_of(enc)[2]) <= max_row

    def run(recs, c):
        def one(rec):
            t = time.perf_counter(); server.probs(rec); return time.perf_counter() - t
        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter(); lat = sorted(pool.map(one, recs)); wall = time.perf_counter() - start
        return {"p50_ms": round(1000 * statistics.median(lat), 1), "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
                "requests_per_s": round(len(lat) / wall, 2)}

    out = {}
    for name, recs in samples.items():
        if only and not any(o in name for o in only):
            continue
        ok = [r for r in recs if admissible(r)]
        if not ok:
            out[name] = {"inadmissible": f"0 of {len(recs)} requests fit this program"}; print(f"[throughput] {name}: inadmissible", flush=True); continue
        try:
            first = {c: run(ok, c) for c in levels}
            server.wait_idle()
            for c in levels:
                sampler = power_sampler()
                if sampler: sampler.__enter__()
                row = {**run(ok, c), "first": first[c], "requests": len(ok), "rejected": len(recs) - len(ok)}
                if sampler:
                    sampler.__exit__(None, None, None)
                    row["power"] = {r: {"mean_w": v["mean_w"], "j_per_request": round(v["energy_j"] / len(ok), 4)}
                                    for r, v in sampler.summary()["rails"].items()}
                out[f"{name} @ {c} clients"] = row
                print(f"[throughput] {name} @ {c}: {json.dumps(row)[:300]}", flush=True)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}; print(f"[throughput] {name}: {out[name]}", flush=True)
    return out


def sb_serve_max():
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    return SERVE_MAX_STATE, SERVE_MAX_BRANCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--backend", default="trt-static", choices=["torch", "trt-dynamic", "trt-static", "trt-combined", "executorch"])
    ap.add_argument("--program", default=None, help="executorch backend: the .pte (XNNPACK / MLX / TensorRT delegate)")
    ap.add_argument("--dtype", default="bf16", help="torch backend only")
    ap.add_argument("--merge", type=int, default=1, help="torch backend: fold the LoRA in fp32 at load (kev's default)")
    ap.add_argument("--engine-dir")
    ap.add_argument("--score-buckets", default="64,256,1024")
    ap.add_argument("--max-prefix", type=int, default=384)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--max-branch", type=int, default=None)
    ap.add_argument("--mixed-fp16", action="store_true", help="the engines were built with trt_probe --mixed-fp16")
    ap.add_argument("--raw-engines", action="store_true", help="load plain .engine files (scripts/trt_extract_engine.py)")
    ap.add_argument("--parity-ref", default=None, help="fp32 reference (trt_probe --make-reference) checked before timing")
    ap.add_argument("--load-device")
    ap.add_argument("--executorch-dir", default=os.environ.get("EXECUTORCH_DIR"))
    ap.add_argument("--suite", default=str(HERE.parent / "evals/v7/decision-v7"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--levels", default="1,8,32,64")
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--deviation", default=None, help="free-text note recorded in the report when the run departs from "
                                                      "upstream's procedure (e.g. fewer client levels)")
    ap.add_argument("--throughput-samples", default=None,
                    help="comma-separated substrings of the upstream sample names to run (e.g. '2,200' for a long-state program)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cpu" if a.backend == "executorch" else "cuda"   # an ExecuTorch program runs where its delegate runs
    from kev.serve import Server
    t0 = time.perf_counter()
    ck, tok, model, info = build_model(a, device)
    import platform
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU {platform.machine()} ({os.cpu_count()} cores)"
    report = {"run": a.run, "device": dev_name, "torch": torch.__version__, **info,
              "load_s": round(time.perf_counter() - t0, 1), "procedure": "kev scripts/serving_bench.py (latency: CASES, median of reps, new/cached)",
              "reps": a.reps, "levels": a.levels}
    if a.deviation:
        report["deviation_from_upstream_procedure"] = a.deviation
    try:
        import tensorrt; report["tensorrt"] = tensorrt.__version__
    except Exception:
        pass
    if a.parity_ref:                                   # accuracy through this exact program, before any timing
        import trt_probe as tp
        ref = json.loads(Path(a.parity_ref).read_text())
        report["parity_vs_fp32"] = tp.parity(model, tok, tp.records(ref["records"] if Path(ref["records"]).exists()
                                                                    else str(HERE.parent / "evals/smoke-v1/development.jsonl"), ref["n"]), ref["probs"])
        print(f"[parity] {json.dumps(report['parity_vs_fp32'])}", flush=True)
    server = Server(ck, tok, model, device)
    save = lambda: (Path(a.out).parent.mkdir(parents=True, exist_ok=True), Path(a.out).write_text(json.dumps(report, indent=2)))
    try:
        if not a.skip_latency:
            report["latency"] = latency_cases(server, a.reps); save()
        if not a.skip_throughput:
            only = [x for x in a.throughput_samples.split(",")] if a.throughput_samples else None
            report["throughput"] = throughput(server, a.suite, [int(x) for x in a.levels.split(",")], only); save()
        if device == "cuda":
            report["peak_cuda_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        else:
            report["peak_rss_mb"] = round(int(next(l for l in open("/proc/self/status") if l.startswith("VmHWM")).split()[1]) / 1024, 1)
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"; report["trace"] = traceback.format_exc()[-2000:]
    finally:
        server.close()
    save()
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
