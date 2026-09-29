# TensorRT backend and NVIDIA Jetson support — work log

Branch `jetson-tensorrt` of the `ZexinLi0w0/kev` fork, based on
[jaredpalmer/kev#69](https://github.com/jaredpalmer/kev/pull/69) (`executorch/export-contract`).
Tracking issue: [jaredpalmer/kev#71](https://github.com/jaredpalmer/kev/issues/71). This branch has not been proposed
upstream and the issue has not been updated from it.

## Goal

Issue #71 asks for Kev on NVIDIA Jetson: end-to-end latency, memory, GPU utilisation and throughput, installation on
aarch64 / JetPack, and a TensorRT backend compared against PyTorch. On the issue,
mergennachin suggested waiting for #69 and
pytorch/executorch#23023, then trying the Torch-TensorRT backend, which can write an ExecuTorch `.pte`
(`torch_tensorrt.save(..., output_format="executorch")`).

That route means: take ExecuTorch's `examples/kev` model (Qwen3.5 backbone with the LoRA merged, the pointer head, the
checkpoint temperature; methods `prefill` and `score`), compile it with Torch-TensorRT, and load it through #69's
`kev.executorch_model.ExecuTorchDecisionModel`, so the TensorRT program is held to the same full-partition parity bar
as the XNNPACK and MLX programs.

## Status (2026-09-29)

| piece | state |
|---|---|
| executorch#23023 | merged 2026-09-23 |
| kev#69 | open, draft (this branch is based on it) |
| TensorRT-lowerable DeltaNet recurrence (`kev/trt_ops.py`) | done, exact to fp32 noise; 11 unit tests |
| dynamic-shape program (`kev/trt_program.py`) | x86, TensorRT 11.3: parity 1.0e-6 / 0 flips; README benchmark done |
| static-shape program (`kev/trt_static.py`), fp32 | Orin AGX, TensorRT 10.3: README benchmark done |
| static program, mixed fp16 (`--mixed-fp16`) | Orin AGX and **Orin Nano**: README benchmark done |
| raw-engine runtime for 8 GB boards (`kev/trt_runtime.py`) | done — what makes the Nano run at all |
| kev's own PyTorch path, same benchmark | server3 (0.8B); AGX (0.8B / 4B / 9B); Nano: does not run |
| ExecuTorch XNNPACK `.pte` on the Orin AGX CPU | done: parity 3.0e-6, 0 flips; latency via #69's backend; throughput not measured (stopped — deviation noted) |
| TensorRT-delegate `.pte` (`output_format="executorch"`) | x86: export in progress; Jetson: blocked (no exporter / runtime build for JetPack 6) |
| INT8 / INT4 / FP8 (ModelOpt) on the Orin AGX | INT8 weight-only and SmoothQuant built, 0 flips; INT4 fails (accuracy + converter); FP8 refused by the hardware |
| results | `runs/trt-jetson/serving/README.md` (all tables), raw JSON and logs in `runs/trt-jetson/` |

## Why upstream's program cannot go to TensorRT as is

`examples/kev/model.py` runs the Gated DeltaNet recurrence as a custom op: `llama::gated_delta_rule` (ExecuTorch CPU
kernel) for XNNPACK, `mlx::gated_delta_rule` for MLX. TensorRT can execute neither. transformers' pure-PyTorch
`torch_chunk_gated_delta_rule` is not compilable either: a Python loop over a data-dependent number of chunks, and a
sequential in-place forward substitution inside each chunk.

### `kev/trt_ops.py` — the recurrence TensorRT can compile

The same chunked algorithm (chunk 64) with:

* **a static chunk loop** — the sequence is right-padded to a fixed `max_len`; pads get β = 0 and log-decay 0, so
  they neither update nor decay the recurrent state, and their outputs are cut before returning;
* **a matmul-only, exact within-chunk inverse** — (I + N)⁻¹ for strictly lower-triangular N by blocked recursion,
  [[A,0],[B,D]]⁻¹ = [[A⁻¹,0],[−D⁻¹BA⁻¹, D⁻¹]], log₂C levels;
* **no `torch.where`, no `cumsum`** — masks are multiplications by constant 0/1 matrices (the exponent clamped first
  so a masked entry is exp(0)·0, never inf·0), and the cumulative log-decay is a matmul with a constant
  upper-triangular ones matrix. Both were needed for TensorRT 10.3 (below).

Verified against the token-by-token definition on random and adversarial inputs (near-identical keys, β≈1, almost no
decay): max |Δ| ≤ 6e-7 on outputs, ≤ 9e-6 on the state, all finite.

## Problems hit, in order, and what fixed them

1. **The power-series inverse is exact but unstable.** First version used (I+N)⁻¹ = ∏(I + (−N)^(2^i)) (N is
   nilpotent). It passed random tests but, on real Kev activations, gave NaN probabilities and 8–9 flipped answers of
   40: L2-normalised keys of neighbouring tokens make N's entries ≈ 1, powers of N grow like binomial coefficients
   (~1e17 at C = 64), and the small true inverse is left as a difference of huge terms. Replaced with the blocked
   recursion above.
2. **TensorRT 10.3 (JetPack 6) cannot build the dynamic-shape graph.** Torch-TensorRT 2.8 converts it, then the
   builder returns no engine: `Could not find any implementation for node {ForeignNode[...]}` / "No Myelin Error
   exists". Capping the workspace (8 GB) did not help. The failing node moved as each op was rewritten
   (`where` → `cumsum` → `select`), which is what identified the dynamic region itself as the cause: the same
   recurrence **builds in 16 s with static shapes**. TensorRT 11.3 (the version tested on x86) builds the dynamic program; versions in between were not tested.
3. **The engine rejected one-question requests.** `torch.export` records `Dim(min=1)` as `min=2` (0/1
   specialisation), so the dynamic engine's question and option dimensions start at 2. The probe duplicates a single
   row / option and slices the result back.
4. **The fp32 embedding table does not fit an 8 GB Orin Nano as an engine constant** (~1 GB, 248k × 1024 × 4 B;
   `'...-consts' region allocation failed`). The static program runs the embedding lookup in PyTorch and feeds the
   engines embeddings; the Nano build also exports every method first and compiles with `offload_module_to_cpu`, and
   the LoRA merge happens on the CPU (`--load-device cpu`).
5. **One failed bucket cost a whole run** (the 1024-row score engine on the AGX after three good builds, ~30 min).
   Built engines are now cached on disk (`--engine-dir`) and a bucket that fails is skipped; questions that need it
   count as rejected.
6. **Loading, not running, is what did not fit the Orin Nano.** Torch-TensorRT's loader (`torch.export.load`) holds
   the archive, the base64 text of the engine and the deserialised engine at once (~3x the engine) before the device
   copy; the Nano failed to load a 1.3 GB fp16 engine asking for 962 MB it did not have. The engines themselves need
   273 MB (prefill) and 155 MB (score128) of activation memory. `scripts/trt_extract_engine.py` writes the raw
   engine (base64-decoded) once, on a board with the same TensorRT; `kev/trt_runtime.py` deserialises it directly and
   gives all engines one shared activation buffer (they never run concurrently). Peak GPU memory on the Nano: 1.0 GB.
   The engines were built on the Orin AGX (same sm_87, same TensorRT 10.3.0) and copied over; the Nano cannot hold the
   PyTorch weights and an engine under construction at the same time. The serving path also never loads the backbone
   (`build_light`: tokenizer, embedding tensor, head) — loading it only to drop it left ~3 GB of host allocations on the
   Nano's shared DRAM.
7. **Mixed fp16.** fp16 GEMMs with the recurrence, norms and pointer head in fp32, as a strongly typed engine
   (`use_explicit_typing`: TensorRT keeps the dtypes the graph gives it). Storing fp16 weights while computing in fp32
   does not save memory: Torch-TensorRT constant-folds the cast back to fp32.
8. **TF32.** TensorRT enables TF32 for fp32 GEMMs by default; that moved the recurrent state by 4e-4 on Orin. With
   `disable_tf32=True` it is 9e-7. The probe now always disables it: a decision model's output is the probability.

### `kev/trt_static.py` — the JetPack 6 program

Every shape fixed at export, padding made exact by explicit lengths:

* `prefill(tokens[1,P], n)` — pads get β = 0 / no decay in DeltaNet layers; the conv state is gathered at the true
  end; attention is causal so pads (after every real token) are never seen; pad keys/values are dropped outside.
* `score(rows[Q,B], n, conv, recurrent, kv[P])` — prefix keys past the true length are masked additively and the
  rows' rotary positions start at the true length; returns hidden states. The pointer head (two Linear layers) and
  the option gather run in PyTorch in fp32, so engines do not depend on the number of options.
* one `prefill` engine (P = 384, Kev's training state limit) and one `score` engine per branch bucket
  (64 / 256 / 1024 on AGX; 64 / 256 / 512 on Nano).
* `StaticProgram` exposes the engines through the ExecuTorch Method interface, so `ExecuTorchDecisionModel` scores
  through it unchanged: same encoder, same `rows_of`, same chunking of >8 questions, same contract checks.

## Parity (Kev-0.8B, `evals/smoke-v1` development, 30 records / 40 questions, vs fp32 PyTorch `DecisionModel`)

| program | device | max \|Δp\| | mean | flips |
|---|---|---|---|---|
| dynamic, eager | RTX 6000 Ada | 7.7e-7 | 1.6e-7 | 0 |
| dynamic, eager | Orin AGX | 1.2e-6 | 3.0e-7 | 0 |
| static, eager | RTX 6000 Ada | 1.1e-6 | 2.6e-7 | 0 |
| static, eager | Orin AGX | 9.2e-7 | 2.6e-7 | 0 |
| **dynamic, TensorRT 11.3 engines** (TF32 off) | RTX 6000 Ada | **1.04e-6** | 2.1e-7 | **0** |
| static, TensorRT 10.3 engines | Orin AGX / Orin Nano | *building* | | |

## Latency, reference request (kev README shape: 270-token state, 5 questions × 3 options)

| program | device | p50 | p99 | peak GPU | note |
|---|---|---|---|---|---|
| dynamic, TensorRT 11.3 fp32 | RTX 6000 Ada | 205.8 ms | 223.8 ms | 2.9 GB | shared GPU; energy not quoted |
| static, TensorRT 10.3 fp32 | Orin AGX / Nano | *building* | | | |

For scale: kev's own PyTorch bf16 path on the same card and request shape is ~58 ms (RT-jev measurements). The TensorRT
program is fp32 end to end — the only precision that has passed parity for this model under TensorRT so far — and pads
its recurrence to the exported maximum, so it is not expected to beat bf16 PyTorch on a desktop GPU. The question it
answers is the Jetson one: whether the current Qwen3.5 family can run under TensorRT there at all.

## Results — kev's README serving benchmark

Measured with kev's own `scripts/serving_bench.py` procedure (imported, not re-written; `scripts/trt_serving_bench.py`
only swaps the model behind `kev.serve.Server`): model time per request (`latency_ms`), median of 20 after two warm-ups,
**new state / same state again**, on upstream's four request shapes; then requests/s at 1 / 8 / 32 / 64 clients. Full
tables including throughput and energy: [`runs/trt-jetson/serving/README.md`](../runs/trt-jetson/serving/README.md).

Kev-0.8B, latency in ms (new / cached):

| device | path | 2 q, short | 6 q, short | 5 q, 370 tok | 5 q, 2,200 tok |
|---|---|---|---|---|---|
| RTX 6000 Ada | TensorRT 11.3, fp32, dynamic | **46.1 / 11.2** | **71.7 / 36.1** | **81.8 / 31.2** | **234.6 / 35.5** |
| RTX 6000 Ada | PyTorch bf16 (kev.serve, reference DeltaNet) | 94.3 / 48.5 | 105.1 / 59.9 | 128.8 / 57.7 | 278.6 / 60.4 |
| Orin AGX | PyTorch bf16 (kev.serve, flash-linear-attention) | 245.9 / **126.1** | 250.8 / **130.8** | 250.0 / **129.9** | **448.4 / 133.2** |
| Orin AGX | TensorRT 10.3, mixed fp16, static P=384 | **235.4** / 155.4 | **235.6** / 156.4 | **235.5** / 156.4 | n/a (state > 384) |
| Orin AGX | TensorRT 10.3, fp32, static P=384 | 561.9 / 392.6 | 562.0 / 393.8 | 562.1 / 393.6 | n/a |
| Orin AGX | TensorRT 10.3, fp32, static P=2240 | 1342.2 / 444.0 | 1342.0 / 446.4 | 1341.4 / 446.2 | 1342.7 / 446.8 |
| **Orin Nano** | **TensorRT 10.3, mixed fp16, static P=384** | **766.2 / 513.3** | **767.0 / 514.7** | **767.2 / 514.8** | n/a |
| Orin Nano | PyTorch bf16 (kev.serve) | does not run | | | |

Larger checkpoints on the Orin AGX, kev's PyTorch path (bf16): Kev-4B 315.9 / 162.1 · 421.2 / 266.8 · 453.0 / 233.8 ·
1511.5 / 288.1 ms; Kev-9B (LoRA unmerged: the fp32 merge needs ~36 GB) 472.3 / 240.3 · 724.4 / 492.8 · 811.7 / 411.6 ·
2798.8 / 466.7 ms.

Parity of each TensorRT program against fp32 PyTorch (`evals/smoke-v1`, measured through the program itself):

| program | device | max \|Δp\| | flips |
|---|---|---|---|
| dynamic, fp32 | RTX 6000 Ada | 1.0e-6 | 0 / 40 |
| static, fp32 (eager) | Orin AGX | 9.2e-7 | 0 / 40 |
| static, mixed fp16 | RTX 6000 Ada (TensorRT 11.3) | 4.1e-3 | 0 / 32 |
| static, mixed fp16 | **Orin Nano (TensorRT 10.3, raw-engine runtime)** | **4.8e-3** | **0 / 32** |

For scale, kev's README documents its own bf16 serving path at up to ~0.03 from fp32.

### What the numbers say

1. **The current (Qwen3.5) Kev family now runs on an 8 GB Orin Nano** — under TensorRT, mixed fp16, 1.0 GB peak GPU
   memory, 0 flips on smoke-v1, ~8.2 J per request on the module rail, 1.30 requests/s sustained. kev's own PyTorch path
   does not run there (JetPack 6's cuSOLVER cannot back the reference DeltaNet's `torch.linalg.solve_triangular`, and
   flash-linear-attention's Triton kernels cannot load against that board's torch). TensorRT is the only route found.
2. **On a desktop GPU the fp32 TensorRT program beats kev's PyTorch bf16 path** (46 vs 94 ms, 11 vs 49 ms cached on the
   short case), with fp32-exact answers. Caveat: that PyTorch baseline uses the reference DeltaNet kernels; upstream's
   README numbers use flash-linear-attention and CUDA graphs.
3. **On the Orin AGX, TensorRT only matches PyTorch, and only in fp16.** Mixed fp16 is 2.4x faster than fp32 engines
   and on par with kev's PyTorch + flash-linear-attention for a new state (235 vs 246 ms), slower for a cached one
   (156 vs 126 ms). Static shapes pay for padding: every request costs the same whatever its size.
4. **The static engines do not batch.** kev.serve batches queued requests; the TensorRT programs answer one at a time,
   so their throughput is flat from 1 to 64 clients (AGX fp16: 4.2 req/s vs PyTorch's 7.0 at 64 clients).
5. **Energy per request on the AGX** (5 V system rail): PyTorch bf16 1.3–1.7 J vs TensorRT mixed fp16 1.9–2.0 J on the
   short cases; fp32 static engines 4.2 J.

## ExecuTorch on the Jetsons

**XNNPACK (#69's validated program) runs on the Orin AGX's ARM CPU.** ExecuTorch's aarch64 wheels are cp313 / cp314 and
JetPack 6 is Python 3.10, so it runs from its own environment (`uv python install 3.13`, `executorch==1.5.1`,
`torch==2.14`, kev installed `--no-deps`); that torch is a CUDA 13 build and cannot use the Orin's CUDA 12.6 driver,
which does not matter for a CPU delegate. #69's recipe step was needed: ExecuTorch `main`'s
`backends/xnnpack/partition/config/{gemm,generic_node}_configs.py` copied over the wheel's (without it the export fails
with *"XNNPACK reshape only supports 1 dynamic dimension"*). The tokenizer step of the recipe was skipped: it fixes the
C++ tokenizer, and Kev's Python path uses the Hugging Face one. Export with ExecuTorch's own
`examples/kev/export.py --backend xnnpack --dtype fp32` on the AGX: 11 min 54 s, 3.08 GB `model.pte`, default limits
384 / 1024. Served through `kev.executorch_model.ExecuTorchDecisionModel` (`scripts/trt_serving_bench.py --backend executorch`):

| | Kev-0.8B, XNNPACK fp32, Orin AGX CPU (12x A78AE) |
|---|---|
| parity vs fp32 PyTorch (smoke-v1) | **max \|Δp\| 3.0e-6, 0 / 40 flips** (fp32-exact, as #69 reports on an M5) |
| 2 questions, short state (new / cached) | 882.7 / 659.6 ms |
| 6 questions, short state | 3406.4 / 3179.2 ms |
| 5 questions, 370-token state | 4517.4 / 2816.7 ms |
| 5 questions, 2,200-token state | n/a — the default export admits states of at most 384 tokens |

> **Deviation from upstream's procedure — no throughput for this program.** Upstream sweeps 1 / 8 / 32 / 64 clients. At
> 0.9–4.5 s per request on the CPU, with this backend answering one request at a time, that sweep would have taken ~4 h and
> would approach 1 / latency. It was stopped at the user's request and restarted at 1 client only (recorded as
> `deviation_from_upstream_procedure` in that run's report); that single-client run was then also stopped, before it
> finished, when all Orin AGX workloads were cleared. **So no throughput figure exists for the XNNPACK program**; its
> latency above is the full upstream procedure and stands. Expected single-client throughput from those latencies:
> roughly 0.2–1.1 requests/s.

**TensorRT-delegate `.pte` on Jetson: blocked** — see *Pending*: no Jetson build of the exporter
(`torch_tensorrt.executorch`, Torch-TensorRT >= 2.15) or of the runtime (`torch_tensorrt_executorch_runtime`), and a .pte's
engines load only on the TensorRT version / GPU that built them, so a server-built one cannot be used on the boards.

**TensorRT-delegate `.pte` on x86: in progress.** `scripts/trt_export_pte.py` writes `prefill` / `score` as TensorRT engines
plus the contract's constant methods; `kev/executorch_model.load_program` now registers the TensorRT delegate when
`torch_tensorrt_executorch_runtime` is installed (confirmed: the runtime then lists `TensorRTBackend`). Making the program
admit what #69's backend actually sends — a one-token `prefill` on load and one-row `score` calls — needed three changes that
leave the numbers unchanged (eager parity 8.2e-7, 0 flips; 11 unit tests pass): `unflatten` instead of a `reshape` whose
view-vs-copy decision depends on length == 1; the cached recurrent state passed un-expanded (batch 1, static); and the per-row
copy of it made with an index gather — `expand()` brings back the `batch != 1` guard and TensorRT cannot convert `repeat()`
with a dynamic count. With `backed_size_oblivious` the program now exports with questions [1, 8], options [1, 255],
prefix [1, 384].

## Low-precision quantization (NVIDIA ModelOpt 0.47, Kev-0.8B)

`scripts/trt_quant_probe.py`: the static mixed-fp16 program with ModelOpt applied to the backbone's `nn.Linear` layers only
(the DeltaNet recurrence, norms and pointer head keep their precision), calibrated on 64 decision-v7 development records
through Kev's own serving path; each config checked fake-quantized first, then as TensorRT engines on the device. Parity on
smoke-v1 (32 questions; 8 rejected by the 512-token rows of this program). Reference: mixed fp16 on the AGX, 235.5 ms p50,
2.97 GB peak.

| config | device | fake-quant max \|Δp\| / flips | TensorRT | engine parity | p50 | peak GPU |
|---|---|---|---|---|---|---|
| INT8 weight-only | Orin AGX (sm_87, TRT 10.3) | 0.012 / 0 | **built** (364 + 347 s) | **0.013 / 0** | 234.5 ms | 2.06 GB |
| INT8 SmoothQuant (W8A8) | Orin AGX | 0.071 / 0 | **built** (416 + 361 s) | 0.059 / 0 | 224.2 ms | 3.03 GB |
| INT4 AWQ (W4) | Orin AGX | 0.33 / **1** | failed: Torch-TensorRT 2.8 *"quantize converter currently only accept INT8 or FP8 based quantize, got num_bits=4"* | — | — | — |
| INT4 blockwise weight-only | Orin AGX | 0.23 / **2** | failed: same converter limit | — | — | — |
| FP8 | Orin AGX | 0.050 / 0 | **failed: TensorRT *"Networks with FP8 Q/DQ layers require hardware with FP8 support"*** | — | — | — |
| FP8 | RTX 6000 Ada (sm_89, TRT 11.3) | 0.037 / 0 | built (167 + 154 s) | **NaN, 19 / 32 flips** | 22.4 ms | 2.01 GB |
| INT8 weight-only | RTX 6000 Ada | 0.012 / 0 | failed: ModelOpt CUDA extension (`fake_tensor_quant_with_axis`) missing in that nightly env | — | — | — |

What this says:

* **INT8 works on Orin.** Weight-only INT8 stays well inside the ~0.03 kev accepts for its own bf16 serving and changes no
  answer; SmoothQuant W8A8 is looser (0.059) but also changes none. At this request size neither is faster than mixed fp16
  (224–235 vs 235 ms): the GEMMs are not the bottleneck. Weight-only INT8 lowers peak memory (2.06 vs 2.97 GB).
* **INT4 is out on two counts** at 0.8B: it already changes answers fake-quantized (1–2 of 32), and Torch-TensorRT 2.8 — the
  only Jetson build — cannot convert 4-bit Q/DQ at all.
* **FP8 on Orin fails for the hardware reason stated**, now measured: TensorRT refuses FP8 Q/DQ networks on sm_87. On an
  Ada GPU the engines build and run (22 ms) but return NaNs: ModelOpt's default FP8 recipe is not enough for this model
  (fp16 activations around the FP8 GEMMs overflow somewhere); making FP8 usable would need per-layer exclusions, which is
  not Jetson work.

## Jetson toolchain

| | Orin AGX (orin1) | Orin Nano (nano2) |
|---|---|---|
| JetPack / L4T | R36.4.7 | R36.5.0 |
| CUDA / TensorRT | 12.6 / 10.3.0 | 12.6 / 10.3.0 (runtime only; Python bindings added) |
| venv `kev-trt` | torch 2.8.0 + torch_tensorrt 2.8.0+cu126 (jetson-ai-lab jp6/cu126), numpy 1.26 | same |

* The `output_format="executorch"` path of Torch-TensorRT is in the 2.15 nightlies (CUDA 12.9/13.0, server-ARM
  builds); the only Jetson build is 2.8 (cu126), and there is no ExecuTorch runtime with the TensorRT backend for
  Jetson. So on the boards the engines are measured through Torch-TensorRT's own runtime, in the same backend class.
* JetPack's torch 2.8 wheel is built against NumPy 1.x — pin `numpy<2`.
* `--system-site-packages` leaks JetPack's old Pillow (breaks transformers imports) — install Pillow in the venv.
* Orin Nano ships TensorRT runtime libraries but not `python3-libnvinfer`; install it (`apt install
  python3-libnvinfer=10.3.0.30-1+cuda12.5`).
* kev declares Python ≥ 3.12; JetPack 6 has 3.10 and every Jetson torch wheel is cp310. The modules used here import
  fine on 3.10, so kev goes on the path (`.pth`) instead of `pip install`.

## Reproducing

```bash
# x86 reference + dynamic program (Torch-TensorRT 2.15 nightly, cu130; ExecuTorch from pytorch/executorch main)
uv venv -p 3.12 venvs/kev-trt
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 "torch_tensorrt[executorch]"
uv pip install --no-deps -e . && uv pip install "transformers>=5.17,<6" "peft>=0.21" accelerate
export EXECUTORCH_DIR=/path/to/executorch        # for examples/kev/model.py
python scripts/trt_probe.py --make-reference ref.json --run jaredpalmer/kev-0.8b
python scripts/trt_probe.py --reference ref.json --run jaredpalmer/kev-0.8b --out probe.json

# Jetson (JetPack 6): static program
python scripts/trt_probe.py --program static --score-buckets 64,256,1024 --workspace-mb 8192 \
    --reference ref.json --run jaredpalmer/kev-0.8b --out orin-agx.json
python scripts/trt_probe.py --program static --score-buckets 64,256,512 --max-context 512 --workspace-mb 2048 \
    --reference ref.json --run jaredpalmer/kev-0.8b --out orin-nano.json
```

Set `RT_JEV_BENCH` to a directory containing `power.py` (INA3221 / nvidia-smi sampler) to record energy.

## Pending

### ExecuTorch route (the one issue #71 describes)

- [~] **x86: write the TensorRT program as a `.pte`** (export in progress, see *ExecuTorch on the Jetsons*) (`torch_tensorrt.save(..., output_format="executorch")` with
      `prefill` / `score` methods and the `get_*` constant methods) and read it back through
      `kev.executorch_model.ExecuTorchDecisionModel` with the ExecuTorch runtime; full-partition parity with
      `scripts/backend_parity.py --backend executorch`. The toolchain is installed (Torch-TensorRT 2.15 nightly,
      TensorRT 11.3, ExecuTorch); the engines run today only through Torch-TensorRT's runtime.
- [ ] **Jetson: `.pte` with the TensorRT delegate.** Blocked as of 2026-09-28 by three independent missing pieces:
      1. *Exporter* -- `output_format="executorch"` / `torch_tensorrt.executorch` exists only in recent Torch-TensorRT
         (the 2.15 nightlies). NVIDIA's JetPack 6 index has Torch-TensorRT 2.8.0 (cu126) only, without it; PyTorch's
         aarch64 nightlies are server-ARM (SBSA) builds for CUDA 12.9 / 13.0, which do not run on JetPack 6's CUDA 12.6
         nor on Orin's integrated-GPU driver.
      2. *Engine* -- a TensorRT-delegate .pte embeds serialised TensorRT engines, which load only on the TensorRT
         version and GPU architecture that built them (TRT 11.3 / sm_89 on the server; the Orins need TRT 10.3 / sm_87).
         A server-exported .pte is therefore not usable on the boards; it has to be produced with Jetson's TensorRT,
         which leads back to (1).
      3. *Runtime* -- running it needs an ExecuTorch runtime with the TensorRT delegate registered, which ships as a
         separate library (`torch_tensorrt_executorch_runtime`); the stock ExecuTorch runtime registers QNN / OpenVINO /
         CUDA / XNNPACK / VGF only. No Jetson build of that library exists; it would have to be compiled from source
         against JetPack's CUDA 12.6 / TensorRT 10.3 (untested that old). ExecuTorch's aarch64 wheels are also cp313/314,
         while JetPack 6 is Python 3.10.
      Not impossible, but it needs from-source builds of both the exporter toolchain and the runtime on the board;
      JetPack 7 on Orin would likely remove most of it.
- [x] **Jetson: ExecuTorch XNNPACK `.pte` on the ARM CPU** — done on the AGX (parity 3.0e-6, 0 flips; latency above).
      Throughput not measured (stopped at the user's request); the Orin Nano was not tried for this route. (the program #69 validates, exported by
      `examples/kev/export.py --backend xnnpack`). Possible today: aarch64 ExecuTorch 1.5 wheels exist for cp313/cp314,
      so it needs a Python 3.13 venv on the board next to JetPack's 3.10. A CPU baseline for the same README cases.

### Low-precision quantization — status per format

- [x] **FP16 (mixed)** — GEMMs in fp16, DeltaNet recurrence / norms / pointer head in fp32, strongly typed engine
      (`--mixed-fp16`). Max |Δp| 4.1e-3 (server) / 4.8e-3 (Orin Nano), 0 flips; README benchmark on AGX and Nano done.
- [x] **INT8** — done on the Orin AGX (weight-only and SmoothQuant both build and keep 0 flips; results above). Tried earlier only on the attention-only `kev-0.6b` through ONNX on
      Orin (TensorRT 10.3 implicit calibration: engine identical to FP16, 10.7% flips; RT-jev work). To try: explicit
      Q/DQ (NVIDIA ModelOpt) weight-only INT8 first, then W8A8 with SmoothQuant, each held to the parity bar.
- [x] **INT4** — tried on the Orin AGX: AWQ and blockwise weight-only both change answers fake-quantized and cannot be
      converted by Torch-TensorRT 2.8. Would need a newer Torch-TensorRT on Jetson (and a larger checkpoint to hold accuracy). Weight-only INT4 (AWQ / blockwise) via ModelOpt Q/DQ; the memory win that would make the
      4B checkpoint fit an Orin Nano. Parity bar as above.
- [x] **FP8** — tried: TensorRT 10.3 refuses FP8 Q/DQ on the Orin (sm_87) — "require hardware with FP8 support";
      on an RTX 6000 Ada (sm_89) it builds but the default ModelOpt FP8 recipe gives NaNs. Background: **no FP8 compute on Orin.** Orin's GPU is Ampere-class (sm_87): its tensor cores do FP16 / BF16 / TF32 /
      INT8 but have no FP8 instructions, and TensorRT's FP8 kernels need Ada (sm_89) or Hopper (sm_90). FP8 *storage*
      with dequantisation to fp16 before each GEMM would still save memory but gives no speed-up there -- weight-only
      INT8 / INT4 does the same job and Ampere has native INT8 tensor cores. The server's RTX 6000 Ada (sm_89) can run
      FP8 natively, as a reference point.

### Other

- [x] Kev-9B on the AGX PyTorch path (`--merge 0`: bf16 without the fp32 LoRA merge, which needs ~36 GB).
- [x] Orin Nano: README benchmark on the mixed-fp16 engines (built on the AGX, raw-engine runtime).
- [ ] Full-partition parity (`scripts/backend_parity.py`, decision-v7 development) for each TensorRT program.
- [ ] Kev-0.8B 2,200-token case on the Nano (needs a P=2240 mixed-fp16 prefill engine) and request batching in the
      static programs (their throughput is flat because they answer one request at a time).
- [ ] Kev-4B under TensorRT: an fp32 build needs ~2x its 16 GB weights, more than the AGX has; mixed-fp16 or INT8/INT4
      are the routes to try.
