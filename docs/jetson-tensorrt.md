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

## Status (2026-09-28)

| piece | state |
|---|---|
| executorch#23023 | merged 2026-09-23 |
| kev#69 | open, draft (this branch is based on it) |
| TensorRT-lowerable DeltaNet recurrence (`kev/trt_ops.py`) | done, exact to fp32 noise |
| dynamic-shape program for TensorRT ≥ 10.13 (`kev/trt_program.py`) | compiles on x86 (TRT 10.13 via Torch-TensorRT 2.15 nightly); parity/timing run in progress |
| static-shape program for TensorRT 10.3 / JetPack 6 (`kev/trt_static.py`) | eager parity exact; Orin AGX and Orin Nano builds in progress |
| probe (`scripts/trt_probe.py`) | done |
| `.pte` via `output_format="executorch"` | x86 only for now — see *Jetson toolchain* |
| results | `runs/trt-jetson/` (filled in as runs finish) |

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
   recurrence **builds in 16 s with static shapes**. TensorRT 10.13 builds the dynamic program.
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
6. **TF32.** TensorRT enables TF32 for fp32 GEMMs by default; that moved the recurrent state by 4e-4 on Orin. With
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
| **dynamic, TensorRT 10.13 engines** (TF32 off) | RTX 6000 Ada | **1.04e-6** | 2.1e-7 | **0** |
| static, TensorRT 10.3 engines | Orin AGX / Orin Nano | *building* | | |

## Latency, reference request (kev README shape: 270-token state, 5 questions × 3 options)

| program | device | p50 | p99 | peak GPU | note |
|---|---|---|---|---|---|
| dynamic, TensorRT 10.13 fp32 | RTX 6000 Ada | 205.8 ms | 223.8 ms | 2.9 GB | shared GPU; energy not quoted |
| static, TensorRT 10.3 fp32 | Orin AGX / Nano | *building* | | | |

For scale: kev's own PyTorch bf16 path on the same card and request shape is ~58 ms (RT-jev measurements). The TensorRT
program is fp32 end to end — the only precision that has passed parity for this model under TensorRT so far — and pads
its recurrence to the exported maximum, so it is not expected to beat bf16 PyTorch on a desktop GPU. The question it
answers is the Jetson one: whether the current Qwen3.5 family can run under TensorRT there at all.

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

## Open

* TensorRT-engine parity, latency, memory and energy on all three devices (running).
* Re-run the x86 dynamic program with TF32 off (the in-flight run predates that fix).
* Write the x86 program as a `.pte` (`torch_tensorrt.save(output_format="executorch")`) and read it back through
  `ExecuTorchDecisionModel` with the ExecuTorch runtime — the exact artifact the issue suggested.
* Full-partition parity (`scripts/backend_parity.py`, decision-v7 dev) once the engines pass smoke-v1.
* Kev-4B.
