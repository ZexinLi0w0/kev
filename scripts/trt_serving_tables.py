"""Markdown tables from scripts/trt_serving_bench.py reports (runs/trt-jetson/serving/*.json): kev's README serving
procedure (latency: new / same state; throughput at 1 / 8 / 32 / 64 clients) per device, model and backend.

    python scripts/trt_serving_tables.py runs/trt-jetson/serving/*.json > runs/trt-jetson/serving/README.md
"""
import json
import sys
from pathlib import Path

CASES = ["2 questions, short state", "6 questions, short state", "5 questions, 370-token state", "5 questions, 2,200-token state"]
DEVICE = {"server3": "RTX 6000 Ada (server3)", "orin1": "Jetson AGX Orin 32 GB", "nano2": "Jetson Orin Nano 8 GB"}


def label(path, d):
    host = path.name.split("-")[0]
    model = next((p for p in path.stem.split("-") if p.startswith("0.") or p in ("4b", "9b")), "")
    model = "Kev-" + model.upper().replace("B", "B") if model else d.get("run", "")
    be = d.get("backend", "?")
    if be == "torch":
        be = f"PyTorch {d.get('dtype', 'bf16')} (kev.serve){'' if d.get('lora_merged', True) else ', LoRA unmerged'}"
    elif be == "trt-dynamic":
        be = f"TensorRT {d.get('tensorrt', '')} dynamic, fp32"
    elif be == "trt-static":
        prec = "mixed fp16" if "fp16" in str(d.get("precision", "")) else "fp32"
        rt = ", raw-engine runtime" if "trt_runtime" in str(d.get("runtime", "")) else ""
        be = f"TensorRT {d.get('tensorrt', '')} static P={d.get('max_prefix')}, {prec}{rt}"
    elif be == "trt-combined":
        be = f"TensorRT {d.get('tensorrt', '')} combined P={d.get('max_prefix')}"
    return DEVICE.get(host, host), d.get("run", model).replace("jaredpalmer/", ""), be


def cell(v):
    if not v:
        return "—"
    if "inadmissible" in v:
        return "n/a (too long for program)"
    if "error" in v:
        return "error"
    return f"{v['new_ms']:.1f} / {v['cached_ms']:.1f}"


def main():
    rows = []
    for f in sorted(sys.argv[1:]):
        p = Path(f)
        d = json.loads(p.read_text())
        rows.append((p, d, *label(p, d)))
    print("# kev README serving benchmark on TensorRT and Jetson\n")
    print("Procedure: kev `scripts/serving_bench.py` (imported, not re-written) — model time per request as `kev.serve.Server`")
    print("reports it (`latency_ms`), median of 20 after two warm-ups; **new state / same state again** (prefix-cache hit).\n")
    print("## Latency (ms, new / cached)\n")
    print("| device | model | backend | " + " | ".join(CASES) + " |")
    print("|---|---|---|" + "---|" * len(CASES))
    for p, d, dev, model, be in rows:
        lat = d.get("latency")
        if not lat:
            if d.get("error"):
                print(f"| {dev} | {model} | {be} | " + " | ".join(["error"] * len(CASES)) + " |")
            continue
        print(f"| {dev} | {model} | {be} | " + " | ".join(cell(lat.get(c)) for c in CASES) + " |")
    print("\n## Throughput (requests/s at 1 / 8 / 32 / 64 concurrent clients; p50 / p99 ms at 64)\n")
    print("| device | model | backend | traffic | req/s @1 / 8 / 32 / 64 | p50 / p99 @64 | rejected |")
    print("|---|---|---|---|---|---|---|")
    for p, d, dev, model, be in rows:
        tp = d.get("throughput") or {}
        names = sorted({k.split(" @ ")[0] for k in tp})
        for n in names:
            if isinstance(tp.get(n), dict) and ("error" in tp[n] or "inadmissible" in tp[n]):
                print(f"| {dev} | {model} | {be} | {n} | {('error: ' + tp[n].get('error', '')[:60]) if 'error' in tp[n] else 'n/a'} | | |"); continue
            lv = [tp.get(f"{n} @ {c} clients") for c in (1, 8, 32, 64)]
            if not any(lv):
                continue
            rps = " / ".join(f"{v['requests_per_s']:.2f}" if v else "—" for v in lv)
            last = lv[-1] or {}
            rej = (lv[0] or {}).get("rejected", 0)
            print(f"| {dev} | {model} | {be} | {n} | {rps} | {last.get('p50_ms', '—')} / {last.get('p99_ms', '—')} | {rej} |")
    print("\n## Energy per request (J, module rail on Jetson)\n")
    print("| device | model | backend | " + " | ".join(CASES) + " |")
    print("|---|---|---|" + "---|" * len(CASES))
    for p, d, dev, model, be in rows:
        lat = d.get("latency") or {}
        def e(v):
            pw = (v or {}).get("power") or {}
            rail = next((r for r in pw if r.upper() in ("VIN_SYS_5V0", "VDD_IN")), None)
            return f"{pw[rail]['j_per_request']:.3f}" if rail else "—"
        if any(e(lat.get(c)) != "—" for c in CASES):
            print(f"| {dev} | {model} | {be} | " + " | ".join(e(lat.get(c)) for c in CASES) + " |")
    print("\nRails: Orin Nano `VDD_IN` (whole-module input); Orin AGX `VIN_SYS_5V0` (5 V system rail; the GPU/SoC and CPU")
    print("rails are in each JSON). Different scopes, so compare energy within a board, not across boards. The desktop GPU")
    print("(server3) is shared and `nvidia-smi` sees card power only; its energy is not reported.")


if __name__ == "__main__":
    main()
