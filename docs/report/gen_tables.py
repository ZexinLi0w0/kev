"""Generate the report's LaTeX tables directly from the measurement JSON files (no hand transcription)."""
import json, glob, math
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "runs" / "trt-jetson"
out = []
esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
CASES = ["2 questions, short state", "6 questions, short state", "5 questions, 370-token state", "5 questions, 2,200-token state"]
DEV = {"server3": "RTX 6000 Ada", "orin1": "Orin AGX 32GB", "nano2": "Orin Nano 8GB"}

def backend(d):
    b = d.get("backend")
    if b == "torch": return "PyTorch " + d.get("dtype", "bf16") + ("" if d.get("lora_merged", True) else " (LoRA unmerged)")
    if b == "trt-dynamic": return "TensorRT " + d.get("tensorrt", "").split(".0.99")[0] + " dyn. fp32"
    if b == "trt-static":
        p = "mixed fp16" if "fp16" in str(d.get("precision", "")) else "fp32"
        return f"TensorRT {d.get('tensorrt','')} static P={d.get('max_prefix')} {p}"
    if b == "executorch": return "ExecuTorch XNNPACK fp32 (CPU)"
    return b

def lat(v):
    if not v: return "--"
    if "inadmissible" in v: return "n/a$^\\dagger$"
    if "error" in v: return "error"
    return f"{v['new_ms']:.1f} / {v['cached_ms']:.1f}"

rows = []
for f in sorted(glob.glob(str(R / "serving/*.json")) + glob.glob(str(R / "executorch/*.latency.json"))):
    d = json.load(open(f)); host = Path(f).name.split("-")[0]
    if not d.get("latency") and not d.get("error"): continue
    model = d.get("run", "").replace("jaredpalmer/", "Kev-").replace("kev-", "").upper().replace("KEV-", "Kev-")
    rows.append((DEV.get(host, host), model, backend(d), d))
order = {"RTX 6000 Ada": 0, "Orin AGX 32GB": 1, "Orin Nano 8GB": 2}
rows.sort(key=lambda r: (order.get(r[0], 9), r[1], r[2]))

# ---- latency table
t = [r"\begin{table*}[t]\centering\small", r"\caption{Latency, kev README serving procedure: median model time per request (ms), \textbf{new state / same state again}. $^\dagger$ state longer than the program admits.}\label{tab:latency}",
     r"\begin{tabular}{lllcccc}\toprule", r"Device & Model & Path & 2 q, short & 6 q, short & 5 q, 370 tok & 5 q, 2{,}200 tok \\\midrule"]
for dev, m, be, d in rows:
    L = d.get("latency") or {}
    cells = [lat(L.get(c)) for c in CASES] if L else ["does not run"] * 4
    t.append(f"{dev} & {m} & {esc(be)} & " + " & ".join(cells) + r" \\")
t += [r"\bottomrule\end{tabular}\end{table*}"]
out.append("\n".join(t))

# ---- throughput table
t = [r"\begin{table*}[t]\centering\small", r"\caption{Throughput (requests/s) at 1 / 8 / 32 / 64 concurrent clients, and p50 / p99 latency (ms) at 64 clients. Rejected = requests with a question row longer than the program's largest engine.}\label{tab:throughput}",
     r"\begin{tabular}{llllcc c}\toprule", r"Device & Model & Path & Traffic & req/s @1/8/32/64 & p50/p99 @64 & rej. \\\midrule"]
for dev, m, be, d in rows:
    tp = d.get("throughput") or {}
    for n in sorted({k.split(" @ ")[0] for k in tp}):
        v = tp.get(n)
        if isinstance(v, dict) and ("error" in v or "inadmissible" in v):
            t.append(f"{dev} & {m} & {esc(be)} & {esc(n)} & " + ("error" if "error" in v else "n/a") + r" & & \\"); continue
        lv = [tp.get(f"{n} @ {c} clients") for c in (1, 8, 32, 64)]
        if not any(lv): continue
        rps = " / ".join(f"{x['requests_per_s']:.2f}" if x else "--" for x in lv)
        last = lv[-1] or {}
        t.append(f"{dev} & {m} & {esc(be)} & {esc(n)} & {rps} & {last.get('p50_ms','--')} / {last.get('p99_ms','--')} & {(lv[0] or {}).get('rejected',0)} \\\\")
t += [r"\bottomrule\end{tabular}\end{table*}"]
out.append("\n".join(t))

# ---- energy
t = [r"\begin{table}[t]\centering\small", r"\caption{Energy per request (J) on the Jetson power rails (Orin AGX: \texttt{VIN\_SYS\_5V0}; Orin Nano: \texttt{VDD\_IN}, whole module). Compare within a board only.}\label{tab:energy}",
     r"\begin{tabular}{lllcccc}\toprule", r"Device & Model & Path & 2 q & 6 q & 370 tok & 2{,}200 tok \\\midrule"]
for dev, m, be, d in rows:
    L = d.get("latency") or {}
    def e(v):
        pw = (v or {}).get("power") or {}
        rail = next((r for r in pw if r.upper() in ("VIN_SYS_5V0", "VDD_IN")), None)
        return f"{pw[rail]['j_per_request']:.2f}" if rail else "--"
    vals = [e(L.get(c)) for c in CASES]
    if any(x != "--" for x in vals) and dev != "RTX 6000 Ada":
        t.append(f"{dev} & {m} & {esc(be)} & " + " & ".join(vals) + r" \\")
t += [r"\bottomrule\end{tabular}\end{table}"]
out.append("\n".join(t))

# ---- quantization
t = [r"\begin{table*}[t]\centering\small", r"\caption{Low-precision quantization (NVIDIA ModelOpt 0.47) of the static mixed-fp16 Kev-0.8B program: backbone Linear layers only. Parity vs fp32 PyTorch on smoke-v1 (max $|\Delta p|$ / argmax flips, 32 questions).}\label{tab:quant}",
     r"\begin{tabular}{llcllcc}\toprule", r"Config & Device & Fake-quant & TensorRT build & Engine parity & p50 (ms) & Peak GPU (GB) \\\midrule"]
names = {"INT8_WEIGHT_ONLY_CFG": "INT8 weight-only", "INT8_SMOOTHQUANT_CFG": "INT8 SmoothQuant (W8A8)", "INT4_AWQ_CFG": "INT4 AWQ",
         "INT4_BLOCKWISE_WEIGHT_ONLY_CFG": "INT4 blockwise W-only", "FP8_DEFAULT_CFG": "FP8"}
def par(p):
    if not p: return "--"
    m = p["max_abs_dp"]; m = "NaN" if (isinstance(m, float) and math.isnan(m)) else f"{m:.3f}"
    return f"{m} / {p['argmax_flips']}"
for f in ("orin1-quant.json", "orin1-quant-int4bw.json", "server3-quant.json"):
    d = json.load(open(R / "quant" / f)); dev = f"{DEV[f.split('-')[0]]} (sm\\_{d['capability'].replace('.','')})"
    for c, v in d["configs"].items():
        fail = (v.get("build_failed") or {}).get("prefill")
        if fail:
            if "FP8" in c and "Orin" in dev: b = "failed: no FP8 hardware"
            elif "num_bits=4" in fail: b = "failed: no 4-bit converter"
            elif "fake_tensor_quant" in fail: b = "failed: env (ModelOpt ext.)"
            else: b = "failed"
        else: b = "built" if v.get("build_s") else "--"
        tr = v.get("timing_reference_request") or {}
        pk = tr.get("cuda_peak_alloc_mb"); pk = f"{pk/1024:.2f}" if pk else "--"
        t.append(f"{names.get(c,c)} & {dev} & {par(v.get('fake_quant_parity'))} & {b} & {par(v.get('trt_parity'))} & {tr.get('p50_ms','--') if tr else '--'} & {pk} \\\\")
t += [r"\bottomrule\end{tabular}\end{table*}"]
out.append("\n".join(t))
Path(__file__).with_name("tables.tex").write_text("\n\n".join(out) + "\n")
print("tables.tex written:", len(rows), "serving rows")
