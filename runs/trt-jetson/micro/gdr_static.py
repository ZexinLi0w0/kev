import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/experiment/zexin/kev-fork")
from kev.trt_ops import gated_delta_rule, reference_recurrent
import torch_tensorrt
from torch.export import Dim
class M(torch.nn.Module):
    def forward(self, q, k, v, g, beta, s):
        return gated_delta_rule(q, k, v, g, beta, s, max_len=384)
dev = "cuda"; H, K, V = 16, 128, 128
def inputs(L):
    q = F.normalize(torch.randn(1, L, H, K, device=dev), dim=-1) * K**-0.5
    k = F.normalize(torch.randn(1, L, H, K, device=dev), dim=-1)
    return q, k, torch.randn(1, L, H, V, device=dev), -F.softplus(torch.randn(1, L, H, device=dev)), torch.rand(1, L, H, device=dev), torch.zeros(1, H, K, V, device=dev)
ex = inputs(100)
L = Dim("L", min=1, max=384)
ex = inputs(384); ep = torch.export.export(M(), ex, strict=True)
spec = lambda shp, d: torch_tensorrt.Input(shape=(1, 384) + shp, dtype=d)
ins = [spec((H, K), torch.float32), spec((H, K), torch.float32), spec((H, V), torch.float32), spec((H,), torch.float32), spec((H,), torch.float32),
       torch_tensorrt.Input(shape=(1, H, K, V), dtype=torch.float32)]
t = time.time()
gm = torch_tensorrt.dynamo.compile(ep, arg_inputs=ins, enabled_precisions={torch.float32}, min_block_size=1, truncate_double=True, disable_tf32=True)
print(f"GDR compiled with TRT {__import__('tensorrt').__version__} in {time.time()-t:.1f}s", flush=True)
x = inputs(384)
o, s = gm(*x); o_r, s_r = reference_recurrent(*x)
print("TRT vs reference: out", (o - o_r).abs().max().item(), "state", (s - s_r).abs().max().item())
