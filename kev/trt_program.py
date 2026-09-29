"""TensorRT-lowerable Qwen3.5 Kev backbone for an ExecuTorch program (`prefill` / `score`, the contract of
kev.executorch_model).

It is ExecuTorch's `examples/kev/model.py` Backbone (pytorch/executorch#23023) with one change: the DeltaNet recurrence
is `kev.trt_ops.gated_delta_rule` (static chunk count, matmul-only, exact) instead of a custom op TensorRT cannot run.
Each method gets its own padding target for the recurrence (`max_len`): the state length it was exported for in
`prefill`, the longest branch row in `score`.
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .trt_ops import gated_delta_rule

CHUNK = 64


def load_upstream_model(executorch_dir=None):
    """Import pytorch/executorch's examples/kev/model.py (Backbone, Prefill, Score) from a checkout."""
    d = Path(executorch_dir or os.environ.get("EXECUTORCH_DIR", "")) / "examples" / "kev" / "model.py"
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: set EXECUTORCH_DIR to a pytorch/executorch checkout (>= #23023)")
    spec = importlib.util.spec_from_file_location("executorch_kev_model", d)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def round_up(n, m=CHUNK):
    return -(-n // m) * m


def make_backbone_class(upstream):
    class TRTBackbone(upstream.Backbone):
        """upstream Backbone, backend "tensorrt": explicit SDPA mask for attention, static-shape delta rule."""

        def __init__(self, lm, max_len):
            nn.Module.__init__(self)
            if lm.config.model_type != "qwen3_5_text":
                raise ValueError("This backend supports Kev's dense Qwen3.5 backbone")
            self.embed_tokens, self.layers, self.norm = lm.embed_tokens, lm.layers, lm.norm
            self.register_buffer("inv_freq", lm.rotary_emb.inv_freq.float().clone())
            self.backend = "tensorrt"           # anything but "mlx" takes upstream's explicit-mask SDPA path
            self.max_len = round_up(max_len)

        def _linear_attention(self, attn, x, conv, recurrent):
            batch, length, _ = x.shape
            qkv = attn.in_proj_qkv(x).transpose(1, 2)
            if conv is None:
                conv = qkv.new_zeros(batch, attn.conv_dim, attn.conv_kernel_size)
                recurrent = torch.zeros(batch, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim,
                                        device=x.device, dtype=torch.float32)
            history = torch.cat((conv.expand(batch, -1, -1), qkv), dim=-1)
            conv_out = history[:, :, -attn.conv_kernel_size:].contiguous()
            qkv = F.silu(F.conv1d(history, attn.conv1d.weight, groups=attn.conv_dim)[:, :, -length:]).transpose(1, 2)
            q, k, v = qkv.split((attn.key_dim, attn.key_dim, attn.value_dim), dim=-1)
            # unflatten the last dim, not reshape: reshaping these non-contiguous slices has to decide view-vs-copy, and
            # that depends on whether `length` is 1 (size-1 dims can have any stride), so torch.export would guard
            # length != 1 and rule out a one-token prefill / one-question score in the exported program
            q = upstream.l2norm(q.unflatten(-1, (attn.num_k_heads, attn.head_k_dim)).float())
            k = upstream.l2norm(k.unflatten(-1, (attn.num_k_heads, attn.head_k_dim)).float())
            v = v.unflatten(-1, (attn.num_v_heads, attn.head_v_dim)).float()
            q = q * attn.head_k_dim ** -0.5
            repeats = attn.num_v_heads // attn.num_k_heads
            if repeats > 1:
                q = q.repeat_interleave(repeats, dim=2)
                k = k.repeat_interleave(repeats, dim=2)
            beta = attn.in_proj_b(x).sigmoid().float()
            g = -attn.A_log.float().exp() * F.softplus(attn.in_proj_a(x).float() + attn.dt_bias)
            # the state goes in unexpanded: batch 1 from the prefix cache (static); gated_delta_rule repeats it per row.
            # expand() here would make the batch symbolic and cost the exported program its one-question score.
            y, recurrent_out = gated_delta_rule(q, k, v, g, beta, recurrent, max_len=self.max_len, chunk=CHUNK)
            z = attn.in_proj_z(x).reshape(-1, attn.head_v_dim)
            y = attn.norm(y.to(x.dtype).reshape(-1, attn.head_v_dim), z)
            return attn.out_proj(y.reshape(batch, length, -1)), conv_out, recurrent_out

    return TRTBackbone
