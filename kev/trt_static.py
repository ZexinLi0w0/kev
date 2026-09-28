"""Static-shape TensorRT program for Kev's Qwen3.5 backbones, for TensorRT versions that cannot build the dynamic one.

JetPack 6 ships TensorRT 10.3. Its Myelin compiler builds the delta rule of `kev.trt_ops` with static shapes but fails on
every dynamic-shape variant with "Could not find any implementation for node {ForeignNode[...]}" (the failing node moves
from op to op as each is rewritten, which is how the dynamic region itself was identified as the cause). TensorRT 10.13
builds the dynamic program (`kev.trt_program`). This module is the same computation with every shape fixed at export and
the padding made exact by explicit lengths:

* `prefill` runs a state padded to `max_prefix` tokens with its true length `n`. In the DeltaNet layers the pads get
  beta = 0 and log-decay 0, so they neither update nor decay the recurrent state; the convolution state is gathered at
  the true end of the sequence; attention is causal, so pads (after every real token) are never seen. The keys/values
  of the pads are returned too and dropped outside the engine.
* `score` runs up to `max_questions` right-padded rows of `branch` tokens against a prefix padded to `max_prefix`: prefix
  keys past the true length are masked, and the rows' rotary positions start at the true length, not at the padded one.
  It returns hidden states; the pointer head (two Linear layers, microseconds) and the option gather run in PyTorch in
  fp32, so the engine does not depend on the number of options.

`StaticProgram` wraps the engines in the ExecuTorch Method interface `kev.executorch_model.ExecuTorchDecisionModel`
expects, choosing the smallest `score` bucket that fits each call, so Kev's backend scores through it unchanged.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .trt_ops import gated_delta_rule

CHUNK = 64


class StaticBackbone(nn.Module):
    def __init__(self, lm, l2norm, apply_rotary_pos_emb):
        super().__init__()
        if lm.config.model_type != "qwen3_5_text":
            raise ValueError("expected Kev's dense Qwen3.5 backbone")
        self.embed_tokens, self.layers, self.norm = lm.embed_tokens, lm.layers, lm.norm
        self.register_buffer("inv_freq", lm.rotary_emb.inv_freq.float().clone())
        self._l2norm, self._rope = l2norm, apply_rotary_pos_emb

    def _linear(self, attn, x, conv, recurrent, valid, conv_end):
        """valid [B, L] 1/0; conv_end [B] index (in history coordinates) one past the last real token."""
        batch, length, _ = x.shape
        qkv = attn.in_proj_qkv(x).transpose(1, 2)
        if conv is None:
            conv = qkv.new_zeros(batch, attn.conv_dim, attn.conv_kernel_size)
            recurrent = torch.zeros(batch, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim, device=x.device, dtype=torch.float32)
        history = torch.cat((conv.expand(batch, -1, -1), qkv), dim=-1)            # [B, C, kernel + L]
        # real token t sits at history position kernel + t, so the `kernel` entries ending at the real end n are
        # history[n : n + kernel]
        idx = conv_end[:, None] + torch.arange(attn.conv_kernel_size, device=x.device)[None]
        conv_out = torch.gather(history, 2, idx[:, None, :].expand(-1, history.shape[1], -1)).contiguous()
        qkv = F.silu(F.conv1d(history, attn.conv1d.weight, groups=attn.conv_dim)[:, :, -length:]).transpose(1, 2)
        q, k, v = qkv.split((attn.key_dim, attn.key_dim, attn.value_dim), dim=-1)
        q = self._l2norm(q.reshape(batch, length, -1, attn.head_k_dim).float()) * attn.head_k_dim ** -0.5
        k = self._l2norm(k.reshape(batch, length, -1, attn.head_k_dim).float())
        v = v.reshape(batch, length, -1, attn.head_v_dim).float()
        repeats = attn.num_v_heads // attn.num_k_heads
        if repeats > 1:
            q, k = q.repeat_interleave(repeats, dim=2), k.repeat_interleave(repeats, dim=2)
        m = valid[..., None].float()
        beta = attn.in_proj_b(x).sigmoid().float() * m                              # pads: no update
        g = -attn.A_log.float().exp() * F.softplus(attn.in_proj_a(x).float() + attn.dt_bias) * m   # pads: no decay
        y, recurrent_out = gated_delta_rule(q, k, v, g, beta, recurrent.expand(batch, -1, -1, -1), max_len=length, chunk=CHUNK)
        z = attn.in_proj_z(x).reshape(-1, attn.head_v_dim)
        y = attn.norm(y.to(x.dtype).reshape(-1, attn.head_v_dim), z)
        return attn.out_proj(y.reshape(batch, length, -1)), conv_out, recurrent_out

    def _attention(self, attn, x, cos, sin, kv, allow):
        batch, length, _ = x.shape
        q, gate = attn.q_proj(x).reshape(batch, length, -1, 2 * attn.head_dim).chunk(2, dim=-1)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).reshape(batch, length, -1, attn.head_dim)).transpose(1, 2)
        v = attn.v_proj(x).reshape(batch, length, -1, attn.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, cos, sin)
        if kv is not None:
            k = torch.cat((kv[0].expand(batch, -1, -1, -1), k), dim=2)
            v = torch.cat((kv[1].expand(batch, -1, -1, -1), v), dim=2)
        bias = (1.0 - allow.float()) * -1e30                                        # additive mask, no where()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=attn.scaling, enable_gqa=True)
        y = y.transpose(1, 2).reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
        return attn.o_proj(y), torch.stack((k, v))

    def forward(self, tokens, valid, conv_end, start, allow, conv=None, recurrent=None, kv=None):
        x = self.embed_tokens(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None] + start[:, None]   # [B or 1, L]
        freqs = positions.float()[..., None] * self.inv_freq
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
        convs, recs, kvs = [], [], []
        li = fi = 0
        for layer in self.layers:
            h = layer.input_layernorm(x)
            if layer.block_type == "linear_attention":
                h, c, r = self._linear(layer.linear_attn, h, None if conv is None else conv[li],
                                       None if recurrent is None else recurrent[li], valid, conv_end)
                convs.append(c); recs.append(r); li += 1
            else:
                h, s = self._attention(layer.self_attn, h, cos, sin, None if kv is None else kv[fi], allow)
                kvs.append(s); fi += 1
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x), torch.stack(convs), torch.stack(recs), torch.stack(kvs)


class StaticPrefill(nn.Module):
    """tokens [1, P] (right-padded), n [1] true length -> conv, recurrent, kv [.., P, ..]."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens, n):
        P = tokens.shape[1]
        ar = torch.arange(P, device=tokens.device)
        valid = (ar[None] < n[:, None]).float()
        allow = (ar[None, :] <= ar[:, None])[None, None]                            # causal
        return self.backbone(tokens, valid, n, torch.zeros_like(n), allow)[1:]


class StaticScore(nn.Module):
    """rows [Q, B] (right-padded), prefix state padded to P with true length n [1] -> hidden [Q, B, d]."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens, n, conv, recurrent, kv):
        Q, B = tokens.shape
        P = kv.shape[-2]
        kj = torch.arange(P + B, device=tokens.device)
        qi = torch.arange(B, device=tokens.device)
        allow = ((kj[None, :] < n) | ((kj[None, :] >= P) & (kj[None, :] - P <= qi[:, None])))[None, None]
        valid = torch.ones(Q, B, device=tokens.device)
        conv_end = torch.full((Q,), B, dtype=torch.long, device=tokens.device)
        return self.backbone(tokens, valid, conv_end, n.expand(Q), allow, conv, recurrent, kv)[0]


def pick_bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} tokens exceed the largest exported bucket {buckets[-1]}")


class StaticProgram:
    """ExecuTorch Program interface over static engines: `prefill`, `score` and the constant methods."""

    def __init__(self, prefill_engine, score_engines, head, constants, max_prefix, max_questions, pad_id, device="cuda"):
        self.pre, self.scores, self.head = prefill_engine, dict(sorted(score_engines.items())), head
        self.P, self.Q, self.pad, self.device = max_prefix, max_questions, pad_id, device
        self.constants = constants
        self.method_names = ["prefill", "score", *constants]

    def load_method(self, name):
        prog = self

        class M:
            def execute(self, args):
                with torch.no_grad():
                    if name == "prefill":
                        return prog._prefill(*args)
                    if name == "score":
                        return prog._score(*args)
                    return [prog.constants[name]]
        return M()

    def _prefill(self, tokens):
        S = tokens.shape[1]
        t = torch.full((1, self.P), self.pad, dtype=torch.long, device=self.device)
        t[:, :S] = tokens.to(self.device)
        conv, rec, kv = self.pre(t, torch.tensor([S], device=self.device))
        return [conv, rec, kv[..., :S, :]]                  # the contract's kv is the true state length

    def _score(self, tokens, decide, options, conv, rec, kv):
        Qn, L = tokens.shape
        B = pick_bucket(L, list(self.scores))
        S = kv.shape[-2]
        t = torch.full((self.Q, B), self.pad, dtype=torch.long, device=self.device)
        t[:Qn, :L] = tokens.to(self.device)
        kvp = F.pad(kv, (0, 0, 0, self.P - S))
        h = self.scores[B](t, torch.tensor([S], device=self.device), conv, rec, kvp)[:Qn].float()
        rows = torch.arange(Qn, device=self.device)
        hd = h[rows, decide.to(self.device)]                                        # [Q, d]
        ho = h[rows[:, None], options.to(self.device)]                              # [Q, K, d]
        z = (self.head.k(ho) * self.head.q(hd)[:, None]).sum(-1) * self.head.scale / self.head.temperature
        return [z]
