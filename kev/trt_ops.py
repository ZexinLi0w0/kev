"""A gated delta rule that TensorRT can compile: static shapes, matmuls and elementwise ops only.

The Qwen3.5 DeltaNet layers carry a recurrent state S (per head, K x V) that each token updates:

    S <- exp(g_t) * S;   S <- S + k_t (beta_t (v_t - S^T k_t))^T;   o_t = S^T q_t

Upstream runs this as a custom op (ExecuTorch's `llama::gated_delta_rule` on CPU, `mlx::gated_delta_rule` on Apple
GPUs) or as flash-linear-attention's Triton kernel on CUDA. TensorRT can execute none of them, and the pure-PyTorch
reference in transformers has two things TensorRT cannot compile: a Python loop over the (dynamic) number of chunks,
and inside each chunk a forward substitution with a data-dependent in-place update.

This module is the same chunked algorithm (transformers' `torch_chunk_gated_delta_rule`, chunk size C) with both
removed:

* **The chunk loop has a static trip count.** The sequence is right-padded to a fixed `max_len` (a multiple of C) that
  the program is exported for, so the loop unrolls at export. Pads get beta = 0 and log-decay g = 0 (decay 1) and zero
  q/k/v: they add nothing to the state and do not decay it, so the state after `max_len` tokens is exactly the state
  after the last real token, and the outputs at pad positions are cut off before they are returned.
* **The within-chunk inverse is matmul-only and exact.** The chunk needs T = (I + N)^-1 with N strictly lower
  triangular (C x C). It is computed by blocked recursion on the unit lower-triangular I + N (log2(C) levels of
  batched matmuls) instead of C sequential row updates. See `_unit_lower_inverse` for why the shorter power-series
  form is not used.

Everything is computed in fp32, as the reference does.
"""
import torch
import torch.nn.functional as F


def _unit_lower_inverse(m):
    """Inverse of a unit lower-triangular m [..., C, C] (C a power of two) by blocked recursion:

        [[A, 0], [B, D]]^-1 = [[A^-1, 0], [-D^-1 B A^-1, D^-1]]

    log2(C) levels of batched matmuls, static at export. This is the numerically stable way to do it. A power-series
    shortcut ((I + N)^-1 = prod (I + (-N)^(2^i)), exact in exact arithmetic because N is nilpotent) is not: with
    Kev's L2-normalised keys, neighbouring tokens give N entries near 1, powers of N grow like binomial coefficients
    (~1e17 at C = 64), and the small true inverse is left as a difference of huge terms -- NaNs and flipped answers
    on real activations even though random tests pass."""
    c = m.shape[-1]
    if c == 1:
        return torch.ones_like(m)
    h = c // 2
    a_inv = _unit_lower_inverse(m[..., :h, :h])
    d_inv = _unit_lower_inverse(m[..., h:, h:])
    lower_left = -(d_inv @ m[..., h:, :h] @ a_inv)
    top = torch.cat((a_inv, torch.zeros_like(a_inv)), -1)
    bottom = torch.cat((lower_left, d_inv), -1)
    return torch.cat((top, bottom), -2)


def gated_delta_rule(q, k, v, g, beta, state, max_len, chunk=64):
    """Chunked gated delta rule on [B, L, H, D] inputs, L <= max_len.

    q, k: [B, L, H, K] (q already scaled and L2-normalised, as upstream passes it); v: [B, L, H, V];
    g: [B, L, H] log-decay (<= 0); beta: [B, L, H]; state: [B, H, K, V] fp32.
    Returns (o [B, L, H, V] fp32, final state [B, H, K, V] fp32)."""
    if chunk & (chunk - 1):
        raise ValueError(f"chunk {chunk} must be a power of two")
    if max_len % chunk:
        raise ValueError(f"max_len {max_len} must be a multiple of the chunk size {chunk}")
    B, L, H, Kd = q.shape
    Vd = v.shape[-1]
    pad = max_len - L
    q, k, v = (F.pad(x.float(), (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
    g = F.pad(g.float(), (0, 0, 0, pad))           # g = 0 at pads: no decay
    beta = F.pad(beta.float(), (0, 0, 0, pad))     # beta = 0 at pads: no update
    # [B, H, n_chunks, C, D]
    n = max_len // chunk
    q, k, v = (x.transpose(1, 2).reshape(B, H, n, chunk, -1) for x in (q, k, v))
    g = g.transpose(1, 2).reshape(B, H, n, chunk)
    beta = beta.transpose(1, 2).reshape(B, H, n, chunk)

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    # within-chunk cumulative log-decay as a matmul with a constant upper-triangular ones matrix, not cumsum: Torch-
    # TensorRT lowers cumsum to a TensorRT loop, which TensorRT 10.3's Myelin cannot fuse ("Could not find any
    # implementation for node {ForeignNode[/cumsum_trip_limit...]}").
    g = g @ torch.triu(torch.ones(chunk, chunk, dtype=torch.float32, device=q.device))
    # Masks are multiplications by constant 0/1 matrices, never torch.where: TensorRT 10.3 (JetPack 6) cannot build a
    # Myelin kernel for the fused where/-inf/exp pattern ("Could not find any implementation for node
    # {ForeignNode[...where_condition...]}"), while 10.13 can. Above the diagonal g_i - g_j >= 0 (g is a cumulative
    # sum of non-positive log-decays), so it is clamped to 0 before exp and then zeroed: exp(0) * 0, never inf * 0.
    lower = torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device=q.device))
    strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device=q.device), -1)
    diff = g[..., :, None] - g[..., None, :]
    decay_mask = diff.clamp(max=0).exp() * lower                       # exp(g_i - g_j) for j <= i, else 0
    nmat = (k_beta @ k.transpose(-1, -2)) * decay_mask * strict
    eye = torch.eye(chunk, dtype=nmat.dtype, device=nmat.device)
    t = _unit_lower_inverse(eye + nmat)                                # (I + N)^-1, blocked and stable
    value = t @ v_beta
    k_cumdecay = t @ (k_beta * g.exp()[..., None])

    s = state.float()
    outs = []
    for i in range(n):                                                 # static trip count: unrolled at export
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        attn = (qi @ ki.transpose(-1, -2)) * decay_mask[:, :, i]        # decay_mask is already zero above the diagonal
        v_new = value[:, :, i] - k_cumdecay[:, :, i] @ s
        outs.append((qi * gi.exp()[..., None]) @ s + attn @ v_new)
        g_last = gi[..., -1:]                                          # [B, H, 1]
        s = s * g_last.exp()[..., None] + (ki * (g_last - gi).exp()[..., None]).transpose(-1, -2) @ v_new
    o = torch.stack(outs, 2).reshape(B, H, max_len, Vd).transpose(1, 2)[:, :L]
    return o, s


def reference_recurrent(q, k, v, g, beta, state):
    """Token-by-token recurrence, the definition (transformers' torch_recurrent_gated_delta_rule), for tests."""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    B, L, H, _ = q.shape
    s = state.float().clone()
    out = []
    for t in range(L):
        s = s * g[:, t].exp()[..., None, None]
        kv_mem = (s * k[:, t][..., None]).sum(-2)
        delta = (v[:, t] - kv_mem) * beta[:, t][..., None]
        s = s + k[:, t][..., None] * delta[..., None, :]
        out.append((s * q[:, t][..., None]).sum(-2))
    return torch.stack(out, 1), s
