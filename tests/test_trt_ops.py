"""kev.trt_ops.gated_delta_rule against the token-by-token definition. No weights; CPU is enough."""
import pytest
import torch
import torch.nn.functional as F

from kev.trt_ops import gated_delta_rule, reference_recurrent


def _inputs(B, L, H, K, V, adversarial, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    if adversarial:   # near-identical keys, beta ~ 1, almost no decay: what broke the power-series inverse
        k = F.normalize(F.normalize(r(1, 1, H, K), dim=-1) + 0.02 * r(B, L, H, K), dim=-1)
        beta = 0.9 + 0.1 * torch.rand(B, L, H, generator=g_)
        g = -0.001 * torch.rand(B, L, H, generator=g_)
    else:
        k = F.normalize(r(B, L, H, K), dim=-1)
        beta = torch.rand(B, L, H, generator=g_)
        g = -F.softplus(r(B, L, H))
    q = F.normalize(r(B, L, H, K), dim=-1) * K ** -0.5
    return q, k, r(B, L, H, V), g, beta, 0.1 * r(B, H, K, V)


@pytest.mark.parametrize("L,max_len", [(1, 64), (37, 64), (64, 64), (100, 128), (200, 384)])
@pytest.mark.parametrize("adversarial", [False, True])
def test_matches_recurrence_with_padding(L, max_len, adversarial):
    x = _inputs(2, L, 4, 32, 16, adversarial)
    o_ref, s_ref = reference_recurrent(*x)
    o, s = gated_delta_rule(*x, max_len=max_len)
    assert o.shape == o_ref.shape and torch.isfinite(o).all() and torch.isfinite(s).all()
    assert (o - o_ref).abs().max() < 1e-5
    assert (s - s_ref).abs().max() < 1e-4


def test_rejects_bad_shapes():
    x = _inputs(1, 10, 2, 8, 8, False)
    with pytest.raises(ValueError):
        gated_delta_rule(*x, max_len=100)          # not a multiple of the chunk
    with pytest.raises(ValueError):
        gated_delta_rule(*x, max_len=96, chunk=48)  # chunk not a power of two
