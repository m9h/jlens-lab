"""Is post-training's change to the J-space STRUCTURED (low-rank) or DIFFUSE (full-rank)?

cos(base, instruct)=0.76 says J moves. It cannot say whether it moved 'toward a point of
view' (a specific, low-dimensional change) or just drifted (weights changed everywhere).
Effective rank of dJ = J_instruct - J_base separates them:

  low effective rank  -> the change lives in a few directions -> structured
  ~full rank          -> diffuse weight drift

Null: the effective rank of the change between two REFITS of the same model (fitting
noise). Structured change must be much lower-rank than that floor to mean anything.
"""
import torch
import pytest
from jlens_lab import viewpoint


def _lens(jac):
    class L:
        def __init__(s): s.jacobians, s.d_model = jac, next(iter(jac.values())).shape[-1]
    return L()


def test_identical_lenses_have_zero_change():
    d = 16
    J = {0: torch.randn(d, d)}
    r = viewpoint.delta_effective_rank(_lens(J), _lens({0: J[0].clone()}))
    assert r[0]["delta_norm"] == pytest.approx(0.0, abs=1e-6)
    assert r[0]["effective_rank"] != r[0]["effective_rank"] or r[0]["effective_rank"] == 0  # nan or 0


def test_rank_one_change_has_effective_rank_near_one():
    d = 64
    base = torch.randn(d, d)
    u, v = torch.randn(d, 1), torch.randn(1, d)
    instruct = base + 5.0 * (u @ v)                      # a single added direction
    r = viewpoint.delta_effective_rank(_lens({0: base}), _lens({0: instruct}))
    assert r[0]["effective_rank"] < 3.0, "a rank-1 change must read as low effective rank"


def test_isotropic_change_has_high_effective_rank():
    d = 64
    base = torch.randn(d, d)
    instruct = base + 0.5 * torch.randn(d, d)            # diffuse change in all directions
    r = viewpoint.delta_effective_rank(_lens({0: base}), _lens({0: instruct}))
    # An iid Gaussian d x d matrix has Marchenko-Pastur effective-rank fraction ~0.50
    # (measured 0.496 +/- 0.008 over 200 draws), so it STRADDLES 0.5 -- the old > d*0.5
    # bar failed ~72% of the time. A diffuse change need only read as high-rank relative to
    # the rank-1 case (which is < 3, i.e. fraction < 0.05); 0.4*d cleanly separates them.
    assert r[0]["effective_rank"] > d * 0.4, "diffuse change must read as high effective rank"


def test_effective_rank_is_scale_invariant():
    """Doubling the magnitude of the change must not change its effective rank."""
    d = 48
    base = torch.randn(d, d)
    delta = torch.randn(d, d)
    r1 = viewpoint.delta_effective_rank(_lens({0: base}), _lens({0: base + delta}))
    r2 = viewpoint.delta_effective_rank(_lens({0: base}), _lens({0: base + 3 * delta}))
    assert r1[0]["effective_rank"] == pytest.approx(r2[0]["effective_rank"], rel=1e-4)


def test_requires_shared_layers():
    with pytest.raises(ValueError, match="no layers in common"):
        viewpoint.delta_effective_rank(_lens({0: torch.randn(4, 4)}),
                                       _lens({9: torch.randn(4, 4)}))
