"""Tests that encode the traps. Each one is a bug we actually shipped."""
import itertools
import torch
import pytest

from jlens_lab import geometry, controls


def test_blockiness_rejects_singleton_blocks():
    """An unconstrained search isolates single layers (CKA 1.0 with themselves)."""
    L = 27
    C = torch.rand(L, L) * 0.1
    C = (C + C.T) / 2
    C.fill_diagonal_(1.0)
    score, cuts = geometry.blockiness(C, min_frac=0.15)
    a, b = cuts
    m = max(2, round(0.15 * L))
    assert a >= m and b - a >= m and L - b >= m, "a block smaller than min_frac survived"


def test_distance_null_has_no_blocks_but_scores_nonzero():
    """The whole reason `excess` exists: pure drift still scores on raw blockiness."""
    L = 40
    i = torch.arange(L)
    C = torch.exp(-(i[:, None] - i[None, :]).abs().float() / 8.0)  # pure |i-j| decay
    raw, _ = geometry.blockiness(C)
    null = controls.distance_null(C)
    null_raw, _ = geometry.blockiness(null)
    assert raw > 0.05, "sanity: smooth drift does score on raw blockiness"
    assert abs(raw - null_raw) < 1e-4, "its own null must reproduce it exactly"
    assert abs(geometry.excess_over_null(C)["excess"]) < 1e-4


def test_distance_null_preserves_the_decay_profile():
    L = 12
    C = torch.rand(L, L)
    C = (C + C.T) / 2
    N = controls.distance_null(C)
    for d in range(L):
        obs = torch.stack([C[i, i + d] for i in range(L - d)]).mean()
        assert torch.allclose(N[0, d], obs, atol=1e-5)


def test_cka_is_one_against_itself():
    X = torch.randn(256, 32)
    assert geometry.linear_cka(X, X) == pytest.approx(1.0, abs=1e-5)


# --- regression: fit_converged must not swallow real errors -------------------
# A bare `except Exception: continue` in the fit loop hid a source_layers bug for
# 600 prompts, then cost a downstream user half a day on a GPU box. The probe must
# surface a real error after ONE prompt.

class _BoomModel:
    n_layers = 4

def test_fit_converged_raises_immediately_on_a_real_error(monkeypatch):
    import jlens_lab.fitting as F
    calls = []

    def boom(model, prompt, layers, **kw):
        calls.append(prompt)
        raise RuntimeError("bad layer index")

    monkeypatch.setattr(F, "jacobian_for_prompt", boom)
    with pytest.raises(RuntimeError, match="bad layer index"):
        F.fit_converged(_BoomModel(), ["a", "b", "c"], source_layers=[0, 1],
                        verbose=False)
    assert len(calls) == 1, "must fail on the FIRST prompt, not churn the whole list"


def test_fit_converged_rejects_empty_prompts():
    import jlens_lab.fitting as F
    with pytest.raises(ValueError, match="no prompts"):
        F.fit_converged(_BoomModel(), [], source_layers=[0], verbose=False)
