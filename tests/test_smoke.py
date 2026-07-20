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


# --- artifacts: catch a mid-fit checkpoint masquerading as a lens ---------------
# qwen3-32b's published .pt is a checkpoint at n_done=80 against a config claiming 615.
# An 80-prompt J is near-identity, so the J-lens degenerates into a logit lens -- and we
# nearly published an architectural conclusion from it.

def _fake_checkpoint(n_done=80, d=8, layers=(0, 1)):
    return {"jacobian_sum": {l: torch.eye(d) * n_done for l in layers},
            "n_done": n_done, "next_idx": n_done, "source_layers": list(layers)}


def test_detects_a_checkpoint_is_not_a_lens():
    from jlens_lab import artifacts
    assert artifacts._is_checkpoint(_fake_checkpoint())
    assert not artifacts._is_checkpoint({"jacobians": {}, "n_prompts": 615})


def test_recovers_the_running_mean_faithfully():
    from jlens_lab import artifacts
    n, d = 80, 8
    ck = _fake_checkpoint(n_done=n, d=d)
    lens = artifacts.recover_from_checkpoint(ck)
    assert lens.n_prompts == n
    # jacobian_sum was eye(d)*n, so the mean must be exactly eye(d)
    assert torch.allclose(lens.jacobians[0], torch.eye(d), atol=1e-6)


def test_recovery_rejects_an_empty_checkpoint():
    from jlens_lab import artifacts
    with pytest.raises(ValueError, match="n_done=0"):
        artifacts.recover_from_checkpoint(_fake_checkpoint(n_done=0))


def test_claimed_prompts_parsed_from_config():
    from jlens_lab import artifacts
    assert artifacts._claimed_prompts("results:\n  prompts_fitted: 615\n") == 615
    assert artifacts._claimed_prompts("no such key") is None


# --- validation anchor ---------------------------------------------------------
# Fit one model that already has a published lens, compare, and only then fit the
# rest. A silent fitting bug (wrong RoPE, bf16 backward, off-by-one source_layers)
# yields a lens that looks fine in isolation and is wrong.

class _L:
    def __init__(self, jac, n=616):
        self.jacobians, self.n_prompts = jac, n
        self.d_model = next(iter(jac.values())).shape[-1]


def test_compare_is_perfect_against_itself():
    from jlens_lab import artifacts
    j = {0: torch.randn(8, 8), 1: torch.randn(8, 8)}
    r = artifacts.compare(_L(j), _L(j))
    assert r["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert r["mean_rel_error"] == pytest.approx(0.0, abs=1e-5)


def test_compare_flags_a_divergent_fit():
    from jlens_lab import artifacts
    a = {0: torch.randn(8, 8), 1: torch.randn(8, 8)}
    b = {0: torch.randn(8, 8), 1: torch.randn(8, 8)}   # independent -> ~0 cosine
    r = artifacts.compare(_L(a), _L(b))
    assert abs(r["mean_cosine"]) < 0.5, "independent lenses must not look similar"


def test_compare_requires_shared_layers():
    from jlens_lab import artifacts
    with pytest.raises(ValueError, match="no layers in common"):
        artifacts.compare(_L({0: torch.randn(4, 4)}), _L({7: torch.randn(4, 4)}))
