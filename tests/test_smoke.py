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


def test_identity_distance_is_zero_for_identity_and_grows():
    from jlens_lab import artifacts
    d = 16
    assert artifacts.identity_distance(_L({0: torch.eye(d)})) == pytest.approx(0.0, abs=1e-6)
    far = artifacts.identity_distance(_L({0: torch.eye(d) * 3.0}))
    assert far > 1.0, "a scaled identity must register as far from I"


def test_cosine_alone_cannot_gate_an_underfit_lens():
    """Measured on real lenses: a 100-prompt fit scores 0.956-0.972 against a converged
    reference. Any cosine threshold admitting a correct fit also admits a 4.7x under-fit
    one -- which is why validate_fit also gates on identity_distance."""
    from jlens_lab import artifacts
    d = 32
    base = torch.eye(d) + 0.05 * torch.randn(d, d)
    underfit = base + 0.05 * torch.randn(d, d)     # same structure, different fit
    r = artifacts.compare(_L({0: base}), _L({0: underfit}))
    assert r["mean_cosine"] > 0.95, "structure alone already clears a 0.95 bar"


def test_random_share_is_the_null_for_a_capacity_claim():
    from jlens_lab import capacity
    assert capacity.random_share(256, 4096) == pytest.approx(0.0625)
    r = capacity.capacity_report(0.08, k=256, d_model=4096)
    assert r["under_paper_10pct"], "8% is under the paper's 10% ceiling"
    assert r["excess_ratio"] == pytest.approx(1.28, abs=0.01), \
        "but it is only 1.28x a random subspace of the same size -- barely a bottleneck"


def test_variance_fraction_is_one_for_a_full_basis():
    from jlens_lab import capacity
    d = 16
    acts = torch.randn(64, d)
    full = torch.linalg.qr(torch.randn(d, d))[0]
    assert capacity.variance_fraction(acts, full) == pytest.approx(1.0, abs=1e-4)


def test_variance_fraction_is_small_for_a_thin_random_basis():
    from jlens_lab import capacity
    d, k = 64, 4
    acts = torch.randn(512, d)
    thin = torch.linalg.qr(torch.randn(d, k))[0][:, :k]
    f = capacity.variance_fraction(acts, thin)
    assert 0.02 < f < 0.15, f"isotropic acts: expect ~k/d={k/d:.3f}, got {f:.3f}"


def test_fit_converged_checkpoints_and_resumes(tmp_path, monkeypatch):
    """A 20-hour 7B fit must not live only in GPU memory. jlens.fit has checkpoint_path;
    this wrapper dropped it, so a dead pod lost everything."""
    import jlens_lab.fitting as F
    d, calls = 4, {"n": 0}

    def fake(model, prompt, layers, **kw):
        calls["n"] += 1
        return {l: torch.eye(d) for l in layers}, 128, 100

    monkeypatch.setattr(F, "jacobian_for_prompt", fake)
    ck = tmp_path / "ck.pt"

    class M: n_layers = 2
    F.fit_converged(M(), [f"p{i}" for i in range(60)], source_layers=[0, 1],
                    checkpoint_path=ck, checkpoint_every=10,
                    min_prompts=1000, verbose=False)
    assert ck.exists(), "must write a checkpoint"

    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["n_done"] >= 50 and "jacobian_sum" in st

    before = calls["n"]
    F.fit_converged(M(), [f"p{i}" for i in range(60)], source_layers=[0, 1],
                    checkpoint_path=ck, min_prompts=1000, verbose=False)
    assert calls["n"] - before < 30, "resume must skip already-processed prompts"


def test_workspace_ratio_divides_out_embedding_norm():
    """The whole point: a token with a huge embedding norm must NOT dominate the ratio,
    because norm cancels between workspace and motor. Motor J = c*I makes the ratio depend
    only on the workspace transport, not on ||W_U[t]||."""
    from jlens_lab import workspace_content as wc
    d, V = 8, 50
    W_U = torch.randn(V, d)
    W_U[0] *= 100.0                       # an embedding-norm outlier
    class L:
        jacobians = {5: torch.randn(d, d), 9: torch.eye(d) * 2.0}   # motor ~ scaled I
    r = wc.workspace_vs_output(L(), W_U, workspace_layer=5, motor_layer=9)
    # token 0's ratio must be within the bulk, not an extreme outlier
    z = (r[0] - r.mean()) / r.std()
    assert abs(z) < 3, f"embedding-norm outlier leaked into the ratio (z={z:.1f})"
