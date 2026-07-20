"""Claim 7: is the workspace actually capacity-limited?

The paper states this numerically, and as far as we know nobody has checked it:

    "the J-space component typically accounts for only a small fraction of total
     activation variance (varying by layer, but never more than 10%)"

    "it is limited in capacity: it holds on the order of tens of concepts at a time"

This matters more than most rows on the scorecard. On Butlin & Long's operationalisation
of global workspace theory, GWT-2 -- a limited-capacity workspace, entailing a bottleneck
-- is one of the two properties that make a workspace a workspace rather than a shared
representation with a good name. (The other, GWT-3 broadcast-back, the paper explicitly
concedes it does not have.) So the bottleneck is load-bearing, and it is checkable from a
published lens plus forward passes. No fitting, no training.

Method. At layer l the J-space is spanned by the lens vectors

    v[l, t] = J_l^T W_U[t]

over the vocabulary. Take its principal subspace (top-k by SVD of a token sample), project
real activations onto it, and measure the fraction of activation variance it captures:

    variance_fraction(l, k) = || P_k h_l ||^2 / || h_l ||^2      (centred, averaged)

Against the paper: this should stay under ~0.10 at every layer.

Reporting a variance fraction alone is not enough -- a random k-dimensional subspace also
captures roughly k/d of the variance. The honest quantity is the EXCESS over that null,
exactly as with the CKA drift null in `geometry`. A J-space that captures 8% of variance
in 200 of 4096 dimensions is capturing 1.6x its random share, not "8% -- therefore
limited."
"""

from __future__ import annotations

import torch


def jspace_basis(lens, W_U, layer: int, *, k: int = 256, n_tokens: int = 8192,
                 seed: int = 0, device="cuda", dtype=torch.float32):
    """Top-k orthonormal basis of the J-space at ``layer``.

    Returns [d_model, k]. Built from the lens vectors v[l,t] = J_l^T W_U[t] over a random
    vocabulary sample, then reduced by SVD.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(W_U.shape[0], generator=g)[:n_tokens]
    Wt = W_U[idx].to(device, dtype)                       # [n_tokens, d_model]
    J = lens.jacobians[layer].to(device, dtype)           # [d_model, d_model]
    V = Wt @ J                                            # [n_tokens, d_model]
    V = V - V.mean(0, keepdim=True)
    # Right singular vectors span the J-space in activation coordinates.
    _, _, Vh = torch.linalg.svd(V, full_matrices=False)
    return Vh[:k].T.contiguous()                          # [d_model, k]


@torch.no_grad()
def variance_fraction(acts: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of centred activation variance captured by ``basis``.

    acts: [n_positions, d_model]. basis: [d_model, k], orthonormal columns.
    """
    X = acts.to(basis.device, basis.dtype)
    X = X - X.mean(0, keepdim=True)
    total = (X ** 2).sum()
    if total == 0:
        return float("nan")
    return ((X @ basis) ** 2).sum().item() / total.item()


def random_share(k: int, d_model: int) -> float:
    """What a RANDOM k-dimensional subspace captures in expectation: k/d.

    The null for any 'the workspace is only X% of variance' claim. Without it, a small
    percentage in a small subspace reads as a bottleneck when it is arithmetic.
    """
    return k / d_model


def capacity_report(variance_frac: float, k: int, d_model: int) -> dict:
    """Excess over the random-subspace null -- the only number worth quoting."""
    null = random_share(k, d_model)
    return {
        "k": k,
        "d_model": d_model,
        "variance_fraction": variance_frac,
        "random_share": null,
        "excess_ratio": variance_frac / null if null else float("nan"),
        "under_paper_10pct": variance_frac < 0.10,
    }
