"""CKA / block-structure analysis. Not in ``jlens`` at all -- the paper's headline
figure is described only as "geometrical matching" and never specified, so this is a
RECONSTRUCTION. Say so in any writeup; the underspecification is a real complaint.

J-space representation at layer l, over a vocabulary sample:

    v[l, t] = J_l^T @ W_U[t]

(the transposed row -- the same object jlens uses for steering). Then linear CKA
between every pair of layers. No forward passes needed: the lens and the unembedding
suffice.

Two traps, both of which we fell into:

  * An unconstrained 3-block search carves a SINGLE layer off as its own "block". A
    lone layer has CKA 1.0 with itself by construction, so the score is maximised by a
    degenerate split that is not a tripartite structure at all. Hence ``min_frac``.

  * A smooth distance-decaying matrix scores nonzero on within-minus-between contrast
    anyway. Raw blockiness rising 0.08 -> 0.29 across Qwen3 1.7B -> 14B was the decay
    profile STEEPENING, not blocks appearing. Always report ``excess`` against
    ``controls.distance_null``.
"""

from __future__ import annotations

import itertools
import torch

from .controls import distance_null


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two [n, d] representations."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def jspace_reps(lens, W_U: torch.Tensor, *, n_tokens: int = 4096, seed: int = 0,
                device="cuda", dtype=torch.float32) -> dict[int, torch.Tensor]:
    """v[l, t] = J_l^T @ W_U[t] over a random vocabulary sample."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(W_U.shape[0], generator=g)[:n_tokens]
    Wt = W_U[idx].to(device, dtype)
    return {l: Wt @ lens.jacobians[l].to(device, dtype) for l in sorted(lens.jacobians)}


def cka_matrix(reps: dict[int, torch.Tensor]) -> torch.Tensor:
    layers = sorted(reps)
    L = len(layers)
    C = torch.zeros(L, L)
    for i in range(L):
        for j in range(i, L):
            C[i, j] = C[j, i] = linear_cka(reps[layers[i]], reps[layers[j]])
    return C


def blockiness(C: torch.Tensor, *, min_frac: float = 0.15):
    """Best contiguous 3-block split: within-block mean CKA minus between-block mean.

    Each block must hold >= ``min_frac`` of the layers, or the search cheats by
    isolating single layers (see module docstring).
    """
    L = C.shape[0]
    m = max(2, int(round(min_frac * L)))
    if L < 3 * m:
        return float("nan"), None
    best, cuts = -1e9, None
    for a, b in itertools.combinations(range(m, L - m + 1), 2):
        if b - a < m or L - b < m:
            continue
        seg = [(0, a), (a, b), (b, L)]
        win = wn = bet = bn = 0
        for i, (s0, e0) in enumerate(seg):
            for j, (s1, e1) in enumerate(seg):
                blk = C[s0:e0, s1:e1]
                if i == j:
                    win += blk.sum().item(); wn += blk.numel()
                else:
                    bet += blk.sum().item(); bn += blk.numel()
        score = win / wn - bet / bn
        if score > best:
            best, cuts = score, (a, b)
    return best, cuts


def excess_over_null(C: torch.Tensor, *, min_frac: float = 0.15) -> dict:
    """THE number. Raw blockiness minus the blockiness of a pure-drift matrix.

    On Anthropic's published lenses:

        qwen3-1.7b  real 0.117  null 0.092  excess +0.025
        qwen3-4b    real 0.078  null 0.062  excess +0.016
        qwen3-8b    real 0.283  null 0.249  excess +0.033
        qwen3-14b   real 0.293  null 0.267  excess +0.026
        gpt-oss-20b real 0.210  null 0.157  excess +0.053
        qwen3.5-27b real 0.267  null 0.217  excess +0.050   <- "nose" emerges here

    The null recovers most of the raw score everywhere. The excess roughly doubles at
    >=20B -- at the same scale the behaviour appears -- but stays small in absolute
    terms. The sharp tripartite figure does not replicate on any open model.
    """
    real, cuts = blockiness(C, min_frac=min_frac)
    null, _ = blockiness(distance_null(C), min_frac=min_frac)
    return {"real": real, "null": null, "excess": real - null, "cuts": cuts}
