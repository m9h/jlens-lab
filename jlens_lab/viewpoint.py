"""Structured vs diffuse: how did post-training change the J-space?

cos(base, instruct) = 0.76 (mean over 8 published pairs) says the workspace moves under
post-training, well beyond the ~0.96 refit-noise floor. It does NOT say whether it moved
"toward a point of view" -- a specific, low-dimensional change -- or simply drifted as the
weights changed everywhere.

The effective rank of dJ = J_instruct - J_base separates the two. For singular values
s_1..s_d of dJ, the participation-ratio effective rank is

    r_eff = (sum s_i^2)^2 / sum s_i^4          (in [1, d]; scale-invariant)

  r_eff small (few directions carry the change)  -> STRUCTURED
  r_eff ~ d  (change spread over all directions) -> DIFFUSE weight drift

This needs only the two lens files -- no model, no GPU. The honest comparison is against
the same statistic computed between two REFITS of one model (fitting noise): a structured
viewpoint shift must be markedly lower-rank than that null.
"""

from __future__ import annotations

import torch


def effective_rank(singular_values: torch.Tensor) -> float:
    """Participation-ratio effective rank of a spectrum. Scale-invariant; in [1, len]."""
    s2 = singular_values.double() ** 2
    denom = (s2 ** 2).sum()
    if denom == 0:
        return float("nan")
    return (s2.sum() ** 2 / denom).item()


def delta_effective_rank(lens_base, lens_instruct) -> dict:
    """Per-layer effective rank of J_instruct - J_base, plus the change magnitude.

    Returns {layer: {effective_rank, d_model, rank_fraction, delta_norm, rel_delta}}.
    ``rank_fraction`` = r_eff / d_model makes layers/models comparable; low = structured.
    """
    shared = sorted(set(lens_base.jacobians) & set(lens_instruct.jacobians))
    if not shared:
        raise ValueError("no layers in common")

    out = {}
    for l in shared:
        a = lens_base.jacobians[l].float()
        b = lens_instruct.jacobians[l].float().to(a.device)
        d = a.shape[-1]
        delta = b - a
        dn = delta.norm().item()
        if dn == 0:
            out[l] = {"effective_rank": float("nan"), "d_model": d,
                      "rank_fraction": float("nan"), "delta_norm": 0.0,
                      "rel_delta": 0.0}
            continue
        s = torch.linalg.svdvals(delta.double())
        r = effective_rank(s)
        out[l] = {"effective_rank": r, "d_model": d, "rank_fraction": r / d,
                  "delta_norm": dn, "rel_delta": (dn / a.norm()).item()}
    return out


def summarise(per_layer: dict) -> dict:
    """Mean effective rank fraction over layers with a real change."""
    vals = [v["rank_fraction"] for v in per_layer.values()
            if v["rank_fraction"] == v["rank_fraction"]]   # drop nan
    return {"n_layers": len(vals),
            "mean_rank_fraction": sum(vals) / len(vals) if vals else float("nan"),
            "min_rank_fraction": min(vals) if vals else float("nan")}
