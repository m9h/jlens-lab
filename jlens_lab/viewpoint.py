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

This needs only the two lens files -- no model, no GPU.

TWO NULLS ARE REQUIRED, and the second already bites:

1. Refit noise -- the same statistic between two REFITS of one model. Not yet run (needs
   one GPU fit, ~$3). A structured shift must be markedly lower-rank than this.

2. J itself. dJ being low-rank means little if J is ALREADY low-rank. Measured on three
   published base/instruct pairs, it largely is:

       model          rank(J_base)   rank(dJ)   ratio
       gemma-3-270m       0.048       0.022     0.45x
       gemma-3-1b         0.019       0.019     0.98x   <- dJ == J rank
       gemma-2-2b         0.119       0.003     0.02x   <- only here concentrated

   So for 2 of 3, dJ's low rank is INHERITED from J being low-rank, not evidence that
   post-training is a structured, low-dimensional shift. The raw dJ effective rank
   (0.003-0.02) must NOT be reported on its own -- it looked like a strong Claim 6
   confirmation and mostly is not. Report dJ rank relative to J rank, and against the
   refit-noise null.
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
