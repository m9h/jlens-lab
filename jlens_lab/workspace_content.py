"""What does the workspace express, relative to the output?

A free, no-GPU probe of a published lens. The naive version -- rank tokens by
||J_l^T W_U[t]|| -- is CONFOUNDED: it just surfaces the highest-embedding-norm tokens
(special/format junk), because the norm of the lens vector tracks ||W_U[t]||. Verified:
the top of that ranking is identical to the top of ||W_U[t]|| with no lens involved.

The fix divides the embedding norm out. At the motor layers J -> I, so ||J_motor^T W_U[t]||
~= ||W_U[t]|| ~= how much the OUTPUT expresses t. So

    ratio(t) = || J_workspace^T W_U[t] || / || J_motor^T W_U[t] ||

is how much more the WORKSPACE band expresses t than the output does -- embedding norm
cancels. High ratio = held by the workspace but suppressed at output ("unstated"); low =
pushed by the motor layers, i.e. the literal next token.

Preliminary, OLMo-3-7B, published lens, layers 18 (workspace) vs 30 (motor):
  high ratio: instead, again, either, similarly, than, themselves, anyway, herself
              -- discourse / contrastive / reflexive markers
  low ratio:  the, that, for, making, taking, going, The, That, Let
              -- function words and literal continuations

CAVEATS: one model, two hand-picked layers, a norm ratio, a heuristic word filter, no
null. A suggestive signal, not a result. And it works on ANY published lens -- it does not
yet use OLMo's unique openness (data / checkpoints). The OLMo-only follow-up is whether
these directions strengthen over training checkpoints, which needs forward passes.
"""

from __future__ import annotations

import torch


def workspace_vs_output(lens, W_U, *, workspace_layer: int, motor_layer: int):
    """Per-token ratio of workspace expression to output expression (embedding-norm free).

    Returns a 1-D tensor over the vocabulary. > 1: held by the workspace more than the
    output expresses it. < 1: pushed toward output, suppressed in the workspace.
    """
    W_U = W_U.float()
    num = (W_U @ lens.jacobians[workspace_layer].float().T).norm(dim=1)
    den = (W_U @ lens.jacobians[motor_layer].float().T).norm(dim=1).clamp_min(1e-6)
    return num / den


def word_tokens(tokenizer, min_len: int = 4):
    """Indices of leading-space alphabetic word-pieces -- excludes specials/fragments,
    which otherwise dominate any norm-based ranking."""
    out = []
    for i in range(tokenizer.vocab_size):
        t = tokenizer.convert_ids_to_tokens([i])[0]
        if t.startswith("Ġ") and t[1:].isalpha() and len(t) > min_len:
            out.append(i)
    return out


# --------------------------------------------------------------------------------
# Phase 0: is the discourse-marker enrichment real, or hand-picked-layer noise?
# --------------------------------------------------------------------------------

# Discourse / contrastive / reflexive markers -- the category the workspace appeared to
# hold in the n=1 probe. A CLOSED list fixed in advance, so the enrichment test is not
# post-hoc: we ask whether THIS predefined set is over-represented at the top of the
# ratio, not whether the top happens to look thematic.
DISCOURSE_MARKERS = {
    "instead", "again", "either", "neither", "similarly", "anyway", "anyways",
    "however", "therefore", "nevertheless", "nonetheless", "meanwhile", "otherwise",
    "moreover", "furthermore", "conversely", "regardless", "whereas", "though",
    "although", "unless", "besides", "rather", "than", "themselves", "herself",
    "himself", "itself", "oneself", "yourself", "myself", "ourselves",
}


def band_ratio(lens, W_U, *, workspace_layers, motor_layers):
    """Average workspace/output ratio over BANDS of layers, not two hand-picked ones.

    Numerator: mean over workspace-band layers of || J_l^T W_U[t] ||.
    Denominator: mean over motor-band layers (J -> I there, so ~ output expression).
    Embedding norm cancels. Returns a per-token tensor.
    """
    W_U = W_U.float()
    num = torch.stack([(W_U @ lens.jacobians[l].float().T).norm(dim=1)
                       for l in workspace_layers]).mean(0)
    den = torch.stack([(W_U @ lens.jacobians[l].float().T).norm(dim=1)
                       for l in motor_layers]).mean(0).clamp_min(1e-6)
    return num / den


def marker_enrichment(ratio, tokenizer, *, top_frac=0.05, n_perm=1000, seed=0,
                      markers=DISCOURSE_MARKERS):
    """Are discourse markers over-represented in the top ``top_frac`` of the ratio?

    Restricts to real word-pieces, ranks by the ratio, counts how many predefined markers
    land in the top fraction, and compares against a permutation null that shuffles the
    ratio<->token assignment. The marker set is compared only to OTHER word tokens (not
    specials/fragments), so casing/length distribution is held roughly constant.

    Returns observed count, null mean/std, enrichment ratio, and an empirical p-value.
    """
    ids = word_tokens(tokenizer)
    if not ids:
        return {"n_word_tokens": 0}
    toks = [tokenizer.convert_ids_to_tokens([i])[0].lstrip("Ġ").lower() for i in ids]
    r = ratio[torch.tensor(ids)]
    is_marker = torch.tensor([t in markers for t in toks])
    n_markers = int(is_marker.sum())
    if n_markers == 0:
        return {"n_word_tokens": len(ids), "n_markers": 0}

    k = max(1, int(top_frac * len(ids)))
    observed = int(is_marker[r.topk(k).indices].sum())

    g = torch.Generator().manual_seed(seed)
    null = torch.empty(n_perm)
    for i in range(n_perm):
        null[i] = is_marker[torch.randperm(len(ids), generator=g)[:k]].sum()
    return {"n_word_tokens": len(ids), "n_markers": n_markers, "top_k": k,
            "observed_in_top": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "enrichment": observed / max(null.mean().item(), 1e-9),
            "p_value": float((null >= observed).float().mean())}
