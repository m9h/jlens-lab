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
