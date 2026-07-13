"""The controls. None of these ship with ``jlens``, and each one changed our answer.

1. ``randomize_blocks`` -- the model-parameter randomization test (Adebayo et al.,
   *Sanity Checks for Saliency Maps*, NeurIPS 2018). Randomize the transformer blocks
   but KEEP the trained embedding / final norm / unembedding, so token identity stays
   meaningful and the control cannot pass for the wrong reason.

   Our result: on randomized blocks the J-lens reads out nothing (next-token 0.0003,
   input-echo 0.0016, both at floor) while the trained model shows a clean
   echo->predict crossover. The J-lens PASSES. Its structure requires learned weights,
   not just a residual stream plus a trained unembedding. Anthropic never ran this,
   and it is a point in their favour.

   TRAP: ``model._init_weights(mod)`` is a **silent no-op** on an already-loaded model
   in transformers v5. The naive control leaves the blocks fully trained and reports a
   confident false PASS. Build from config and transplant; then assert.

2. ``logit_lens_floor`` -- ``lens.apply(..., use_jacobian=False)`` already exists in
   jlens, but nothing tells you to use it as the baseline. Every J-lens number should
   be reported against it, not against chance.

   CAVEAT we found the hard way: rank-based pass@k (min rank over all layers) REWARDS
   NOISE. It hands a diffuse lens one lottery ticket per layer, so a logit lens
   emitting ['Ċ','Âł','..','-','N'] can outscore a J-lens returning
   ['smile','nose','noses','grin']. Score the top-k contents, not just the rank.

3. ``distance_null`` -- for CKA/geometry claims. A matrix whose entries depend ONLY on
   layer distance |i-j| still scores nonzero on any within-block-minus-between-block
   contrast, because nearby layers are more similar than distant ones. On the published
   lenses this null reproduces 79-91% of the apparent "sensory/workspace/motor" block
   structure. Report the EXCESS over the null, or report nothing.
"""

from __future__ import annotations

import torch


def randomize_blocks(model_name: str, *, dtype=torch.bfloat16, device="cuda"):
    """A model whose blocks are random but whose embed / norm / unembed are trained.

    Randomizing everything would make token identity meaningless, and the lens would
    read out garbage for a trivial reason. Keeping the trained input/output maps is
    what makes this a real test of whether the *computation* is doing the work.
    """
    from transformers import AutoModelForCausalLM

    trained = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    rand = AutoModelForCausalLM.from_config(trained.config).to(dtype)

    ref = trained.model.layers[0]
    got = rand.model.layers[0]
    p_ref = next(p for p in ref.parameters() if p.dim() == 2)
    p_got = next(p for p in got.parameters() if p.dim() == 2)
    if torch.equal(p_ref, p_got):
        raise AssertionError(
            "blocks were NOT randomized. transformers' _init_weights() is a no-op on "
            "an already-loaded model -- build from config instead."
        )

    rand.model.embed_tokens.load_state_dict(trained.model.embed_tokens.state_dict())
    rand.model.norm.load_state_dict(trained.model.norm.state_dict())
    rand.lm_head.load_state_dict(trained.lm_head.state_dict())
    if not torch.equal(rand.model.embed_tokens.weight,
                       trained.model.embed_tokens.weight):
        raise AssertionError("embedding transplant failed")

    del trained
    return rand.to(device).eval()


@torch.no_grad()
def logit_lens_floor(lens, model, tokenizer, prompts, layers, *, skip=8):
    """Score the J-lens and the plain logit lens side by side on the same lens.

    Returns (j_lens, logit_lens) dicts, each {layer: {"next_acc", "echo"}}:

      next_acc  top-1 == the true next token   -> requires learned structure
      echo      top-1 == the token at this position -> residual passthrough

    ``use_jacobian=False`` skips the J_l transport, giving vanilla unembed(h_l).
    """
    out = {}
    for use_j, key in ((True, "j_lens"), (False, "logit_lens")):
        hit_next = {l: 0 for l in layers}
        hit_echo = {l: 0 for l in layers}
        total = 0
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt")["input_ids"][0]
            readouts, *_ = lens.apply(
                model, p, layers=layers, max_seq_len=len(ids), use_jacobian=use_j
            )
            n = 0
            for l in layers:
                top1 = readouts[l].argmax(-1).cpu()
                n = min(len(top1), len(ids) - 1)
                for i in range(skip, n):
                    if top1[i].item() == ids[i + 1].item():
                        hit_next[l] += 1
                    if top1[i].item() == ids[i].item():
                        hit_echo[l] += 1
            total += max(0, n - skip)
        out[key] = {
            l: {"next_acc": hit_next[l] / max(total, 1),
                "echo": hit_echo[l] / max(total, 1)}
            for l in layers
        }
    return out["j_lens"], out["logit_lens"]


def distance_null(C: torch.Tensor) -> torch.Tensor:
    """A matrix with the same distance-decay profile as ``C`` and ZERO block structure.

    C_null[i, j] = mean of C[p, q] over all |p - q| == |i - j|.

    Any block metric must be reported as an excess over this. On the published lenses
    it reproduces 79-91% of the raw "tripartite" score -- i.e. most of the structure
    in the paper's headline figure is smooth drift.
    """
    L = C.shape[0]
    prof = torch.zeros(L, dtype=C.dtype)
    for d in range(L):
        prof[d] = torch.stack([C[i, i + d] for i in range(L - d)]).mean()
    N = torch.empty_like(C)
    for i in range(L):
        for j in range(L):
            N[i, j] = prof[abs(i - j)]
    return N
