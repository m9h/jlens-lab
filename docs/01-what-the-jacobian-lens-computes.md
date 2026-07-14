# 1. What the Jacobian lens actually computes

> The most-shared criticism of the global-workspace paper is that the J-lens is "just
> backprop." That criticism is about a different derivative. But the *steelman* of it is
> correct and more interesting, and almost nobody has said it out loud.

## A network is a function of two arguments

Write the model as `logits = f(θ, x)` — weights `θ`, input `x`. You can differentiate
with respect to either one, and the two answer completely different questions. In JAX
this is literally an `argnums` choice.

**Derivative A — the training gradient.** Compute a loss against a target, take `∂L/∂θ`.
Requires a label. Its entire purpose is to tell you how to *change the weights*.

```python
jax.grad(loss_fn, argnums=0)(θ, x)     # w.r.t. PARAMETERS
```

**Derivative B — the Jacobian of the forward map.** Freeze `θ`. Take an internal
activation `h` at layer `l`, and differentiate the model's *future output* with respect
to **that vector**. No loss. No label. Nothing is learned; the weights never move.

```python
jax.jacrev(lambda h: logits_from(h))(h_l)    # w.r.t. ACTIVATIONS
```

**The J-lens is Derivative B.** From the repo's own README:

```
lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

The expectation runs over prompts, source positions, and all current-and-future target
positions of a web-text corpus. It is a sensitivity analysis of a *frozen function*, not
a learning signal.

So *"LLMs literally cannot learn anything that wasn't first encoded as a gradient…
that's backprop"* is a true sentence about the wrong object. The two share the word
"gradient" and the same reverse-mode autodiff machinery, which is why they're easy to
conflate.

## The steelman, which is correct

Look at what `J_l` **is**: an averaged *linear map* from layer `l`'s residual stream into
the final residual stream, followed by the unembedding.

That is, structurally, **an analytically-derived tuned lens.** The tuned lens *learns* an
affine map from `h_l` to the output; the J-lens *computes* one as an expected Jacobian.

And this is why it coheres across layers. A transformer block does not transform its
input wholesale — it *adds* to it:

```
h_{l+1} = h_l + attn(h_l) + mlp(h_l)
```

Every layer writes into the **same** vector space, and that space is the one the
unembedding reads. So `h_l` already lives in near-output coordinates. That is why the
plain logit lens works at all, and it is why "the same J-space directions are meaningful
at many layers" is close to a restatement of "this architecture has residual
connections."

You do not get to rediscover your own skip connections and call it a global workspace.

## So is the whole thing trivial?

No — and here is the test that settles it, which nobody had run.

Randomize the transformer **blocks** but keep the **trained embedding and unembedding**.
(Randomizing everything makes token identity meaningless and the control passes for the
wrong reason.) Then fit a lens and score it two ways at each layer:

| metric | meaning |
|---|---|
| `next_acc` | lens top-1 == the true next token → requires *learned structure* |
| `echo` | lens top-1 == the token *at this position* → pure residual passthrough |

Run it (`jlens_lab.controls.randomize_blocks`), on Qwen3-0.6B:

```
        peak next_acc   peak echo
trained    0.3414         0.2936
random     0.0003         0.0016     <- FLOOR on both
```

The trained model shows a clean crossover — early layers echo the current token (~0.29),
and as depth increases echo decays while next-token prediction climbs to 0.34. **The
random-blocks model reads out nothing at all.**

So the J-lens's structure is *not* an artifact of the residual stream plus a trained
unembedding. It requires the blocks to have learned something. The lens **passes** the
model-randomization sanity check (Adebayo et al., *Sanity Checks for Saliency Maps*,
NeurIPS 2018) — the test that gutted a generation of gradient-based attribution methods.

Anthropic never ran this control. It is a point in their favour.

> **Note on a prediction that failed.** I expected the random-blocks lens to *echo
> strongly* — the embedding riding the additive residual highway up to a trained
> unembedding. It doesn't: echo is zero. Twenty-eight layers of random blocks inject
> enough noise to swamp the embedding entirely. The residual stream does not preserve the
> input when the blocks are junk.

## What the lens does that the logit lens doesn't

Fit both on the same model and read them at the same position — the `^` of the ASCII-art
face from the repo's own examples, on Qwen3.5-27B:

```
J-lens  top-5:  ['smile', 'nose', "'^", 'noses', 'grin']    <- a semantic cluster
logit   top-5:  ['Ċ', 'Âł', '..', '-', 'N']                 <- punctuation
```

The J-lens reads *facial features* at a position whose token is a caret, in a prompt
where none of those words appear. The logit lens has no idea what it is looking at.

That difference is real, and it is what the Jacobian buys you. It is also why the
paper's own metric is broken — see [3. Reading a lens honestly](03-reading-a-lens-honestly.md).

## Reading list

- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021) — the
  residual stream as a shared additive channel. Read this before arguing about lenses.
- nostalgebraist, *interpreting GPT: the logit lens* (2020). Short.
- Belrose et al., *Eliciting Latent Predictions with the Tuned Lens* (2023) — what the
  logit lens gets wrong and how much affine correction it needs.
- Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018) — why any
  gradient-based readout owes you a randomization control.
