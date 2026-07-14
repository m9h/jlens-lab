# 3. Reading a lens honestly

> Three controls. Each one changed our answer, and two of them reversed it. None ship
> with `jlens`.

## Control 1 — the floor

Every J-lens number needs a baseline, and the right one is the **plain logit lens**:
`unembed(h_l)` with no Jacobian transport. It costs nothing — `jlens` already supports it:

```python
lens.apply(model, prompt, layers=layers, use_jacobian=False)
```

Nothing in the repo tells you to use it. Do it anyway. If the Jacobian isn't beating the
raw residual stream, it isn't earning its keep.

## The trap: rank-based `pass@k` rewards noise

The paper's lens-quality metric is:

> pass@k = mean over items of the fraction of `intermediates` whose
> **min-over-layers** lens rank ≤ k

Read that carefully. **Minimum rank over ~35 layers.** That hands a diffuse, noisy
distribution *one lottery ticket per layer*: it will park the target at a middling rank
*somewhere* by chance. A confident lens that is correctly reading the actual content puts
an unrelated word far down at **every** layer.

**The metric systematically rewards the lens that knows less.**

Here it is, unambiguously, on Qwen3.5-27B at the `^` of the ASCII face — using
Anthropic's own published lens:

| lens | rank("nose") | top-5 at that layer |
|---|---|---|
| **J-lens** | **2** | `['smile', 'nose', "'^", 'noses', 'grin']` |
| logit lens | 5 | `['Ċ', 'Âł', '..', '-', 'N']` |

The J-lens has understood it is looking at a face. The logit lens is emitting
punctuation. **Their ranks are comparable.** `pass@k` cannot tell them apart.

This is why, in our first pass, the logit lens "beat" the J-lens on mean pass@10 at 4B
and 8B — on Anthropic's own lenses. We nearly reported that as a refutation. It was an
artifact of the metric.

**If you use `pass@k`, score what is *in* the top-k, not just where the target lands.**

## Control 2 — the randomization test

Randomize the transformer **blocks**; keep the **trained embedding, final norm, and
unembedding**. Then a lens that still reads out coherent content is reading the
*architecture*, not the model.

```python
from jlens_lab.controls import randomize_blocks
hf_random = randomize_blocks("Qwen/Qwen3-0.6B")
```

Result (Qwen3-0.6B, 100 prompts, 32 held out):

```
        peak next_acc   peak echo
trained    0.3414         0.2936
random     0.0003         0.0016     <- floor on BOTH
```

The J-lens **passes**. Its structure requires learned weights. This is a point in
Anthropic's favour that they never claimed, because they never ran the control.

> ### The trap that produces a confident false PASS
>
> `model._init_weights(mod)` is a **silent no-op** on an already-loaded model in
> transformers v5. The naive control leaves the blocks **fully trained**, finds the lens
> works beautifully, and reports that it passes.
>
> We wrote that version first. It was caught only by asserting the weights had actually
> changed. Build from `AutoModelForCausalLM.from_config()` and transplant the trained
> embed/norm/head — then assert both halves.

## Control 3 — the distance-only null

For any claim about **geometry** — the CKA layer × layer matrix, the sensory / workspace /
motor block structure — you need this one, and it is brutal.

A matrix whose entries depend **only on layer distance** `|i−j|` will still score nonzero
on any within-block-minus-between-block contrast, because nearby layers are more similar
than distant ones. So build that matrix from the model's own decay profile — zero blocks
by construction — and score it identically.

```python
from jlens_lab import geometry
geometry.excess_over_null(C)     # {'real':…, 'null':…, 'excess':…}
```

On Anthropic's published lenses:

| model | real | null | **excess** |
|---|---|---|---|
| qwen3-1.7b | 0.117 | 0.092 | +0.025 |
| qwen3-4b | 0.078 | 0.062 | +0.016 |
| olmo-3-7b | 0.184 | 0.154 | +0.030 |
| qwen3-8b | 0.283 | 0.249 | +0.033 |
| qwen3-14b | 0.293 | 0.267 | +0.026 |
| gpt-oss-20b | 0.210 | 0.157 | **+0.053** |
| qwen3.5-27b | 0.267 | 0.217 | **+0.050** |

**The null recovers 79–91% of the score.** Most of the "tripartite structure" is smooth
drift.

And note what this does to a finding we nearly published: raw blockiness rises
0.08 → 0.29 from 4B to 14B, which looks exactly like *"block structure emerges with
scale."* It is not. It is the **decay profile steepening**. The excess over the null is
flat (0.016–0.033) across that whole range.

> ### A second trap, in the block metric itself
>
> An unconstrained 3-block search maximises its score by carving a **single layer** off as
> its own "block" — a lone layer has CKA 1.0 with itself by construction. On qwen3-1.7b
> (27 layers) the unconstrained optimum was cuts `(1, 26)`: `{L0} {L1..25} {L26}`. Enforce
> a minimum block size (`min_frac=0.15`).

## The honest summary of what the controls found

- The J-lens **passes** randomization. It is reading learned structure, not architecture.
- The J-lens **genuinely beats** the logit lens — qualitatively, at mid layers, on content
  the logit lens cannot see at all.
- The paper's **`pass@k` metric is broken**, and rewards noise.
- The **tripartite block structure is mostly drift** — though real excess appears at ≥20B,
  at the same scale the behaviour emerges.

Both the paper and its critics overclaimed. You only find that out by running the
controls.
