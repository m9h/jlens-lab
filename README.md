# jlens-lab

**Convergence fitting, architecture layouts, and the controls the Jacobian lens needs
before anyone should believe a figure made with it.**

Companion to Anthropic's [`jlens`](https://github.com/anthropics/jacobian-lens) — the
reference code for *"Verbalizable Representations Form a Global Workspace in Language
Models"*. Upstream is Apache-2.0 and explicitly **"not maintained and not accepting
contributions."** This does not fork it. It depends on it.

Everything here comes from a failure we actually hit while trying to replicate the
paper on open weights.

## 1. `fit_converged` — the lens does not tell you when it is under-fit

`jlens.fit(model, prompts)` averages over whatever list you give it. There is no
stopping rule, and the README says quality "saturates quickly" and "~100 prompts is
usable."

That is not what Anthropic did. The command line is recorded verbatim in every
published lens's `config.yaml`:

```
--n_prompts 1000 --min_prompts 100 --stop_window 10 --stop_at_delta 0.002
```

They fit **to convergence**, and the prompts actually consumed grow with model width:

| model | prompts | model | prompts |
|---|---|---|---|
| gemma-2-2b | 454 | qwen3-8b | 461 |
| gemma-3-12b | 775 | qwen3-14b | 615 |
| qwen3-1.7b | 466 | qwen3-32b | 615 |
| qwen3-4b | 479 | qwen3.5-27b | 672 |

`100` is their **floor**, the point where convergence checking begins — not their fit
size. Fitting on 100 leaves the lens under-fit by 4.6–6×, and it fails *silently*: the
lens still produces plausible figures. It cost us ~10 GPU-hours and an entire scaling
sweep before we noticed.

```python
lens, report = fit_converged(model, wikitext(tok, 1000))
assert report.converged            # otherwise the lens is under-fit
report.write_csv("convergence.csv")  # diff against Anthropic's published trace
```

## 2. `layouts` — the hybrids are where the science is

`jlens.from_hf` sniffs six transformer shapes and raises on anything else. That is a
naming mismatch, not an incompatibility — Nemotron-H calls its embedding `embeddings`
and its final norm `norm_f`.

It matters because the models that would *test* the architectural objection are exactly
the ones `jlens` cannot load:

- **Qwen3.5-27B** — 48 of 64 layers are *linear attention*; only 16 are softmax.
- **Nemotron-H** — 21 Mamba-2 mixers, 17 MLPs, 4 attention layers.

The standing objection to this research programme is that a "global workspace" is just an
artifact of transformer topology. **You cannot test that with a library that refuses to
load non-transformers.** (An earlier version of this README claimed the workspace is
*clearest* in hybrids. That was based on a single ASCII-art prompt and is retracted — the
102-prompt `association` eval shows a smooth scale effect in which a dense 32B beats the
27B hybrid. See `docs/04`.)

```python
from jlens_lab import from_hf, describe
describe(hf)   # {'Mamba2Mixer': 21, 'MLP': 17, 'Attention': 4}
model = from_hf(hf, tok)   # drop-in; auto-detects, falls back to jlens
```

## 3. `controls` — none of these ship, and each one changed our answer

**`randomize_blocks`** — the model-randomization sanity check (Adebayo et al. 2018).
Randomize the blocks, keep the trained embedding/unembedding. Our result: the J-lens
reads out *nothing* on random blocks (next-token 0.0003, echo 0.0016). It **passes** —
its structure requires learned weights. Anthropic never ran this, and it is a point in
their favour.

> **Trap:** `model._init_weights()` is a **silent no-op** on an already-loaded model in
> transformers v5. The naive control leaves the blocks fully trained and reports a
> confident false PASS.

**`logit_lens_floor`** — `use_jacobian=False` exists in `jlens` but nothing tells you
to baseline against it.

> **Trap:** rank-based `pass@k` (min rank over all layers) **rewards noise** — it hands
> a diffuse lens one lottery ticket per layer. At 27B the logit lens scores rank 5 while
> emitting `['Ċ','Âł','..','-','N']`; the J-lens scores rank 2 with
> `['smile','nose','noses','grin']`. The metric cannot tell them apart.

**`distance_null`** — for any CKA/geometry claim. A matrix depending only on `|i−j|`
still scores on a within-minus-between-block contrast. On the published lenses it
reproduces **79–91%** of the apparent sensory/workspace/motor structure.

## 4. `geometry` — the headline figure, reconstructed

The CKA analysis behind the paper's tripartite figure **is not in the repo**. The paper
describes it only as "geometrical matching." This is a reconstruction; say so.

```python
C = geometry.cka_matrix(geometry.jspace_reps(lens, W_U))
geometry.excess_over_null(C)   # the only number worth quoting
```

| model | real | null | **excess** |
|---|---|---|---|
| qwen3-1.7b | 0.117 | 0.092 | +0.025 |
| qwen3-4b | 0.078 | 0.062 | +0.016 |
| qwen3-8b | 0.283 | 0.249 | +0.033 |
| qwen3-14b | 0.293 | 0.267 | +0.026 |
| gpt-oss-20b | 0.210 | 0.157 | **+0.053** |
| qwen3.5-27b | 0.267 | 0.217 | **+0.050** ← "nose" emerges here |

Raw blockiness rising 0.08 → 0.29 with scale is the **decay profile steepening, not
blocks appearing**. The excess roughly doubles at ≥20B — the same scale at which the
behaviour appears — but stays small. The sharp tripartite structure of the Sonnet 4.5
figure does not replicate on any open model.

## Install

`jlens` is not on PyPI:

```bash
pip install git+https://github.com/anthropics/jacobian-lens.git
pip install -e .
```

## Licence

Apache-2.0, matching upstream.
