# jlens-lab

**Convergence fitting, architecture layouts, and the controls the Jacobian lens needs
before anyone should believe a figure made with it.**

Companion to Anthropic's [`jlens`](https://github.com/anthropics/jacobian-lens) — the
reference code for *"Verbalizable Representations Form a Global Workspace in Language
Models"*. Upstream is Apache-2.0 and explicitly **"not maintained and not accepting
contributions."** This does not fork it. It depends on it.

Everything here comes from a failure we actually hit while trying to replicate the
paper on open weights. Every module pairs a measure with the null that keeps it honest.

| module | what it gives you | § |
|---|---|---|
| **`fitting`** | `fit_converged` — fit to convergence (the real protocol), checkpoint/resume | 1 |
| **`layouts`** | load the Mamba/linear-attention hybrids `from_hf` refuses | 2 |
| **`controls`** | the three controls no published result ships with | 3 |
| **`geometry`** | CKA layer×layer structure, and the distance-null most of it turns out to be | 4 |
| **`artifacts`** | validate a fit vs a published lens; audit all 38; recover checkpoints | 5 |
| **`capacity`** | test the "≤10% of variance" bottleneck claim vs a random-subspace null | 6 |
| **`workspace_content`** | what the workspace expresses *relative to the output*, norm-free | 7 |
| **`viewpoint`** | did post-training move the J-space in few directions or many | 8 |
| **`attribution`** | trace J-space content to the training documents that produced it | 9 |

New here? The two that bite first are **§1 fitting** (the published guidance under-fits
by ~5×) and **§3 controls** (no published result ships with a null). What actually held up
vs what didn't is in [`FINDINGS.md`](FINDINGS.md).

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
lens, report = fit_converged(model, wikitext(tok, 1000), checkpoint_path="ckpt.pt")
assert report.converged            # otherwise the lens is under-fit
report.write_csv("convergence.csv")  # diff against Anthropic's published trace
```

Fits are long and pods die. `checkpoint_path=` writes atomically every `checkpoint_every`
prompts and `resume=True` picks up where a killed run stopped — added after a dead Modal
pod ate an 11-hour fit with nothing to show.

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

## 5. `artifacts` — validate before you trust; audit what's published

Cosine cannot gate a fit: a 100-prompt under-fit scores **0.96** against the converged
reference, so any cosine bar that admits a correct lens admits a 5×-under-fit one.
`validate_fit` gates on `identity_distance` (`mean_l ||J_l − I||_F / ||I||_F`) — a scalar
the fit must *land on*, which cosine cannot — as well as cosine.

```python
from jlens_lab import artifacts
gate = artifacts.validate_fit(lens, "olmo-3-1025-7b")   # vs the published lens
assert gate["pass"]        # checks identity_distance, not just cosine
artifacts.audit()          # all 38 published lenses; finds the one that is broken
```

The `audit` found **qwen3-32b's published `.pt` is a mid-fit checkpoint** (n=80, vs a
config claiming 615) — a lens that silently degenerates toward a plain logit lens. Every
other published lens checks out, including qwen3.5-27b. `recover_from_checkpoint` salvages
a partial fit from a dead run.

## 6. `capacity` — is the workspace actually a *bottleneck*?

The paper's framing needs the workspace to be low-dimensional — "≤10% of activation
variance." `capacity` builds the J-space basis, measures the variance fraction the top
directions carry, and compares it against a **random-subspace null** of the same rank.
A number is only a bottleneck if it beats what a random subspace of equal size captures.

```python
from jlens_lab import capacity
capacity.capacity_report(lens, W_U)   # variance_fraction vs random_share
```

## 7. `workspace_content` — what the workspace holds, relative to the output

Ranking tokens by `||J_l^T W_U[t]||` is confounded: it just surfaces the highest
embedding-norm tokens (format junk), because the lens-vector norm tracks `||W_U[t]||`.
The fix divides the embedding norm out through the **motor band** (where `J → I`, so the
denominator ≈ how much the *output* expresses the token):

```python
from jlens_lab import workspace_content as wc
ratio = wc.band_ratio(lens, W_U, workspace_layers=ws, motor_layers=mo)
wc.marker_enrichment(ratio, tok, n_perm=1000)   # permutation null over a CLOSED marker set
```

High ratio = held by the workspace but suppressed at output. The `marker_enrichment` test
uses a marker set **fixed in advance** so the result is not post-hoc. Running it across 7
published lenses is what turned an n=1 "the workspace holds discourse markers" observation
into a specific, falsifiable finding — see [`FINDINGS.md`](FINDINGS.md).

## 8. `viewpoint` — did post-training move the J-space few directions or many?

`delta_effective_rank(base, instruct)` asks whether post-training reshaped the J-space in
a low-dimensional (structured, "toward a point of view") or diffuse way — Anthropic's
Claim 6. The catch, documented in the module: `dJ`'s low rank is *mostly inherited* from
`J` already being low-rank (`rank(dJ) ≈ rank(J)` for 2/3 pairs). **Report it relative to
J's own rank**, or it reads as a finding when it is arithmetic.

## 9. `attribution` — trace J-space content to training data

Wraps the free [infini-gram](https://infini-gram.io) API (the engine under AI2's
OlmoTrace). Given the tokens a workspace direction expresses, pull the actual training
documents — and their **Dolma source** (`cc_en_tail` web-tail vs `wiki`/`books`) — to test
whether a workspace register is a coherent slice of the training data or decoding noise.

```python
from jlens_lab import attribution
attribution.documents("disgusting")           # real training contexts, with source
attribution.source_distribution(tokens)        # which Dolma subsets they come from
```

*Caveat baked into the module: the free hosted index is OLMo-2 / Dolma-1.7, which overlaps
but is not OLMo-3's exact corpus — prototype provenance. The free tier rate-limits; the
wrapper backs off and paces bulk calls.*

## Tutorials

`docs/` — hands-on, each runs one control yourself. The most consequential (the
distance-null that dissolves most of a headline figure) needs **no GPU**, because the
lenses are published.

## Install

`jlens` is not on PyPI:

```bash
pip install git+https://github.com/anthropics/jacobian-lens.git
pip install -e .
```

## Licence

Apache-2.0, matching upstream.
