# Findings

An independent replication of Anthropic's *"Verbalizable Representations Form a Global
Workspace in Language Models"* (Jul 2026) on open weights, using [`jlens`](https://github.com/anthropics/jacobian-lens)
+ this package. The point of the exercise was to run the controls the paper doesn't ship,
and to say plainly which of its claims survive them on open models.

**Reading guide.** ✅ survived its control · ⚠️ holds but narrow/caveated · ❌ does not hold
as stated · ⬜ untested. Every "real" result here is reported as *excess over a null*, not
raw. The one-line detail docs live in `jacobian-lens/results/` (this package's sibling
repo); the paper's own claim inventory is
`societies-of-thought/docs/anthropic_claims_scorecard.md`.

## Method / instrument

| # | finding | status | control | detail |
|---|---|---|---|---|
| M1 | **The J-lens is a real instrument.** Randomize the transformer blocks (keep trained embed/unembed) and it reads out ~nothing (next-token 0.0003). Its structure requires learned weights. Anthropic never ran this — it's a point *for* them. | ✅ | model-randomization (Adebayo 2018) | `results/randomization_control.json` |
| M2 | **The published fitting guidance under-fits by ~5×.** "~100 prompts is usable" is the *floor*; Anthropic's own lenses converged at 454–775 prompts, growing with width. A 100-prompt lens still makes plausible figures — it fails silently. | ✅ | convergence trace vs published `config.yaml` | README §1 |
| M3 | **The paper's `pass@k` metric rewards noise.** Min-rank over ~35 layers hands a diffuse lens one lottery ticket per layer; a logit lens emitting punctuation outscores a J-lens reading the content. Use identity_distance to gate, and baseline against the logit lens. | ✅ | logit-lens floor | README §3 |
| M4 | **One of the 38 published lenses is broken.** qwen3-32b's `.pt` is a mid-fit checkpoint (n=80 vs a config claiming 615) that degenerates toward a logit lens. The other 37, including qwen3.5-27b, check out. | ✅ | full audit (`artifacts.audit`) | README §5 |

## The paper's structural claims

| # | finding | status | control | detail |
|---|---|---|---|---|
| S1 | **The tripartite sensory/workspace/motor geometry is mostly a distance-null.** A matrix depending only on \|i−j\| reproduces 79–91% of the apparent block structure. Real excess exists and roughly doubles at ≥20B, but stays small (~+0.05); the sharp Sonnet-4.5 figure does not replicate on any open model. | ⚠️ | distance-only null (CKA) | README §4 |
| S2 | **The "nose" content demo reproduces — with a sharp scale threshold.** Absent below 14B, present at qwen3.5-27B (rank 2, a semantic cluster: smile/nose/grin). Emergence is between 14B and 27B, not gradual. | ⚠️ n=1 demo | logit-lens floor + eval | `results/scaling/` |
| S3 | **Capacity / bottleneck claim** ("J-space is never >10% of activation variance; holds tens of concepts"). The `capacity` module tests it against a random-subspace null; not yet run to a conclusion on open weights. | ⬜ | random-subspace null | README §6 |
| S4 | **Broadcast-back** is *not* claimed by the paper — it concedes the opposite (broadcast within one feedforward pass, no recurrent loops). Nothing to refute; don't spend on it. | — | — | scorecard |

## Post-training / "point of view" (Claim 6) — OLMo-3

The only open family with base + post-trained variants, public Dolma data, and ~1,486
checkpoints. This is where the paper's Claim 6 becomes externally testable.

| # | finding | status | control | detail |
|---|---|---|---|---|
| C1 | **Post-training does move the J-space.** mean cos(base, instruct) = 0.76 over 8 pairs, vs a ~0.96 refit-noise floor. First outside quantification of the claim. Cosine can't say *how* it moved. | ✅ | same-model refit floor | `results/posttrain/viewpoint_finding.md` |
| C2 | **The "low-rank / structured viewpoint shift" reading is confounded.** dJ looks strongly low-rank (0.003–0.02) but J is *already* low-rank; rank(dJ) ≈ rank(J) for 2/3 pairs. The low rank is mostly inherited. Do not report alone. | ❌ as stated | rank(dJ) relative to rank(J) | `viewpoint_finding.md` |
| C3 | **"The workspace holds discourse markers" is NOT general.** Pre-registered gate across 7 cross-family models: significant in 1. An n=1 observation over-generalized. Dropped as a general claim. | ❌ | permutation null, 7 models | `results/posttrain/phase0_gate.md` |
| C4 | **OLMo-3-7B specifically holds a suppressed informal/negative/profane/connective register in its workspace** that its output layers suppress — 9–12× beyond chance across four predefined lexicons, and flat (~1×) in six other models. Survives all three free controls: real, OLMo-specific, and *not* a frequency artifact (register beats frequency-matched neutral words in 5/5 Zipf bins; corr with Zipf = −0.15). | ⚠️ n=1 model | permutation null · cross-model flatness · frequency-matched control | `results/posttrain/olmo_register_finding.md` |
| C5 | **Data attribution of that register** — do the workspace-held tokens trace to a distinct slice of Dolma (raw web / toxic register)? Direct test of Hoel's "just an output transformation." The `attribution` module makes this runnable on the free infini-gram index (OLMo-2/Dolma-1.7 — a *prototype*, not OLMo-3's exact corpus). **Open:** the one experiment not yet completed. | ⬜ | Dolma source distribution vs neutral control | README §9 |

## The through-line

The J-lens is a genuine instrument (M1) — but the paper's most striking *pictures* soften
under a null. The headline geometry is mostly drift (S1); the scaling story is real but
gradual, not the sharp partition of the flagship figure (S1, S2); the "point of view"
shift is real in magnitude (C1) but not in the low-rank form it's read as (C2); and the
one vivid content result held for exactly one model (C3→C4). That model is OLMo, the one
whose data can close the loop (C5). What's solid is the *skepticism infrastructure*; what's
open is whether the register traces to the training data.

*Not peer-reviewed; open-weights only (no frontier-model activations). Corrections
welcome — the whole point is that these ran their own controls.*
