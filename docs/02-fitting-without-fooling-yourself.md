# 2. Fitting a lens without fooling yourself

> This one cost us ten GPU-hours and an entire scaling sweep. The failure is silent: an
> under-fit lens still produces plausible figures.

## The library will not tell you when you are wrong

`jlens.fit(model, prompts)` averages the per-prompt Jacobians over whatever list you hand
it, and stops. There is **no stopping rule**. The README's only guidance:

> The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like corpus.
> Quality saturates quickly (§9.3); **~100 prompts is usable.**

We read that, fit every lens on 100 prompts, ran a scaling sweep across Qwen3
0.6B → 14B, and got a beautiful anomaly: the J-lens got monotonically **worse** with
model size while the logit lens got better. On `typo`, J-lens pass@10 fell
0.750 → 0.573 → 0.292 while the logit lens climbed 0.479 → 0.594 → 0.750.

A 4B model obviously knows "langauge" → "language" better than a 0.6B one. A lens that
reads it *less* well at 4B is telling you about the lens.

## What Anthropic actually did

Every published lens ships a `config.yaml` recording the **exact command** that produced
it. Here is the one that matters:

```
--n_prompts 1000 --min_prompts 100 --stop_window 10 --stop_at_delta 0.002
```

They do not use a fixed count. They fit **to convergence** — until the running mean of
`J` stops moving (`mean_rel_change < 0.002` for 10 consecutive prompts). `100` is
`--min_prompts`: the **floor** at which convergence *checking begins*. It is not the fit
size.

And the prompts actually consumed grow with model width:

| model | d_model | prompts to converge |
|---|---|---|
| pythia-70m | 512 | **1000** (never converged — hit the cap) |
| gemma-2-2b | 2304 | 454 |
| qwen3-1.7b | 2048 | 466 |
| qwen3-4b | 2560 | 479 |
| qwen3-8b | 4096 | 461 |
| gemma-3-12b | 3840 | 775 |
| qwen3-14b | 5120 | 615 |
| qwen3.5-27b | 5120 | 672 |
| qwen3-32b | 5120 | 615 |

`J_l` is a `d_model × d_model` object. The estimator's variance grows with width, so a
wider model needs more prompts. Fitting all of them on 100 under-fits by **4.6–6×**.

## Fit to convergence, and check that it did

```python
from jlens_lab import fit_converged, wikitext

lens, report = fit_converged(model, wikitext(tok, 1000))

assert report.converged        # <- otherwise the lens is under-fit
print(report.n_prompts, report.final_mean_rel_change)
report.write_csv("convergence.csv")
```

The trace has the same columns as Anthropic's published `convergence.csv`, so you can
diff your fit against theirs for the same model. That is the strongest possible check
and it costs nothing.

## The twist: our diagnosis was also wrong

Having found the under-fitting, we confidently blamed it for the anomaly. Then we
downloaded Anthropic's own converged lenses and compared:

```
cosine(our 100-prompt J, their 466-prompt J)  =  0.9719   (qwen3-1.7b)
typo pass@10:  ours 0.573    theirs 0.562
```

**Essentially identical.** The under-fitting was real, but it explained *nothing* about
the anomaly. The anomaly was the *metric* — see
[3. Reading a lens honestly](03-reading-a-lens-honestly.md).

Two lessons, and the second is the important one:

1. **Fit to convergence.** It is cheap insurance and the library will not do it for you.
2. **Do not stop at the first plausible explanation.** We had a real bug, a real fix, and
   a story that hung together — and it was still the wrong story. The only thing that
   settled it was downloading the artifact we were trying to reproduce and comparing
   directly against it.

## Which is the real rule

**Reproduce a published number before you run a single variant.**

We ran three original experiments before establishing that our harness could reproduce
anything Anthropic published. When the results came back strange, we could not tell a
finding from a bug — because we had never calibrated the instrument. Everything after
that point was recoverable only because their artifacts (lenses, `config.yaml`,
`convergence.csv`) are public enough to calibrate against.

The flagship demo is the gate: on the ASCII-art face, the lens should read out **"nose"**
at the `^`. If it doesn't, stop.
