"""Convergence-based fitting. The single most important thing missing from ``jlens``.

``jlens.fit(model, prompts)`` averages over whatever list you hand it and stops. It
has no stopping rule, and the README's only guidance is that quality "saturates
quickly" and "~100 prompts is usable".

That guidance is wrong in a way that silently corrupts results. Anthropic's *own*
production fitter -- Neuronpedia's ``fit_lens.py``, which produced the published
lenses -- does not use a fixed count. Its command line, recorded verbatim in every
published ``config.yaml``:

    --n_prompts 1000 --min_prompts 100 --stop_window 10 --stop_at_delta 0.002

It fits until the running mean of ``J`` stops moving. And the prompts actually
consumed grow with model width:

    gemma-2-2b   454      qwen3-1.7b  466      qwen3-8b   461
    gemma-3-12b  775      qwen3-4b    479      qwen3-14b  615
                          qwen3-32b   615      qwen3.5-27b 672

100 is their *floor* -- the point at which convergence checking begins -- not their
fit size. Fitting on 100 prompts leaves the lens under-fit by 4.6-6x, and the
failure is silent: it produces a lens that works well enough to generate plausible
figures. (We lost ~10 GPU-hours and a whole scaling sweep to this before noticing.)

So: expose the stopping rule, and emit the convergence trace so the fit can be
audited against Anthropic's published ``convergence.csv`` for the same model.

``mean_rel_change`` is computed exactly as ``jlens.fitting.fit`` computes it
internally (it already tracks the quantity -- it just never acts on it):

    max_l  ||J_p - Jbar_l|| / ((n+1) * ||Jbar_l||)
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

import torch
from jlens import JacobianLens
from jlens.fitting import jacobian_for_prompt

# Anthropic's published defaults, from config.yaml.
STOP_AT_DELTA = 0.002
MIN_PROMPTS = 100
STOP_WINDOW = 10
N_MAX = 1000
MAX_SEQ_LEN = 128
DIM_BATCH = 128
SKIP_FIRST = 16
MAX_CHARS = 2000


@dataclass
class FitReport:
    """What actually happened during a fit -- so it can be audited, not assumed."""

    n_prompts: int
    converged: bool
    final_mean_rel_change: float
    trace: list[tuple[int, int, int, float]]  # (n_done, seq_len, n_valid, mrc)

    def write_csv(self, path: str | pathlib.Path) -> None:
        """Emit a trace in the same shape as Anthropic's published convergence.csv."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n_done", "seq_len", "n_valid_positions", "mean_rel_change"])
            w.writerows(self.trace)


def fit_converged(
    model,
    prompts,
    source_layers=None,
    *,
    stop_at_delta: float = STOP_AT_DELTA,
    min_prompts: int = MIN_PROMPTS,
    stop_window: int = STOP_WINDOW,
    max_seq_len: int = MAX_SEQ_LEN,
    dim_batch: int = DIM_BATCH,
    skip_first: int = SKIP_FIRST,
    checkpoint_path: str | pathlib.Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    verbose: bool = True,
) -> tuple[JacobianLens, FitReport]:
    """Fit a Jacobian lens until it stops moving, not until the prompts run out.

    Stops when ``mean_rel_change < stop_at_delta`` for ``stop_window`` consecutive
    prompts, after at least ``min_prompts``. Returns the lens and a FitReport whose
    trace can be diffed against Anthropic's published convergence.csv.

    If ``prompts`` is exhausted before convergence, ``report.converged`` is False --
    check it. A lens that never converged is under-fit, and under-fitting does not
    announce itself.
    """
    if source_layers is None:
        source_layers = list(range(model.n_layers))

    jac_sum: dict[int, torch.Tensor] | None = None
    n_done, under, trace = 0, 0, []
    n_skipped, last_error = 0, None
    mrc = float("nan")

    prompts = list(prompts)
    if not prompts:
        raise ValueError("no prompts given")

    # Checkpoint the running sum. jlens.fit takes a checkpoint_path and this wrapper
    # dropped it -- a 20-hour 7B fit then lives only in GPU memory, and a dead pod loses
    # all of it. Same class of mistake as a cloud function timing out with nothing saved.
    ckpt = pathlib.Path(checkpoint_path) if checkpoint_path else None
    start_idx = 0
    if ckpt is not None and resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        jac_sum = {int(k): v for k, v in st["jacobian_sum"].items()}
        n_done, trace, start_idx = st["n_done"], st.get("trace", []), st["next_idx"]
        if verbose:
            print(f"  resuming from {ckpt}: {n_done} prompts already done", flush=True)

    # PROBE: run the first prompt with NO exception handling, so a real error --
    # bf16 backward, accelerate device_map dispatch hooks, a bad layer index, a
    # source layer that includes the target -- surfaces after ONE forward pass
    # instead of being swallowed by the loop below and reappearing hundreds of
    # prompts later as an unrelated symptom.
    #
    # This is not hypothetical. A bare `except Exception: continue` here hid a
    # source_layers bug for 600 prompts and surfaced it as "TypeError: 'NoneType'
    # object is not subscriptable"; a downstream user then lost half a day on a GPU
    # box to "no prompt produced a Jacobian". Fail fast, loudly, with the real error.
    jacobian_for_prompt(
        model, prompts[0], source_layers,
        dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
    )

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < start_idx:
            continue
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, source_layers,
                dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
            )
        except Exception as e:
            # Only legitimate per-prompt skips (too short for skip_first) should land
            # here -- the probe above has already ruled out config errors. Keep the
            # error so an all-skipped run can report WHY instead of "produced nothing".
            n_skipped += 1
            last_error = e
            continue

        if jac_sum is None:
            jac_sum = {l: torch.zeros_like(per_prompt[l]) for l in source_layers}

        if n_done:
            mrc = max(
                (
                    (per_prompt[l] - jac_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jac_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        for l in source_layers:
            jac_sum[l] += per_prompt[l]
        n_done += 1
        trace.append((n_done, seq_len, n_valid, mrc))

        if n_done >= min_prompts and mrc == mrc and mrc < stop_at_delta:
            under += 1
            if under >= stop_window:
                if verbose:
                    print(f"  converged at {n_done} prompts "
                          f"(mean_rel_change={mrc:.6f} < {stop_at_delta} "
                          f"for {stop_window} consecutive)", flush=True)
                break
        else:
            under = 0

        if verbose and n_done % 50 == 0:
            print(f"    {n_done:4d} prompts  mean_rel_change={mrc:.6f}", flush=True)

        if ckpt is not None and n_done % checkpoint_every == 0:
            tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")     # atomic: never a torn file
            torch.save({"jacobian_sum": jac_sum, "n_done": n_done,
                        "trace": trace, "next_idx": prompt_idx + 1}, tmp)
            tmp.replace(ckpt)

    if jac_sum is None:
        raise RuntimeError(
            f"no prompt produced a Jacobian ({n_skipped}/{len(prompts)} skipped). "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    converged = under >= stop_window
    if verbose and not converged:
        print(f"  WARNING: exhausted {n_done} prompts WITHOUT converging "
              f"(mean_rel_change={mrc:.6f}, target <{stop_at_delta}). "
              f"The lens is under-fit. Supply more prompts.", flush=True)

    jac_mean = {l: jac_sum[l] / n_done for l in source_layers}
    d_model = next(iter(jac_mean.values())).shape[-1]
    lens = JacobianLens(jacobians=jac_mean, n_prompts=n_done, d_model=d_model)
    return lens, FitReport(n_done, converged, mrc, trace)


def wikitext(tokenizer, n: int = N_MAX, max_chars: int = MAX_CHARS,
             min_tokens: int = MAX_SEQ_LEN):
    """The corpus Anthropic's published lenses were fit on, per config.yaml:
    Salesforce/wikitext, wikitext-103-raw-v1, train, text[:2000].
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    out, it = [], iter(ds)
    while len(out) < n:
        try:
            text = next(it)["text"].strip()[:max_chars]
        except StopIteration:
            break
        if len(text) < 400 or text.startswith("="):
            continue
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) >= min_tokens:
            out.append(text)
    return out
