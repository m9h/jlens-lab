"""Verify a published lens before you trust it.

Anthropic's lenses are fit and hosted by Neuronpedia (an independent MIT project), and
each ships a ``config.yaml`` recording the exact fit command and, crucially,
``results.prompts_fitted`` -- how many prompts the fit actually consumed. Nothing checks
the ``.pt`` against it.

It should. Auditing all 38 published lenses on 2026-07-18 found one that disagrees:

    qwen3-32b   config says prompts_fitted: 615   .pt actually contains n_done: 80

The uploaded file is not a lens at all -- it is a raw ``fit()`` **checkpoint**, keys
``(jacobian_sum, n_done, next_idx, source_layers)``, saved 13% of the way through. 80
prompts is below Anthropic's own ``--min_prompts 100`` floor, so ``J`` is still close to
identity, the transport does nothing, and the J-lens silently degenerates into a plain
logit lens.

That failure is invisible without this check. We used that lens as a *dense control* in a
scaling experiment and it returned J-lens and logit-lens top-5s that were **character for
character identical** -- which read as a real architectural finding until we looked at
``n_prompts``. A near-published conclusion, from a file that was never a lens.

(A second artifact, ``qwen3.6-27b``, contains only ``.DS_Store`` files and no ``.pt`` at
all. Both are third-party pipeline errors, not defects in the method.)

    from jlens_lab.artifacts import load_lens, verify, audit

    lens = load_lens(path)          # recovers a checkpoint transparently, and says so
    verify("qwen3-32b")             # -> Report(ok=False, reason="checkpoint, not a lens")
    audit()                         # all published lenses; exit non-zero if any fail
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

REPO = "neuronpedia/jacobian-lens"
SUBDIR = "jlens/Salesforce-wikitext"

# Keys that mark a torch file as a mid-fit checkpoint rather than a saved lens.
_CHECKPOINT_KEYS = {"jacobian_sum", "n_done"}

# Anthropic's own floor: below this, convergence checking has not even begun.
MIN_PROMPTS = 100


@dataclass
class Report:
    """The outcome of auditing one published lens."""

    np_id: str
    ok: bool
    reason: str = ""
    claimed_prompts: int | None = None      # from config.yaml results.prompts_fitted
    actual_prompts: int | None = None       # from the .pt itself
    was_checkpoint: bool = False
    d_model: int | None = None
    n_layers: int | None = None

    def __str__(self) -> str:
        flag = "OK  " if self.ok else "FAIL"
        c, a = self.claimed_prompts, self.actual_prompts
        return (f"{flag} {self.np_id:22s} claimed={str(c):>5} actual={str(a):>5}"
                + (f"  {self.reason}" if self.reason else ""))


def _is_checkpoint(obj) -> bool:
    return isinstance(obj, dict) and _CHECKPOINT_KEYS.issubset(obj.keys())


def recover_from_checkpoint(state: dict):
    """Rebuild a JacobianLens from a mid-fit checkpoint: J = jacobian_sum / n_done.

    The running mean is exactly what ``fit`` would have returned had it stopped there, so
    the recovery is faithful -- but the result is a lens fit on ``n_done`` prompts, which
    may be far fewer than intended. Check ``n_prompts`` before using it.
    """
    from jlens import JacobianLens

    n = int(state["n_done"])
    if n <= 0:
        raise ValueError(f"checkpoint has n_done={n}; nothing to recover")
    jac = {int(l): v / n for l, v in state["jacobian_sum"].items()}
    d_model = next(iter(jac.values())).shape[-1]
    return JacobianLens(jacobians=jac, n_prompts=n, d_model=d_model)


def load_lens(path: str, *, allow_checkpoint: bool = True, warn=print):
    """Load a lens, transparently recovering a mid-fit checkpoint and saying so.

    ``jlens.JacobianLens.load`` rejects a checkpoint outright:

        ValueError: ... is not a JacobianLens file (found keys ['jacobian_sum',
        'n_done', 'next_idx', 'source_layers']; a fit() checkpoint?)

    which is correct but leaves you stuck on a file that *is* recoverable. This recovers
    it and warns loudly, because a recovered checkpoint is usually under-fit.
    """
    from jlens import JacobianLens

    try:
        return JacobianLens.load(path)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not _is_checkpoint(state):
            raise
        if not allow_checkpoint:
            raise
        lens = recover_from_checkpoint(state)
        warn(f"  WARNING: {path} is a mid-fit CHECKPOINT, not a saved lens. "
             f"Recovered at n_prompts={lens.n_prompts}."
             + (f" That is below Anthropic's own min_prompts={MIN_PROMPTS} floor -- "
                "J is near-identity and the J-lens will behave like a logit lens."
                if lens.n_prompts < MIN_PROMPTS else ""))
        return lens


def _claimed_prompts(config_text: str) -> int | None:
    m = re.search(r"prompts_fitted:\s*(\d+)", config_text)
    return int(m.group(1)) if m else None


def verify(np_id: str, *, repo: str = REPO, subdir: str = SUBDIR) -> Report:
    """Audit one published lens: does the .pt match what config.yaml claims?"""
    from huggingface_hub import HfApi, hf_hub_download
    import pathlib

    api = HfApi()
    files = [f for f in api.list_repo_files(repo) if f.startswith(np_id + "/")]
    pts = [f for f in files if f.endswith(".pt")]
    cfgs = [f for f in files if f.endswith("config.yaml")]

    if not pts:
        return Report(np_id, False, "no .pt in the repo at all")

    claimed = None
    if cfgs:
        try:
            claimed = _claimed_prompts(
                pathlib.Path(hf_hub_download(repo, cfgs[0])).read_text())
        except Exception:
            pass

    try:
        state = torch.load(hf_hub_download(repo, pts[0]),
                           map_location="cpu", weights_only=False)
    except Exception as e:
        return Report(np_id, False, f"{type(e).__name__} loading .pt",
                      claimed_prompts=claimed)

    if _is_checkpoint(state):
        n = int(state["n_done"])
        return Report(np_id, False, "mid-fit CHECKPOINT, not a lens",
                      claimed_prompts=claimed, actual_prompts=n, was_checkpoint=True)

    actual = state.get("n_prompts")
    jac = state.get("jacobians") or {}
    rep = Report(np_id, True, claimed_prompts=claimed, actual_prompts=actual,
                 d_model=state.get("d_model"), n_layers=len(jac) or None)

    if claimed is not None and actual is not None and int(actual) != int(claimed):
        rep.ok, rep.reason = False, "n_prompts disagrees with config.yaml"
    elif actual is not None and int(actual) < MIN_PROMPTS:
        rep.ok, rep.reason = False, f"under-fit: {actual} < min_prompts {MIN_PROMPTS}"
    return rep


def audit(np_ids=None, *, repo: str = REPO, verbose: bool = True) -> list[Report]:
    """Audit every published lens (or a given subset). Returns the failures last."""
    from huggingface_hub import HfApi

    if np_ids is None:
        files = HfApi().list_repo_files(repo)
        np_ids = sorted({f.split("/")[0] for f in files if "/" in f})

    reports = []
    for np_id in np_ids:
        try:
            r = verify(np_id, repo=repo)
        except Exception as e:
            r = Report(np_id, False, f"{type(e).__name__}: {e}")
        reports.append(r)
        if verbose:
            print(r, flush=True)

    bad = [r for r in reports if not r.ok]
    if verbose:
        print(f"\n{len(reports) - len(bad)}/{len(reports)} OK")
        for r in bad:
            print(f"  BROKEN  {r.np_id}: {r.reason}")
    return reports


def _cli() -> int:
    """`jlens-audit [np_id ...]` -- exit non-zero if any published lens is defective."""
    import sys
    reports = audit(sys.argv[1:] or None)
    return 1 if any(not r.ok for r in reports) else 0
