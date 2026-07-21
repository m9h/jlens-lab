"""Trace J-space content back to the training data that produced it.

A J-lens direction is not a string, but infini-gram (the engine under AI2's OlmoTrace)
maps n-grams to the actual training documents that contain them. This module wraps the
free hosted API at ``https://api.infini-gram.io/`` so that, given the tokens/phrases a
workspace direction expresses, we can ask *where in the training corpus they come from*.

Why it matters. Hoel's deflationary reading of the J-space is "just a relatively constant
transformation of internal processing into the output." If workspace-held tokens instead
trace to a *specific, coherent slice of the training data* (e.g. OLMo-3-7B's workspace
retains an informal / web / toxic register that its output suppresses), then the workspace
is encoding something about the training distribution -- not merely accumulating toward
output. That is a data-grounded rebuttal, and OLMo is the only model whose training data
is public enough to run it.

CAVEAT baked into the design: the free hosted indexes are OLMo-2 / Dolma-1.7 era
(``v4_dolma-v1_7_llama``, ``v4_olmo-mix-1124_llama``), which overlap heavily with but are
not identical to OLMo-3's Dolma-3. So results are a *prototype* of provenance, not exact
OLMo-3 attribution; exact provenance needs a self-built index (deferred). ``count`` and
``prob`` still let us compare a token's corpus prevalence, which is enough to test the
register hypothesis directionally.

No network at import; every call is explicit. Tests mock the HTTP layer.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

API_URL = "https://api.infini-gram.io/"
# Hosted indexes that exist today (OLMo-2 / Dolma era). Not OLMo-3's exact corpus.
DEFAULT_INDEX = "v4_dolma-v1_7_llama"


def _post(payload: dict, *, timeout: float = 30.0, url: str = API_URL) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"infini-gram error: {out['error']}")
    return out


def count(query: str, *, index: str = DEFAULT_INDEX, **kw) -> int:
    """Number of occurrences of an n-gram in the training corpus."""
    out = _post({"index": index, "query_type": "count", "query": query}, **kw)
    return int(out.get("count", 0))


def corpus_prevalence(tokens, *, index: str = DEFAULT_INDEX, **kw) -> dict:
    """Corpus counts for a list of tokens/phrases -- the directional test of the register
    hypothesis. Returns {token: count}. A register that OLMo *holds but suppresses* should
    be genuinely present in the training data (high count), not a decoding artifact.
    """
    return {t: count(t, index=index, **kw) for t in tokens}


def find(query: str, *, index: str = DEFAULT_INDEX, **kw) -> dict:
    """Locate matches for an n-gram; returns the API's segment/shard pointers for retrieval
    via :func:`documents`. (``find`` + ``get_doc_by_ptr`` is the attribution primitive.)
    """
    return _post({"index": index, "query_type": "find", "query": query}, **kw)


def documents(query: str, *, index: str = DEFAULT_INDEX, max_docs: int = 5, **kw) -> list:
    """Actual training documents containing ``query``, with the matched span.

    Uses ``find`` to get the match count/pointers, then ``get_doc_by_rank`` to pull real
    documents. Returns up to ``max_docs`` dicts (the API's document payloads). The point is
    to *read* the contexts a workspace-held token came from and judge whether they form a
    coherent register.
    """
    f = find(query, index=index, **kw)
    cnt = int(f.get("cnt", f.get("count", 0)))
    if cnt == 0:
        return []
    docs = []
    for rank in range(min(max_docs, cnt)):
        try:
            d = _post({"index": index, "query_type": "get_doc_by_rank",
                       "query": query, "rank": rank,
                       "max_disp_len": kw.get("max_disp_len", 200)}, **kw)
            docs.append(d)
        except Exception:
            break
    return docs


def register_provenance(workspace_tokens, control_tokens, *, index: str = DEFAULT_INDEX,
                        **kw) -> dict:
    """Do workspace-held tokens have systematically different corpus prevalence than a
    control set? The directional, quota-cheap test before pulling documents.

    Returns median counts for each set and the ratio. The register hypothesis predicts the
    workspace-held (informal/charged) tokens are *well represented* in the (web-heavy)
    training corpus -- i.e. present, traceable, not decoding noise.
    """
    ws = corpus_prevalence(workspace_tokens, index=index, **kw)
    ct = corpus_prevalence(control_tokens, index=index, **kw)

    def med(d):
        v = sorted(d.values())
        return v[len(v) // 2] if v else 0

    return {"index": index,
            "workspace_counts": ws, "control_counts": ct,
            "workspace_median": med(ws), "control_median": med(ct),
            "n_workspace_absent": sum(c == 0 for c in ws.values()),
            "n_control_absent": sum(c == 0 for c in ct.values())}
