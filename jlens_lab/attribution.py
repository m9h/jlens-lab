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


def _post(payload: dict, *, timeout: float = 30.0, url: str = API_URL,
          retries: int = 4, backoff: float = 1.5) -> dict:
    """POST to the infini-gram API with backoff on 403/429/5xx.

    The free tier rate-limits: a burst of a few hundred rapid calls returns HTTP 403.
    Bulk callers (source_distribution over many tokens) MUST pace themselves -- this
    retries with exponential backoff, but for large sweeps also pass a per-call ``pause``
    or self-host the index.
    """
    import time
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(f"infini-gram error: {out['error']}")
            return out
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(backoff ** (attempt + 1))   # 1.5, 2.25, 3.4, ... seconds
                continue
            raise
    raise last


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


def documents(query: str, *, index: str = DEFAULT_INDEX, max_docs: int = 5,
              max_disp_len: int = 200, **kw) -> list:
    """Actual training documents containing ``query``, with the matched span.

    ``find`` returns ``segment_by_shard`` -- a [start, end) rank range per shard.
    ``get_doc_by_rank`` must be told which shard ``s`` and a rank WITHIN that shard's
    segment (a global rank 500s a shard that does not hold it). We walk the non-empty
    shards and pull the first ``max_docs`` documents so their contexts can be read and
    judged for register coherence.
    """
    f = find(query, index=index, **kw)
    if int(f.get("cnt", f.get("count", 0))) == 0:
        return []
    docs = []
    for s, seg in enumerate(f.get("segment_by_shard", [])):
        if not seg or seg[1] <= seg[0]:
            continue
        for rank in range(seg[0], min(seg[1], seg[0] + max_docs - len(docs))):
            try:
                docs.append(_post({"index": index, "query_type": "get_doc_by_rank",
                                   "query": query, "s": s, "rank": rank,
                                   "max_disp_len": max_disp_len}, **kw))
            except Exception:
                break
        if len(docs) >= max_docs:
            break
    return docs


def _source_of(doc: dict) -> str:
    """Dolma source subset for a document, from its metadata.path prefix.

    e.g. 'cc_en_tail/cc_en_tail-0164.json.gz' -> 'cc_en_tail' (Common Crawl low-quality
    tail); 'wiki/...' -> 'wiki'; 'books/...' -> 'books'. This is the field that turns
    attribution into a REGISTER test: web-tail / forum sources vs curated (wiki/books/
    papers).
    """
    meta = doc.get("metadata", {})
    path = meta.get("path", "") if isinstance(meta, dict) else ""
    return path.split("/")[0] if path else "unknown"


def source_distribution(tokens, *, n_docs: int = 10, index: str = DEFAULT_INDEX,
                        **kw) -> dict:
    """Tally which Dolma sources a set of tokens' occurrences come from.

    For each token, sample up to ``n_docs`` documents and record their source subset.
    Returns {source: fraction} aggregated over all tokens. The register hypothesis
    predicts workspace-held tokens draw disproportionately from web-tail / social sources
    vs curated ones, relative to a neutral control set.
    """
    import time
    from collections import Counter
    pause = kw.pop("pause", 0.3)          # space calls; the free tier 403s on bursts
    c = Counter()
    for t in tokens:
        for d in documents(t, index=index, max_docs=n_docs, max_disp_len=1, **kw):
            c[_source_of(d)] += 1
            time.sleep(pause)
    total = sum(c.values())
    return {s: n / total for s, n in c.most_common()} if total else {}


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
