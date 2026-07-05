"""Citation reconciliation for the synthesis task.

The synthesis prompt asks the model to support every claim with an inline
`[P###]` citation drawn ONLY from the supplied evidence excerpts. Listing the
retrieved paper ids as "the citations" is not the same as the citations the model
actually wrote: a model can omit some evidence papers, or emit an id that was
never in its evidence (a fabricated citation). This module reconciles the two.

`reconcile_citations` parses the bracketed `[P###]` ids the model actually wrote,
intersects them with the ids that were genuinely retrieved, and reports:
  - cited            : written AND backed by retrieved evidence (trustworthy provenance)
  - invented         : written but NOT among the evidence (fabricated / unsupported)
  - uncited_evidence : retrieved but never cited in the narrative (informational)

Pure / dependency-free so it can be unit-tested without an LLM call.
"""

import re

# Bracketed citation groups, e.g. [P016] or [P016, P018]. Matches the UI's
# linkify pattern (a letter + 2-4 digits) so server and client agree on what
# counts as a citation; bare "P50 papers" in prose is intentionally not a citation.
_CITE_GROUP = re.compile(r"\[([A-Za-z]\d{2,4}(?:\s*,\s*[A-Za-z]\d{2,4})*)\]")
_ID = re.compile(r"[A-Za-z]\d{2,4}")


def _norm_id(x: str) -> str:
    return str(x or "").strip().upper()


def citations_in(narrative: str) -> list[str]:
    """Return the bracketed paper ids the narrative actually cites (order preserved)."""
    out = []
    for group in _CITE_GROUP.findall(narrative or ""):
        for tok in _ID.findall(group):
            out.append(_norm_id(tok))
    return out


def reconcile_citations(narrative: str, retrieved_ids) -> dict:
    """Reconcile the narrative's inline citations against the retrieved evidence ids.

    Returns a dict with cited / invented / uncited_evidence id-lists and counts.
    `cited` is the trustworthy provenance: ids the model wrote AND that were
    actually supplied as evidence. `invented` flags ids the model wrote that were
    never retrieved (a fabricated citation the old retrieved-ids list would hide).
    """
    retrieved = {_norm_id(p) for p in (retrieved_ids or [])}
    written = set(citations_in(narrative))
    cited = sorted(written & retrieved)
    invented = sorted(written - retrieved)
    uncited_evidence = sorted(retrieved - written)
    return {
        "cited": cited,
        "invented": invented,
        "uncited_evidence": uncited_evidence,
        "n_cited": len(cited),
        "n_invented": len(invented),
    }
