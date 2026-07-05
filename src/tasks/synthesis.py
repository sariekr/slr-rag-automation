"""
Task 3: Thematic synthesis per research question.

Gold (manual reference): the human-coded themes in slr/analysis/rq_narratives.md.
RAG retrieves evidence chunks across the WHOLE corpus per RQ and synthesizes a
narrative; direct/long-context is infeasible at corpus scale (~938K tokens), so
this is the task where retrieval is structurally required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import LLM_MODEL, SLR_DATA
from src.llm import call_llm, SYSTEM_ROLE, synthesis_prompt
from src.evaluation.citations import reconcile_citations

HUMAN_THEMES = Path(SLR_DATA).parent / "analysis" / "rq_narratives.md"


def load_human_themes(path=HUMAN_THEMES):
    """Return the human RQ-narrative synthesis (gold reference) text, or '' if absent."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _diverse_by_paper(hits, k):
    """Pick k chunks round-robin across papers (each paper's best first, then its
    second-best, ...), maximizing the number of distinct studies represented within
    the k budget. Counters the failure mode where one dominant paper's chunks fill
    the context window and starve rare themes: the broad-RQ synthesis coverage gap.
    Input hits must already be relevance-ranked.
    """
    from collections import OrderedDict
    by_paper = OrderedDict()
    for h in hits:                       # keep each paper's chunks in relevance order
        by_paper.setdefault(h.paper_id, []).append(h)
    selected = []
    while len(selected) < k:
        progressed = False
        for chunks in by_paper.values():
            if chunks:
                selected.append(chunks.pop(0))
                progressed = True
                if len(selected) >= k:
                    break
        if not progressed:               # every paper exhausted before reaching k
            break
    return selected


def synthesize(research_question, retriever, k=20, model=LLM_MODEL, allowed_ids=None,
               diversity=True):
    """Retrieve k cross-corpus evidence chunks for the RQ and synthesize a narrative.

    retriever is a GLOBAL retriever over the full corpus (not paper-scoped).
    allowed_ids: if given, only chunks from these paper_ids are used; SLR synthesis
    is restricted to the included studies.
    diversity: when True (default), select the k chunks round-robin across papers
    (source diversity) rather than top-k by raw relevance, so broad RQs draw on more
    studies and miss fewer rare themes. The k budget (token cost) is unchanged.

    Returns (narrative, citation_info, response). citation_info reconciles the ids
    the model ACTUALLY cited against the retrieved evidence (see
    src.evaluation.citations): cited = written AND retrieved (trustworthy
    provenance), invented = written but never retrieved (fabricated citation),
    uncited_evidence = retrieved but never cited. The earlier version returned the
    raw retrieved ids, which silently passed off retrieval as the model's citations.
    """
    hits = retriever.retrieve(research_question, k=k * 5)   # over-retrieve, then select k
    if allowed_ids is not None:
        hits = [h for h in hits if h.paper_id in allowed_ids]
    hits = _diverse_by_paper(hits, k) if diversity else hits[:k]
    evidence = [f"[{h.paper_id}] {h.text}" for h in hits]
    r = call_llm(synthesis_prompt(research_question, evidence), system=SYSTEM_ROLE, model=model)
    citation_info = reconcile_citations(r.text, [h.paper_id for h in hits])
    return r.text, citation_info, r
