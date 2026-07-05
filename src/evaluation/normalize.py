"""
Canonical normalization for extraction scoring.

Uses the vendored canonical normalizers (src/evaluation/normalizers.py) so
extraction output is scored against the reference labels on the SAME canonical
vocabulary the reference was built with, with no external dependency. Without
this, taxonomy fields score near zero against the reference (raw "history"
!= "history_culture", "hybrid search" != "dense_sparse").

  domain      -> single canonical label  (DomainNormalizer)
  hybrid_type -> set-F1 over the canonical taxonomy (HybridTypeNormalizer)
  set fields  -> set-F1 over canonical members (vector_db / llm_used / embedding_model):
                 naming one of N gold values no longer earns full credit

Also exposes the taxonomies (DOMAIN_VALUES / HYBRID_VALUES) used to constrain the
extraction prompt: post-hoc normalization fixes domain but cannot recover
free-text hybrid_type, so the prompt itself must offer the allowed types (the
same framing that produced the reference labels -> a fair comparison).
"""

import re

from .normalizers import DomainNormalizer, HybridTypeNormalizer

# Canonical taxonomies (match slr/scripts/09b baseline prompt -> fair comparison)
DOMAIN_VALUES = [
    "general", "healthcare", "education", "construction", "manufacturing",
    "telecom", "legal", "finance", "agriculture", "energy", "transportation",
    "cybersecurity", "scientific_research", "software_engineering",
    "history_culture", "government", "other",
]
HYBRID_VALUES = [
    "dense_sparse", "graph_vector", "multi_stage", "adaptive", "multimodal",
    "reranking", "generation_ensemble",
]


def _norm(s) -> str:
    # Normalize separators (hyphen, en/em dash, dot, slash, underscore, comma) to
    # spaces so hyphenated names tokenize correctly: "GPT-3.5" == "gpt 3.5",
    # an en-dashed "Qwen3-8B" == its hyphenated form, "e5-base-v2" == "e5 base v2".
    # Without this, correct extractions of dashed model/embedding names scored 0.
    s = str(s or "").lower()
    for ch in ("\u2013", "\u2014", "-", "_", ",", ".", "/"):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _split(value) -> list[str]:
    parts = re.split(r"[,;/]| and |\+", str(value or ""))
    return [p.strip() for p in parts if p.strip()]


def canonical_domain(value) -> str:
    return DomainNormalizer.normalize(str(value or ""))


def canonical_hybrid_set(value) -> set[str]:
    return {HybridTypeNormalizer.normalize(p) for p in _split(value)}


def _lenient(pred, gold):
    """LEGACY lenient match (whole-string containment). Kept only to reproduce the
    earlier headline numbers for before/after comparison; superseded by per-member
    set scoring (_set_f1), because containment over the comma-joined gold credited
    naming one of N gold values with a full 1.0."""
    g, p = _norm(gold), _norm(pred)
    if not g:
        return None
    if not p:
        return 0.0
    if g in p or p in g:
        return 1.0
    gt, pt = set(g.split()), set(p.split())
    return len(pt & gt) / len(gt) if gt else None


# Fields whose value is a SET of entities (one or many, comma/"and"-separated):
# model names, embedding names, vector stores, and the hybrid-retrieval types.
# Scored by set agreement, not whole-string containment.
SET_FIELDS = ("hybrid_type", "vector_db", "llm_used", "embedding_model")


def _members(value) -> set[str]:
    """Canonical member set of a (possibly multi-value) field: split on
    comma/semicolon/slash/'and'/'+', normalize each part to its token form."""
    out = set()
    for part in _split(value):
        n = _norm(part)
        if n:
            out.add(n)
    return out


def _member_match(a: str, b: str) -> bool:
    """Whether two normalized value-members denote the same entity.

    Exact token-set equality ('gpt 3.5' == 'gpt-3.5'), or a multi-word name fully
    contained in a longer one ('gpt 3 5' in 'gpt 3 5 turbo'). The containment arm
    requires >=2 tokens so a bare family token ('gpt') cannot match a versioned
    name ('gpt 4'): the leniency that previously over-credited 1-of-N answers.
    """
    if a == b:
        return True
    sa, sb = set(a.split()), set(b.split())
    if sa == sb:
        return True
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(short) >= 2 and short <= long_


def _set_f1(pred, gold, recall_only: bool = False):
    """Symmetric set agreement over canonical members, in [0, 1] (None if no gold).

    recall    = fraction of gold members matched by some prediction member.
    precision = fraction of prediction members matched by some gold member.
    Returns F1 = 2PR/(P+R) (penalizes over- AND under-prediction). recall_only
    returns recall (gold-coverage view), for sensitivity reporting.
    """
    g = _members(gold)
    if not g:
        return None
    p = _members(pred)
    if not p:
        return 0.0
    rec = sum(1 for gi in g if any(_member_match(gi, pj) for pj in p)) / len(g)
    if recall_only:
        return rec
    prec = sum(1 for pj in p if any(_member_match(pj, gi) for gi in g)) / len(p)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _hybrid_set_f1(pred, gold, recall_only: bool = False):
    """Set agreement over the canonical hybrid-type taxonomy. Labels are already
    canonical, so exact set intersection (no fuzzy member match needed)."""
    g = canonical_hybrid_set(gold)
    if not g:
        return None
    p = canonical_hybrid_set(pred)
    if not p:
        return 0.0
    inter = len(p & g)
    rec = inter / len(g)
    if recall_only:
        return rec
    prec = inter / len(p)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def field_score(field: str, pred, gold, recall_only: bool = False):
    """Score one (pred, gold) cell in [0, 1], or None when there is no gold.

    domain      : exact canonical-label match (single categorical).
    hybrid_type : set-F1 over the canonical taxonomy labels.
    set fields  : set-F1 over canonical members (vector_db / llm_used /
                  embedding_model): symmetric precision+recall, so naming one of
                  N gold values no longer earns full credit.
    recall_only : return set recall instead of F1 (sensitivity reporting only).
    """
    if gold is None or str(gold).strip() == "":
        return None
    if field == "domain":
        return 1.0 if canonical_domain(pred) == canonical_domain(gold) else 0.0
    if field == "hybrid_type":
        return _hybrid_set_f1(pred, gold, recall_only)
    return _set_f1(pred, gold, recall_only)


def field_score_legacy(field: str, pred, gold):
    """The PRE-2026-06 scorer (domain exact, hybrid_type set-recall, others
    whole-string lenient). Retained ONLY to reproduce the earlier headline numbers
    for before/after comparison; not used in production scoring."""
    if gold is None or str(gold).strip() == "":
        return None
    if field == "domain":
        return 1.0 if canonical_domain(pred) == canonical_domain(gold) else 0.0
    if field == "hybrid_type":
        g = canonical_hybrid_set(gold)
        if not g:
            return None
        p = canonical_hybrid_set(pred)
        return len(p & g) / len(g)
    return _lenient(pred, gold)


def lenient_score(pred, gold):
    """Pre-normalization lenient score (for before/after comparison)."""
    return _lenient(pred, gold)
