"""Evaluation metrics for the SLR automation arms.

Real implementations (not a skeleton): these centralize the computations the
research/ scripts perform inline, so the package exposes one tested metric layer.

Screening : precision / recall / F1 (vs gold); McNemar (direct vs rag).
Extraction: macro per-field agreement (vs the reference); hallucination rate
            (deterministic source-span grounding); Wilcoxon (direct vs rag).
Agreement : Cohen's kappa (each arm vs manual gold).

Heavy deps (scikit-learn, scipy) are imported lazily inside each function so the
module loads even before they are installed, matching the rest of the package.
"""


def precision_recall_f1(pred_labels, gold_labels):
    """Binary screening metrics (pos_label=1=include). Returns (precision, recall, f1)."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, average="binary", pos_label=1, zero_division=0)
    return float(p), float(r), float(f)


def cohen_kappa(arm_labels, manual_labels):
    """Inter-rater agreement (Cohen's kappa) between an automation arm and the manual gold."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(arm_labels, manual_labels))


def field_f1(pred_fields, gold_fields):
    """Macro-averaged per-field agreement in [0,1] over fields present in gold.

    pred_fields / gold_fields: {field: value}. Uses the canonical field_score
    (exact match for domain, set-recall for hybrid_type, lenient otherwise): the
    same per-field score the extraction experiments report. Fields with no gold
    are skipped; returns None when nothing is scorable.
    """
    from src.evaluation.normalize import field_score
    scores = []
    for f, gold in (gold_fields or {}).items():
        s = field_score(f, (pred_fields or {}).get(f), gold)
        if s is not None:
            scores.append(s)
    return float(sum(scores) / len(scores)) if scores else None


def hallucination_rate(extractions, source_texts):
    """Share of non-null extracted values whose source quote is NOT a contiguous span
    of the source text (deterministic grounding; lower is better).

    extractions: {paper_id: {field: {value, source}}}; source_texts: {paper_id: text}.
    """
    from src.evaluation.grounding import grounded, normalize_text
    flags = []
    for pid, obj in (extractions or {}).items():
        tn = normalize_text(source_texts.get(pid, ""))
        for cell in (obj or {}).values():
            if isinstance(cell, dict):
                value, source = cell.get("value"), cell.get("source")
            else:
                value, source = cell, None
            if value is None or str(value).strip().lower() in ("", "null", "none"):
                continue
            flags.append(0.0 if grounded(source, value, tn) else 1.0)
    return float(sum(flags) / len(flags)) if flags else 0.0


def mcnemar(direct_correct, rag_correct):
    """Exact McNemar for paired screening (direct vs rag). Inputs: per-item correctness bools.

    Returns (stat, p): stat = min(b, c) discordant count (b = direct-right/rag-wrong,
    c = direct-wrong/rag-right), p = two-sided exact binomial. Mirrors research/statistical_tests.py.
    """
    from scipy.stats import binomtest
    b = sum(1 for d, r in zip(direct_correct, rag_correct) if d and not r)
    c = sum(1 for d, r in zip(direct_correct, rag_correct) if not d and r)
    n = b + c
    if n == 0:
        return 0, 1.0
    return min(b, c), float(binomtest(min(b, c), n, 0.5).pvalue)


def wilcoxon_f1(direct_f1s, rag_f1s):
    """Paired Wilcoxon signed-rank on per-item extraction scores (rag vs direct).

    Returns (stat, p) as floats. On an all-ties input (every paired difference is
    zero) scipy 1.17.1 does not raise: it emits a RuntimeWarning and returns
    (0.0, 1.0), i.e. p = 1.0 = no difference. Callers that want to special-case
    all-ties should check for an all-zero difference up front rather than relying
    on an exception.
    """
    from scipy.stats import wilcoxon
    w, p = wilcoxon(rag_f1s, direct_f1s, zero_method="wilcox")
    return float(w), float(p)
