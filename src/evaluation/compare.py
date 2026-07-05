"""Pure arm-comparison helpers built on metrics.py.

The actual arm RUNS (corpus + LLM) live in research/ (ablation_rag.py for the
extraction arms, screening_h1.py for the screening arms). This module computes the
two comparisons from already-produced labels (no corpus or LLM dependency), so it
is importable and unit-testable:
  (1) each arm vs the manual gold  (precision/recall/F1 + Cohen's kappa)
  (2) direct vs rag                (paired McNemar)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import precision_recall_f1, cohen_kappa, mcnemar


def compare_screening(direct_labels, rag_labels, gold_labels) -> dict:
    """Compare the direct and rag screening arms against the manual gold.

    Returns per-arm precision/recall/f1 and kappa-vs-gold, plus the paired McNemar
    test between the two arms. All three label lists must be aligned (same order).
    """
    def arm(labels):
        p, r, f = precision_recall_f1(labels, gold_labels)
        return {"precision": p, "recall": r, "f1": f,
                "kappa_vs_gold": cohen_kappa(labels, gold_labels)}

    direct_correct = [a == g for a, g in zip(direct_labels, gold_labels)]
    rag_correct = [a == g for a, g in zip(rag_labels, gold_labels)]
    stat, p = mcnemar(direct_correct, rag_correct)
    return {
        "direct": arm(direct_labels),
        "rag": arm(rag_labels),
        "mcnemar": {"stat": stat, "p": p},
        "n": len(gold_labels),
    }
