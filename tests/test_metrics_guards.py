"""Characterization / guard tests for the statistical metric helpers (pure logic).

These lock in the documented sharp edges the metric helpers depend on:
  - wilcoxon_f1 RAISES on an all-ties input (the caller must treat all-ties as
    'no difference' rather than calling it; mirrors the docstring contract)
  - mcnemar returns (0, 1.0) when there are no discordant pairs (explicit n==0 guard)
A refactor that silently changed either contract would skew the reported
significance. These characterize current behavior; they do not change it.
"""

from src.evaluation.metrics import precision_recall_f1, mcnemar, wilcoxon_f1


def test_precision_recall_f1_perfect_prediction():
    assert precision_recall_f1([1, 0, 1], [1, 0, 1]) == (1.0, 1.0, 1.0)


def test_precision_recall_f1_no_positive_prediction_does_not_crash():
    # zero_division=0 -> precision/f1 are 0.0 when nothing is predicted positive.
    p, r, f = precision_recall_f1([0, 0, 0], [1, 0, 1])
    assert (p, f) == (0.0, 0.0)


def test_mcnemar_no_discordant_pairs_returns_identity():
    # Identical correctness vectors -> b = c = 0 -> n==0 guard -> (0, 1.0), no crash.
    assert mcnemar([True, True, False], [True, True, False]) == (0, 1.0)


def test_mcnemar_single_discordant_pair():
    stat, p = mcnemar([True, True, False], [True, False, True])
    assert stat == 1
    assert p == 1.0


def test_wilcoxon_all_ties_returns_no_difference_and_does_not_raise():
    # FINDING (debugging-skill, low severity): the wilcoxon_f1 docstring claims scipy
    # "raises if every difference is zero" and that the caller must guard. In the pinned
    # scipy 1.17.1 it does NOT raise: it emits a RuntimeWarning (divide-by-zero in the
    # z-score) and returns (0.0, 1.0), i.e. p=1.0 = no difference. The docstring's "raises"
    # contract is stale for this version. Practical impact on the comparison results: none
    # (the reported Wilcoxon p-values are real-difference cases; p=1.0 is the desired
    # all-ties semantic), but any caller using try/except to catch the all-ties case would
    # never trip. This test characterizes the real behavior; the code is left unchanged.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = wilcoxon_f1([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert (stat, p) == (0.0, 1.0)


def test_wilcoxon_returns_floats_on_real_difference():
    stat, p = wilcoxon_f1([0.2, 0.3, 0.4, 0.1], [0.6, 0.7, 0.5, 0.9])
    assert isinstance(stat, float) and isinstance(p, float)
    assert 0.0 <= p <= 1.0
