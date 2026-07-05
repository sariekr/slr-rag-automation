"""Characterization tests for round-robin chunk diversification (pure logic).

src.tasks.synthesis._diverse_by_paper maximizes the number of distinct studies within
the k budget (the broad-RQ synthesis coverage fix; counters one dominant paper starving
rare themes). These lock in: distinct-paper-first ordering, the k cap, in-paper relevance
order, and graceful handling when fewer than k chunks exist. These characterize
current behavior only.
"""

from types import SimpleNamespace

from src.tasks.synthesis import _diverse_by_paper


def _h(paper_id, tag):
    # _diverse_by_paper only reads .paper_id and treats the hit as opaque.
    return SimpleNamespace(paper_id=paper_id, text=tag)


def test_empty_hits_returns_empty():
    assert _diverse_by_paper([], 5) == []


def test_k_zero_returns_empty():
    assert _diverse_by_paper([_h("P1", "a")], 0) == []


def test_round_robin_prefers_distinct_papers_first():
    hits = [_h("P1", "a"), _h("P1", "b"), _h("P2", "c")]
    out = _diverse_by_paper(hits, 2)
    assert [h.paper_id for h in out] == ["P1", "P2"]   # distinct first, not ["P1", "P1"]


def test_respects_k_budget():
    hits = [_h("P1", "a"), _h("P1", "b"), _h("P2", "c"), _h("P2", "d")]
    out = _diverse_by_paper(hits, 3)
    assert len(out) == 3


def test_returns_all_when_fewer_than_k():
    hits = [_h("P1", "a"), _h("P2", "b")]
    out = _diverse_by_paper(hits, 5)
    assert [h.text for h in out] == ["a", "b"]          # no padding, no crash


def test_interleaves_in_paper_then_relevance_order():
    # by_paper = {P1: [a, b], P2: [c]}; round-robin -> P1a, P2c, then P1b. k=3.
    hits = [_h("P1", "a"), _h("P1", "b"), _h("P2", "c")]
    out = _diverse_by_paper(hits, 3)
    assert [h.text for h in out] == ["a", "c", "b"]
