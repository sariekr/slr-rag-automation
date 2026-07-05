"""Characterization tests for the screening decision mapping (LLM call mocked).

src.tasks.screening.screen maps a model response to a binary decision (1=include,
0=exclude). The LLM call is replaced with a stub, so the test is a fast,
deterministic unit (TDD test-pyramid base).

It deliberately characterizes the silent-exclude behavior: an
UNPARSEABLE response maps to 0 (exclude), because the screening decision is binary
against the human gold (parse_json -> None -> {} -> 0). This is intentional and is
distinct from the web product's PRISMA flow, which routes an unparseable response to
a separate 'could not assess' bucket. Locking it in stops a refactor from silently
flipping either side. These characterize current behavior only.
"""

import types

from src.tasks import screening


def _patch_llm(monkeypatch, text):
    # screen() looks up call_llm in the screening module globals.
    monkeypatch.setattr(screening, "call_llm", lambda *a, **k: types.SimpleNamespace(text=text))


def test_valid_include_maps_to_1(monkeypatch):
    _patch_llm(monkeypatch, '{"decision": "include"}')
    decision, _ = screening.screen("title", "abstract")
    assert decision == 1


def test_valid_exclude_maps_to_0(monkeypatch):
    _patch_llm(monkeypatch, '{"decision": "exclude"}')
    decision, _ = screening.screen("title", "abstract")
    assert decision == 0


def test_unparseable_response_is_silently_excluded(monkeypatch):
    # Documented, intentional experiment behavior: unparseable -> exclude (not a third bucket).
    _patch_llm(monkeypatch, "I cannot determine this from the abstract alone.")
    decision, _ = screening.screen("title", "abstract")
    assert decision == 0


def test_missing_decision_key_maps_to_0(monkeypatch):
    _patch_llm(monkeypatch, '{"reason": "looks relevant"}')
    decision, _ = screening.screen("title", "abstract")
    assert decision == 0


def test_decision_match_is_prefix_and_case_insensitive(monkeypatch):
    # "Included (meets IC1)" lowercased startswith "incl" -> 1.
    _patch_llm(monkeypatch, '{"decision": "Included (meets IC1)"}')
    decision, _ = screening.screen("title", "abstract")
    assert decision == 1
