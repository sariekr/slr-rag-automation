"""Characterization tests for the paragraph-aware overlapping chunker (pure logic).

src.indexing.chunker.chunk_text feeds the index; chunk boundaries and the overlap
carry-forward affect retrieval. These lock in current behavior, including the
degenerate-overlap termination guard (step = max(target - overlap, 1)), so a refactor
cannot reintroduce an empty chunk or a non-terminating window. These tests
characterize current behavior; they do not change it.
"""

from src.indexing.chunker import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("", "P1") == []


def test_whitespace_only_yields_no_chunks():
    assert chunk_text("   \n\n  \n", "P1") == []


def test_single_short_paragraph_is_one_chunk():
    out = chunk_text("hello world", "P1", target_words=10)
    assert len(out) == 1
    assert out[0].text == "hello world"
    assert out[0].id == "P1:0"
    assert out[0].chunk_index == 0


def test_metadata_is_merged_with_provenance():
    out = chunk_text("a b", "P7", metadata={"year": 2024}, target_words=10)
    assert out[0].metadata == {"year": 2024, "paper_id": "P7", "chunk_index": 0}


def test_overlap_carries_words_forward():
    # target 3, overlap 1: "a b c" fills, then "d e f" forces a split carrying "c".
    out = [c.text for c in chunk_text("a b c\n\nd e f", "P1", target_words=3, overlap_words=1)]
    assert out == ["a b c", "c d e", "e f"]


def test_chunk_ids_and_indices_are_sequential():
    out = chunk_text("a b c\n\nd e f", "P1", target_words=3, overlap_words=1)
    assert [c.chunk_index for c in out] == [0, 1, 2]
    assert [c.id for c in out] == ["P1:0", "P1:1", "P1:2"]


def test_degenerate_overlap_ge_target_still_terminates():
    # overlap_words (5) >= target_words (2): the step guard max(target - overlap, 1) = 1
    # prevents a non-terminating / empty-chunk window. Reaching the assert proves termination.
    out = chunk_text("a b c d e", "P1", target_words=2, overlap_words=5)
    assert len(out) > 0
    assert all(c.text.strip() for c in out)              # no empty chunk emitted
    assert all(len(c.text.split()) <= 2 for c in out)    # no chunk exceeds the target
