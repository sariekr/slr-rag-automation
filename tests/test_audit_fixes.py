"""Unit tests for the audit P1 + F7 fixes (pure logic, no LLM/network).

Covers:
  F1  citation reconciliation  -> src.evaluation.citations.reconcile_citations
  F3  source grounding         -> src.evaluation.grounding.grounded
  F4  truncation surfacing     -> src.tasks.extraction.extraction_status
"""

from src.evaluation.citations import reconcile_citations, citations_in
from src.evaluation.grounding import grounded, normalize_text
from src.tasks.extraction import extraction_status


# ── F1: citation reconciliation ──────────────────────────────────────────────

def test_citations_parsed_from_brackets_only():
    # Bare "P50" in prose is not a citation; only bracketed ids count.
    assert citations_in("We found P50 papers, see [P003] and [P082, P011].") == ["P003", "P082", "P011"]


def test_cited_is_intersection_of_written_and_retrieved():
    info = reconcile_citations("Hybrid retrieval helps [P003]. Reranking too [P082].",
                               retrieved_ids=["P003", "P082", "P099"])
    assert info["cited"] == ["P003", "P082"]            # written AND retrieved
    assert info["uncited_evidence"] == ["P099"]          # retrieved, never cited
    assert info["invented"] == []


def test_invented_citation_is_flagged():
    # The core bug: a model citing an id that was never in the evidence must be caught,
    # not silently passed through as a clean reference.
    info = reconcile_citations("As shown in [P999], RAG wins. Also [P003].",
                               retrieved_ids=["P003", "P082"])
    assert info["invented"] == ["P999"]
    assert info["n_invented"] == 1
    assert info["cited"] == ["P003"]                     # P999 excluded from trustworthy provenance


def test_no_citations_yields_empty_cited():
    info = reconcile_citations("A narrative with no citations at all.",
                               retrieved_ids=["P003", "P082"])
    assert info["cited"] == []
    assert info["uncited_evidence"] == ["P003", "P082"]


def test_reconcile_handles_empty_inputs():
    info = reconcile_citations("", retrieved_ids=[])
    assert info == {"cited": [], "invented": [], "uncited_evidence": [], "n_cited": 0, "n_invented": 0}


# ── F3: source grounding ─────────────────────────────────────────────────────

PAPER = normalize_text(
    "The system uses a FAISS vector database with the BGE embedding model. "
    "Reranking is performed by a cross-encoder over the dense candidates."
)


def test_grounded_exact_substring():
    assert grounded("FAISS vector database", "FAISS", PAPER) is True


def test_ungrounded_fabricated_quote_is_rejected():
    # A plausible but fabricated source quote that is not a contiguous span.
    assert grounded("we deployed a Pinecone serverless index in production", "Pinecone", PAPER) is False


def test_grounded_value_verbatim_when_no_source():
    assert grounded(None, "cross-encoder", PAPER) is True


def test_ungrounded_when_value_absent_and_no_source():
    assert grounded(None, "Milvus", PAPER) is False


def test_grounded_tolerates_minor_paraphrase_via_windows():
    # >=50% of 5-word windows present -> grounded despite a small edit.
    quote = "the system uses a FAISS vector database with the BGE embedding model"
    assert grounded(quote, "FAISS", PAPER) is True


# ── F4: truncation / parse-failure surfacing ─────────────────────────────────

def test_status_ok_when_parsed():
    assert extraction_status({"domain": {"value": "healthcare"}}, finish_reason="stop") == "ok"


def test_status_truncated_when_length_capped():
    assert extraction_status(None, finish_reason="length") == "truncated"


def test_status_parse_failed_when_unparseable_not_truncated():
    assert extraction_status(None, finish_reason="stop") == "parse_failed"


def test_status_parse_failed_defaults_without_finish_reason():
    assert extraction_status(None) == "parse_failed"


# ── F6: evaluation metrics (real implementations, no longer skeleton) ─────────

from src.evaluation.metrics import (precision_recall_f1, cohen_kappa, field_f1,
                                     hallucination_rate, mcnemar, wilcoxon_f1)
from src.evaluation.compare import compare_screening


def test_precision_recall_f1_known():
    # gold=[1,1,0,0], pred=[1,0,0,0] -> TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=0.667
    p, r, f = precision_recall_f1([1, 0, 0, 0], [1, 1, 0, 0])
    assert p == 1.0 and r == 0.5 and abs(f - 2 / 3) < 1e-6


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1]) == 1.0


def test_field_f1_macro_exact_and_lenient():
    pred = {"domain": "healthcare", "llm_used": "gpt-4"}
    gold = {"domain": "healthcare", "llm_used": "gpt-4"}
    assert field_f1(pred, gold) == 1.0


def test_field_f1_skips_none_gold():
    assert field_f1({"x": "a"}, {"x": None}) is None


def test_hallucination_rate_half():
    ext = {"P1": {"a": {"value": "FAISS", "source": "uses FAISS vector store"},
                  "b": {"value": "Pinecone", "source": "deployed Pinecone cluster"}}}
    txt = {"P1": "the system uses FAISS vector store for retrieval"}
    assert hallucination_rate(ext, txt) == 0.5      # FAISS grounded, Pinecone fabricated


def test_mcnemar_discordant_pair():
    stat, p = mcnemar([True, True, False, True], [False, True, True, True])
    assert stat == 1 and 0.0 <= p <= 1.0


def test_mcnemar_no_discordant():
    stat, p = mcnemar([True, True], [True, True])
    assert stat == 0 and p == 1.0


def test_wilcoxon_f1_runs():
    _, p = wilcoxon_f1([0.5, 0.6, 0.7, 0.4], [0.7, 0.8, 0.9, 0.5])
    assert 0.0 <= p <= 1.0


def test_compare_screening_integration():
    res = compare_screening(direct_labels=[1, 1, 0, 0], rag_labels=[1, 0, 0, 0], gold_labels=[1, 1, 0, 0])
    assert res["n"] == 4
    assert res["direct"]["f1"] == 1.0                # direct == gold
    assert 0.0 <= res["mcnemar"]["p"] <= 1.0


# ── F9: embedding backend resolution (API vs local fallback) ─────────────────

from src.indexing.embedder import _resolve_backend


def test_backend_api_when_key_present():
    assert _resolve_backend("sk-xxx") == "api"


def test_backend_local_when_no_key():
    assert _resolve_backend("") == "local"


def test_backend_prefer_overrides_key_presence():
    assert _resolve_backend("sk-xxx", prefer="local") == "local"
    assert _resolve_backend("", prefer="api") == "api"


# ── 2026-06 review: multi-value set scoring + grounding dash parity ───────────

from src.evaluation.normalize import field_score, field_score_legacy


def test_multivalue_one_of_n_is_not_full_credit():
    # Naming 1 of 3 gold LLMs must NOT score 1.0 (the over-credit the review found).
    # set-F1: precision 1.0, recall 1/3 -> 0.5.
    s = field_score("llm_used", "GPT-4", "GPT-4, LLaMA, Qwen")
    assert abs(s - 0.5) < 1e-9
    # The legacy scorer is the one that wrongly gave full credit (kept for before/after).
    assert field_score_legacy("llm_used", "GPT-4", "GPT-4, LLaMA, Qwen") == 1.0


def test_multivalue_order_independent_full_match():
    assert field_score("llm_used", "LLaMA, GPT-4", "GPT-4, LLaMA") == 1.0


def test_multivalue_over_prediction_is_penalized():
    # Predicting all 7 hybrid types when gold is one must NOT score 1.0 (precision penalty).
    all7 = ("dense_sparse, graph_vector, multi_stage, adaptive, "
            "multimodal, reranking, generation_ensemble")
    s = field_score("hybrid_type", all7, "dense_sparse")
    assert abs(s - 0.25) < 1e-9          # recall 1.0, precision 1/7 -> F1 0.25


def test_norm_dash_dot_equality_in_set_field():
    # The dash/dot _norm fix: a hyphenated embedding name matches its spaced spelling.
    assert field_score("embedding_model", "e5-base-v2", "e5 base v2") == 1.0


def test_grounding_dash_normalization_parity():
    # The review's F2/③: grounding must strip dashes like the extraction scorer, so a
    # source quote "GPT-3.5" is grounded against a paper that spells it "gpt 3.5".
    paper = normalize_text("the proposed system uses gpt 3.5 as the language model")
    assert grounded("GPT-3.5", "GPT-3.5", paper) is True
    paper2 = normalize_text("we fine-tune the e5-base-v2 encoder for retrieval")
    assert grounded("e5-base-v2 encoder", "e5-base-v2", paper2) is True
