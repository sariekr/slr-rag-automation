#!/usr/bin/env python3
"""
End-to-end demonstration of the SLR-RAG pipeline on the bundled sample corpus.

Runs the four stages (index -> screen -> extract -> synthesize) plus the verification
and provenance layer on the license-clean sample corpus in data/sample/, and prints a
per-stage throughput funnel. This is a behaviour/throughput demonstration that the
software runs end to end; it is NOT an accuracy evaluation (no F1 / recall / gold
comparison). The result is also written to data/sample/example_output.json so that a
reviewer without an API key can inspect the expected output shape.

The screening / extraction / synthesis stages call an LLM through the configured
OpenAI-compatible endpoint, so an OPENROUTER_API_KEY (or equivalent) must be set;
indexing and retrieval run offline. Usage:

    export OPENROUTER_API_KEY=...
    python demonstration.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Always demonstrate on the bundled sample corpus (deterministic across machines).
config.CHROMA_COLLECTION = "sample_demo"
import src.indexing.corpus as corpus
corpus.PAPER_LIST = config.SAMPLE_DATA / "paper_list.json"
corpus.PDFS_TXT_DIR = config.SAMPLE_DATA / "txt"

from src.indexing.chunker import chunk_text
from src.indexing.embedder import Embedder
from src.indexing.vector_store import ChromaStore
from src.retrieval import build_retriever
from src.tasks.screening import screen, IC_EC
from src.tasks.extraction import extract, format_context
from src.tasks.synthesis import synthesize
from config import LLM_MODEL

EXTRACT_FIELDS = ["domain", "hybrid_type", "vector_db", "llm_used", "embedding_model"]
K_EXTRACT = 12
RESEARCH_QUESTION = ("What hybrid retrieval strategies and components (dense, sparse, "
                     "reranking, graph) do the included systems use?")


def bar(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set. Indexing/retrieval would run, but the "
              "screening/extraction/synthesis stages need a key. Set it and re-run.")
        return 1
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

    t0 = time.time()
    bar("Stage 1: Indexing")
    papers = corpus.load_corpus()
    store = ChromaStore()
    store.reset()
    store = ChromaStore()
    embedder = Embedder()
    chunks = []
    for p in papers:
        meta = {"paper_id": p.paper_id, "doi": p.doi, "title": p.title, "year": p.year, "author": p.author}
        chunks.extend(chunk_text(p.text, p.paper_id, metadata=meta))
    retriever = build_retriever("reranker", store=store, embedder=embedder, corpus_chunks=chunks)
    print(f"  papers indexed : {len(papers)}")
    print(f"  chunks         : {len(chunks)}  ({len(chunks)/max(len(papers),1):.1f} per paper)")
    print(f"  embedder       : {embedder.backend}  ({config.EMBEDDING_MODEL})")

    bar("Stage 2: Screening (PRISMA title/abstract decision)")
    included, excluded, parse_failed = [], [], 0
    screen_rows = []
    for p in papers:
        decision, resp = screen(p.title, p.text, ic_ec=IC_EC)
        (included if decision == 1 else excluded).append(p)
        screen_rows.append({"paper_id": p.paper_id, "title": p.title, "decision": "include" if decision == 1 else "exclude"})
    print(f"  screened       : {len(papers)}")
    print(f"  included       : {len(included)}")
    print(f"  excluded       : {len(excluded)}")

    bar("Stage 3: Extraction (structured fields, source-grounded)")
    extractions = []
    for p in included:
        query = f"{p.title}. " + ", ".join(EXTRACT_FIELDS)
        hits = [h for h in retriever.retrieve(query, k=40) if h.paper_id == p.paper_id][:K_EXTRACT]
        ctx = format_context(hits) if hits else None
        r, status, fields, sources = _extract(p, ctx)
        extractions.append({"paper_id": p.paper_id, "title": p.title, "status": status,
                            "fields": fields, "sources": sources, "n_chunks": len(hits)})
    ok = sum(1 for e in extractions if e["status"] == "ok")
    print(f"  papers extracted : {len(extractions)}")
    print(f"  parseable (ok)   : {ok}")
    print(f"  fields per paper : {len(EXTRACT_FIELDS)}")

    bar("Stage 4: Synthesis (per research question, included papers only)")
    allowed = [p.paper_id for p in included]
    narrative, cinfo, r = synthesize(RESEARCH_QUESTION, retriever, allowed_ids=allowed)
    cited = cinfo.get("cited", [])
    invented = cinfo.get("invented", [])
    print(f"  research question : {RESEARCH_QUESTION[:60]}...")
    print(f"  narrative length  : {len(narrative)} chars")
    print(f"  [P###] cited (reconciled to retrieved evidence) : {len(cited)}")
    print(f"  fabricated citations flagged                    : {len(invented)}")

    bar("Verification and provenance: SHA-256 integrity")
    export = {"screening": screen_rows, "extraction": extractions,
              "synthesis": {"research_question": RESEARCH_QUESTION, "narrative": narrative,
                            "cited": cited, "invented_flagged": invented}}
    payload = json.dumps(export, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    print(f"  SHA-256 of the exported result : {digest}")
    print(f"  re-hash matches (tamper check) : {hashlib.sha256(payload).hexdigest() == digest}")

    out = config.SAMPLE_DATA / "example_output.json"
    out.write_text(json.dumps({"sha256": digest, **export}, ensure_ascii=False, indent=2), encoding="utf-8")

    bar("Funnel")
    print(f"  {len(papers)} indexed -> {len(included)} included / {len(excluded)} excluded "
          f"-> {len(extractions)} extracted -> synthesis citing {len(cited)} papers "
          f"(SHA-256 verified)")
    print(f"  wall clock: {time.time()-t0:.0f}s   example output: {out.relative_to(config.BASE_DIR)}")
    return 0


def _extract(paper, ctx):
    """Call extract() with retrieved context (or the paper's own text as fallback)."""
    parsed, r = extract(EXTRACT_FIELDS, paper_text=(None if ctx else paper.text),
                        retrieved_context=ctx, model=LLM_MODEL)
    parsed = parsed or {}
    fields = {f: (parsed.get(f, {}) or {}).get("value") if isinstance(parsed.get(f), dict) else parsed.get(f)
              for f in EXTRACT_FIELDS}
    sources = {f: (parsed.get(f, {}) or {}).get("source") if isinstance(parsed.get(f), dict) else None
               for f in EXTRACT_FIELDS}
    status = "ok" if parsed else "parse_failed"
    return r, status, fields, sources


if __name__ == "__main__":
    raise SystemExit(main())
