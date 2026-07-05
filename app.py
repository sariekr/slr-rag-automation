"""
Web Application + RESTful API for the RAG-LLM systematic literature review
automation system.

Exposes the three pipeline tasks (screening / extraction / synthesis) over the
indexed corpus through a browser dashboard and a REST API, and exports results as
CSV / JSON / PDF, on top of the rag_pipeline modules (indexing / retrieval / llm /
tasks). This is the integration and delivery layer; no new research logic.

Endpoints:
  GET  /                 -> dashboard HTML
  GET  /api/health       -> readiness, model, corpus size
  GET  /api/papers       -> indexed papers (id, title, year) for the extraction picker
  POST /api/screen       -> screening decision (include/exclude) for a title+abstract
  POST /api/extract      -> RAG field extraction for one paper (paper-scoped retrieval)
  POST /api/synthesize   -> cross-corpus narrative synthesis for a research question
  GET  /api/retrieve     -> raw retrieval hits for an arbitrary query (transparency)
  POST /api/export       -> last result as csv | json | pdf

Run:
    cd rag_pipeline && source .venv/bin/activate
    python3 app.py                      # serves http://127.0.0.1:8000

Requires: build_index.py already run (data/chroma/ populated) and OPENROUTER_API_KEY
(read from ../slr/.env if present). LLM endpoints make real OpenRouter calls (cost).
"""

import os
import sys
import io
import csv
import json
import hashlib
import time
import asyncio
import logging
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("slr_rag.app")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))


# ── Load OPENROUTER_API_KEY from slr/.env BEFORE importing config ────────────
# config reads the key at import time; the experiment scripts source slr/.env
# manually, so the web app does the same (config itself does not load dotenv).
def _load_env():
    """Load the API key from an environment variable or a .env file. Priority:
    already-set environment variable > rag_pipeline/.env (portable install) > ../slr/.env (development repo)."""
    for env_path in (BASE_DIR / ".env", PROJECT_ROOT / "slr" / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import LLM_MODEL, EMBEDDING_MODEL
from src.indexing.corpus import load_corpus, Paper
from src.indexing.chunker import chunk_text
from src.indexing.embedder import Embedder
from src.indexing.vector_store import ChromaStore
from src.retrieval import build_retriever
from src.retrieval.retriever import _wrap_chroma, RetrievedChunk
from src.tasks.screening import screen, IC_EC
from src.tasks.extraction import extract, format_context, get_value, extraction_status
from src.tasks.synthesis import synthesize
from src.evaluation.normalize import DOMAIN_VALUES, HYBRID_VALUES
from src.evaluation.grounding import grounded, normalize_text
from src.llm import parse_json

STATIC_DIR = BASE_DIR / "static"
SESSION_COLLECTION = "session_corpus"   # live-ingested corpus, kept separate from the default slr_corpus

# Split the default IC/EC into separate include / exclude blocks for the UI.
_ICEC_PARTS = IC_EC.split("EXCLUDE if", 1)
DEFAULT_IC = _ICEC_PARTS[0].strip()
DEFAULT_EC = ("EXCLUDE if" + _ICEC_PARTS[1]).strip() if len(_ICEC_PARTS) > 1 else ""


def _compose_ic_ec(ic: str, ec: str) -> str:
    """Build a screening criteria block from separate include/exclude inputs.

    Accepts either keywords ("RAG, dense+sparse retrieval") or full sentences;
    keywords are wrapped in a 'relates to or satisfies' instruction so the LLM
    can judge relevance. Empty inputs fall back to the default hybrid-RAG IC/EC.
    """
    ic, ec = ic.strip(), ec.strip()
    if not ic and not ec:
        return IC_EC
    parts = []
    if ic:
        parts.append(
            f"INCLUDE the paper ONLY if it actually USES or implements {ic} as part of its own "
            "method or contribution. Judge conceptually by what the paper does: synonyms, "
            "abbreviations and equivalent techniques count (e.g. 'reranker' = a cross-encoder, "
            "re-ranking or passage re-ordering stage; 'dense + sparse' = 'hybrid search' or "
            "combining BM25 with dense embeddings), so the exact words need not appear. But "
            "EXCLUDE papers that only mention it in passing or as related work, or that address "
            "merely the broader area (e.g. general RAG/retrieval) WITHOUT employing it"
        )
    if ec:
        parts.append(f"EXCLUDE the paper if: {ec}")
    return ". ".join(parts) + "."


def _require_api_key():
    """Reject an LLM stage early when no OpenRouter key is configured, instead of
    returning an HTTP 200 whose per-paper results are all wrong. Reads the live
    environment value (populated by _load_env before config import)."""
    if not os.environ.get("OPENROUTER_API_KEY", ""):
        raise HTTPException(
            status_code=503,
            detail="This stage needs an OPENROUTER_API_KEY. Set it in .env "
                   "(indexing and raw retrieval work without a key; screening, "
                   "extraction and synthesis do not).")

# Retrieval depth: paper-scoped for extraction, corpus-wide for synthesis.
# Reranker retrieval with k=12 for extraction and k=20 for synthesis.
K_EXTRACT = 12
K_SYNTH = 20

# Default closed extraction fields for the built-in review schema. The first
# five are canonical taxonomy/entity fields; frameworks and metrics_used are
# extra open fields. "hybrid_type" is singular to match the canonical schema.
EXTRACT_FIELDS = [
    "domain", "hybrid_type", "llm_used",
    "vector_db", "embedding_model", "frameworks", "metrics_used",
]

# Allowed-value taxonomies for the closed fields, injected into the extraction
# prompt so the model emits canonical labels. Without this, free-text labels
# cannot be normalized post-hoc, so domain/hybrid_type extraction is weaker.
FIELD_HINTS = {
    "domain": "choose exactly one of: " + ", ".join(DOMAIN_VALUES),
    "hybrid_type": "choose all that apply (comma-separated) from: " + ", ".join(HYBRID_VALUES),
}

# The four review research questions (same wording as synthesis_demo.py).
RESEARCH_QUESTIONS = {
    "RQ1": "What are the main application domains of RAG systems, and what key challenges arise in these domains?",
    "RQ2": "What methods, approaches, and tools are commonly used to build RAG systems?",
    "RQ3": "What hybrid methods are used to enhance RAG performance, and how are they applied?",
    "RQ4": "What integration methods, tools, and techniques are used to deploy RAG systems?",
}

# In-memory application state (single-process demo).
STATE = {
    "retriever": None,
    "embedder": None,
    "papers": [],            # [{paper_id, title, year}] for the UI
    "corpus_papers": [],     # Paper objects (with text) of the indexed corpus: PRISMA/report screen these
    "n_chunks": 0,
    "ready": False,
    "last_result": None,
    "corpus_source": "",
    "pool_size": 0,
}

# Guards the multi-field STATE swap during (re)indexing so a concurrent request
# (sync handlers run in FastAPI's threadpool) never reads a half-rebuilt corpus.
STATE_LOCK = threading.RLock()

# Restart persistence: the expensive PRISMA included-set + last result survive a
# restart (the Chroma index itself is already persistent on disk).
SESSION_STATE_PATH = BASE_DIR / "data" / "session_state.json"


@asynccontextmanager
async def lifespan(app):
    """Build the corpus chunks + reranker retriever once at startup (synthesis_demo pattern)."""
    logger.info("Loading corpus and building retriever...")
    store = ChromaStore()
    embedder = Embedder()
    STATE["embedder"] = embedder
    STATE["store"] = store
    try:
        papers = load_corpus()                       # baseline corpus when the development repo is mounted
    except Exception as e:
        papers = []
        logger.warning("Baseline corpus not found (%s); starting with an empty corpus.", e.__class__.__name__)
    if papers:
        chunks = []
        for p in papers:
            meta = {"paper_id": p.paper_id, "doi": p.doi, "title": p.title,
                    "year": p.year, "author": p.author}
            chunks.extend(chunk_text(p.text, p.paper_id, metadata=meta))
        STATE["retriever"] = build_retriever("reranker", store=store,
                                             embedder=embedder, corpus_chunks=chunks)
        STATE["papers"] = [{"paper_id": p.paper_id, "title": p.title, "year": p.year} for p in papers]
        STATE["corpus_papers"] = papers
        STATE["n_chunks"] = len(chunks)
        STATE["corpus_source"] = f"default corpus · {len(papers)} papers"
        STATE["pool_size"] = len(papers)
    else:                                            # portable install: the user builds the corpus from the Indexing tab
        STATE["retriever"] = None
        STATE["papers"] = []
        STATE["corpus_papers"] = []
        STATE["n_chunks"] = 0
        STATE["corpus_source"] = "empty, build a corpus from the Indexing tab (arXiv / PDF)"
        STATE["pool_size"] = 0
    STATE["ready"] = True
    _restore_session()                               # restart persistence: PRISMA + last result (if any)
    logger.info("Ready: %d papers, model=%s", len(papers), LLM_MODEL)
    yield


app = FastAPI(title="RAG-LLM SLR Automation Web Interface", lifespan=lifespan)


# ── Request models ───────────────────────────────────────────────────────────
class ScreenReq(BaseModel):
    title: str
    abstract: str = ""
    ic: str = ""
    ec: str = ""
    model: str = ""        # empty -> config.LLM_MODEL; an OpenRouter slug overrides the model


class ExtractReq(BaseModel):
    paper_id: str
    fields: list[str] = []              # empty -> default SLR schema; fill in to adapt to a new domain
    field_hints: dict[str, str] = {}    # allowed-value hints for closed fields (optional)
    model: str = ""                     # empty -> config.LLM_MODEL


class SynthReq(BaseModel):
    research_question: str
    model: str = ""        # empty -> config.LLM_MODEL


class ExportReq(BaseModel):
    fmt: str  # csv | json | pdf


class PoolIngestReq(BaseModel):
    n: int = 20


class PrismaReq(BaseModel):
    n: int = 15
    ic: str = ""
    ec: str = ""
    paper_ids: list[str] = []   # selected screening pool; empty -> first n papers
    model: str = ""             # empty -> config.LLM_MODEL


class FetchReq(BaseModel):
    query: str
    max: int = 10
    full_text: bool = True


class ReportReq(BaseModel):
    n: int = 10
    ic: str = ""
    ec: str = ""
    rqs: list[str] = []
    fields: list[str] = []              # empty -> default SLR schema; fill in to adapt to a new domain
    field_hints: dict[str, str] = {}    # allowed-value hints for closed fields (optional)
    model: str = ""                     # empty -> config.LLM_MODEL


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the dashboard."""
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse({"error": "dashboard not found"}, status_code=404)


@app.get("/api/health")
async def health():
    """Readiness probe and pipeline configuration."""
    return {
        "status": "ok" if STATE["ready"] else "loading",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "n_papers": len(STATE["papers"]),
        "n_chunks": STATE["n_chunks"],
        "corpus_source": STATE["corpus_source"],
        "pool_size": STATE["pool_size"],
        "default_ic": DEFAULT_IC,
        "default_ec": DEFAULT_EC,
    }


@app.get("/api/papers")
async def papers():
    """Indexed papers for the extraction picker."""
    return STATE["papers"]


def _reindex(papers, source_label: str) -> dict:
    """Stage 1-3: chunk + embed + index `papers` into the session collection,
    then rebuild the live retriever and corpus state. Returns timing telemetry."""
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up.")
    t0 = time.time()
    chunks = []
    for p in papers:
        meta = {"paper_id": p.paper_id, "doi": p.doi, "title": p.title,
                "year": p.year, "author": p.author}
        chunks.extend(chunk_text(p.text, p.paper_id, metadata=meta))
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not chunk any text (empty corpus).")

    embedder = STATE["embedder"]
    embeddings = embedder.embed([c.text for c in chunks])
    store = ChromaStore(collection=SESSION_COLLECTION)
    store.reset()
    store = ChromaStore(collection=SESSION_COLLECTION)
    store.add(ids=[c.id for c in chunks], embeddings=embeddings,
              documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks])
    # Build the new retriever into a local before touching STATE, then swap all
    # corpus fields together under the lock so a concurrent request never reads a
    # half-rebuilt corpus (retriever updated but corpus_papers not, etc.).
    retriever = build_retriever("reranker", store=store, embedder=embedder, corpus_chunks=chunks)
    papers_meta = [{"paper_id": p.paper_id, "title": p.title, "year": p.year} for p in papers]
    with STATE_LOCK:
        STATE["store"] = store
        STATE["retriever"] = retriever
        STATE["papers"] = papers_meta
        STATE["corpus_papers"] = papers
        STATE["n_chunks"] = len(chunks)
        STATE["corpus_source"] = source_label
        STATE["last_result"] = None
        STATE["prisma"] = None          # new corpus -> the old PRISMA included-set is invalid
    _persist_session()                  # also flush the now-invalid session state to disk
    dt = time.time() - t0
    return {
        "n_papers": len(papers),
        "n_chunks": len(chunks),
        "seconds": round(dt, 1),
        "ms_per_chunk": round(dt / max(len(chunks), 1) * 1000, 1),
        "corpus_source": source_label,
    }


@app.post("/api/ingest/pool")
def ingest_pool(req: PoolIngestReq):
    """Index N papers from the existing corpus pool into the session collection."""
    n = max(1, min(req.n, STATE["pool_size"] or 101))
    try:
        papers = load_corpus(limit=n)
    except Exception:                                       # a portable install may have no built-in pool
        papers = []
    if not papers:
        raise HTTPException(status_code=400, detail="No built-in pool available; fetch from arXiv or upload PDFs.")
    return _reindex(papers, f"{len(papers)} papers from the pool")


@app.post("/api/ingest/upload")
async def ingest_upload(files: list[UploadFile] = File(...)):
    """Index user-uploaded PDFs (text extracted with PyMuPDF) into the session collection."""
    datas = [((f.filename or ""), await f.read()) for f in files]   # async read, then offload blocking work
    return await asyncio.to_thread(_process_uploads, datas)


def _process_uploads(datas):
    """Extract text from uploaded PDF bytes and index them (runs in a worker thread so the
    PDF parse + embed + index never block the event loop)."""
    import fitz  # PyMuPDF, lazy import
    papers = []
    for i, (filename, data) in enumerate(datas, 1):
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception:
            continue
        if text.strip():
            papers.append(Paper(paper_id=f"U{i:03d}", doi="", title=(filename or f"upload_{i}"),
                                year="", author="", txt_path=None, text=text))
    if not papers:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded PDFs.")
    return _reindex(papers, f"{len(papers)} uploaded PDFs")


@app.post("/api/prisma")
def api_prisma(req: PrismaReq):
    """Batch screening over N pool papers -> PRISMA 2020 flow counts + per-paper decisions.

    Produces the PRISMA process-flow output: the corpus is
    screened paper-by-paper against IC/EC, then the identification -> screening ->
    included flow is reported with exclusion reasons.
    """
    _require_api_key()
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up.")
    corpus = STATE["corpus_papers"]
    if not corpus:
        raise HTTPException(status_code=400, detail="Index a corpus first (Indexing tab).")
    if req.paper_ids:                                  # the pool the user selected
        sel = set(req.paper_ids)
        papers = [p for p in corpus if p.paper_id in sel]
    else:                                              # backward-compatible: first n
        n = max(1, min(req.n, len(corpus)))
        papers = corpus[:n]
    if not papers:
        raise HTTPException(status_code=400, detail="No papers selected to screen.")

    t0 = time.time()
    ic_ec = _compose_ic_ec(req.ic, req.ec)
    model = req.model or LLM_MODEL
    tin = tout = 0

    # RAG screening: the decision is made over the chunks most relevant to the IC,
    # retrieved from the vector DB (paper-scoped retrieval), instead of the first N
    # characters. The method is found wherever it appears in the paper (higher recall)
    # and screening itself becomes retrieval-based.
    ic_query = (req.ic or "retrieval augmented generation hybrid retrieval method").strip()
    def _ctx(p, k):
        hits = _paper_chunks(p.paper_id, ic_query, k)
        return ("\n\n".join(h.text for h in hits)) if hits else (p.text or "")[:1500]

    def _reason(resp):                                 # rationale for each decision (transparency)
        return ((parse_json(resp.text) or {}).get("reason") or "").strip()

    # Phase 1: Screening (title + abstract level filtering).
    screened_in, excl_screening = [], []
    for p in papers:
        try:
            dec, r = screen(p.title, _ctx(p, 5), ic_ec=ic_ec, model=model)   # RAG: 5 IC-relevant chunks
            tin += r.input_tokens; tout += r.output_tokens
        except Exception as e:                                  # one paper failing must not abort screening
            excl_screening.append({"paper_id": p.paper_id, "title": p.title,
                                   "reason": f"could not be assessed ({e.__class__.__name__})"})
            continue
        if parse_json(r.text) is None:        # response could not be parsed -> not a silent EXCLUDE; separate bucket
            excl_screening.append({"paper_id": p.paper_id, "title": p.title,
                                   "reason": "could not be assessed (response not parseable)"})
            continue
        if dec == 1:
            screened_in.append(p)
        else:
            excl_screening.append({"paper_id": p.paper_id, "title": p.title, "reason": _reason(r)})

    # Phase 2: Eligibility, full-text assessment with rationale (PRISMA 2020).
    included_papers, included, excl_eligibility = [], [], []
    for p in screened_in:
        try:
            dec, r = screen(p.title, _ctx(p, 12), ic_ec=ic_ec, model=model)  # RAG: deeper, 12 IC-relevant chunks
            tin += r.input_tokens; tout += r.output_tokens
        except Exception as e:
            excl_eligibility.append({"paper_id": p.paper_id, "title": p.title,
                                     "reason": f"could not be assessed ({e.__class__.__name__})"})
            continue
        if parse_json(r.text) is None:        # response could not be parsed -> not a silent EXCLUDE
            excl_eligibility.append({"paper_id": p.paper_id, "title": p.title,
                                     "reason": "could not be assessed (response not parseable)"})
            continue
        if dec == 1:
            included_papers.append(p)
            included.append({"paper_id": p.paper_id, "title": p.title, "reason": _reason(r)})
        else:
            excl_eligibility.append({"paper_id": p.paper_id, "title": p.title, "reason": _reason(r)})

    summary = {
        "identification": len(papers),
        "screening_excluded": len(excl_screening),
        "eligibility_assessed": len(screened_in),
        "eligibility_excluded": len(excl_eligibility),
        "included": len(included),
    }
    # the included set + criteria are stored here -> the Report step consumes this and does not re-screen
    STATE["prisma"] = {"summary": summary, "included_papers": included_papers,
                       "ic": req.ic, "ec": req.ec}
    result = {
        "task": "prisma",
        **summary,
        "excl_screening": excl_screening,
        "excl_eligibility": excl_eligibility,
        "included_list": included,
        "input_tokens": tin, "output_tokens": tout,
        "latency_s": round(time.time() - t0, 1),
    }
    _set_last_result(result)
    return result


@app.post("/api/screen")
def api_screen(req: ScreenReq):
    """Task 1: include/exclude decision for a title+abstract (direct, zero-shot)."""
    _require_api_key()
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up; try again in a few seconds.")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    decision, r = screen(req.title, req.abstract, ic_ec=_compose_ic_ec(req.ic, req.ec),
                         model=req.model or LLM_MODEL)
    parsed = parse_json(r.text)
    result = {
        "task": "screening",
        "title": req.title,
        "decision": "include" if decision == 1 else "exclude",
        "reason": ((parsed or {}).get("reason") or "").strip(),
        "status": "ok" if parsed is not None else "parse_failed",   # transparency
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_s": round(r.latency_s, 2),
    }
    _set_last_result(result)
    return result


@app.post("/api/extract")
def api_extract(req: ExtractReq):
    """Task 2: RAG field extraction for one paper (paper-scoped reranker retrieval)."""
    _require_api_key()
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up; try again in a few seconds.")
    paper = next((p for p in STATE["papers"] if p["paper_id"] == req.paper_id), None)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {req.paper_id}")

    # fields are configurable; empty request falls back to the default SLR schema
    # so existing behaviour is unchanged but a foreign domain can supply its own fields.
    field_list = req.fields or EXTRACT_FIELDS
    hints = req.field_hints or (FIELD_HINTS if not req.fields else {})

    # Paper-scoped retrieval anchored on the title + the requested fields (so the
    # retrieval query generalizes with the schema), keeping only this paper's chunks
    # (fallback: its own text if none match).
    query = f"{paper['title']}. " + ", ".join(field_list)
    ctx_hits = _paper_chunks(req.paper_id, query, K_EXTRACT)
    if not ctx_hits:      # paper has no own chunks in the store → fall back to its own text
        pobj = next((p for p in STATE.get("corpus_papers", []) if p.paper_id == req.paper_id), None)
        if pobj:
            ctx_hits = [RetrievedChunk(chunk_id=req.paper_id, text=(pobj.text or "")[:4000],
                                       paper_id=req.paper_id, score=0.0, metadata={})]

    r, status, fields, sources, grounding, ungrounded = _extract_one(
        field_list, hints, ctx_hits, _paper_full_text(req.paper_id), model=req.model or LLM_MODEL)

    result = {
        "task": "extraction",
        "paper_id": req.paper_id,
        "title": paper["title"],
        "fields": fields,
        "sources": sources,
        "grounding": grounding,              # per-field source-grounded (True/False/None)
        "ungrounded_fields": ungrounded,     # non-null values whose source is NOT in the text
        "status": status,                    # ok | truncated | parse_failed
        "n_chunks_retrieved": len(ctx_hits),
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_s": round(r.latency_s, 2),
    }
    _set_last_result(result)
    return result


@app.post("/api/synthesize")
def api_synthesize(req: SynthReq):
    """Task 3: cross-corpus narrative synthesis for a research question (RAG-only)."""
    _require_api_key()
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up; try again in a few seconds.")
    if STATE["retriever"] is None:
        raise HTTPException(status_code=400, detail="Index a corpus first (Indexing tab).")
    rq = req.research_question.strip()
    if not rq:
        raise HTTPException(status_code=400, detail="Research question cannot be empty.")
    # Allow shortcut keys (RQ1..RQ4) or free-text questions.
    rq = RESEARCH_QUESTIONS.get(rq.upper(), rq)

    narrative, cinfo, r = synthesize(rq, STATE["retriever"], k=K_SYNTH, model=req.model or LLM_MODEL)
    cited = cinfo["cited"]                       # written AND retrieved (verified provenance)
    result = {
        "task": "synthesis",
        "research_question": rq,
        "narrative": narrative,
        "cited_papers": cited,
        "n_cited": len(cited),
        "invented_citations": cinfo["invented"],         # cited in text but never retrieved (flagged)
        "uncited_evidence": cinfo["uncited_evidence"],   # retrieved but not cited (informational)
        "references": _refs_for(cited),
        "truncated": (getattr(r, "finish_reason", "") == "length"),   # F4
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_s": round(r.latency_s, 2),
    }
    _set_last_result(result)
    return result


@app.get("/api/retrieve")
def api_retrieve(query: str, k: int = 8):
    """Raw retrieval hits for an arbitrary query (no LLM call): retrieval transparency."""
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up; try again in a few seconds.")
    if STATE["retriever"] is None:
        raise HTTPException(status_code=400, detail="Index a corpus first (Indexing tab).")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    hits = STATE["retriever"].retrieve(query, k=k)
    return {
        "query": query,
        "k": k,
        "hits": [
            {"chunk_id": h.chunk_id, "paper_id": h.paper_id,
             "score": round(h.score, 4), "text": h.text[:400]}
            for h in hits
        ],
    }


@app.post("/api/export")
def api_export(req: ExportReq):
    """Export the most recent task result as csv | json | pdf."""
    res = STATE["last_result"]
    if res is None:
        raise HTTPException(status_code=400, detail="No result to export; run a task first.")
    fmt = req.fmt.lower().strip()
    base = f"slr_result_{res.get('task', 'result')}"
    cs = res.get("checksum") or _checksum(res)      # output-integrity hash
    hcs = {"X-Content-SHA256": cs}

    if fmt == "json":
        out = dict(res); out["checksum_sha256"] = cs
        body = json.dumps(out, ensure_ascii=False, indent=2)
        return Response(body, media_type="application/json",
                        headers={**hcs, "Content-Disposition": f"attachment; filename={base}.json"})

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["key", "value"])
        for k, v in res.items():
            w.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
        w.writerow(["checksum_sha256", cs])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={**hcs, "Content-Disposition": f"attachment; filename={base}.csv"})

    if fmt == "pdf":
        return Response(_make_pdf(res), media_type="application/pdf",
                        headers={**hcs, "Content-Disposition": f"attachment; filename={base}.pdf"})

    raise HTTPException(status_code=400, detail="fmt must be one of: csv | json | pdf.")


@app.post("/api/verify")
async def api_verify(payload: dict):
    """Output-integrity verification. Recomputes the SHA-256 of the submitted result JSON
    (excluding checksum fields) and compares it to the embedded hash, confirming a downloaded output is unchanged."""
    embedded = payload.get("checksum") or payload.get("checksum_sha256")
    actual = _checksum(payload)
    return {"embedded": embedded, "actual": actual,
            "match": (embedded == actual) if embedded else None}


def _make_pdf(res: dict) -> bytes:
    """Render a result dict to a simple one-or-more-page PDF (reportlab)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"SLR Automation Result: {res.get('task', '')}")
    y -= 0.9 * cm
    c.setFont("Helvetica", 9)

    def wrap(text, n=95):
        text = str(text)
        return [text[i:i + n] for i in range(0, len(text), n)] or [""]

    for key, val in res.items():
        val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
        for line in wrap(f"{key}: {val_str}"):
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 9)
            # reportlab's base-14 Helvetica is Latin-1; drop chars it cannot encode.
            c.drawString(margin, y, line.encode("latin-1", "replace").decode("latin-1"))
            y -= 0.5 * cm

    c.showPage()
    c.save()
    return buf.getvalue()


@app.post("/api/fetch")
def api_fetch(req: FetchReq):
    """Fetch papers from arXiv (open API, no key) by query and index them.

    Pulls title + abstract for each hit and indexes them as a fresh corpus, so the
    user can build a review corpus directly from a database search.
    """
    import urllib.parse
    import urllib.request
    import urllib.error
    import feedparser  # lazy

    q = urllib.parse.quote(req.query.strip())
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    n = max(1, min(req.max, 500))
    entries = []
    batch = 100   # pages of 100 return reliably from arXiv; 200 can time out on large responses
    try:
        for start in range(0, n, batch):
            url = (f"http://export.arxiv.org/api/query?search_query=all:{q}"
                   f"&start={start}&max_results={min(batch, n - start)}&sortBy=relevance")
            hdr = {"User-Agent": "SLR-RAG-Automation/1.0 (academic research prototype)"}
            for _attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=90) as resp:
                        feed = feedparser.parse(resp.read())
                    break
                except urllib.error.HTTPError as he:
                    if he.code == 429 and _attempt < 2:
                        time.sleep(6 * (_attempt + 1))
                        continue
                    raise
            if not feed.entries:
                break
            entries.extend(feed.entries)
            if start + batch < n:
                time.sleep(3)   # arXiv rate-limit courtesy between pages
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach arXiv: {e}")

    import fitz  # PyMuPDF: PDF parsing for full text
    papers = []
    for i, e in enumerate(entries, 1):
        title = (e.get("title") or "").replace("\n", " ").strip()
        summary = (e.get("summary") or "").replace("\n", " ").strip()
        if not title:
            continue
        eid = e.get("id") or ""
        arxiv_id = eid.split("/abs/")[-1] if "/abs/" in eid else ""
        body = ""
        if req.full_text and arxiv_id:
            try:
                pdf_req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers=hdr)
                with urllib.request.urlopen(pdf_req, timeout=90) as presp:
                    doc = fitz.open(stream=presp.read(), filetype="pdf")
                    body = "\n".join(p.get_text() for p in doc)
                    doc.close()
                time.sleep(3)   # arXiv rate limit: a PDF download is also a request
            except Exception:
                body = ""   # fall back to the abstract if the PDF is unavailable
        text = body.strip() or f"{title}. {summary}"
        authors = e.get("authors") or []
        author = ", ".join(a.get("name", "") for a in authors[:3]) if authors else (e.get("author") or "")
        papers.append(Paper(
            paper_id=f"A{i:03d}", doi=eid, title=title,
            year=(e.get("published") or "")[:4], author=author, txt_path=None, text=text))
    if not papers:
        raise HTTPException(status_code=404, detail="No results from arXiv; try a different query.")
    label = "full text" if req.full_text else "abstract"
    return _reindex(papers, f"arXiv ({label}): \"{req.query}\" -> {len(papers)} papers")


def _paper_chunks(paper_id: str, query: str, k: int):
    """Paper-scoped dense retrieval: that paper's own most-relevant chunks via a
    Chroma where-filter. Replaces the global top-k + filter pattern, which fell back
    to *other* papers' chunks when a paper didn't rank in its own global top-k
    (leaving its extraction empty, e.g. P034)."""
    store, embedder = STATE.get("store"), STATE.get("embedder")
    if not store or not embedder:
        return []
    res = store.query(embedder.embed_one(query), k=k, where={"paper_id": paper_id})
    return _wrap_chroma(res)


def _paper_full_text(paper_id: str) -> str:
    """The indexed paper's full text (for source-grounding), or '' if unavailable."""
    p = next((x for x in STATE.get("corpus_papers", []) if x.paper_id == paper_id), None)
    return ((getattr(p, "text", "") or "") if p else "")


def _extract_one(field_list, hints, ctx_hits, source_text, model=LLM_MODEL):
    """Extract field_list over ctx_hits and attach value / source / grounding / status.

    Each non-null value's "source" quote is verified to be a contiguous span of
    source_text (the paper's full text when available, else the retrieved context) via
    a deterministic substring check, so "grounds every claim" is enforced on the
    product path, not merely requested in the prompt.
    extraction_status surfaces a truncated/parse_failed outcome instead of a silent
    all-null record.
    Returns (response, status, fields, sources, grounding, ungrounded_fields).
    """
    parsed, r = extract(field_list, retrieved_context=format_context(ctx_hits), field_hints=hints, model=model)
    status = extraction_status(parsed, getattr(r, "finish_reason", ""))
    tn = normalize_text(source_text or format_context(ctx_hits))
    fields, sources, grounding, ungrounded = {}, {}, {}, []
    for f in field_list:
        val = get_value(parsed, f)
        fields[f] = val
        v = parsed.get(f) if parsed else None
        src = v.get("source") if isinstance(v, dict) else None
        sources[f] = src
        if val is None or str(val).strip().lower() in ("", "null", "none"):
            grounding[f] = None                      # n/a: no value to ground
        else:
            ok = grounded(src, val, tn)
            grounding[f] = ok
            if not ok:
                ungrounded.append(f)
    return r, status, fields, sources, grounding, ungrounded


def _refs_for(paper_ids):
    """Bibliographic references {paper_id: {author, year, title, doi, url}} for the
    cited/included papers, so the UI can render clickable, traceable citations."""
    by_id = {p.paper_id: p for p in STATE.get("corpus_papers", [])}
    refs = {}
    for pid in dict.fromkeys(paper_ids):          # de-dup, preserve order
        p = by_id.get(pid)
        if not p:
            continue
        doi = (p.doi or "").strip()
        url = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
        refs[pid] = {"paper_id": pid, "title": p.title, "author": (p.author or "").strip(),
                     "year": (p.year or "").strip(), "doi": doi, "url": url}
    return refs


def _checksum(obj) -> str:
    """SHA-256 output integrity over the result's canonical JSON (excluding checksum fields).
    Identical content always yields the same hash, so a downloaded output can be re-verified via /api/verify."""
    payload = {k: v for k, v in obj.items() if k not in ("checksum", "checksum_sha256")}
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _persist_session():
    """Snapshot the expensive session state (PRISMA included-set + last result) to disk so
    it survives a restart. Best-effort: any failure is logged and ignored."""
    try:
        pr = STATE.get("prisma")
        prisma_snap = None
        if pr and pr.get("included_papers"):
            prisma_snap = {"summary": pr.get("summary"),
                           "included_ids": [p.paper_id for p in pr["included_papers"]],
                           "ic": pr.get("ic", ""), "ec": pr.get("ec", "")}
        snap = {"corpus_source": STATE.get("corpus_source", ""),
                "prisma": prisma_snap, "last_result": STATE.get("last_result")}
        SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STATE_PATH.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not persist session state: %s", e)


def _restore_session():
    """Restore last_result + the PRISMA included-set after a restart, re-resolving included
    papers from the freshly loaded corpus by id. Guarded: a missing, corrupt, or
    corpus-mismatched state file is ignored so startup never breaks."""
    try:
        if not SESSION_STATE_PATH.exists():
            return
        snap = json.loads(SESSION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read session state: %s", e)
        return
    if snap.get("corpus_source") and snap["corpus_source"] != STATE.get("corpus_source"):
        return                                       # different corpus -> cannot be re-resolved, skip
    if snap.get("last_result") is not None:
        STATE["last_result"] = snap["last_result"]
    pr = snap.get("prisma")
    if pr and pr.get("included_ids"):
        by_id = {p.paper_id: p for p in STATE.get("corpus_papers", [])}
        included = [by_id[i] for i in pr["included_ids"] if i in by_id]
        if included:
            STATE["prisma"] = {"summary": pr.get("summary"), "included_papers": included,
                               "ic": pr.get("ic", ""), "ec": pr.get("ec", "")}
            logger.info("Session restored: %d included papers.", len(included))


def _set_last_result(result):
    """Set the last result and persist the session (so a restart keeps it)."""
    STATE["last_result"] = result
    _persist_session()


@app.post("/api/report")
def api_report(req: ReportReq):
    """Integrated SLR report: the system's final output.

    End-to-end in one call: determine the included papers via PRISMA (two-stage
    screening), extract each paper's classification data with RAG, produce
    cross-corpus synthesis for the research questions, and merge everything into a
    single report.
    """
    _require_api_key()
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="The system is still warming up.")
    pr = STATE.get("prisma")
    if not pr or not pr.get("included_papers"):
        raise HTTPException(status_code=400,
            detail="Run the PRISMA screening step first; the report is built from the papers it includes.")
    included = pr["included_papers"]          # output of the PRISMA step; not re-screened
    summary = pr["summary"]
    t0 = time.time(); tin = tout = 0

    # configurable fields (empty -> default SLR schema, behaviour unchanged).
    field_list = req.fields or EXTRACT_FIELDS
    hints = req.field_hints or (FIELD_HINTS if not req.fields else {})
    model = req.model or LLM_MODEL

    # 1) Data extraction, for each paper PRISMA included (source + grounding).
    extractions = []
    for p in included:
        query = f"{p.title}. " + ", ".join(field_list)
        ph = _paper_chunks(p.paper_id, query, K_EXTRACT)
        if not ph:        # paper has no own chunks in the store → use its own text
            ph = [RetrievedChunk(chunk_id=p.paper_id, text=(p.text or "")[:4000],
                                 paper_id=p.paper_id, score=0.0, metadata={})]
        try:
            r, status, fields, srcs, grounding, ungrounded = _extract_one(
                field_list, hints, ph, (p.text or ""), model=model)
            tin += r.input_tokens; tout += r.output_tokens
        except Exception as e:                              # one paper failing must not abort the report
            extractions.append({"paper_id": p.paper_id, "title": p.title,
                                "fields": {f: None for f in field_list}, "sources": {},
                                "grounding": {}, "ungrounded_fields": [], "status": "error",
                                "error": e.__class__.__name__})
            continue
        extractions.append({"paper_id": p.paper_id, "title": p.title,
                            "fields": fields, "sources": srcs,
                            "grounding": grounding, "ungrounded_fields": ungrounded,
                            "status": status})

    # 2) Synthesis: research questions, only over the included papers.
    rqs = req.rqs or ["RQ1", "RQ2", "RQ3", "RQ4"]
    included_ids = {p.paper_id for p in included}
    syntheses = []
    for rq in rqs:
        q = RESEARCH_QUESTIONS.get(rq.upper(), rq)
        invented = []
        try:
            narrative, cinfo, r = synthesize(q, STATE["retriever"], k=K_SYNTH, allowed_ids=included_ids, model=model)
            cited, invented = cinfo["cited"], cinfo["invented"]    # verified provenance + flags
            tin += r.input_tokens; tout += r.output_tokens
        except Exception as e:                              # one RQ failing must not stop the others
            narrative, cited = f"(synthesis failed: {e.__class__.__name__})", []
        syntheses.append({"rq": rq, "question": q, "narrative": narrative,
                          "cited": cited, "invented_citations": invented})

    ref_ids = [p.paper_id for p in included]
    for s in syntheses:
        ref_ids.extend(s["cited"])
    result = {
        "task": "report",
        "corpus_source": STATE["corpus_source"],
        "prisma": summary,                    # summary produced by the PRISMA step, not re-screened
        "included_list": [{"paper_id": p.paper_id, "title": p.title} for p in included],
        "extractions": extractions,
        "syntheses": syntheses,
        "references": _refs_for(ref_ids),
        "input_tokens": tin, "output_tokens": tout,
        "latency_s": round(time.time() - t0, 1),
    }
    result["checksum"] = _checksum(result)          # output-integrity checksum
    _set_last_result(result)
    return result


# Static assets (after API routes so they don't shadow them).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
