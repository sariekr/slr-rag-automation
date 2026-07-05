"""
Central configuration for the SLR-RAG backend.

Corpus source is resolved automatically: if an external baseline corpus is present
at ../slr/data/ it is read read-only; otherwise the pipeline runs standalone on the
bundled, license-clean sample corpus in data/sample/. The package writes only into
rag_pipeline/data/ and rag_pipeline/experiments/.
"""

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
SLR_DATA = PROJECT_ROOT / "slr" / "data"        # optional external baseline corpus (read-only if present)
SAMPLE_DATA = BASE_DIR / "data" / "sample"      # bundled sample corpus (always available)

# Corpus + optional ground truth. Falls back to the bundled sample when no baseline
# corpus is mounted, so a fresh clone can run the full pipeline out of the box.
if SLR_DATA.exists():
    CORPUS_SOURCE = "baseline"
    PAPER_LIST = SLR_DATA / "extraction" / "paper_list.json"
    EXTRACTED_DATA = SLR_DATA / "extraction" / "extracted_data.json"  # extraction reference, if present
    PDFS_TXT_DIR = SLR_DATA / "pdfs_txt"
    SCREENED_DIR = SLR_DATA / "screened"
    STATS_JSON = SLR_DATA / "stats.json"
else:
    CORPUS_SOURCE = "sample"
    PAPER_LIST = SAMPLE_DATA / "paper_list.json"
    EXTRACTED_DATA = SAMPLE_DATA / "extracted_data.json"   # not shipped; evaluation scripts guard on it
    PDFS_TXT_DIR = SAMPLE_DATA / "txt"
    SCREENED_DIR = SAMPLE_DATA
    STATS_JSON = SAMPLE_DATA / "stats.json"

# Outputs (this package writes only here)
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"                     # persistent vector store
RESULTS_DIR = BASE_DIR / "experiments" / "results"

# ── Embedding model ──────────────────────────────────────────────────────────
# E5-Base-v2 (intfloat), 768-d. Runs locally via sentence-transformers, or through
# the OpenRouter embeddings endpoint when an API key is set. E5 prefixes the input
# with "passage: " / "query: ".
EMBEDDING_MODEL = "intfloat/e5-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_API_URL = "https://openrouter.ai/api/v1/embeddings"

# ── Chunking (paragraph-aware, overlapping) ──────────────────────────────────
CHUNK_TARGET_WORDS = 350        # ~512 tokens, the cross-encoder input limit
CHUNK_OVERLAP_WORDS = 50

# ── Vector store / retrieval ─────────────────────────────────────────────────
CHROMA_COLLECTION = "slr_corpus"
DISTANCE_METRIC = "cosine"     # Chroma hnsw:space
TOP_K = 10                     # candidates retrieved
RERANK_TOP_N = 50              # candidates passed to the cross-encoder
RETRIEVAL_CONFIGS = ("dense", "hybrid", "reranker")   # three retrieval strategies
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60                     # reciprocal rank fusion constant

# ── LLM (OpenRouter) ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.5-flash-lite"   # OpenRouter slug; override per request
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0          # deterministic for reproducibility

# ── Task configuration ───────────────────────────────────────────────────────
TASKS = ("screening", "extraction", "synthesis")
