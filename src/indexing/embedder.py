"""Embedding wrapper: API-based (OpenRouter) with a local fallback.

E5 convention (intfloat models):
  documents → prefix "passage: "
  queries   → prefix "query: "

By default vectors come from OpenRouter's OpenAI-compatible /embeddings endpoint
and are L2-normalized client-side so inner product == cosine (matching the Chroma
collection's cosine space). If no API key is set, or the API call fails, the
embedder falls back to a LOCAL sentence-transformers model of the SAME name
(e.g. intfloat/e5-base-v2): same weights, same 768-dim space, so a locally-built
or API-built index stays compatible. This removes the single point of failure on
OpenRouter and lets indexing/retrieval run fully offline.
"""

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EMBEDDING_MODEL, EMBEDDING_API_URL

logger = logging.getLogger("slr_rag.embed")

BATCH_SIZE = 96
MAX_RETRIES = 4
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 60

# E5-family convention: prefix documents with "passage: " and queries with "query: ".
# Other API encoders (perplexity/pplx-embed, openai/text-embedding-*, voyage-*) are
# trained without prefixes: sending them confuses the encoder.
_PREFIX_PATTERNS = ("intfloat/e5-", "intfloat/multilingual-e5-")
DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_LOCAL_MODELS = {}        # model_name -> SentenceTransformer (lazy, cached)


def _needs_prefix(model_name: str) -> bool:
    return any(p in model_name for p in _PREFIX_PATTERNS)


def _resolve_backend(api_key: str, prefer: str = "auto") -> str:
    """Pick the embedding backend: 'api' when a key is present, else 'local'.

    prefer='api'/'local' forces a backend; 'auto' (default) decides by key presence.
    """
    if prefer in ("api", "local"):
        return prefer
    return "api" if api_key else "local"


def _normalize(vec):
    s = sum(v * v for v in vec) ** 0.5
    return [v / s for v in vec] if s > 0 else vec


def _embed_batch(texts, model, api_key):
    import requests
    payload = {"model": model, "input": texts, "encoding_format": "float"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(EMBEDDING_API_URL, json=payload,
                                 headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_err = f"request error: {e}"
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue
        if resp.status_code == 429:
            last_err = "429 rate limited"
            time.sleep(RETRY_DELAY * (attempt + 1) * 2)
            continue
        if resp.status_code >= 500:
            last_err = f"{resp.status_code} server error"
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"embeddings API {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return [_normalize(d["embedding"]) for d in data["data"]]
    raise RuntimeError(f"embeddings API failed after {MAX_RETRIES} attempts ({last_err}).")


def _local_encode(prepared_texts, model_name):
    """Encode already-prefixed texts with a local sentence-transformers model.

    The model is loaded once per name and cached. normalize_embeddings=True applies
    L2 normalization, matching the API path's _normalize so cosine stays consistent.
    """
    from sentence_transformers import SentenceTransformer
    model = _LOCAL_MODELS.get(model_name)
    if model is None:
        logger.info("Loading local embedding model %s (first use; downloads weights once).", model_name)
        model = SentenceTransformer(model_name)
        _LOCAL_MODELS[model_name] = model
    vecs = model.encode(list(prepared_texts), normalize_embeddings=True)
    return [[float(x) for x in v] for v in vecs]


class Embedder:
    """Embeddings client: OpenRouter API with automatic local fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, api_key: str = None,
                 backend: str = "auto"):
        self.model = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.use_prefix = _needs_prefix(model_name)
        self.backend = _resolve_backend(self.api_key, prefer=backend)

    def _embed_prepared(self, prepared, batch_size=BATCH_SIZE):
        """Embed already-prefixed texts via the API, falling back to local on failure."""
        if self.backend == "api":
            try:
                out = []
                for i in range(0, len(prepared), batch_size):
                    out.extend(_embed_batch(prepared[i:i + batch_size], self.model, self.api_key))
                    logger.debug("embedded %d/%d", min(i + batch_size, len(prepared)), len(prepared))
                return out
            except RuntimeError as e:
                logger.warning("Embedding API failed (%s); falling back to local model %s.",
                               e, self.model)
                self.backend = "local"
        return _local_encode(prepared, self.model)

    def embed(self, texts, batch_size: int = BATCH_SIZE, show_progress_bar: bool = False):
        """Documents: prefix only for E5-family encoders."""
        prepared = [DOC_PREFIX + t for t in texts] if self.use_prefix else list(texts)
        return self._embed_prepared(prepared, batch_size=batch_size)

    def embed_query(self, text: str):
        """Queries: prefix only for E5-family encoders."""
        q = QUERY_PREFIX + text if self.use_prefix else text
        return self._embed_prepared([q])[0]

    # Legacy alias: callers historically use embed_one() for queries.
    embed_one = embed_query
