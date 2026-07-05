"""
Retrieval layer: dense / hybrid / reranker.

Three strategies behind one Retriever.retrieve() interface, so the task modules
(screening / extraction / synthesis) can swap configs (config.RETRIEVAL_CONFIGS)
without changing their logic.

Adapted from the SE experiments (slr/experiments/pipelines/), which implement the
same three strategies over a code corpus with FAISS. Here the dense index already
lives in the persistent Chroma store built by build_index.py, so the dense arm
queries Chroma instead of building an in-memory FAISS index; BM25 is rebuilt in
memory over the same chunks at construction time.

  - p1_naive_rag.py    -> DenseRetriever     (Chroma cosine top-k)
  - p2_hybrid_rag.py   -> HybridRetriever    (dense + BM25, RRF fusion)
  - p3_reranker_rag.py -> RerankerRetriever   (hybrid candidates + cross-encoder)

Heavy dependencies (rank_bm25, sentence-transformers CrossEncoder) are imported
lazily inside the classes that need them, matching the indexing layer: the dense
arm runs without rank_bm25, and the module imports cleanly before `pip install`.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TOP_K, RERANK_TOP_N, RRF_K, RERANKER_MODEL, RETRIEVAL_CONFIGS


@dataclass
class RetrievedChunk:
    """A retrieval hit with score and provenance."""
    chunk_id: str
    text: str
    paper_id: str
    score: float
    metadata: dict


def _tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens for BM25.

    The corpus here is academic prose, so this keeps digits (years, "f1", "95")
    and drops punctuation, unlike the identifier-tuned tokenizer in
    slr/experiments (which kept underscores for source code).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _rrf(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion contribution for a 0-based rank: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def _wrap_chroma(res: dict) -> list[RetrievedChunk]:
    """Turn a single-query Chroma result into a RetrievedChunk list.

    Chroma's cosine space returns distance = 1 - cosine_similarity, so the
    similarity score reported here is 1 - distance.
    """
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        hits.append(RetrievedChunk(
            chunk_id=cid,
            text=doc,
            paper_id=meta.get("paper_id", ""),
            score=1.0 - float(dist),
            metadata=meta,
        ))
    return hits


class Retriever:
    """Common interface for all retrieval strategies."""

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    """Dense-only: embed the query, cosine top-k from Chroma."""

    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        emb = self.embedder.embed_one(query)
        res = self.store.query(emb, k=k)
        return _wrap_chroma(res)


class HybridRetriever(Retriever):
    """Dense (Chroma) + sparse (BM25) fused with Reciprocal Rank Fusion.

    corpus_chunks must be the same chunks held in the Chroma collection
    (load_corpus -> chunk_text). They back both the BM25 index and the
    text/metadata lookup for fused hits, keyed by Chunk.id.
    """

    def __init__(self, store, embedder, corpus_chunks):
        from rank_bm25 import BM25Okapi
        self.store = store
        self.embedder = embedder
        self.chunks = list(corpus_chunks)
        self._by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def _dense_ranks(self, query: str, n: int) -> dict[str, int]:
        res = self.store.query(self.embedder.embed_one(query), k=n)
        ids = (res.get("ids") or [[]])[0]
        return {cid: rank for rank, cid in enumerate(ids)}

    def _sparse_ranks(self, query: str, n: int) -> dict[str, int]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return {self.chunks[i].id: rank for rank, i in enumerate(order)}

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        n = max(k * 5, RERANK_TOP_N)          # over-retrieve from both arms, then fuse
        dense = self._dense_ranks(query, n)
        sparse = self._sparse_ranks(query, n)

        fused: dict[str, float] = {}
        for cid in set(dense) | set(sparse):
            if cid not in self._by_id:        # keep fusion aligned to known chunks
                continue
            score = 0.0
            if cid in dense:
                score += _rrf(dense[cid])
            if cid in sparse:
                score += _rrf(sparse[cid])
            fused[cid] = score

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [self._as_hit(cid, score) for cid, score in ranked]

    def _as_hit(self, cid: str, score: float) -> RetrievedChunk:
        c = self._by_id[cid]
        return RetrievedChunk(
            chunk_id=c.id, text=c.text, paper_id=c.paper_id,
            score=float(score), metadata=c.metadata,
        )


class RerankerRetriever(Retriever):
    """Hybrid candidates (top RERANK_TOP_N) reranked by a cross-encoder."""

    def __init__(self, hybrid: HybridRetriever, reranker_model: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        self.hybrid = hybrid
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        candidates = self.hybrid.retrieve(query, k=RERANK_TOP_N)
        if not candidates:
            return []
        scores = self.reranker.predict(
            [(query, c.text) for c in candidates],
            show_progress_bar=False,
        )
        for c, s in zip(candidates, scores):
            c.score = float(s)
        candidates.sort(key=lambda c: -c.score)
        return candidates[:k]


def build_retriever(config: str, *, store=None, embedder=None,
                    corpus_chunks=None, reranker_model: str = RERANKER_MODEL) -> Retriever:
    """Factory: 'dense' | 'hybrid' | 'reranker' (see config.RETRIEVAL_CONFIGS).

    - dense    needs store + embedder
    - hybrid   needs store + embedder + corpus_chunks
    - reranker needs the hybrid inputs, then wraps them in a cross-encoder
    """
    if config == "dense":
        return DenseRetriever(store, embedder)
    if config == "hybrid":
        return HybridRetriever(store, embedder, corpus_chunks)
    if config == "reranker":
        hybrid = HybridRetriever(store, embedder, corpus_chunks)
        return RerankerRetriever(hybrid, reranker_model)
    raise ValueError(
        f"unknown retrieval config {config!r}; expected one of {RETRIEVAL_CONFIGS}"
    )
