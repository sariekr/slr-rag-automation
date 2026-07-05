"""
Chroma persistent vector store wrapper.

Uses a cosine-space HNSW collection. Metadata values must be scalars
(str/int/float/bool), so chunk metadata is kept flat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CHROMA_DIR, CHROMA_COLLECTION, DISTANCE_METRIC


class ChromaStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, persist_dir=CHROMA_DIR, collection: str = CHROMA_COLLECTION):
        import chromadb
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )

    def add(self, ids, embeddings, documents, metadatas, batch_size: int = 512):
        """Add chunks in batches. embeddings: iterable of float vectors."""
        emb = [list(map(float, e)) for e in embeddings]
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=emb[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

    def query(self, query_embedding, k: int = 10, where=None):
        """Return Chroma's native query result dict for one query vector.

        where: optional metadata filter, e.g. {"paper_id": "P034"}, to scope retrieval
        to a single paper (used by per-paper extraction so it never falls back to
        another paper's chunks).
        """
        kwargs = {"query_embeddings": [list(map(float, query_embedding))], "n_results": k}
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop the collection (used with build_index.py --reset)."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
