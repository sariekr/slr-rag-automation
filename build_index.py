"""
Build the Chroma index from the configured corpus.

Pipeline stages 1-3: corpus -> chunk -> embed -> index. The corpus is resolved
through config: the external baseline corpus when mounted, otherwise the bundled
sample corpus. Reads via config paths; writes to the configured Chroma directory.

Usage:
    python3 build_index.py            # index the whole corpus
    python3 build_index.py --limit 5  # quick test on first 5 papers
    python3 build_index.py --reset    # drop and rebuild the collection
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.indexing.corpus import load_corpus
from src.indexing.chunker import chunk_text
from src.indexing.embedder import Embedder
from src.indexing.vector_store import ChromaStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Chroma index from SLR corpus.")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N papers")
    ap.add_argument("--reset", action="store_true", help="drop the collection before indexing")
    args = ap.parse_args()

    print("=" * 60)
    print("RAG-LLM SLR: Index builder (stages 1-3)")
    print("=" * 60)

    # 1. Corpus
    print("\n1. Loading corpus...")
    papers = load_corpus(limit=args.limit)
    print(f"   {len(papers)} papers loaded")
    if not papers:
        sys.exit("No papers found. Check config.PAPER_LIST and config.PDFS_TXT_DIR.")

    # 2. Chunk
    print("\n2. Chunking (paragraph-aware, overlapping)...")
    all_chunks = []
    for p in papers:
        meta = {
            "paper_id": p.paper_id, "doi": p.doi, "title": p.title,
            "year": p.year, "author": p.author,
        }
        all_chunks.extend(chunk_text(p.text, p.paper_id, metadata=meta))
    per_paper = len(all_chunks) / max(len(papers), 1)
    print(f"   {len(all_chunks)} chunks ({per_paper:.1f} per paper)")

    # 3. Embed
    from config import EMBEDDING_MODEL
    print(f"\n3. Embedding ({EMBEDDING_MODEL})...")
    t0 = time.time()
    embedder = Embedder()
    embeddings = embedder.embed([c.text for c in all_chunks])
    dt = time.time() - t0
    ms_per_chunk = dt / max(len(all_chunks), 1) * 1000
    print(f"   done in {dt:.1f}s ({ms_per_chunk:.1f} ms/chunk)")

    # 4. Index
    print("\n4. Indexing into Chroma...")
    store = ChromaStore()
    if args.reset:
        store.reset()
        store = ChromaStore()
    store.add(
        ids=[c.id for c in all_chunks],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[c.metadata for c in all_chunks],
    )
    print(f"   collection now holds {store.count()} vectors")
    print("\nDone.")


if __name__ == "__main__":
    main()
