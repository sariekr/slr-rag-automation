"""
Paragraph-aware, overlapping text chunker.

Paragraph-based chunking with word overlap. Chunk size is expressed in words (approximating
tokens); the 256/512/1024-token sweep maps to CHUNK_TARGET_WORDS in config.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CHUNK_TARGET_WORDS, CHUNK_OVERLAP_WORDS


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    id: str
    text: str
    paper_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    paper_id: str,
    metadata: dict | None = None,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split text into overlapping, paragraph-aware chunks of ~target_words.

    Paragraphs (blank-line separated) are accumulated until the target size is
    reached. Oversized paragraphs are windowed. Each new chunk carries the last
    overlap_words of the previous one to preserve boundary context.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    idx = 0

    def emit(words: list[str]) -> None:
        nonlocal idx
        chunk_str = " ".join(words).strip()
        if not chunk_str:
            return
        chunks.append(Chunk(
            id=f"{paper_id}:{idx}",
            text=chunk_str,
            paper_id=paper_id,
            chunk_index=idx,
            metadata={**metadata, "paper_id": paper_id, "chunk_index": idx},
        ))
        idx += 1

    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(buf) + len(words) <= target_words:
            buf.extend(words)
            continue
        # current buffer is full: emit it, then carry overlap forward
        if buf:
            emit(buf)
            buf = buf[-overlap_words:] if 0 < overlap_words < len(buf) else []
        buf.extend(words)
        # window any paragraph larger than the target
        while len(buf) > target_words:
            emit(buf[:target_words])
            step = max(target_words - overlap_words, 1)
            buf = buf[step:]

    if buf:
        emit(buf)
    return chunks
