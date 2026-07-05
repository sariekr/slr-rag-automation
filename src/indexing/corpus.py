"""
Load the SLR corpus (101 final papers) with metadata.

Reads the read-only ground-truth mapping at slr/data/extraction/paper_list.json
and the matching full-text files in slr/data/pdfs_txt/. Never writes to slr/.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PAPER_LIST, PDFS_TXT_DIR


@dataclass
class Paper:
    """One corpus paper with its metadata and full text."""
    paper_id: str
    doi: str
    title: str
    year: str
    author: str
    txt_path: Path
    text: str


def load_corpus(limit: int | None = None) -> list[Paper]:
    """Load papers listed in paper_list.json whose text file exists and is non-empty.

    Args:
        limit: if set, return at most this many papers (for quick test runs).
    """
    records = json.loads(Path(PAPER_LIST).read_text(encoding="utf-8"))
    papers: list[Paper] = []
    missing = 0
    for rec in records:
        txt_path = Path(PDFS_TXT_DIR) / rec.get("txt_file", "")
        if not txt_path.exists():
            missing += 1
            continue
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        papers.append(Paper(
            paper_id=rec.get("paper_id", ""),
            doi=rec.get("doi", ""),
            title=rec.get("title", ""),
            year=rec.get("year", ""),
            author=rec.get("author", ""),
            txt_path=txt_path,
            text=text,
        ))
        if limit and len(papers) >= limit:
            break
    if missing:
        print(f"  (warning) {missing} papers skipped: text file not found")
    return papers
