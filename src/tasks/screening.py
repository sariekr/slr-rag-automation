"""
Task 1: Screening (include / exclude).

Pool: slr/data/screened title/abstract CSVs; gold = the human PRISMA decision
(after_title_abstract = include, excluded_title_abstract = exclude).
Arms:
  - direct: zero-shot IC/EC over title+abstract (examples=None).
  - rag:    + k nearest already-decided papers as few-shot examples.
Decisions are binary (1=include, 0=exclude) for precision / recall / F1.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SCREENED_DIR, LLM_MODEL
from src.llm import call_llm, SYSTEM_ROLE, screening_prompt
from src.llm.parsing import parse_json

IC_EC = (
    "INCLUDE if the paper: (IC1) presents a Retrieval-Augmented Generation (RAG) system; "
    "(IC2) employs a HYBRID approach (e.g. dense+sparse retrieval, reranking, graph+vector, "
    "multi-stage) as a core contribution; (IC3) is in English; (IC4) published 2020-2026; "
    "(IC5) peer-reviewed or a full arXiv paper.\n"
    "EXCLUDE if the paper: (EC1) uses traditional information retrieval only (no generation); "
    "(EC2) mentions RAG only in related work/background; (EC3) is not a full research paper "
    "(poster, extended abstract, tutorial); (EC4) is a survey or review; (EC5) where RAG or the "
    "hybrid method is not the main contribution."
)


def load_pool(screened_dir=SCREENED_DIR):
    """Return [{title, abstract, gold}] from the PRISMA title/abstract CSVs (1=include, 0=exclude)."""
    rows = []
    for fn, gold in [("after_title_abstract.csv", 1), ("excluded_title_abstract.csv", 0)]:
        path = Path(screened_dir) / fn
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            for r in csv.DictReader(f):
                title = (r.get("Title") or "").strip()
                abstract = (r.get("Abstract Note") or "").strip()
                if title and abstract:
                    rows.append({"title": title, "abstract": abstract, "gold": gold})
    return rows


def nearest_examples(test_emb, bank, bank_emb, k=3, balanced=False):
    """Select k few-shot examples (with decisions) from a decided bank by abstract similarity.

    bank: [{title, abstract, gold}]; bank_emb: (n, dim) normalized; test_emb: 1-D normalized.
    balanced=True -> k//2 nearest include + k//2 nearest exclude (contrastive boundary).
    """
    import numpy as np
    sims = np.asarray(bank_emb) @ np.asarray(test_emb)

    def ex(i):
        return {"title": bank[i]["title"], "abstract": bank[i]["abstract"][:400],
                "decision": "include" if bank[i]["gold"] == 1 else "exclude"}

    if balanced:
        h = max(k // 2, 1)
        inc = [i for i, b in enumerate(bank) if b["gold"] == 1]
        exc = [i for i, b in enumerate(bank) if b["gold"] == 0]
        idx = sorted(inc, key=lambda i: -sims[i])[:h] + sorted(exc, key=lambda i: -sims[i])[:h]
    else:
        idx = sorted(range(len(bank)), key=lambda i: -sims[i])[:k]
    return [ex(i) for i in idx]


def screen(title, abstract, ic_ec=IC_EC, examples=None, model=LLM_MODEL):
    """One screening decision. examples=None -> direct (zero-shot); list -> RAG few-shot.

    Returns (decision, response): decision is 1 (include) / 0 (exclude).
    """
    r = call_llm(screening_prompt(title, abstract, ic_ec, examples=examples),
                 system=SYSTEM_ROLE, model=model)
    obj = parse_json(r.text) or {}
    decision = 1 if str(obj.get("decision", "")).lower().startswith("incl") else 0
    return decision, r
