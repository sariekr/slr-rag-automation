# Sample corpus

A small, redistributable corpus of **7 open-access arXiv papers** (all **CC BY 4.0**) so the
pipeline can be run end-to-end (index → screen → extract → synthesize) on a fresh clone without
the full private baseline corpus. Each `txt/<arxiv_id>.txt` file contains that paper's title and
abstract; `paper_list.json` holds the metadata the loader reads.

This sample is meant to **demonstrate that the software runs**, not to reproduce any evaluation
numbers (those require a full corpus and are out of scope for this software artifact).

## Attribution (CC BY 4.0)

Each paper below is © its authors, licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Text is reproduced (title + abstract) unmodified under that license.

| ID | arXiv | License | Authors | Title |
|----|-------|---------|---------|-------|
| P001 | [2606.13550](https://arxiv.org/abs/2606.13550) | CC BY 4.0 | Jung & Wang | Uncertainty-Aware Hybrid Retrieval for Long-Document RAG |
| P002 | [2606.13249](https://arxiv.org/abs/2606.13249) | CC BY 4.0 | Kim & Kim | Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cause Analysis |
| P003 | [2605.01664](https://arxiv.org/abs/2605.01664) | CC BY 4.0 | Irany & Akwafuo | A Hybrid Retrieval and Reranking Framework for Evidence-Grounded RAG |
| P004 | [2606.10381](https://arxiv.org/abs/2606.10381) | CC BY 4.0 | Jiang et al. | Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis |
| P005 | [2606.01240](https://arxiv.org/abs/2606.01240) | CC BY 4.0 | Puspitasari et al. | Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking |
| P006 | [2605.00529](https://arxiv.org/abs/2605.00529) | CC BY 4.0 | Zhao & Yang | Hierarchical Abstract Tree for Cross-Document RAG |
| P007 | [2504.14891](https://arxiv.org/abs/2504.14891) | CC BY 4.0 | Gan et al. | Retrieval Augmented Generation Evaluation in the Era of LLMs: A Comprehensive Survey |

Papers P001–P006 present hybrid-retrieval RAG *systems*; P007 is a *survey* (a natural
exclude example when screening against the default "hybrid RAG system" criteria). The license of
each was verified on its arXiv abstract page on 2026-07-05.
