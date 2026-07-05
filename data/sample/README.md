# Sample corpus

A small, redistributable corpus of **32 open-access arXiv papers** (all under a
Creative Commons redistributable licence) so the pipeline can be run end to end
(index -> screen -> extract -> synthesize) on a fresh clone without the full private baseline
corpus. Each `txt/<arxiv_id>.txt` file contains that paper's title and abstract; `paper_list.json`
holds the metadata the loader reads.

The set mixes hybrid-retrieval RAG *systems* (natural includes under the default "hybrid RAG system"
screening criteria) with surveys, benchmarks, evaluation frameworks, and retrieval-only papers
(natural excludes), so the screening stage produces both decisions. This sample is meant to
**demonstrate that the software runs**, not to reproduce any evaluation numbers.

## Attribution (Creative Commons)

Each paper below is © its authors under the stated Creative Commons licence
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless noted). Titles and abstracts are
reproduced unmodified under those licences. Every licence was verified on the paper's arXiv
abstract page on 2026-07-05.

| ID | arXiv | Licence | Authors | Title |
|----|-------|---------|---------|-------|
| P001 | [2606.13550](https://arxiv.org/abs/2606.13550) | CC BY 4.0 | Jung & Wang | Uncertainty-Aware Hybrid Retrieval for Long-Document RAG |
| P002 | [2606.13249](https://arxiv.org/abs/2606.13249) | CC BY 4.0 | Kim & Kim | Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cau |
| P003 | [2605.01664](https://arxiv.org/abs/2605.01664) | CC BY 4.0 | Irany & Akwafuo | A Hybrid Retrieval and Reranking Framework for Evidence-Grounded Retrieval-Augme |
| P004 | [2606.10381](https://arxiv.org/abs/2606.10381) | CC BY 4.0 | Jiang et al. | Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis |
| P005 | [2606.01240](https://arxiv.org/abs/2606.01240) | CC BY 4.0 | Puspitasari et al. | Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking |
| P006 | [2605.00529](https://arxiv.org/abs/2605.00529) | CC BY 4.0 | Zhao & Yang | Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation |
| P007 | [2504.14891](https://arxiv.org/abs/2504.14891) | CC BY 4.0 | Gan et al. | Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A |
| P008 | [2309.15217](https://arxiv.org/abs/2309.15217) | CC BY 4.0 | Es et al. | Ragas: Automated Evaluation of Retrieval Augmented Generation |
| P009 | [2506.06962](https://arxiv.org/abs/2506.06962) | CC BY 4.0 | Qi et al. | AR-RAG: Autoregressive Retrieval Augmentation for Image Generation |
| P010 | [2402.12317](https://arxiv.org/abs/2402.12317) | CC BY 4.0 | Su et al. | EVOR: Evolving Retrieval for Code Generation |
| P011 | [2601.05264](https://arxiv.org/abs/2601.05264) | CC BY 4.0 | Wampler et al. | Engineering the RAG Stack: A Comprehensive Review of the Architecture and Trust  |
| P012 | [2411.18583](https://arxiv.org/abs/2411.18583) | CC BY 4.0 | Ali et al. | Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augment |
| P013 | [2507.23334](https://arxiv.org/abs/2507.23334) | CC BY 4.0 | Kwon et al. | MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation |
| P014 | [2601.19535](https://arxiv.org/abs/2601.19535) | CC BY 4.0 | Chandra et al. | LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG |
| P015 | [2401.15391](https://arxiv.org/abs/2401.15391) | CC BY-SA 4.0 | Tang & Yang | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries |
| P016 | [2606.30093](https://arxiv.org/abs/2606.30093) | CC BY 4.0 | Bonifazi et al. | Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs |
| P017 | [2602.20735](https://arxiv.org/abs/2602.20735) | CC BY 4.0 | Ran et al. | RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition |
| P018 | [2410.20724](https://arxiv.org/abs/2410.20724) | CC BY 4.0 | Li et al. | Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge- |
| P019 | [2606.21553](https://arxiv.org/abs/2606.21553) | CC BY 4.0 | Shaikh | Dissecting Agentic RAG: A Component Ablation for Multi-Hop QA with a Local 7B Mo |
| P020 | [2606.29328](https://arxiv.org/abs/2606.29328) | CC BY 4.0 | Zhang et al. | Covering the Unseen: Information Demand Coverage Optimization for Retrieval-Augm |
| P021 | [2607.00052](https://arxiv.org/abs/2607.00052) | CC BY 4.0 | Huu & Hashimoto | AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generatio |
| P022 | [2606.16409](https://arxiv.org/abs/2606.16409) | CC BY 4.0 | Wang et al. | PathRouter: Aligning Rewards with Retrieval Quality in Agentic Graph Retrieval-A |
| P023 | [2606.26458](https://arxiv.org/abs/2606.26458) | CC BY 4.0 | Wang et al. | MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph-Augmented Ge |
| P024 | [2504.05181](https://arxiv.org/abs/2504.05181) | CC BY 4.0 | Mekonnen et al. | Lightweight and Direct Document Relevance Optimization for Generative Informatio |
| P025 | [2108.06279](https://arxiv.org/abs/2108.06279) | CC BY 4.0 | Macdonald et al. | On Single and Multiple Representations in Dense Passage Retrieval |
| P026 | [2403.13468](https://arxiv.org/abs/2403.13468) | CC BY 4.0 | Kasela et al. | DESIRE-ME: Domain-Enhanced Supervised Information REtrieval using Mixture-of-Exp |
| P027 | [2406.18960](https://arxiv.org/abs/2406.18960) | CC BY 4.0 | Kostric & Balog | A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversatio |
| P028 | [2301.05508](https://arxiv.org/abs/2301.05508) | CC BY 4.0 | Penha & Hauff | Do the Findings of Document and Passage Retrieval Generalize to the Retrieval of |
| P029 | [2504.17137](https://arxiv.org/abs/2504.17137) | CC BY 4.0 | Park et al. | MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluati |
| P030 | [2210.09877](https://arxiv.org/abs/2210.09877) | CC BY 4.0 | Ahmed & Bulathwela | Towards Proactive Information Retrieval in Noisy Text with Wikipedia Concepts |
| P031 | [2412.16708](https://arxiv.org/abs/2412.16708) | CC BY 4.0 | Su et al. | Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversa |
| P032 | [2505.04680](https://arxiv.org/abs/2505.04680) | CC BY 4.0 | Ceresa et al. | Retrieval Augmented Generation Evaluation for Health Documents |
