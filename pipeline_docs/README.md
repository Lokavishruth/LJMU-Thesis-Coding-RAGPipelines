# RAG Pipeline Documentation — LJMU Thesis
## HotPotQA Multi-Hop Question Answering

This folder contains detailed thesis documentation for each of the 6 RAG pipeline architectures evaluated in this study. All pipelines use the **HotPotQA `distractor` validation split** (500 samples, seed=42), the **Groq `llama-3.3-70b-versatile`** LLM, and a **14-key API carousel** for parallel processing.

---

## Pipeline Overview

| # | Pipeline | Retriever(s) | LLM Calls | Retrieval Calls |
|---|----------|-------------|-----------|-----------------|
| 1 | [Vanilla RAG](01_VanillaRAG.md) | BM25 (sparse) | 1 | 1 |
| 2 | [Single-Stage RAG](02_SingleStageRAG.md) | E5 Dense + ChromaDB | 1 | 1 |
| 3 | [Multi-Stage RAG](03_MultiStageRAG.md) | E5 Dense × 2 stages | 2 | 2 |
| 4 | [Hybrid RAG](04_HybridRAG.md) | BM25 + E5 Dense + RRF | 1 | 1 |
| 5 | [Graph RAG](05_GraphRAG.md) | BM25 + E5 Dense + GraphPPR + 3-Way RRF | 1 | 1 |
| 6 | [Multi-Stage Graph RAG](06_MultiStageGraphRAG.md) | BM25 + E5 Dense + GraphPPR × 2 stages | 2 | 2 |

---

## Complexity Progression

```
Vanilla RAG  →  Single-Stage RAG  →  Hybrid RAG  →  Graph RAG
   (BM25)         (Dense only)        (+Fusion)    (+Graph PPR)
                                                         ↓
                              Multi-Stage Graph RAG  ←  Multi-Stage RAG
                              (Everything combined)      (+Refinement)
```

---

## Common Evaluation Metrics (All Pipelines)

| Metric | Formula Reference |
|--------|------------------|
| Answer EM | Section 5 of each doc |
| Answer F1 | Section 5 of each doc |
| SP EM | Section 5 of each doc |
| SP F1 | Section 5 of each doc |
| SP Precision | Section 5 of each doc |
| SP Recall | Section 5 of each doc |
| Joint EM | Answer EM × SP EM |
| Joint F1 | Answer F1 × SP F1 |
| RAGAS Faithfulness | LLM-as-judge, claim entailment |
| MRR | Mean Reciprocal Rank |
| Latency (mean/median/p95) | Wall-clock per sample |
| Cost Proxy (mean/total) | Weighted token count |

---

## Shared Infrastructure

| Component | Value |
|-----------|-------|
| **LLM** | `llama-3.3-70b-versatile` (Groq) |
| **Embedding Model** | `intfloat/e5-base-v2` (768-dim) |
| **Vector Store** | ChromaDB EphemeralClient (HNSW, cosine) |
| **API Keys** | 14-key carousel |
| **Concurrency** | ThreadPoolExecutor (14 workers) |
| **Dataset** | HotPotQA distractor, validation split |
| **Samples** | 500 (SEED=42) |
| **Top-K** | 4 |
| **RRF k** | 60 (Pipelines 4, 5, 6) |
| **Graph Weight** | 1.5 (Pipelines 5, 6) |
| **PPR α** | 0.85 (Pipelines 5, 6) |
