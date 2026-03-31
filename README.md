# LLM Research Assistant

> Production-grade LLM pipeline built on published research in hyperspectral imaging and image super-resolution.
> **PhD, ECE, UMKC** | CVPR / WACV / Infrared Physics & Technology publications

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)](https://langchain.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-orange)](https://qdrant.tech)

---

## 7 Production Demos

### Demo 1 — RAG Pipeline with Reranker
**Files:** `rag_my_research.py`, `rag_with_reranker.py`

| Metric | Baseline | + Query Rewriting | + Reranker |
|--------|----------|-------------------|------------|
| Faithfulness | 0.52 | 0.84 | 0.71 |
| Answer Relevancy | 0.66 | 0.95 | **0.98** |
| Context Recall | 0.00 | **1.00** | **1.00** |

- Multi-query rewriting generates 3 search variants per question
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) for precision
- Evaluated with RAGAS framework

### Demo 2 — Multi-Agent System (LangGraph)
**File:** `multi_agent_research.py`
```
Question → RAG Agent → Draft Agent → Critique Agent → Synthesis Agent → Answer
```
- 4-node LangGraph DAG with self-correction loop
- Critique agent evaluates factuality before synthesis
- Tested on 5 published research papers

### Demo 3 — QLoRA Fine-tuning
**Files:** `finetune_qlora.py`, `generate_finetune_dataset.py`

| Config | Value |
|--------|-------|
| Base model | Qwen2.5-1.5B-Instruct |
| Method | QLoRA (4-bit NF4) |
| Trainable params | 4.36M / 1.55B (0.28%) |
| Training loss | 5.66 → 3.98 |
| Hardware | RTX 3090 24GB |

### Demo 4 — Production LLM Monitoring
**File:** `llm_monitoring.py`

- Real-time hallucination detection (LLM-as-judge)
- Retrieval drift detection with baseline comparison
- Latency tracking: avg 1853ms, P95 2593ms
- SQLite logging with alert system
- Inspired by production MLOps at Flix/Greyhound ($20M revenue impact)

### Demo 5 — GraphRAG Knowledge Graph
**File:** `graph_rag.py`

- Extracts entities and relations from research papers via GPT-4o-mini
- Builds NetworkX knowledge graph: **39 nodes, 50 edges**
- Graph traversal retrieval vs vector baseline comparison
- Multi-hop queries across document collections

### Demo 6 — LLM Guardrails & Safety
**File:** `llm_guardrails.py`

- **Input:** Prompt injection detection, PII anonymization, topic relevance filtering
- **Output:** Hallucination filtering, PII removal, answer quality check
- 7-layer guardrail pipeline with logging
- Tested: 2/7 injection attempts correctly blocked

### Demo 7 — Research Assistant REST API
**File:** `research_api.py`
```
POST /query    → Guardrails → RAG/Agent → Monitoring → Response
GET  /health   → System health check
GET  /metrics  → Live monitoring dashboard
GET  /papers   → Indexed collections
```
- FastAPI with automatic Swagger docs
- Integrates all 6 demos into one service
- Avg latency: 2150ms | Faithfulness: 0.73

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector DB | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | GPT-4o-mini |
| Agent Framework | LangGraph 1.1 |
| Fine-tuning | QLoRA, PEFT, TRL |
| Evaluation | RAGAS |
| API | FastAPI |
| Knowledge Graph | NetworkX |
| Safety | Presidio, better-profanity |
| Hardware | RTX 3090 (24GB VRAM) |

## Quick Start
```bash
# Start vector DB
docker run -d -p 6333:6333 qdrant/qdrant

# Load research papers
python my_papers_data.py

# Run RAG evaluation
python rag_my_research.py

# Start API server
python research_api.py
# Docs at http://localhost:8000/docs
```

## Research Papers Indexed

1. **PTST** — Pivotal Token Selective Transformer, Medical Image Super-Resolution (2024)
2. **SF-GPT** — Spatial-Frequency Guided Pixel Transformer, NIR-to-RGB (Infrared Physics & Technology, 2025)
3. **SMLP-KAN** — Diffusion-based Hyperspectral Restoration, 40x compression (CVPR Workshop 2026)
4. **HAT** — Hybrid Attention Transformer, Thermal Super-Resolution (CVPR Workshop, 2024)
5. **Pansharpening** — Spectral Diffusion Priors (WACV Workshop, 2025)
