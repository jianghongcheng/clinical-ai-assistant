# Clinical AI Research Assistant

> **Production-grade LLM platform** for clinical knowledge retrieval, multi-agent reasoning, and AI safety — built on medical imaging research with 5 published papers.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Architecture
```
                    ┌─────────────────────────────────────────┐
                    │         REST API (FastAPI)               │
                    │  POST /query  GET /health  GET /metrics  │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │           7-Layer Guardrails             │
                    │  Injection → PII → Profanity → Topic     │
                    └──────────────────┬──────────────────────┘
                                       │
               ┌───────────────────────▼───────────────────────┐
               │            LangGraph Multi-Agent DAG           │
               │  ┌─────────┐  ┌───────┐  ┌───────────────┐   │
               │  │RAG Agent│→ │ Draft │→ │   Critique    │   │
               │  │(Qdrant) │  │ Agent │  │    Agent      │   │
               │  └─────────┘  └───────┘  └──────┬────────┘   │
               │                                  │            │
               │                        ┌─────────▼──────┐    │
               │                        │Synthesis Agent │    │
               │                        └────────────────┘    │
               └───────────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │           LLM Monitoring                 │
                    │  Hallucination · Drift · Latency · Logs  │
                    └─────────────────────────────────────────┘
```

---

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| RAG Pipeline | Answer Relevancy | **0.98** |
| RAG Pipeline | Context Recall | **1.00** |
| RAG Pipeline | Faithfulness | **0.84** |
| Guardrails | Injection Block Rate | **100%** |
| API | Avg Latency | 2150ms |
| API | P95 Latency | 2593ms |
| Fine-tuning | Training Loss | 5.66 → **3.98** |
| Fine-tuning | Trainable Params | **0.28%** (4.36M / 1.55B) |

---

## Components

### 1. RAG Pipeline with Reranker (`rag/`)
- Multi-query rewriting: generates 3 search variants per question
- Vector retrieval: Qdrant with `all-MiniLM-L6-v2` embeddings
- Cross-encoder reranking: `ms-marco-MiniLM-L-6-v2` for precision
- Evaluation: RAGAS framework (Faithfulness, Answer Relevancy, Context Recall)

### 2. Multi-Agent System (`agents/`)
- 4-node LangGraph DAG: RAG → Draft → Critique → Synthesis
- Self-correction loop: Critique agent validates factuality before synthesis
- State management: full conversation history preserved across nodes

### 3. QLoRA Fine-tuning (`finetune/`)
- Base model: Qwen2.5-1.5B-Instruct
- Method: QLoRA (4-bit NF4, LoRA r=16, alpha=32)
- Dataset: 50 domain-specific instruction-answer pairs
- Hardware: RTX 3090 24GB VRAM

### 4. LLM Monitoring (`monitoring/`)
- Hallucination detection: LLM-as-judge scoring
- Retrieval drift: baseline comparison across sessions
- Latency tracking: per-query timing with P95 alerts
- Persistent logging: SQLite with automated anomaly detection

### 5. GraphRAG Knowledge Graph (`graph/`)
- Entity extraction: GPT-4o-mini relationship mining
- Graph: NetworkX directed graph — 39 nodes, 50 edges
- Multi-hop queries: cross-document reasoning

### 6. LLM Guardrails (`guardrails/`)
- Input: Prompt injection detection, PII anonymization (Presidio), topic relevance
- Output: Hallucination filter, PII removal, answer quality check
- 7-layer pipeline with structured logging

### 7. REST API (`api/`)
```
POST /query   → Guardrails → RAG/Agent → Monitor → Response
GET  /health  → System health check
GET  /metrics → Live monitoring dashboard
GET  /papers  → Indexed paper collections
```

---

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add your OPENAI_API_KEY

# 3. Start vector database
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Index research papers
python rag/my_papers_data.py

# 5. Run RAG evaluation
python rag/rag_with_reranker.py

# 6. Start API server
python api/research_api.py
# Swagger UI: http://localhost:8000/docs
```

---

## Research Papers Indexed

| # | Paper | Venue | Role |
|---|-------|-------|------|
| 1 | SMLP-KAN: Spectral MLP-KAN Diffusion Prior | CVPR Workshop 2026 | First Author |
| 2 | THAT: Token-wise High-frequency Augmentation Transformer | IEEE SMC 2025 | Co-First Author |
| 3 | Transformer-based Diffusion & Spectral Priors | IEEE JSTARS 2025 | First Author |
| 4 | SF-GPT: Spatial-Frequency Guided Pixel Transformer | Infrared P&T 2025 | First Author |
| 5 | FW-SAT: Flexible Window-based Self-attention Transformer | CVPR Workshop 2024 | First Author |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini |
| Agent Framework | LangGraph 1.1 |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Fine-tuning | QLoRA, PEFT, TRL |
| Evaluation | RAGAS |
| API | FastAPI |
| Knowledge Graph | NetworkX |
| Safety | Presidio, better-profanity |
| Hardware | RTX 3090 (24GB VRAM) |

---

## Author

**Hongcheng Jiang** — PhD ECE, University of Missouri-Kansas City (GPA: 4.0)

9 publications: CVPR · IEEE JSTARS · IEEE SMC · WACV · Infrared Physics & Technology

[![GitHub](https://img.shields.io/badge/GitHub-jianghongcheng-black)](https://github.com/jianghongcheng)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/hongcheng-jiang-a31860181)
