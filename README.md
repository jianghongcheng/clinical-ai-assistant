# Clinical AI Research Assistant

> **Production-grade LLM platform** for clinical knowledge retrieval, multi-agent reasoning, and AI safety — built on medical imaging AI research with 5 published papers.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange)](https://qdrant.tech)

---

## Demo

![Demo](docs/demo.png)

---

## Architecture
```
┌─────────────────────────────────────────┐
│         React Frontend (frontend/)      │
│  Chat UI · PDF Upload · Live Metrics    │
└──────────────────┬──────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────┐
│         FastAPI Backend (api/)          │
├─────────────────────────────────────────┤
│  7-Layer Guardrails                     │
│  Injection → PII → Topic → Quality      │
├─────────────────────────────────────────┤
│  LangGraph Multi-Agent DAG              │
│  RAG Agent → Draft → Critique → Synth  │
├─────────────────────────────────────────┤
│  RAG Pipeline                           │
│  Qdrant · Multi-Query · Reranker        │
├─────────────────────────────────────────┤
│  GraphRAG Knowledge Graph               │
│  39 nodes · 50 edges · GPT-4o-mini      │
├─────────────────────────────────────────┤
│  LLM Monitoring                         │
│  Hallucination · Drift · Latency · Logs │
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
| Fine-tuning | Trainable Params | **0.28%** (4.36M/1.55B) |
| GraphRAG | Knowledge Graph | **39 nodes / 50 edges** |

---

## Components

### 1. RAG Pipeline (`rag/`)
- Multi-query rewriting: 3 search variants per question
- Qdrant vector DB with `all-MiniLM-L6-v2` embeddings
- Cross-encoder reranking: `ms-marco-MiniLM-L-6-v2`
- RAGAS evaluation: Faithfulness 0.84 / Relevancy 0.98 / Recall 1.00

### 2. Multi-Agent System (`agents/`)
- 4-node LangGraph DAG: RAG → Draft → Critique → Synthesis
- Self-correction loop with factuality validation

### 3. QLoRA Fine-tuning (`finetune/`)
- Base: Qwen2.5-1.5B-Instruct
- QLoRA 4-bit NF4, LoRA r=16
- Training loss 5.66→3.98, 0.28% trainable params, RTX 3090

### 4. LLM Monitoring (`monitoring/`)
- Hallucination detection (LLM-as-judge)
- Retrieval drift detection with baseline comparison
- SQLite persistent logging with automated alerts
- Avg latency 1853ms, P95 2593ms

### 5. GraphRAG (`graph/`)
- GPT-4o-mini entity and relation extraction
- NetworkX directed graph: 39 nodes, 50 edges
- Multi-hop cross-document queries

### 6. LLM Guardrails (`guardrails/`)
- Input: Prompt injection, PII anonymization (Presidio), topic filter
- Output: Hallucination filter, answer quality check
- 7-layer pipeline, 100% injection block rate

### 7. REST API + React Frontend (`api/`, `frontend/`)
- FastAPI with 4 endpoints
- React UI with PDF upload, chat interface, live metrics
- Swagger docs at `/docs`

---

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY
docker run -d -p 6333:6333 qdrant/qdrant
python rag/my_papers_data.py
python api/research_api.py
# Open http://localhost:8000
```

---

## Research Papers Indexed

| # | Paper | Venue | Role |
|---|-------|-------|------|
| 1 | SMLP-KAN: Spectral MLP-KAN Diffusion Prior | CVPR Workshop 2026 | **First Author** |
| 2 | THAT: Token-wise High-frequency Augmentation Transformer | IEEE SMC 2025 | **Co-First Author** |
| 3 | Transformer-based Diffusion & Spectral Priors | IEEE JSTARS 2025 | **First Author** |
| 4 | SF-GPT: Spatial-Frequency Guided Pixel Transformer | Infrared P&T 2025 | **First Author** |
| 5 | FW-SAT: Flexible Window-based Self-attention Transformer | CVPR Workshop 2024 | **First Author** |

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

**Hongcheng Jiang** — PhD ECE, UMKC (GPA: 4.0)

9 publications: CVPR · IEEE JSTARS · IEEE SMC · WACV · Infrared Physics & Technology

[![GitHub](https://img.shields.io/badge/GitHub-jianghongcheng-black)](https://github.com/jianghongcheng)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/hongcheng-jiang-a31860181)
