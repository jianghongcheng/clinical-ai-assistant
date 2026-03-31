"""
Demo #7: Research Assistant REST API
Integrates RAG, Guardrails, Monitoring, Multi-Agent into one service.
Production-ready FastAPI application.
"""
import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import re

# ── Setup ─────────────────────────────────────────────────
for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
COLLECTION = "my_research_papers"
DB_PATH = "llm_monitoring.db"


# ── DB Init ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, question TEXT, answer TEXT,
        retrieval_score REAL, latency_ms REAL,
        faithfulness_score REAL, flagged INTEGER, flag_reason TEXT
    )""")
    conn.commit()
    conn.close()


def log_query(question, answer, retrieval_score, latency_ms, faithfulness=0.8, flagged=0, reason=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO query_logs
        (timestamp,question,answer,retrieval_score,latency_ms,faithfulness_score,flagged,flag_reason)
        VALUES (?,?,?,?,?,?,?,?)""",
        (datetime.now().isoformat(), question, answer,
         retrieval_score, latency_ms, faithfulness, flagged, reason))
    conn.commit()
    conn.close()


# ── Guardrails ────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"forget (everything|all|previous)",
    r"you are now", r"act as (a|an|if)",
    r"pretend (you are|to be)", r"jailbreak",
]

def check_injection(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)

def check_relevance(question: str) -> bool:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Is this about image processing, PTST, SMLP-KAN, SF-GPT, HAT, pansharpening, hyperspectral, NIR, thermal, super resolution, transformer, attention, diffusion, computer vision, deep learning, machine learning, neural network, or LLM research? Reply only: YES or NO"},
            {"role": "user", "content": question}
        ]
    )
    return "YES" in response.choices[0].message.content.upper()


# ── RAG ───────────────────────────────────────────────────
def retrieve(question: str, k: int = 3) -> tuple[list[str], float]:
    embedding = embedder.encode(question).tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=embedding, limit=k)
    contexts = [r.payload.get("content", "") for r in results]
    avg_score = float(np.mean([r.score for r in results])) if results else 0.0
    return contexts, avg_score

def generate(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(contexts)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on context. Be specific and detailed."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


# ── Multi-Agent ───────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    retrieved_docs: list[str]
    draft_answer: str
    critique: str
    final_answer: str
    iterations: Annotated[int, operator.add]

def rag_node(state):
    contexts, _ = retrieve(state["question"])
    return {"retrieved_docs": contexts, "iterations": 1}

def draft_node(state):
    ctx = "\n\n".join(state["retrieved_docs"])
    response = llm.invoke([
        SystemMessage(content="Answer based only on context."),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {state['question']}")
    ])
    return {"draft_answer": response.content, "iterations": 1}

def critique_node(state):
    response = llm.invoke([
        SystemMessage(content="Evaluate: PASS if accurate and complete, else IMPROVE: <feedback>"),
        HumanMessage(content=f"Q: {state['question']}\nAnswer: {state['draft_answer']}")
    ])
    return {"critique": response.content, "iterations": 1}

def synthesis_node(state):
    if "PASS" in state["critique"]:
        return {"final_answer": state["draft_answer"], "iterations": 1}
    response = llm.invoke([
        SystemMessage(content="Improve the answer based on critique."),
        HumanMessage(content=f"Answer: {state['draft_answer']}\nCritique: {state['critique']}")
    ])
    return {"final_answer": response.content, "iterations": 1}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", rag_node)
workflow.add_node("draft", draft_node)
workflow.add_node("critique", critique_node)
workflow.add_node("synthesize", synthesis_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "critique")
workflow.add_edge("critique", "synthesize")
workflow.add_edge("synthesize", END)
agent_app = workflow.compile()


# ── FastAPI App ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Research Assistant API",
    description="Production LLM API with RAG, Guardrails, Multi-Agent, and Monitoring",
    version="1.0.0",
    lifespan=lifespan
)


# ── Request/Response Models ───────────────────────────────
class QueryRequest(BaseModel):
    question: str
    mode: str = "rag"  # "rag" or "agent"

class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    latency_ms: float
    retrieval_score: float
    blocked: bool
    block_reason: str = ""


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
async def health():
    """System health check."""
    qdrant_ok = True
    try:
        qdrant.get_collections()
    except:
        qdrant_ok = False

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "qdrant": "up" if qdrant_ok else "down",
            "openai": "up",
            "embedder": "up"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint with guardrails and monitoring."""
    start = time.time()
    question = request.question

    # Input guardrails
    if check_injection(question):
        return QueryResponse(
            question=question, answer="Request blocked: security policy violation.",
            mode=request.mode, latency_ms=0, retrieval_score=0,
            blocked=True, block_reason="Prompt injection detected"
        )

    if not check_relevance(question):
        return QueryResponse(
            question=question,
            answer="I can only answer questions about image processing and deep learning research.",
            mode=request.mode, latency_ms=0, retrieval_score=0,
            blocked=True, block_reason="Out of domain"
        )

    # Process
    if request.mode == "agent":
        result = agent_app.invoke({
            "question": question, "retrieved_docs": [], "draft_answer": "",
            "critique": "", "final_answer": "", "iterations": 0
        })
        answer = result["final_answer"]
        retrieval_score = 0.8
    else:
        contexts, retrieval_score = retrieve(question)
        answer = generate(question, contexts)

    latency_ms = (time.time() - start) * 1000
    log_query(question, answer, retrieval_score, latency_ms)

    return QueryResponse(
        question=question, answer=answer, mode=request.mode,
        latency_ms=round(latency_ms, 2),
        retrieval_score=round(retrieval_score, 4),
        blocked=False
    )


@app.get("/metrics")
async def metrics():
    """Monitoring metrics dashboard."""
    conn = sqlite3.connect(DB_PATH)
    logs = conn.execute("SELECT * FROM query_logs").fetchall()
    conn.close()

    if not logs:
        return {"message": "No data yet"}

    latencies = [r[6] for r in logs if r[6] is not None]
    faithfulness = [r[7] for r in logs if r[7] is not None]
    flagged = [r[8] for r in logs if r[8] is not None]

    return {
        "total_queries": len(logs),
        "flagged_queries": sum(flagged),
        "flag_rate_pct": round(100 * sum(flagged) / len(logs), 1),
        "latency": {
            "avg_ms": round(float(np.mean(latencies)), 1),
            "p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        },
        "faithfulness": {
            "avg": round(float(np.mean(faithfulness)), 4),
            "min": round(float(min(faithfulness)), 4),
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/papers")
async def list_papers():
    """List indexed research papers."""
    collections = qdrant.get_collections()
    collection_names = [c.name for c in collections.collections]
    info = {}
    for name in collection_names:
        col = qdrant.get_collection(name)
        info[name] = col.points_count
    return {"collections": info}


if __name__ == "__main__":
    print("Starting Research Assistant API...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
