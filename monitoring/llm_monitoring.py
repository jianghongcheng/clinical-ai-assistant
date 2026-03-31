"""
Demo #4: Production LLM Monitoring System
Monitors RAG pipeline quality, detects drift, triggers alerts.
Inspired by production MLOps experience at Flix/Greyhound.
"""
import os
import time
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
COLLECTION = "my_research_papers"
DB_PATH = "llm_monitoring.db"


# ── Database Setup ────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            retrieval_score REAL,
            answer_length INTEGER,
            latency_ms REAL,
            faithfulness_score REAL,
            flagged INTEGER DEFAULT 0,
            flag_reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            message TEXT,
            severity TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(question, answer, retrieval_score, latency_ms, faithfulness_score, flagged=0, flag_reason=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO query_logs 
        (timestamp, question, answer, retrieval_score, answer_length, latency_ms, faithfulness_score, flagged, flag_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        question, answer, retrieval_score,
        len(answer), latency_ms, faithfulness_score,
        flagged, flag_reason
    ))
    conn.commit()
    conn.close()


def log_alert(alert_type, message, severity="WARNING"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO alerts (timestamp, alert_type, message, severity)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), alert_type, message, severity))
    conn.commit()
    conn.close()
    print(f"  🚨 ALERT [{severity}] {alert_type}: {message}")


# ── Monitoring Checks ─────────────────────────────────────
def check_faithfulness(question: str, answer: str, context: str) -> float:
    """LLM-as-judge faithfulness check."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rate faithfulness 0.0-1.0. Return only a number."},
            {"role": "user", "content": f"Context: {context[:500]}\nAnswer: {answer}\nScore:"}
        ]
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5


def check_retrieval_drift(current_scores: list[float], baseline_mean: float = 0.6) -> bool:
    """Detect if retrieval quality has drifted from baseline."""
    if not current_scores:
        return False
    current_mean = np.mean(current_scores)
    drift = abs(current_mean - baseline_mean)
    return drift > 0.15


def check_hallucination(answer: str, context: str) -> tuple[bool, str]:
    """Simple hallucination detection."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Check if the answer contains claims NOT supported by context.
Reply: HALLUCINATION: <reason> or GROUNDED"""},
            {"role": "user", "content": f"Context: {context[:500]}\nAnswer: {answer}"}
        ]
    )
    result = response.choices[0].message.content.strip()
    is_hallucination = result.startswith("HALLUCINATION")
    reason = result.replace("HALLUCINATION: ", "") if is_hallucination else ""
    return is_hallucination, reason


# ── Monitored RAG Pipeline ────────────────────────────────
def monitored_rag(question: str) -> dict:
    """RAG pipeline with full monitoring."""
    start = time.time()

    # Retrieve
    embedding = embedder.encode(question).tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=embedding, limit=3)
    contexts = [r.payload.get("content", "") for r in results]
    retrieval_scores = [r.score for r in results]
    avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0

    # Generate
    ctx = "\n\n".join(contexts)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on context."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
        ]
    )
    answer = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000

    # ── Monitoring checks ─────────────────────────────────
    flagged = False
    flag_reasons = []

    # 1. Hallucination check
    is_hallucination, reason = check_hallucination(answer, ctx)
    if is_hallucination:
        flagged = True
        flag_reasons.append(f"Hallucination: {reason}")
        log_alert("HALLUCINATION", f"Q: {question[:50]} | {reason}", "ERROR")

    # 2. Faithfulness check
    faithfulness = check_faithfulness(question, answer, ctx)
    if faithfulness < 0.5:
        flagged = True
        flag_reasons.append(f"Low faithfulness: {faithfulness:.2f}")
        log_alert("LOW_FAITHFULNESS", f"Score: {faithfulness:.2f} | Q: {question[:50]}", "WARNING")

    # 3. Retrieval drift check
    if check_retrieval_drift(retrieval_scores):
        log_alert("RETRIEVAL_DRIFT", f"Avg score: {avg_retrieval_score:.3f} | Q: {question[:50]}", "WARNING")

    # 4. Latency check
    if latency_ms > 5000:
        log_alert("HIGH_LATENCY", f"{latency_ms:.0f}ms | Q: {question[:50]}", "WARNING")

    # 5. Short answer check
    if len(answer) < 50:
        flagged = True
        flag_reasons.append("Answer too short")
        log_alert("SHORT_ANSWER", f"Length: {len(answer)} | Q: {question[:50]}", "WARNING")

    # Log to DB
    log_query(
        question, answer, avg_retrieval_score,
        latency_ms, faithfulness,
        int(flagged), "; ".join(flag_reasons)
    )

    return {
        "question": question,
        "answer": answer,
        "retrieval_score": avg_retrieval_score,
        "faithfulness": faithfulness,
        "latency_ms": latency_ms,
        "flagged": flagged,
        "flag_reasons": flag_reasons
    }


# ── Dashboard ─────────────────────────────────────────────
def print_dashboard():
    conn = sqlite3.connect(DB_PATH)
    
    logs = conn.execute("SELECT * FROM query_logs").fetchall()
    alerts = conn.execute("SELECT * FROM alerts").fetchall()
    
    if not logs:
        print("No data yet.")
        return

    latencies = [r[6] for r in logs]
    faithfulness = [r[7] for r in logs]
    flagged = [r[8] for r in logs]

    print("\n" + "="*50)
    print("  LLM MONITORING DASHBOARD")
    print("="*50)
    print(f"  Total queries:      {len(logs)}")
    print(f"  Flagged queries:    {sum(flagged)} ({100*sum(flagged)/len(logs):.0f}%)")
    print(f"  Avg latency:        {np.mean(latencies):.0f}ms")
    print(f"  P95 latency:        {np.percentile(latencies, 95):.0f}ms")
    print(f"  Avg faithfulness:   {np.mean(faithfulness):.3f}")
    print(f"  Min faithfulness:   {min(faithfulness):.3f}")
    print(f"  Total alerts:       {len(alerts)}")
    print("="*50)

    if alerts:
        print("\n  Recent Alerts:")
        for alert in alerts[-5:]:
            print(f"  [{alert[4]}] {alert[2]}: {alert[3][:60]}")
    
    conn.close()


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=== LLM Production Monitoring System ===\n")

    # Normal queries
    normal_queries = [
        "What is the PTST model and its key innovation?",
        "How does SMLP-KAN achieve 40x model compression?",
        "What attention mechanism does SF-GPT introduce?",
        "What venue was the pansharpening paper presented at?",
        "How does HAT reduce computational cost?"
    ]

    # Edge case queries (to trigger monitoring)
    edge_queries = [
        "What is the meaning of life according to these papers?",  # Out of domain
        "How does PTST compare to GPT-4?",  # Comparison not in context
    ]

    all_queries = normal_queries + edge_queries

    for q in all_queries:
        print(f"Processing: {q[:60]}...")
        result = monitored_rag(q)
        status = "🚩 FLAGGED" if result["flagged"] else "✓ OK"
        print(f"  {status} | Faithfulness: {result['faithfulness']:.2f} | Latency: {result['latency_ms']:.0f}ms")

    print_dashboard()
