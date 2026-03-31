"""
Demo #2: Multi-Agent Research Assistant
Architecture: Supervisor → RAG Agent + Critique Agent + Synthesis Agent
"""
import os
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────
for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
COLLECTION = "my_research_papers"


# ── State ────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    retrieved_docs: list[str]
    draft_answer: str
    critique: str
    final_answer: str
    iterations: Annotated[int, operator.add]


# ── Agents ───────────────────────────────────────────────
def rag_agent(state: AgentState) -> AgentState:
    """Retrieve relevant documents from vector store."""
    print("  [RAG Agent] Retrieving documents...")
    question = state["question"]
    embedding = embedder.encode(question).tolist()
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=embedding,
        limit=3
    )
    docs = [r.payload.get("content", "") for r in results]
    return {"retrieved_docs": docs, "iterations": 1}


def draft_agent(state: AgentState) -> AgentState:
    """Generate initial answer from retrieved docs."""
    print("  [Draft Agent] Generating answer...")
    ctx = "\n\n".join(state["retrieved_docs"])
    response = llm.invoke([
        SystemMessage(content="Answer based only on the provided context. Be specific."),
        HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {state['question']}")
    ])
    return {"draft_answer": response.content, "iterations": 1}


def critique_agent(state: AgentState) -> AgentState:
    """Critique the draft answer for accuracy and completeness."""
    print("  [Critique Agent] Evaluating answer...")
    response = llm.invoke([
        SystemMessage(content="""You are a critical reviewer. Evaluate the answer for:
1. Factual accuracy based on context
2. Completeness
3. Clarity
Reply with: PASS if good, or IMPROVE: <specific feedback> if needs improvement."""),
        HumanMessage(content=f"""Question: {state['question']}
Context: {state['retrieved_docs'][0][:500]}
Answer: {state['draft_answer']}""")
    ])
    return {"critique": response.content, "iterations": 1}


def synthesis_agent(state: AgentState) -> AgentState:
    """Synthesize final answer incorporating critique feedback."""
    print("  [Synthesis Agent] Finalizing answer...")
    if "PASS" in state["critique"]:
        return {"final_answer": state["draft_answer"], "iterations": 1}

    response = llm.invoke([
        SystemMessage(content="Improve the answer based on the critique feedback."),
        HumanMessage(content=f"""Original answer: {state['draft_answer']}
Critique: {state['critique']}
Question: {state['question']}
Provide an improved answer:""")
    ])
    return {"final_answer": response.content, "iterations": 1}


def should_continue(state: AgentState) -> str:
    """Route: if critique passes or max iterations, end. Else improve."""
    if "PASS" in state["critique"] or state["iterations"] >= 6:
        return "synthesize"
    return "synthesize"


# ── Build Graph ───────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", rag_agent)
workflow.add_node("draft", draft_agent)
workflow.add_node("critique", critique_agent)
workflow.add_node("synthesize", synthesis_agent)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "critique")
workflow.add_conditional_edges("critique", should_continue, {
    "synthesize": "synthesize"
})
workflow.add_edge("synthesize", END)

app = workflow.compile()


# ── Run ───────────────────────────────────────────────────
def run_agent(question: str) -> dict:
    print(f"\nQuestion: {question}")
    result = app.invoke({
        "question": question,
        "retrieved_docs": [],
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "iterations": 0
    })
    return result


if __name__ == "__main__":
    questions = [
        "What is the key innovation of PTST for medical image super-resolution?",
        "How does SMLP-KAN compare to PLRDiff in terms of model size?",
        "What frequency decomposition method does SF-GPT use and why?"
    ]

    print("=== Multi-Agent Research Assistant ===")
    print("Architecture: RAG → Draft → Critique → Synthesis\n")

    for q in questions:
        result = run_agent(q)
        print(f"\nFinal Answer: {result['final_answer'][:300]}...")
        print(f"Iterations: {result['iterations']}")
        print("-" * 60)
