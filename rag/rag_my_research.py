"""
RAG Pipeline on My Own Research Papers
Hongcheng (Gabe) Jiang - LLM Engineer Portfolio Demo
"""

import os

from datasets import Dataset
from openai import OpenAI
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from sentence_transformers import SentenceTransformer

with open(".env") as f:
    for line in f:
        if "OPENAI_API_KEY" in line:
            os.environ["OPENAI_API_KEY"] = line.split("=")[1].strip()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = "my_research_papers"


def rewrite_query(question: str) -> list[str]:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 different search queries for the question. Return only the queries, one per line.",
            },
            {"role": "user", "content": question},
        ],
    )
    queries = response.choices[0].message.content.strip().split("\n")
    return [question] + queries[:2]


def retrieve(question: str, k: int = 3) -> list[str]:
    queries = rewrite_query(question)
    all_results = {}
    for q in queries:
        embedding = embedder.encode(q).tolist()
        results = qdrant.search(collection_name=COLLECTION, query_vector=embedding, limit=k)
        for r in results:
            if r.id not in all_results or r.score > all_results[r.id][0]:
                all_results[r.id] = (r.score, r.payload.get("content", ""))
    sorted_results = sorted(all_results.values(), key=lambda x: x[0], reverse=True)
    return [content for _, content in sorted_results[:k]]


def generate(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(contexts)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. Be specific and detailed."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"},
        ],
    )
    return response.choices[0].message.content


# 针对你自己论文的问答对
qa_pairs = [
    {
        "question": "What is the PTST model and what performance does it achieve?",
        "ground_truth": "PTST (Pivotal Token Selective Transformer) is a transformer for medical image super-resolution that achieves 7.5% PSNR improvement and 30% FLOPs reduction through pivotal token selection and multi-feature aggregation.",
    },
    {
        "question": "How does SF-GPT improve NIR-to-RGB image translation?",
        "ground_truth": "SF-GPT uses spatial-frequency guided pixel transformer with SFG-MSA attention mechanism, DCT/DWT frequency decomposition, and PixelUnshuffle to achieve 5.7% PSNR improvement over state-of-the-art NIR-to-RGB translation.",
    },
    {
        "question": "What makes SMLP-KAN efficient for hyperspectral image restoration?",
        "ground_truth": "SMLP-KAN achieves 40x model compression to 9.5M parameters compared to 391M in PLRDiff and HIR-Diff, while achieving 14% PSNR gain for sharpening using spectral diffusion priors and MLP-KAN architecture.",
    },
    {
        "question": "What is the main contribution of the hyperspectral pansharpening paper?",
        "ground_truth": "The paper proposes transformer-based spectral diffusion priors for hyperspectral pansharpening that preserves spectral fidelity while enhancing spatial resolution, presented at WACV 2025.",
    },
]

q_list, a_list, c_list, gt_list = [], [], [], []

print("\n=== RAG on My Research Papers ===\n")
for pair in qa_pairs:
    q = pair["question"]
    contexts = retrieve(q)
    answer = generate(q, contexts)
    q_list.append(q)
    a_list.append(answer)
    c_list.append(contexts)
    gt_list.append(pair["ground_truth"])
    print(f"  ✓ {q[:60]}...")

dataset = Dataset.from_dict({"question": q_list, "answer": a_list, "contexts": c_list, "ground_truth": gt_list})

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

print("\n=== Final RAGAS Scores ===")
print(f"  Faithfulness:     {result['faithfulness']:.4f}")
print(f"  Answer Relevancy: {result['answer_relevancy']:.4f}")
print(f"  Context Recall:   {result['context_recall']:.4f}")
