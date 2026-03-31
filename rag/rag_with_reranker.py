"""
RAG Pipeline + Cross-Encoder Reranker
Demo #1 Final Version
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from pathlib import Path
import os

# Config
for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
COLLECTION = "my_research_papers"


def rewrite_query(question: str) -> list[str]:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 3 different search queries. Return only queries, one per line."},
            {"role": "user", "content": question}
        ]
    )
    queries = response.choices[0].message.content.strip().split('\n')
    return [question] + queries[:2]


def retrieve_and_rerank(question: str, k: int = 5) -> list[str]:
    # Step 1: Multi-query retrieval
    queries = rewrite_query(question)
    all_results = {}
    for q in queries:
        embedding = embedder.encode(q).tolist()
        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            limit=k
        )
        for r in results:
            if r.id not in all_results or r.score > all_results[r.id][0]:
                all_results[r.id] = (r.score, r.payload.get('content', ''))

    candidates = [content for _, content in all_results.values()]

    # Step 2: Cross-encoder reranking
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:3]]


def generate(question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(contexts)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on context. Be specific and detailed."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"}
        ]
    )
    return response.choices[0].message.content


qa_pairs = [
    {
        "question": "What is the PTST model and what performance does it achieve?",
        "ground_truth": "PTST achieves 7.5% PSNR improvement and 30% FLOPs reduction through pivotal token selection."
    },
    {
        "question": "How does SF-GPT improve NIR-to-RGB image translation?",
        "ground_truth": "SF-GPT uses SFG-MSA attention with DCT/DWT frequency decomposition achieving 5.7% PSNR improvement."
    },
    {
        "question": "What makes SMLP-KAN efficient for hyperspectral image restoration?",
        "ground_truth": "SMLP-KAN achieves 40x model compression to 9.5M parameters with 14% PSNR gain using spectral diffusion priors."
    },
    {
        "question": "What is the main contribution of the hyperspectral pansharpening paper?",
        "ground_truth": "Transformer-based spectral diffusion priors for hyperspectral pansharpening presented at WACV 2025."
    }
]

q_list, a_list, c_list, gt_list = [], [], [], []

print("\n=== RAG + Reranker Pipeline ===\n")
for pair in qa_pairs:
    q = pair["question"]
    contexts = retrieve_and_rerank(q)
    answer = generate(q, contexts)
    q_list.append(q)
    a_list.append(answer)
    c_list.append(contexts)
    gt_list.append(pair["ground_truth"])
    print(f"  ✓ {q[:60]}...")

dataset = Dataset.from_dict({
    'question': q_list,
    'answer': a_list,
    'contexts': c_list,
    'ground_truth': gt_list
})

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print("\n=== Final RAGAS Scores (with Reranker) ===")
print(f"  Faithfulness:     {result['faithfulness']:.4f}")
print(f"  Answer Relevancy: {result['answer_relevancy']:.4f}")
print(f"  Context Recall:   {result['context_recall']:.4f}")
