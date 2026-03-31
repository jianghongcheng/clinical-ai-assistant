"""
Demo #5: GraphRAG - Knowledge Graph + Vector Retrieval
"""
import os
import json
from pathlib import Path
import networkx as nx
import spacy
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')
COLLECTION = "my_research_papers"

papers = [
    {"id": "ptst", "title": "PTST", "content": "Pivotal Token Selective Transformer (PTST) enhances window-based self-attention for medical image super-resolution. Achieves 7.5% PSNR improvement and 30% FLOPs reduction through pivotal token selection and multi-feature aggregation. Deployed at NextTier 2024."},
    {"id": "sfgpt", "title": "SF-GPT", "content": "Spatial-Frequency Guided Pixel Transformer (SF-GPT) for NIR-to-RGB translation. Uses DCT and DWT frequency decomposition with PixelUnshuffle. Introduces SFG-MSA attention mechanism. Achieves 5.7% PSNR improvement. Published in Infrared Physics and Technology 2025."},
    {"id": "smlpkan", "title": "SMLP-KAN", "content": "SMLP-KAN integrates spectral diffusion priors with MLP-KAN for hyperspectral image restoration. Achieves 40x model compression to 9.5M parameters versus PLRDiff 391M parameters. Gains 14% PSNR improvement. Submitted to CVPR Workshop 2026."},
    {"id": "hat", "title": "HAT", "content": "Hybrid Attention Transformer (HAT) with flexible window-based self-attention for thermal image super-resolution. Uses DCT preprocessing. Reduces FLOPs by 20% and improves PSNR by 4%. Published at CVPR Workshop 2024."},
    {"id": "pansharp", "title": "Pansharpening", "content": "Transformer-based spectral diffusion priors for hyperspectral pansharpening. Preserves spectral fidelity while enhancing spatial resolution. Presented at WACV Workshop 2025."}
]


def extract_entities_relations(paper: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Extract entities and relationships. Return JSON:
{"entities": [{"name": "...", "type": "MODEL/TECHNIQUE/METRIC/VENUE"}],
 "relations": [{"source": "...", "relation": "...", "target": "..."}]}"""},
            {"role": "user", "content": f"Paper: {paper['title']}\n{paper['content']}"}
        ]
    )
    text = response.choices[0].message.content.strip()
    start, end = text.find('{'), text.rfind('}') + 1
    return json.loads(text[start:end])


def build_knowledge_graph(papers: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for paper in papers:
        print(f"  Extracting entities from {paper['title']}...")
        G.add_node(paper['id'], type='PAPER', title=paper['title'], content=paper['content'])
        extracted = extract_entities_relations(paper)
        for entity in extracted.get('entities', []):
            node_id = entity['name'].lower().replace(' ', '_')
            G.add_node(node_id, type=entity['type'], name=entity['name'])
            G.add_edge(paper['id'], node_id, relation='HAS_ENTITY')
        for rel in extracted.get('relations', []):
            src = rel['source'].lower().replace(' ', '_')
            tgt = rel['target'].lower().replace(' ', '_')
            if not G.has_node(src):
                G.add_node(src, type='CONCEPT', name=rel['source'])
            if not G.has_node(tgt):
                G.add_node(tgt, type='CONCEPT', name=rel['target'])
            G.add_edge(src, tgt, relation=rel['relation'])
    return G


def graph_retrieve(question: str, G: nx.DiGraph, k: int = 3) -> list:
    query_tokens = [t.lemma_.lower() for t in nlp(question) if not t.is_stop and t.is_alpha and len(t) > 3]
    relevant_papers = set()
    for node, data in G.nodes(data=True):
        node_text = (data.get('name', '') + ' ' + data.get('title', '')).lower()
        if any(token in node_text for token in query_tokens):
            for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
                if G.nodes[neighbor].get('type') == 'PAPER':
                    relevant_papers.add(neighbor)
    contexts = [f"[{G.nodes[p]['title']}] {G.nodes[p]['content']}" for p in list(relevant_papers)[:k]]
    if not contexts:
        embedding = embedder.encode(question).tolist()
        results = qdrant.search(collection_name=COLLECTION, query_vector=embedding, limit=k)
        contexts = [r.payload.get('content', '') for r in results]
    return contexts


def vector_retrieve(question: str, k: int = 3) -> list:
    embedding = embedder.encode(question).tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=embedding, limit=k)
    return [r.payload.get('content', '') for r in results]


def generate(question: str, contexts: list) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on context. Be specific."},
            {"role": "user", "content": f"Context:\n{chr(10).join(contexts)}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


def score_answer(question: str, answer: str, ground_truth: str) -> float:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Score answer vs ground truth 0.0-1.0. Return only a number."},
            {"role": "user", "content": f"Q: {question}\nGT: {ground_truth}\nAnswer: {answer}"}
        ]
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5


if __name__ == "__main__":
    print("=== GraphRAG — Knowledge Graph + Vector Retrieval ===\n")
    print("Step 1: Building knowledge graph...")
    G = build_knowledge_graph(papers)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    qa_pairs = [
        {"question": "What is PTST and its PSNR improvement?", "ground_truth": "PTST achieves 7.5% PSNR improvement and 30% FLOPs reduction through pivotal token selection."},
        {"question": "How does SMLP-KAN compare to PLRDiff in model size?", "ground_truth": "SMLP-KAN has 9.5M parameters, 40x smaller than PLRDiff with 391M parameters."},
        {"question": "Which papers use DCT in their approach?", "ground_truth": "SF-GPT uses DCT for frequency decomposition, and HAT uses DCT preprocessing."},
        {"question": "What venue was the pansharpening paper at?", "ground_truth": "The pansharpening paper was presented at WACV Workshop 2025."}
    ]

    print("Step 2: Comparing GraphRAG vs Vector RAG...\n")
    graph_scores, vector_scores = [], []

    for pair in qa_pairs:
        q, gt = pair["question"], pair["ground_truth"]
        g_ctx = graph_retrieve(q, G)
        g_ans = generate(q, g_ctx)
        g_score = score_answer(q, g_ans, gt)
        graph_scores.append(g_score)

        v_ctx = vector_retrieve(q)
        v_ans = generate(q, v_ctx)
        v_score = score_answer(q, v_ans, gt)
        vector_scores.append(v_score)

        print(f"  Q: {q[:50]}...")
        print(f"    GraphRAG: {g_score:.2f} | VectorRAG: {v_score:.2f}")

    avg_g = sum(graph_scores) / len(graph_scores)
    avg_v = sum(vector_scores) / len(vector_scores)

    print(f"\n{'='*50}")
    print(f"  GraphRAG vs Vector RAG Results")
    print(f"{'='*50}")
    print(f"  GraphRAG avg score:  {avg_g:.4f}")
    print(f"  VectorRAG avg score: {avg_v:.4f}")
    print(f"  GraphRAG improvement: {(avg_g - avg_v)*100:+.1f}%")
    print(f"{'='*50}")

    results = {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "graphrag_score": avg_g,
        "vectorrag_score": avg_v,
        "improvement_pct": (avg_g - avg_v) * 100
    }
    Path("graph_rag_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to graph_rag_results.json")
