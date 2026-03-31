"""
Ch5: Generate instruction-answer dataset from research papers
"""
from openai import OpenAI
from pathlib import Path
import json, os

for line in Path('.env').read_text().splitlines():
    if 'OPENAI_API_KEY' in line:
        os.environ['OPENAI_API_KEY'] = line.split('=')[1].strip()

client = OpenAI()

papers = [
    {
        "title": "PTST - Medical Image Super-Resolution",
        "content": "Pivotal Token Selective Transformer enhances window-based self-attention for medical image super-resolution. Achieves 7.5% PSNR improvement and 30% FLOPs reduction through pivotal token selection and multi-feature aggregation."
    },
    {
        "title": "SF-GPT - NIR-to-RGB Translation", 
        "content": "Spatial-Frequency Guided Pixel Transformer integrates DCT/DWT frequency decomposition with PixelUnshuffle for NIR-to-RGB translation. Novel SFG-MSA attention mechanism achieves 5.7% PSNR improvement over SOTA."
    },
    {
        "title": "SMLP-KAN - Hyperspectral Restoration",
        "content": "SMLP-KAN integrates spectral diffusion priors with MLP-KAN architecture. Achieves 40x model compression (9.5M vs 391M parameters) with 14% PSNR gain for sharpening."
    },
    {
        "title": "HAT - Thermal Image Super-Resolution",
        "content": "Hybrid Attention Transformer with flexible window-based self-attention for thermal image super-resolution. DCT-based preprocessing achieves 4% PSNR improvement with 20% FLOPs reduction."
    },
    {
        "title": "Hyperspectral Pansharpening with Diffusion Priors",
        "content": "Transformer-based spectral diffusion priors for hyperspectral pansharpening. Preserves spectral fidelity while enhancing spatial resolution. Presented at WACV 2025."
    }
]

def generate_qa_pairs(paper: dict, n: int = 10) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Generate {n} instruction-answer pairs for fine-tuning based on this research paper.
Return ONLY a JSON array like this exact format:
[
  {{"instruction": "What is PTST?", "answer": "PTST is..."}},
  {{"instruction": "How does X work?", "answer": "X works by..."}}
]
No other text, just the JSON array."""
            },
            {
                "role": "user",
                "content": f"Paper: {paper['title']}\nContent: {paper['content']}"
            }
        ]
    )
    
    text = response.choices[0].message.content.strip()
    # 找到JSON数组
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        return []
    pairs = json.loads(text[start:end])
    return pairs

all_pairs = []
print("Generating instruction-answer pairs...\n")

for paper in papers:
    pairs = generate_qa_pairs(paper, n=10)
    for pair in pairs:
        pair["source"] = paper["title"]
        all_pairs.append(pair)
    print(f"  ✓ {paper['title']}: {len(pairs)} pairs")

output = Path("finetune_dataset.jsonl")
with output.open("w") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair) + "\n")

print(f"\n=== Dataset Summary ===")
print(f"  Total pairs:  {len(all_pairs)}")
print(f"  Saved to:     {output}")
print(f"\nSample:")
if all_pairs:
    print(json.dumps(all_pairs[0], indent=2))
