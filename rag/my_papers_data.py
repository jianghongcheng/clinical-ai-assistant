from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Initialize
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "my_research_papers"

# Research papers data
papers = [
    {
        "id": str(uuid.uuid4()),
        "title": "SMLP-KAN: Spectral MLP-KAN Diffusion Prior for Hyperspectral Image Restoration",
        "venue": "CVPR Workshop (PBVS) 2026",
        "authors": "Hongcheng Jiang et al.",
        "abstract": "We propose SMLP-KAN, an unsupervised framework combining spectral diffusion priors with MLP-KAN feature extraction. It consists of three components: an MLP-KAN Feature Extractor for hierarchical abstraction, a Spectral Diffusion Prior Module, and an Attentive Function Learning Mechanism. Achieves PSNR 34.74 dB on Botswana sharpening x2 and 31.68 dB on denoising. Model has only 9.5M parameters, 41x smaller than HIR-Diff and PLRDiff (391M).",
        "contributions": "Unsupervised hyperspectral restoration, MLP-KAN feature extractor, spectral diffusion prior, attentive function learning, 9.5M parameters"
    },
    {
        "id": str(uuid.uuid4()),
        "title": "THAT: Token-wise High-frequency Augmentation Transformer for Hyperspectral Pansharpening",
        "venue": "IEEE SMC 2025",
        "authors": "Hongkun Jin, Hongcheng Jiang, Zejun Zhang et al.",
        "abstract": "We propose THAT, a novel framework for hyperspectral pansharpening addressing token redundancy and high-frequency feature representation. THAT introduces Pivotal Token Selective Attention (PTSA) using k-means clustering to prioritize informative tokens, and Multi-level Variance-aware Feed-forward Network (MVFN) for high-frequency detail learning. Achieves PSNR 37.82 dB on PaviaU x2 with SSIM 0.9632. FLOPs 78.42G vs PLRDiff 22.43T.",
        "contributions": "Pivotal Token Selective Attention, k-means token selection, MVFN, 78.42G FLOPs, PSNR 37.82 dB PaviaU"
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Transformer-based Diffusion and Spectral Priors Model for Hyperspectral Pansharpening",
        "venue": "IEEE JSTARS 2025",
        "authors": "Hongcheng Jiang, Zhiqiang Chen",
        "abstract": "We propose uTDSP, an unsupervised transformer-based diffusion model for hyperspectral pansharpening. The model integrates spectral priors learned directly from LR-HSIs as a MAP regularization term. Achieves PSNR 31.61 dB on Botswana and 30.68 dB on PaviaU. Only 1.11M parameters and 0.44 GFLOPs total, 51000x more efficient than PLRDiff (22.43 TFLOPs).",
        "contributions": "Unsupervised pansharpening, spectral diffusion prior, MAP framework, 1.11M parameters, 0.44 GFLOPs"
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Spatial-Frequency Guided Pixel Transformer for NIR-to-RGB Translation",
        "venue": "Infrared Physics and Technology 2025",
        "authors": "Hongcheng Jiang, Zhiqiang Chen",
        "abstract": "We propose SF-GPT integrating DCT/DWT spatial-frequency decomposition with PixelUnshuffle pixel features via Spatial-Frequency Guided Multi-head Self-Attention (SFG-MSA). Achieves PSNR 26.09 dB and SSIM 0.77 on VCIP test dataset, outperforming ColorMamba (24.56 dB) and CoColor (23.54 dB) across 13 SOTA methods. LPIPS 0.132.",
        "contributions": "SF-GPT, SFG-MSA, DCT/DWT decomposition, PixelUnshuffle, PSNR 26.09 dB, LPIPS 0.132"
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution",
        "venue": "CVPR Workshop (PBVS) 2024",
        "authors": "Hongcheng Jiang, Zhiqiang Chen",
        "abstract": "We propose FW-SAT combining Flexible Window-based Self-attention (FWA) for regional features and Channel Spatial Attention Block (CSAB) for global context. Achieves PSNR 28.56 dB and SSIM 0.8698 at x8 scale, surpassing EDSR by 2.90 dB, SwinIR by 3.58 dB, HAN by 2.70 dB, and GRL by 2.97 dB on PBVS-2024 GTISR Challenge.",
        "contributions": "FW-SAT, flexible window self-attention, CSAB, PSNR 28.56 dB, outperforms EDSR SwinIR HAN GRL"
    }
]

# Create collection
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f"Created collection: {COLLECTION_NAME}")

# Index papers
points = []
for paper in papers:
    text = f"{paper['title']} {paper['abstract']} {paper['contributions']}"
    vector = model.encode(text).tolist()
    points.append(PointStruct(
        id=paper["id"],
        vector=vector,
        payload=paper
    ))

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Indexed {len(papers)} papers into '{COLLECTION_NAME}'")

for p in papers:
    print(f"  - {p['title'][:60]}...")
