```

找到这一行：
```
"venue": "Submitted to ICCV 2025",
```

改成：
```
"venue": "CVPR Workshop 2026","""
Load my own published papers into Qdrant for RAG demo.
Papers: PTST, SF-GPT, SMLP-KAN, HAT
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

COLLECTION = "my_research_papers"
VECTOR_SIZE = 384

if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

papers = [
    {
        "title": "Pivotal Token Selective Transformer for Medical Image Super-Resolution",
        "content": """Title: Pivotal Token Selective Transformer (PTST) for Medical Image Super-Resolution
        
Abstract: We propose PTST, a novel transformer architecture that enhances window-based self-attention 
for medical image super-resolution. By selectively identifying pivotal tokens that contribute most 
to image reconstruction, PTST reduces computational redundancy while maintaining high reconstruction quality. 
The method achieves 7.5% PSNR improvement (0.25 dB) over state-of-the-art methods while reducing 
computational cost by 30% FLOPs through pivotal token selection and multi-feature aggregation. 
The token selection mechanism identifies which tokens carry the most discriminative information, 
enabling efficient attention computation without sacrificing reconstruction quality.

Key contributions:
- Pivotal token selection mechanism for efficient self-attention
- 30% FLOPs reduction compared to standard window-based transformers  
- 7.5% PSNR improvement over SOTA on medical image super-resolution
- Multi-feature aggregation for enhanced reconstruction""",
        "venue": "NextTier ML Research, 2024",
        "authors": ["Hongcheng Jiang"],
    },
    {
        "title": "Spatial-Frequency Guided Pixel Transformer for NIR-to-RGB Translation",
        "content": """Title: Spatial-Frequency Guided Pixel Transformer (SF-GPT) for NIR-to-RGB Translation

Abstract: We present SF-GPT, a transformer-based approach for near-infrared to RGB image translation. 
The method integrates spatial-frequency domain features with pixel-level representations to capture 
both local details and global context. By decomposing images into low and high frequency components 
using DCT or DWT, and utilizing PixelUnshuffle for pixel feature extraction, SF-GPT achieves superior 
translation accuracy. We introduce SFG-MSA, a novel self-attention mechanism that enhances feature 
extraction from spatial-frequency and pixel representations simultaneously.

The method achieves 5.7% (1.5 dB) improvement in PSNR over state-of-the-art NIR-to-RGB translation methods.

Key contributions:
- Spatial-Frequency Guided Multi-head Self-Attention (SFG-MSA)
- DCT/DWT-based frequency domain decomposition for image translation
- PixelUnshuffle-based pixel feature extraction
- 5.7% PSNR improvement over SOTA on NIR-to-RGB translation""",
        "venue": "Infrared Physics and Technology, 2025",
        "authors": ["Hongcheng Jiang", "Zhiqiang Chen"],
    },
    {
        "title": "Diffusion-based Hyperspectral Image Restoration with SMLP-KAN",
        "content": """Title: SMLP-KAN: Diffusion-based Hyperspectral Image Restoration

Abstract: We propose SMLP-KAN, a framework that integrates spectral diffusion priors with 
attention-driven refinement and function learning for hyperspectral image restoration. 
The framework achieves 14% (+4.96 dB) PSNR gain for sharpening and 6% (+2.21 dB) for denoising 
on hyperspectral imaging benchmarks. Remarkably, SMLP-KAN reduces model complexity to 9.5M parameters,
over 40x smaller than PLRDiff (391M) and HIR-Diff (391M) while maintaining competitive performance.

MLP-KAN consistently outperforms both MLP and KAN across five benchmark datasets, achieving 
higher PSNR in x4 HSI sharpening (e.g., 32.95 vs 29.73/29.69 on the Botswana dataset).

Key contributions:
- Integration of spectral diffusion priors with attention-driven refinement
- 40x model compression compared to SOTA methods
- 14% PSNR improvement for hyperspectral sharpening
- Novel MLP-KAN architecture for function learning in image restoration""",
        "venue": "CVPR Workshop 2026",
        "authors": ["Hongcheng Jiang", "Zhiqiang Chen"],
    },
    {
        "title": "Flexible Window-based Self-attention Transformer for Thermal Image Super-Resolution",
        "content": """Title: Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution

Abstract: We present a flexible window-based self-attention transformer designed specifically for 
thermal image super-resolution. The method addresses the limitations of fixed window-based attention 
by introducing adaptive window partitioning that adjusts to image content. 
The Hybrid Attention Transformer (HAT) integrates window-based self-attention with adaptive pixel 
activation to improve efficiency. DCT-based preprocessing workflows ensure high-quality inputs,
achieving 4% (1.53 dB) improvement in PSNR compared to SOTA methods while reducing FLOPs by 20%.

Key contributions:
- Flexible window partitioning for adaptive self-attention
- DCT-based preprocessing for thermal image enhancement
- Integration with Hybrid Attention Transformer (HAT) from CVPR 2023
- 4% PSNR improvement with 20% FLOPs reduction""",
        "venue": "CVPR Workshop 2024",
        "authors": ["Hongcheng Jiang", "Zhiqiang Chen"],
    },
    {
        "title": "Hyperspectral Pansharpening with Transformer-based Spectral Diffusion Priors",
        "content": """Title: Hyperspectral Pansharpening with Transformer-based Spectral Diffusion Priors

Abstract: This paper presents a transformer-based diffusion model for hyperspectral pansharpening. 
By incorporating spectral diffusion priors into the pansharpening framework, the method effectively 
preserves spectral fidelity while enhancing spatial resolution. The approach leverages the power of 
diffusion models to generate high-quality hyperspectral images from low-resolution inputs guided by 
high-resolution panchromatic images. Presented at WACV 2025.

Key contributions:
- Spectral diffusion priors for hyperspectral pansharpening
- Transformer-based architecture for spatial-spectral feature fusion
- Superior spectral fidelity preservation compared to CNN-based methods
- Validated on multiple hyperspectral benchmark datasets""",
        "venue": "WACV Workshop 2025",
        "authors": ["Hongcheng Jiang", "Zhiqiang Chen"],
    },
]

points = []
for paper in papers:
    embedding = embedder.encode(paper["content"]).tolist()
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "content": paper["content"],
                "title": paper["title"],
                "venue": paper["venue"],
                "authors": paper["authors"],
            },
        )
    )

qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"Stored {len(points)} papers in '{COLLECTION}'")

# 验证
info = qdrant.get_collection(COLLECTION)
print(f"Collection '{COLLECTION}': {info.points_count} vectors")
