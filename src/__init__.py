"""
Retrieval-Augmented Image Restoration System

A modular pipeline for image restoration using:
- DA-CLIP for degradation-aware embeddings
- FAISS for efficient similarity search
- CNN decoder for patch reconstruction
- Intelligent overlap stitching
"""

__version__ = "1.0.0"
__author__ = "RAG-IR Team"

from .patching import extract_patches, load_image, get_image_shape
from .encoder import DAClipEncoder
from .retrieval import FAISSRetriever
from .fusion import EmbeddingFuser, AdvancedFuser
from .decoder import SimpleDecoder, PretrainedDecoder, load_decoder
from .stitching import PatchStitcher
from .pipeline import RestorationPipeline, run_pipeline

__all__ = [
    "extract_patches",
    "load_image",
    "get_image_shape",
    "DAClipEncoder",
    "FAISSRetriever",
    "EmbeddingFuser",
    "AdvancedFuser",
    "SimpleDecoder",
    "PretrainedDecoder",
    "load_decoder",
    "PatchStitcher",
    "RestorationPipeline",
    "run_pipeline",
]
