"""
Minimal configuration for RAG Image Restoration Pipeline
"""

import torch
from pathlib import Path

# Device auto-detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
INDEXES_DIR = PROJECT_ROOT / "indexes"
TENSORS_DIR = PROJECT_ROOT / "tensors"
SRC_DIR = PROJECT_ROOT / "src"

PATCH_MAP_PATH = DATA_DIR / "patch_map.json"
FAISS_INDEX_PATH = INDEXES_DIR / "clean_patches.index"

# Patch Segmentation
PATCH_SIZE = 64
PATCH_STRIDE = 32

# DA-CLIP Encoder
DACLIP_EMBED_DIM = 512
DACLIP_IMAGE_SIZE = 512
DACLIP_PATCH_SIZE = 16
DACLIP_OUTPUT_SHAPE = (1, DACLIP_EMBED_DIM, DACLIP_PATCH_SIZE, DACLIP_PATCH_SIZE)

# Retrieval
RETRIEVAL_TOP_K = 5

# Context Fusion
FUSION_LAYERS = 2
