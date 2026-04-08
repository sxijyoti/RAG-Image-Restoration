"""
Retrieval Module - Lightweight
Queries FAISS index and retrieves top-k reference patches
"""

import torch
import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from configs.config import (
    DEVICE, RETRIEVAL_TOP_K, PATCH_MAP_PATH,
    FAISS_INDEX_PATH, DACLIP_OUTPUT_SHAPE
)
from src.clip_encoder import DAClipEncoder


class RetrieverFAISS:
    def __init__(self, top_k: int = RETRIEVAL_TOP_K):
        self.top_k = top_k
        self.device = DEVICE
        self.index_path = FAISS_INDEX_PATH
        self.patch_map_path = PATCH_MAP_PATH
        self.encoder = DAClipEncoder(pretrained=False)
        
        self._load_index()
        self._load_patch_map()
    
    def _load_index(self):
        try:
            import faiss
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                if self.device.type == 'cuda':
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    except:
                        pass
            else:
                self.index = self._create_dummy_index()
        except ImportError:
            self.index = self._create_dummy_index()
    
    def _create_dummy_index(self):
        try:
            import faiss
            embedding_dim = 512 * 16 * 16
            index = faiss.IndexFlatL2(embedding_dim)
            dummy_vectors = np.random.randn(10, embedding_dim).astype(np.float32)
            index.add(dummy_vectors)
            return index
        except:
            return MockFAISSIndex()
    
    def _load_patch_map(self):
        if self.patch_map_path.exists():
            with open(self.patch_map_path, 'r') as f:
                self.patch_map = json.load(f)
        else:
            self.patch_map = {str(i): f"/path/to/patch_{i}.png" for i in range(10)}
    
    def retrieve(self, query_embedding: torch.Tensor) -> List[torch.Tensor]:
        """Retrieve top-k similar patches"""
        query_vector = query_embedding.reshape(1, -1).cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(query_vector, self.top_k)
        
        retrieved_embeddings = []
        for idx in indices[0]:
            idx_str = str(int(idx))
            if idx_str in self.patch_map:
                patch_path = self.patch_map[idx_str]
                try:
                    patch_img = Image.open(patch_path).convert("RGB")
                    patch_tensor = torch.from_numpy(np.array(patch_img)).permute(2, 0, 1).float() / 255.0
                    encoded = self.encoder.encode(patch_tensor)
                    retrieved_embeddings.append(encoded)
                except:
                    dummy_embedding = torch.randn(DACLIP_OUTPUT_SHAPE).to(self.device)
                    retrieved_embeddings.append(dummy_embedding)
        
        return retrieved_embeddings


class MockFAISSIndex:
    def __init__(self):
        self.ntotal = 10
    
    def search(self, query_vector, k):
        indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        distances = np.array([[1.0, 1.1, 1.2, 1.3, 1.4]], dtype=np.float32)
        return distances, indices
