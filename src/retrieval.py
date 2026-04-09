"""
Retrieval module using FAISS for similar patch retrieval.

Handles FAISS index loading and searching, patch metadata mapping.
"""

import json
import numpy as np
import faiss
import torch
from typing import List, Tuple, Dict
from PIL import Image


class FAISSRetriever:
    """FAISS-based retriever for similar patch lookup."""
    
    def __init__(self, index_path: str, patch_map_path: str):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file (.index)
            patch_map_path: Path to patch map JSON file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load patch map
        with open(patch_map_path, 'r') as f:
            self.patch_map = json.load(f)
        
        # Ensure index is on CPU for compatibility
        self.index = faiss.index_cpu_to_all_gpus(self.index) if False else self.index
        
        self.index_path = index_path
        self.patch_map_path = patch_map_path
        
        print(f"Loaded FAISS index: {index_path}")
        print(f"Index size: {self.index.ntotal} vectors")
        print(f"Loaded {len(self.patch_map)} patch mappings")
    
    def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors in the index.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,), torch tensor
            k: Number of nearest neighbors to retrieve
            
        Returns:
            distances: Array of shape (k,) - L2 distances
            indices: Array of shape (k,) - FAISS index IDs
        """
        # Convert to numpy and add batch dimension
        query_np = query_embedding.numpy().reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        return distances[0], indices[0]
    
    def search_batch(
        self,
        query_embeddings: torch.Tensor,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.
        
        Args:
            query_embeddings: Query embeddings of shape (batch_size, embedding_dim), torch tensor
            k: Number of nearest neighbors to retrieve
            
        Returns:
            distances: Array of shape (batch_size, k)
            indices: Array of shape (batch_size, k)
        """
        # Convert to numpy
        query_np = query_embeddings.numpy().astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        return distances, indices
    
    def get_patch_metadata(self, faiss_idx: int) -> Dict:
        """
        Get metadata for a patch by FAISS index ID.
        
        Args:
            faiss_idx: FAISS index ID
            
        Returns:
            Dictionary with keys: 'image_path', 'x', 'y', 'size'
        """
        if faiss_idx >= len(self.patch_map):
            raise IndexError(f"FAISS index {faiss_idx} out of range")
        
        return self.patch_map[str(faiss_idx)]
    
    def load_retrieved_patch(
        self,
        faiss_idx: int
    ) -> np.ndarray:
        """
        Load the actual patch image data.
        
        Args:
            faiss_idx: FAISS index ID
            
        Returns:
            Patch array of shape (3, 64, 64), float32 in [0, 1]
        """
        metadata = self.get_patch_metadata(faiss_idx)
        
        image_path = metadata['image_path']
        x = metadata['x']
        y = metadata['y']
        size = metadata['size']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Crop patch
        patch = image_array[y:y + size, x:x + size, :]
        
        # Convert to (C, H, W)
        patch_chw = np.transpose(patch, (2, 0, 1))
        
        return patch_chw
    
    def load_retrieved_patches(
        self,
        faiss_indices: np.ndarray
    ) -> np.ndarray:
        """
        Load multiple retrieved patches.
        
        Args:
            faiss_indices: Array of FAISS index IDs of shape (k,) or (batch_size, k)
            
        Returns:
            Array of patches of shape (k, 3, 64, 64) or (batch_size, k, 3, 64, 64)
        """
        original_shape = faiss_indices.shape
        faiss_indices_flat = faiss_indices.flatten()
        
        patches = []
        for idx in faiss_indices_flat:
            patch = self.load_retrieved_patch(int(idx))
            patches.append(patch)
        
        patches_array = np.array(patches, dtype=np.float32)
        
        # Reshape to original structure
        if len(original_shape) == 2:
            patches_array = patches_array.reshape(original_shape[0], original_shape[1], 3, 64, 64)
        
        return patches_array
