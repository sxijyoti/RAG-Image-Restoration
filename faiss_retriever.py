"""
FAISS-based retrieval for similar patches.

Builds index of clean patch embeddings for fast retrieval during inference.
"""

import numpy as np
import faiss
from typing import List, Tuple


class FAISSRetriever:
    """
    FAISS-based retriever for patch embeddings.
    
    Uses L2 distance to find top-k similar clean patches for each query patch.
    """
    
    def __init__(self, embedding_dim: int = 768, top_k: int = 5):
        """
        Initialize retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
            top_k: Number of similar patches to retrieve
        """
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.index = None
        self.embeddings = None
        self.is_trained = False
    
    def build_index(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Build FAISS index from dataset embeddings.
        
        Args:
            embeddings: (N, 50, 768) token embeddings from clean dataset
        """
        # Flatten token embeddings to create single vector per patch
        # (N, 50, 768) -> (N, 50*768)
        N, num_tokens, embed_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(N, -1)
        
        print(f"Building FAISS index...")
        print(f"  Dataset size: {N} patches")
        print(f"  Flattened embedding dim: {flat_embeddings.shape[1]}")
        
        # Create L2 index
        self.index = faiss.IndexFlatL2(flat_embeddings.shape[1])
        self.index.add(flat_embeddings.astype(np.float32))
        
        self.embeddings = embeddings
        self.is_trained = True
        
        print(f"  ✓ Index built with {self.index.ntotal} patches")
    
    def retrieve(
        self,
        query_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k similar patches for each query.
        
        Args:
            query_embeddings: (M, 50, 768) token embeddings from degraded image
        
        Returns:
            distances: (M, k) L2 distances to retrieved patches
            indices: (M, k) indices of retrieved patches in dataset
        """
        if not self.is_trained:
            raise ValueError("Index not built. Call build_index() first.")
        
        M, num_tokens, embed_dim = query_embeddings.shape
        
        # Flatten query embeddings
        flat_queries = query_embeddings.reshape(M, -1).astype(np.float32)
        
        # Retrieve
        distances, indices = self.index.search(flat_queries, self.top_k)
        
        return distances, indices
    
    def get_retrieved_patches(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        Get the actual retrieved patch embeddings.
        
        Args:
            indices: (M, k) indices from retrieve()
        
        Returns:
            patches: (M, k, 50, 768) retrieved patch embeddings
        """
        M, k = indices.shape
        _, num_tokens, embed_dim = self.embeddings.shape
        
        retrieved = np.zeros((M, k, num_tokens, embed_dim), dtype=np.float32)
        
        for i in range(M):
            for j in range(k):
                patch_idx = indices[i, j]
                retrieved[i, j] = self.embeddings[patch_idx]
        
        return retrieved
    
    def get_config(self) -> dict:
        """Get retriever configuration."""
        return {
            "embedding_dim": self.embedding_dim,
            "top_k": self.top_k,
            "is_trained": self.is_trained,
            "dataset_size": self.index.ntotal if self.index else 0,
            "metric": "L2",
        }


def create_retriever(top_k: int = 5) -> FAISSRetriever:
    """Factory function to create retriever."""
    return FAISSRetriever(embedding_dim=768, top_k=top_k)


if __name__ == "__main__":
    print("=" * 70)
    print("FAISS RETRIEVER TEST")
    print("=" * 70)
    
    # Create dummy dataset
    print("\nTest 1: Build index")
    dataset_embeddings = np.random.randn(100, 50, 768).astype(np.float32)
    
    retriever = FAISSRetriever(top_k=5)
    retriever.build_index(dataset_embeddings)
    
    config = retriever.get_config()
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Test retrieval
    print("\nTest 2: Retrieve similar patches")
    query_embeddings = np.random.randn(10, 50, 768).astype(np.float32)
    distances, indices = retriever.retrieve(query_embeddings)
    
    print(f"  Queries: {query_embeddings.shape[0]}")
    print(f"  Retrieved per query: {retriever.top_k}")
    print(f"  Distances shape: {distances.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Sample distances (first query): {distances[0]}")
    print(f"  Sample indices (first query): {indices[0]}")
    
    # Get actual embeddings
    print("\nTest 3: Get retrieved embeddings")
    retrieved = retriever.get_retrieved_patches(indices)
    print(f"  Retrieved shape: {retrieved.shape}")
    print(f"  Expected: (10, 5, 50, 768)")
    
    print("\n✓ All tests passed!")
