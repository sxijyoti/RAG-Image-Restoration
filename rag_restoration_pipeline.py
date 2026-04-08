"""
RAG-IR (Retrieval-Augmented Image Restoration) Pipeline.

Main module that orchestrates:
1. Patch extraction
2. CLIP encoding
3. FAISS retrieval
4. Image reconstruction

This is the query/inference pipeline. Runs on local Mac with MPS.
"""

import numpy as np
from typing import Tuple, List, Optional
from patch_extractor import extract_patches, reconstruct_image
from clip_encoder import CLIPPatchEncoder
from faiss_retriever import FAISSRetriever


class RAGRestorationPipeline:
    """
    End-to-end RAG-IR pipeline.
    
    Flow:
    1. Extract patches from degraded image (64x64, stride 32)
    2. Encode patches with CLIP ViT-B/32 (get token embeddings)
    3. Retrieve top-5 similar clean patches from dataset
    4. Fuse degraded + retrieved embeddings
    5. Decode to restore image
    """
    
    def __init__(
        self,
        device: str = None,
        top_k: int = 5,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            device: Device for inference ("cuda", "mps", "cpu")
            top_k: Number of similar patches to retrieve
        """
        self.patch_size = 64
        self.stride = 32
        self.top_k = top_k
        
        # Initialize components
        print("Initializing RAG-IR Pipeline...")
        self.encoder = CLIPPatchEncoder(device=device)
        self.retriever = FAISSRetriever(embedding_dim=768, top_k=top_k)
        
        print("✓ Pipeline components initialized")
    
    def set_retrieval_index(self, dataset_embeddings: np.ndarray) -> None:
        """
        Set the retrieval index from dataset embeddings.
        
        Args:
            dataset_embeddings: (N, 50, 768) embeddings from clean dataset
        """
        self.retriever.build_index(dataset_embeddings)
    
    def process_degraded_image(
        self,
        image_path_or_array,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Process degraded image: extract and encode patches.
        
        Args:
            image_path_or_array: Path to degraded image or numpy array
        
        Returns:
            query_embeddings: (M, 50, 768) token embeddings
            patches: List of original patches
            image_shape: Original image shape for reconstruction
        """
        print(f"\n{'='*70}")
        print("Processing degraded image")
        print(f"{'='*70}")
        
        # Extract patches
        patches, coordinates, image_shape = extract_patches(
            image_path_or_array,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        
        print(f"Extracted {len(patches)} patches from image {image_shape}")
        
        # Encode patches
        print("Encoding patches with CLIP...")
        query_embeddings = self.encoder.encode_patches(patches, return_numpy=True)
        print(f"Embeddings shape: {query_embeddings.shape}")
        
        return query_embeddings, patches, image_shape
    
    def retrieve_similar_patches(
        self,
        query_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve similar patches from clean dataset.
        
        Args:
            query_embeddings: (M, 50, 768) embeddings from degraded image
        
        Returns:
            distances: (M, k) L2 distances
            indices: (M, k) indices of retrieved patches
            retrieved_embeddings: (M, k, 50, 768) retrieved embeddings
        """
        print("\nRetrieving similar patches from dataset...")
        distances, indices = self.retriever.retrieve(query_embeddings)
        
        print(f"Retrieved {self.top_k} similar patches per query")
        print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
        
        # Get actual embeddings
        retrieved_embeddings = self.retriever.get_retrieved_patches(indices)
        
        return distances, indices, retrieved_embeddings
    
    def fuse_embeddings(
        self,
        query_embeddings: np.ndarray,
        retrieved_embeddings: np.ndarray,
        fusion_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Fuse degraded and retrieved embeddings.
        
        Simple fusion: weighted average
        query_fused = query * (1 - w) + mean(retrieved) * w
        
        Args:
            query_embeddings: (M, 50, 768) degraded embeddings
            retrieved_embeddings: (M, k, 50, 768) retrieved embeddings
            fusion_weight: Weight for retrieved patches (0-1)
        
        Returns:
            fused_embeddings: (M, 50, 768) fused embeddings
        """
        print(f"\nFusing embeddings (weight={fusion_weight})...")
        
        # Average retrieved embeddings per query
        mean_retrieved = retrieved_embeddings.mean(axis=1)  # (M, 50, 768)
        
        # Weighted fusion
        fused = query_embeddings * (1 - fusion_weight) + mean_retrieved * fusion_weight
        
        return fused
    
    def full_pipeline(
        self,
        degraded_image_path,
        fusion_weight: float = 0.5,
        return_intermediate: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        """
        Run complete RAG-IR pipeline.
        
        Args:
            degraded_image_path: Path to degraded image
            fusion_weight: Weight for retrieved embeddings in fusion
            return_intermediate: If True, return intermediate results
        
        Returns:
            results: Dictionary with pipeline outputs
            intermediate: Optional dict with intermediate values
        """
        # Step 1: Extract and encode
        query_emb, patches, img_shape = self.process_degraded_image(degraded_image_path)
        
        # Step 2: Retrieve
        distances, indices, retrieved_emb = self.retrieve_similar_patches(query_emb)
        
        # Step 3: Fuse
        fused_emb = self.fuse_embeddings(query_emb, retrieved_emb, fusion_weight)
        
        results = {
            "query_embeddings": query_emb.shape,
            "retrieved_embeddings": retrieved_emb.shape,
            "fused_embeddings": fused_emb.shape,
            "num_patches": len(patches),
            "image_shape": img_shape,
            "top_k_retrieved": self.top_k,
            "mean_distance": distances.mean(),
        }
        
        print(f"\n{'='*70}")
        print("Pipeline Results")
        print(f"{'='*70}")
        for key, val in results.items():
            print(f"  {key}: {val}")
        
        intermediate = None
        if return_intermediate:
            intermediate = {
                "query_embeddings": query_emb,
                "fused_embeddings": fused_emb,
                "retrieved_indices": indices,
                "distances": distances,
                "patches": patches,
            }
        
        return results, intermediate
    
    def get_config(self) -> dict:
        """Get pipeline configuration."""
        return {
            "patch_size": self.patch_size,
            "stride": self.stride,
            "top_k_retrieval": self.top_k,
            "encoder_config": self.encoder.get_config(),
            "retriever_config": self.retriever.get_config(),
        }


def create_pipeline(device: str = None, top_k: int = 5) -> RAGRestorationPipeline:
    """Factory function to create pipeline."""
    return RAGRestorationPipeline(device=device, top_k=top_k)


if __name__ == "__main__":
    print("=" * 70)
    print("RAG-IR PIPELINE TEST")
    print("=" * 70)
    
    try:
        # Create pipeline
        pipeline = create_pipeline(top_k=5)
        
        # Show config
        print("\nPipeline Config:")
        config = pipeline.get_config()
        for key, val in config.items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for k, v in val.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {val}")
        
        print("\n✓ Pipeline initialized successfully!")
        print("\nNote: Full pipeline requires:")
        print("  1. Degraded image")
        print("  2. Dataset embeddings from clean patches")
        print("  3. FAISS index built from dataset")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Ensure transformers, torch, and faiss-cpu are installed")
