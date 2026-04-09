"""
Phase 5: Context Fusion Module for DA-CLIP Image Restoration

Fuses retrieved patch embeddings with query embedding to create a rich context
representation for the decoder.

Three fusion strategies:
1. Mean Fusion - Simple average (baseline)
2. Concatenation + Linear Projection - Learnable weighted combination
3. Cross-Attention - Query attends to retrieved patches

Constraints:
- Input: query embedding (1, 512), retrieved embeddings (k, 512)
- Output: fused embedding (1, 512, 16, 16) or (1, 8192)
- Efficient (CPU/MPS compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path


class MeanFusion(nn.Module):
    """
    Mean Fusion Strategy (Baseline)
    
    PROS:
    - Simplest to implement
    - No learnable parameters
    - Very fast
    - Works well as baseline
    
    CONS:
    - Treats all retrieved patches equally
    - No learned weighting
    - Loses individual patch distinctiveness
    - May lose important details
    
    Use when: You want a simple baseline or have limited data
    """
    
    def __init__(self, embedding_dim: int = 512, debug: bool = True):
        """
        Initialize Mean Fusion.
        
        Args:
            embedding_dim: Embedding dimension (default 512)
            debug: Print debug info
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.debug = debug
        
        if debug:
            print("Mean Fusion initialized (baseline strategy)")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse embeddings by simple averaging.
        
        Args:
            query_embedding: (1, 512)
            retrieved_embeddings: (k, 512)
            
        Returns:
            fused_embedding: (1, 512)
        """
        # Stack all embeddings (including query)
        all_embeddings = torch.cat([query_embedding, retrieved_embeddings], dim=0)  # (k+1, 512)
        
        # Simple mean
        fused = all_embeddings.mean(dim=0, keepdim=True)  # (1, 512)
        
        return fused


class ConcatProjectionFusion(nn.Module):
    """
    Concatenation + Linear Projection Strategy
    
    PROS:
    - Learnable weights for each patch
    - Flexible representation
    - Can learn importance of each patch
    - Good balance of simplicity and power
    
    CONS:
    - More parameters (512 * k -> 512)
    - Needs training data
    - Higher memory during training
    - Requires backprop to learn
    
    Use when: You have labeled data and want to learn optimal fusion
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_retrieved: int = 5,
        hidden_dim: int = 256,
        debug: bool = True
    ):
        """
        Initialize Concatenation + Projection Fusion.
        
        Args:
            embedding_dim: Embedding dimension (512)
            num_retrieved: Number of retrieved patches (k)
            hidden_dim: Hidden dimension in projection MLP
            debug: Print debug info
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_retrieved = num_retrieved
        self.hidden_dim = hidden_dim
        
        # MLP for projecting concatenated embeddings back to embedding_dim
        concat_dim = embedding_dim * (num_retrieved + 1)  # query + k retrieved
        
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        if debug:
            print(f"ConcatProjection Fusion initialized")
            print(f"  Input: {concat_dim} (query + {num_retrieved} retrieved)")
            print(f"  Hidden: {hidden_dim}")
            print(f"  Output: {embedding_dim}")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse by concatenating and projecting through MLP.
        
        Args:
            query_embedding: (1, 512)
            retrieved_embeddings: (k, 512)
            
        Returns:
            fused_embedding: (1, 512)
        """
        # Concatenate query with retrieved
        concat = torch.cat([query_embedding, retrieved_embeddings], dim=0)  # (k+1, 512)
        concat_flat = concat.flatten()  # ((k+1)*512,)
        
        # Project through MLP
        fused = self.projection(concat_flat)  # (512,)
        fused = fused.unsqueeze(0)  # (1, 512)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Strategy
    
    PROS:
    - Query attends to retrieved patches
    - Can learn which patches are important
    - Interpretable attention weights
    - Most flexible/powerful
    - Mimics transformer architecture
    
    CONS:
    - Most parameters
    - Requires training
    - Slower than simple methods
    - Complex to debug
    - May overfit on small data
    
    Use when: You have enough labeled data and want state-of-the-art fusion
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        debug: bool = True
    ):
        """
        Initialize Cross-Attention Fusion.
        
        Args:
            embedding_dim: Embedding dimension (512)
            num_heads: Number of attention heads
            debug: Print debug info
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Optional: Post-attention projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        if debug:
            print(f"CrossAttention Fusion initialized")
            print(f"  Embedding dim: {embedding_dim}")
            print(f"  Heads: {num_heads}")
            print(f"  Head dim: {embedding_dim // num_heads}")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse using cross-attention (query attends to retrieved).
        
        Args:
            query_embedding: (1, 512) - query (key, value)
            retrieved_embeddings: (k, 512) - context to attend to
            
        Returns:
            fused_embedding: (1, 512)
            attention_weights: (1, k) - attention distribution
        """
        # Cross-attention: query attends to retrieved patches
        # query=query_embedding, key=retrieved, value=retrieved
        attn_output, attn_weights = self.attention(
            query=query_embedding,           # (1, 512)
            key=retrieved_embeddings,        # (k, 512)
            value=retrieved_embeddings,      # (k, 512)
        )
        # attn_output: (1, 512)
        # attn_weights: (1, k)
        
        # Optional projection
        fused = self.projection(attn_output)  # (1, 512)
        
        return fused, attn_weights


class ContextFusionPipeline(nn.Module):
    """
    Complete Context Fusion Pipeline
    
    Combines query embedding with retrieved embeddings using selected strategy.
    Outputs spatial feature map suitable for decoder.
    """
    
    def __init__(
        self,
        strategy: str = "mean",
        embedding_dim: int = 512,
        num_retrieved: int = 5,
        output_spatial: bool = True,
        spatial_size: int = 16,
        debug: bool = True
    ):
        """
        Initialize Context Fusion Pipeline.
        
        Args:
            strategy: "mean", "concat", or "attention"
            embedding_dim: Embedding dimension (512)
            num_retrieved: Number of retrieved patches
            output_spatial: Whether to reshape to spatial (1, 512, 16, 16)
            spatial_size: Height/width of spatial output (16)
            debug: Print debug info
        """
        super().__init__()
        self.strategy = strategy
        self.embedding_dim = embedding_dim
        self.num_retrieved = num_retrieved
        self.output_spatial = output_spatial
        self.spatial_size = spatial_size
        self.debug = debug
        
        # Select fusion strategy
        if strategy == "mean":
            self.fusion = MeanFusion(embedding_dim, debug=debug)
        elif strategy == "concat":
            self.fusion = ConcatProjectionFusion(
                embedding_dim, num_retrieved, hidden_dim=256, debug=debug
            )
        elif strategy == "attention":
            self.fusion = CrossAttentionFusion(embedding_dim, num_heads=8, debug=debug)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if debug:
            print(f"\nContextFusionPipeline initialized")
            print(f"  Strategy: {strategy}")
            print(f"  Embedding dim: {embedding_dim}")
            print(f"  Num retrieved: {num_retrieved}")
            print(f"  Output spatial: {output_spatial} ({spatial_size}x{spatial_size})")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fuse context.
        
        Args:
            query_embedding: (1, 512)
            retrieved_embeddings: (k, 512)
            
        Returns:
            If strategy="attention": (fused, attention_weights)
                - fused: (1, 512) or (1, 512, 16, 16)
                - attention_weights: (1, k)
            Else: fused (1, 512) or (1, 512, 16, 16)
        """
        # Validate shapes
        assert query_embedding.shape == (1, self.embedding_dim), \
            f"Query shape {query_embedding.shape} != (1, {self.embedding_dim})"
        assert retrieved_embeddings.shape[0] > 0, "No retrieved embeddings"
        assert retrieved_embeddings.shape[1] == self.embedding_dim, \
            f"Retrieved shape {retrieved_embeddings.shape} incompatible"
        
        # Fuse
        if self.strategy == "attention":
            fused, attn_weights = self.fusion(query_embedding, retrieved_embeddings)
        else:
            fused = self.fusion(query_embedding, retrieved_embeddings)
            attn_weights = None
        
        # Optional: reshape to spatial
        if self.output_spatial:
            # (1, 512) -> (1, 512, 16, 16) requires 131,072 dims
            # Tile the embedding across spatial dimensions
            # Option 1: Repeat embedding across spatial positions
            fused_spatial = fused.unsqueeze(-1).unsqueeze(-1)  # (1, 512, 1, 1)
            fused = fused_spatial.repeat(1, 1, self.spatial_size, self.spatial_size)  # (1, 512, 16, 16)
        
        if attn_weights is not None:
            return fused, attn_weights
        else:
            return fused


# Utility function for comparison
def compare_fusion_strategies(
    query_embedding: torch.Tensor,
    retrieved_embeddings: torch.Tensor,
    device: str = "cpu"
) -> dict:
    """
    Compare all three fusion strategies on same input.
    
    Args:
        query_embedding: (1, 512)
        retrieved_embeddings: (k, 512)
        device: "cpu" or "cuda"
        
    Returns:
        Dictionary with results from each strategy
    """
    query_embedding = query_embedding.to(device)
    retrieved_embeddings = retrieved_embeddings.to(device)
    
    results = {}
    
    # Mean Fusion
    mean_fusion = MeanFusion(embedding_dim=512, debug=False).to(device)
    mean_output = mean_fusion(query_embedding, retrieved_embeddings)
    results["mean"] = {
        "output": mean_output,
        "params": sum(p.numel() for p in mean_fusion.parameters()),
        "strategy": "Simple average"
    }
    
    # Concat Fusion
    concat_fusion = ConcatProjectionFusion(
        embedding_dim=512,
        num_retrieved=retrieved_embeddings.shape[0],
        debug=False
    ).to(device)
    concat_output = concat_fusion(query_embedding, retrieved_embeddings)
    results["concat"] = {
        "output": concat_output,
        "params": sum(p.numel() for p in concat_fusion.parameters()),
        "strategy": "Concatenation + MLP"
    }
    
    # Attention Fusion
    attn_fusion = CrossAttentionFusion(embedding_dim=512, debug=False).to(device)
    attn_output, attn_weights = attn_fusion(query_embedding, retrieved_embeddings)
    results["attention"] = {
        "output": attn_output,
        "attn_weights": attn_weights,
        "params": sum(p.numel() for p in attn_fusion.parameters()),
        "strategy": "Cross-Attention"
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create dummy data
    query_emb = torch.randn(1, 512, device=device)
    retrieved_embs = torch.randn(5, 512, device=device)  # k=5 patches
    
    print("="*70)
    print("CONTEXT FUSION - COMPARISON")
    print("="*70)
    
    # Compare strategies
    results = compare_fusion_strategies(query_emb, retrieved_embs, device=device)
    
    print("\nStrategy Comparison:")
    print(f"{'Strategy':<15} {'Params':<10} {'Output Shape':<20}")
    print("-" * 45)
    for name, result in results.items():
        params = result["params"]
        output_shape = tuple(result["output"].shape)
        print(f"{name:<15} {params:<10} {str(output_shape):<20}")
    
    # Test with spatial output
    print("\n" + "="*70)
    print("WITH SPATIAL OUTPUT (1, 512, 16, 16)")
    print("="*70)
    
    pipeline = ContextFusionPipeline(
        strategy="attention",
        embedding_dim=512,
        num_retrieved=5,
        output_spatial=True,
        spatial_size=16,
        debug=True
    )
    pipeline = pipeline.to(device)
    
    fused, attn_weights = pipeline(query_emb, retrieved_embs)
    
    print(f"\nFused embedding shape: {fused.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention distribution: {attn_weights[0].detach().cpu().numpy()}")
