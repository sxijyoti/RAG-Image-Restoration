"""
Fusion module for combining query and retrieved patch embeddings.

Implements baseline mean fusion and optional advanced fusion strategies.
"""

import torch
import torch.nn as nn
from typing import Literal


class EmbeddingFuser:
    """Fuses query embedding with retrieved patch embeddings."""
    
    def __init__(self, method: Literal["mean", "concat"] = "mean"):
        """
        Initialize embedding fuser.
        
        Args:
            method: Fusion method ("mean" for averaging, "concat" for concatenation)
        """
        self.method = method
        
        if method not in ["mean", "concat"]:
            raise ValueError(f"Unknown method: {method}")
    
    def fuse(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse query embedding with retrieved embeddings.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            retrieved_embeddings: Retrieved embeddings of shape (k, embedding_dim)
            
        Returns:
            Fused embedding of shape (embedding_dim,) or (concat_dim,)
        """
        if self.method == "mean":
            return self._fuse_mean(query_embedding, retrieved_embeddings)
        elif self.method == "concat":
            return self._fuse_concat(query_embedding, retrieved_embeddings)
    
    @staticmethod
    def _fuse_mean(
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple mean fusion baseline.
        
        Args:
            query_embedding: Shape (embedding_dim,)
            retrieved_embeddings: Shape (k, embedding_dim)
            
        Returns:
            Fused embedding of shape (embedding_dim,)
        """
        # Stack query with retrieved embeddings
        all_embeddings = torch.stack([query_embedding] + list(retrieved_embeddings))
        
        # Compute mean
        fused = all_embeddings.mean(dim=0)
        
        return fused
    
    @staticmethod
    def _fuse_concat(
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenation-based fusion (requires post-processing with linear layer).
        
        Args:
            query_embedding: Shape (embedding_dim,)
            retrieved_embeddings: Shape (k, embedding_dim)
            
        Returns:
            Concatenated embedding of shape (1 + k) * embedding_dim
        """
        # Flatten and concatenate
        all_embeddings = torch.cat([
            query_embedding.unsqueeze(0),
            retrieved_embeddings
        ], dim=0).flatten()
        
        return all_embeddings


class AdvancedFuser(nn.Module):
    """Learnable fusion module (for future training)."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_retrieved: int = 5
    ):
        """
        Initialize advanced fuser with learned weights.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_retrieved: Number of retrieved samples
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_retrieved = num_retrieved
        
        # Learned attention weights for retrieved samples
        self.retrieved_weights = nn.Parameter(
            torch.ones(num_retrieved) / num_retrieved
        )
        
        # Fusion gate (mix between query and retrieved)
        self.gate = nn.Linear(embedding_dim * 2, 1)
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with learned fusion.
        
        Args:
            query_embedding: Shape (embedding_dim,) or (batch_size, embedding_dim)
            retrieved_embeddings: Shape (k, embedding_dim) or (batch_size, k, embedding_dim)
            
        Returns:
            Fused embedding
        """
        # Handle batch dimension
        batch_mode = query_embedding.dim() == 2
        
        if batch_mode:
            batch_size = query_embedding.shape[0]
            # Fuse for each sample in batch
            fused_list = []
            for i in range(batch_size):
                fused = self._fuse_single(
                    query_embedding[i],
                    retrieved_embeddings[i]
                )
                fused_list.append(fused)
            return torch.stack(fused_list)
        else:
            return self._fuse_single(query_embedding, retrieved_embeddings)
    
    def _fuse_single(
        self,
        query_embedding: torch.Tensor,
        retrieved_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse single query with retrieved embeddings.
        
        Args:
            query_embedding: Shape (embedding_dim,)
            retrieved_embeddings: Shape (k, embedding_dim)
            
        Returns:
            Fused embedding of shape (embedding_dim,)
        """
        # Softmax weights for retrieved samples
        retrieval_weights = torch.softmax(self.retrieved_weights, dim=0)
        
        # Weighted mean of retrieved embeddings
        weighted_retrieved = (retrieved_embeddings * retrieval_weights.view(-1, 1)).sum(dim=0)
        
        # Gate for mixing query and retrieval
        gate_input = torch.cat([query_embedding, weighted_retrieved])
        gate_value = torch.sigmoid(self.gate(gate_input))
        
        # Fused embedding
        fused = gate_value * query_embedding + (1 - gate_value) * weighted_retrieved
        
        return fused
