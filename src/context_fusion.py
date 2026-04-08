"""
Context Fusion Module - Lightweight
Fuses query patch with retrieved reference patches using cross-attention
"""

import torch
import torch.nn as nn
from typing import List

from configs.config import (
    DEVICE,
    DACLIP_OUTPUT_SHAPE,
    FUSION_LAYERS
)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = FUSION_LAYERS):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = DEVICE
        
        self.cross_attn_layers = nn.ModuleList()
        self.layer_norms_1 = nn.ModuleList()
        self.layer_norms_2 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            )
            self.layer_norms_1.append(nn.LayerNorm(embed_dim))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                nn.ReLU(),
                nn.Linear(2 * embed_dim, embed_dim)
            ))
        
        self.to(self.device)
    
    def forward(self, degraded_emb: torch.Tensor, retrieved_embs: List[torch.Tensor]) -> torch.Tensor:
        """Fuse degraded and retrieved embeddings: (1, 512, 16, 16) -> (1, 512, 16, 16)"""
        B, C, H, W = degraded_emb.shape
        query = degraded_emb.flatten(2).transpose(1, 2)  # (1, 256, 512)
        
        kv = torch.cat([
            emb.flatten(2).transpose(1, 2) for emb in retrieved_embs
        ], dim=1)  # (1, k*256, 512)
        
        x = query
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](x, kv, kv)
            x = self.layer_norms_1[i](x + attn_out)
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms_2[i](x + ffn_out)
        
        return x.transpose(1, 2).reshape(B, C, H, W)


class ContextFusionPipeline:
    def __init__(self, num_layers: int = FUSION_LAYERS, embed_dim: int = 512):
        self.fusion = CrossAttentionFusion(embed_dim, num_layers=num_layers)
        self.device = DEVICE
    
    def fuse(self, degraded_emb: torch.Tensor, retrieved_embs: List[torch.Tensor]) -> torch.Tensor:
        """Fuse degraded with retrieved embeddings"""
        assert degraded_emb.shape == torch.Size(DACLIP_OUTPUT_SHAPE)
        self.fusion.eval()
        with torch.no_grad():
            fused = self.fusion(degraded_emb, retrieved_embs)
        return fused
    
    def fuse_batch(self, degraded_embs: List[torch.Tensor], 
                   retrieved_batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Batch fuse multiple degraded embeddings with their retrieved contexts"""
        return [
            self.fuse(deg_emb, ret_embs)
            for deg_emb, ret_embs in zip(degraded_embs, retrieved_batch)
        ]

