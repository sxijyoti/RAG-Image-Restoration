"""
Decoder module for converting embeddings back to image patches.

Implements a simple CNN decoder architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SimpleDecoder(nn.Module):
    """Simple CNN decoder: embedding -> patch."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        patch_size: int = 64
    ):
        """
        Initialize decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for intermediate layers
            patch_size: Output patch size (patch_size x patch_size)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # Expansion from embedding to spatial feature map
        # 512 -> flatten for conv layers
        self.fc = nn.Linear(embedding_dim, hidden_dim * 16 * 16)
        
        # 4 transposed convolution blocks
        self.decoder = nn.Sequential(
            # Input: (batch, hidden_dim, 16, 16)
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim // 2),
            
            # Additional refinement
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim // 2),
            
            # Upsample to 64x64
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim // 4),
            
            # Final refinement
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim // 4),
            
            # Output: 3 channels
            nn.Conv2d(hidden_dim // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to patch.
        
        Args:
            embedding: Input embedding of shape (batch_size, embedding_dim) or (embedding_dim,)
            
        Returns:
            Decoded patch of shape (batch_size, 3, 64, 64) or (3, 64, 64)
        """
        single_sample = embedding.dim() == 1
        if single_sample:
            embedding = embedding.unsqueeze(0)
        
        batch_size = embedding.shape[0]
        
        # Expand to spatial features
        x = self.fc(embedding)  # (batch_size, hidden_dim * 16 * 16)
        x = x.view(batch_size, self.hidden_dim, 16, 16)
        
        # Decode
        patch = self.decoder(x)  # (batch_size, 3, 64, 64)
        
        if single_sample:
            patch = patch.squeeze(0)
        
        return patch


class PretrainedDecoder(nn.Module):
    """
    Decoder using pre-trained backbone (for future use).
    Can be initialized with pretrained weights if available.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        patch_size: int = 64,
        pretrained: bool = False
    ):
        """
        Initialize decoder.
        
        Args:
            embedding_dim: Dimension of input embeddings
            patch_size: Output patch size
            pretrained: Whether to load pretrained weights (placeholder)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        
        # For now, use simple decoder
        self.decoder = SimpleDecoder(embedding_dim, 256, patch_size)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to patch.
        
        Args:
            embedding: Input embedding tensor
            
        Returns:
            Decoded patch tensor in [0, 1]
        """
        return self.decoder(embedding)


def load_decoder(
    checkpoint_path: Optional[str] = None,
    embedding_dim: int = 512,
    patch_size: int = 64,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load decoder model.
    
    Args:
        checkpoint_path: Path to checkpoint file (optional)
        embedding_dim: Embedding dimension
        patch_size: Patch size
        device: Device to load on
        
    Returns:
        Decoder model
    """
    decoder = SimpleDecoder(embedding_dim, 256, patch_size)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            decoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            decoder.load_state_dict(checkpoint)
        print(f"Loaded decoder from {checkpoint_path}")
    else:
        print("Warning: Using untrained decoder. Performance may be poor.")
    
    decoder.to(device)
    decoder.eval()
    
    return decoder
