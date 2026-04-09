"""
Encoder module using DA-CLIP for degradation-aware patch embeddings.

Handles model loading, preprocessing, and batch encoding of patches.
"""

import torch
import numpy as np
from typing import Tuple, Union, List
import open_clip


class DAClipEncoder:
    """DA-CLIP based encoder for degradation-aware image patch embeddings."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = None
    ):
        """
        Initialize DA-CLIP encoder.
        
        Args:
            model_name: Model architecture (default: ViT-B-32)
            pretrained: Pretrained weights variant (default: openai)
            device: Device to use ('cuda', 'cpu', 'mps'). Auto-detect if None.
        """
        if device is None:
            device = self._detect_device()
        
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load DA-CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()
        
        print(f"Loaded {model_name} on device: {device}")
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def encode_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Encode a single patch.
        
        Args:
            patch: Patch array of shape (3, 64, 64) or (64, 64, 3), float32 in [0, 1]
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        # Handle channel ordering
        if patch.shape[0] != 3:
            patch = np.transpose(patch, (2, 0, 1))
        
        # Convert to PIL Image for preprocessing
        patch_uint8 = (patch * 255).astype(np.uint8)
        patch_uint8 = np.transpose(patch_uint8, (1, 2, 0))  # (H, W, C)
        
        from PIL import Image
        image = Image.fromarray(patch_uint8, mode='RGB')
        
        # Preprocess and encode
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Critical: Use control=True for degradation-aware features
            degra_features = self.model.encode_image(image_input, control=True)
        
        return degra_features.squeeze(0).detach().cpu()
    
    def encode_batch(self, patches: np.ndarray) -> torch.Tensor:
        """
        Encode a batch of patches.
        
        Args:
            patches: Batch of patches, shape (batch_size, 3, 64, 64) or (batch_size, 64, 64, 3),
                    float32 in [0, 1]
            
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        batch_size = patches.shape[0]
        embeddings = []
        
        for i in range(batch_size):
            patch = patches[i]
            embedding = self.encode_patch(patch)
            embeddings.append(embedding)
        
        # Stack embeddings
        embeddings_tensor = torch.stack(embeddings, dim=0)
        
        return embeddings_tensor
    
    def encode_batch_optimized(self, patches: np.ndarray) -> torch.Tensor:
        """
        Encode a batch of patches with better memory management.
        
        Args:
            patches: Batch of patches, shape (batch_size, 3, 64, 64), float32 in [0, 1]
            
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Handle channel ordering
        if patches.shape[1] != 3:
            patches = np.transpose(patches, (0, 3, 1, 2))
        
        batch_size = patches.shape[0]
        embeddings = []
        
        # Process in smaller chunks to manage memory
        chunk_size = 16  # Adjust based on available memory
        
        for i in range(0, batch_size, chunk_size):
            chunk = patches[i:i + chunk_size]
            chunk_embeddings = []
            
            for j in range(len(chunk)):
                patch = chunk[j]
                embedding = self.encode_patch(patch)
                chunk_embeddings.append(embedding)
            
            embeddings.extend(chunk_embeddings)
        
        embeddings_tensor = torch.stack(embeddings, dim=0)
        return embeddings_tensor
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        # DA-CLIP ViT-B/32 outputs 512-dim embeddings
        return 512
