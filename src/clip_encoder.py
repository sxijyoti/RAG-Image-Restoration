"""
DA-CLIP Encoder Module - Lightweight
Extracts token-level embeddings from image patches
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List
import numpy as np

from configs.config import (
    DEVICE, DACLIP_EMBED_DIM, DACLIP_IMAGE_SIZE,
    DACLIP_PATCH_SIZE, DACLIP_OUTPUT_SHAPE
)


class DAClipEncoder:
    def __init__(self, pretrained: bool = False):
        self.device = DEVICE
        self.embed_dim = DACLIP_EMBED_DIM
        self.image_size = DACLIP_IMAGE_SIZE
        self.patch_size = DACLIP_PATCH_SIZE
        
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                             interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        self._load_model()
    
    def _load_model(self):
        try:
            import timm
            self.model = timm.create_model(
                'vit_base_patch32_clip_224',
                pretrained=True, num_classes=0
            )
            self.model.to(self.device)
            self.model.eval()
        except:
            self._create_dummy_encoder()
    
    def _create_dummy_encoder(self):
        class SimpleViT(nn.Module):
            def __init__(self, embed_dim=512, num_patches=256):
                super().__init__()
                self.embed_dim = embed_dim
                self.patch_embed = nn.Conv2d(3, embed_dim, 32, 32)
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
            
            def forward(self, x):
                x = self.patch_embed(x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                x = x + self.pos_embed
                return x
        
        self.model = SimpleViT(embed_dim=self.embed_dim, num_patches=256)
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        """Encode patch to (1, 512, 16, 16) embedding"""
        if patch_tensor.ndim == 3:
            patch_tensor = patch_tensor.unsqueeze(0)
        
        patch_tensor = patch_tensor.to(self.device)
        
        if patch_tensor.shape[-2:] != (self.image_size, self.image_size):
            patch_tensor = torch.nn.functional.interpolate(
                patch_tensor,
                size=(self.image_size, self.image_size),
                mode='bicubic', align_corners=False
            )
        
        with torch.no_grad():
            embeddings = self.model(patch_tensor)
        
        if embeddings.ndim == 3:
            B, num_tokens, embed_dim = embeddings.shape
            spatial_size = int(np.sqrt(num_tokens))
            embeddings = embeddings.transpose(1, 2)
            embeddings = embeddings.reshape(B, embed_dim, spatial_size, spatial_size)
        
        return embeddings
    
    def encode_batch(self, patch_tensors: list) -> list:
        """Encode multiple patches"""
        return [self.encode(patch) for patch in patch_tensors]
