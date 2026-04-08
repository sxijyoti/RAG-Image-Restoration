"""
Patch Segmentation Module - Lightweight
Extracts overlapping patches from degraded images using sliding window
"""

import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from configs.config import PATCH_SIZE, PATCH_STRIDE


class PatchSegmentor:
    def __init__(self, patch_size: int = PATCH_SIZE, stride: int = PATCH_STRIDE):
        self.patch_size = patch_size
        self.stride = stride
        self.to_tensor = transforms.ToTensor()
    
    def segment(self, image_path: str) -> List[Tuple[torch.Tensor, int, int]]:
        """Extract overlapping patches from image"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.to_tensor(image)
        
        _, height, width = img_tensor.shape
        patches = []
        
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = img_tensor[:, y:y + self.patch_size, x:x + self.patch_size]
                if patch.shape == (3, self.patch_size, self.patch_size):
                    patches.append((patch, x, y))
        
        if not patches:
            raise ValueError(f"No patches extracted from {image_path}")
        
        return patches
