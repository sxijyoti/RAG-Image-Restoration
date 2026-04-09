"""
Patch extraction module for image restoration.

Handles overlapping patch extraction with configurable patch size and stride.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Dict


def extract_patches(
    image_path: str,
    patch_size: int = 64,
    stride: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract overlapping patches from an image.
    
    Args:
        image_path: Path to input image
        patch_size: Size of each patch (patch_size x patch_size)
        stride: Stride between patches
        
    Returns:
        patches: Array of shape (num_patches, 3, patch_size, patch_size), dtype float32 in [0, 1]
        coordinates: List of dicts with patch metadata {'x': int, 'y': int, 'size': int}
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    height, width, _ = image_array.shape
    patches = []
    coordinates = []
    
    # Extract patches with stride
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image_array[y:y + patch_size, x:x + patch_size, :]
            # Convert to (C, H, W) format for torch compatibility
            patch_chw = np.transpose(patch, (2, 0, 1))
            patches.append(patch_chw)
            coordinates.append({
                'x': x,
                'y': y,
                'size': patch_size
            })
    
    patches_array = np.array(patches, dtype=np.float32)
    
    return patches_array, coordinates


def load_image(image_path: str) -> np.ndarray:
    """
    Load image as numpy array.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array of shape (H, W, 3) normalized to [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array


def get_image_shape(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions.
    
    Args:
        image_path: Path to image file
        
    Returns:
        (height, width) tuple
    """
    image = Image.open(image_path).convert('RGB')
    return image.size[::-1]  # PIL returns (width, height), we want (height, width)
