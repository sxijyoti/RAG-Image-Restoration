"""
Stitching module for combining patches into full image.

Handles overlapping patch stitching with averaging for overlap regions.
"""

import numpy as np
from typing import List, Dict, Tuple
from PIL import Image


class PatchStitcher:
    """Stitches overlapping patches back into a full image."""
    
    @staticmethod
    def stitch_patches(
        patches: np.ndarray,
        coordinates: List[Dict],
        image_shape: Tuple[int, int],
        patch_size: int = 64
    ) -> np.ndarray:
        """
        Stitch patches into full image with overlap averaging.
        
        Args:
            patches: Array of decoded patches, shape (num_patches, 3, 64, 64), float32 in [0, 1]
            coordinates: List of dicts {'x': int, 'y': int, 'size': int}
            image_shape: Tuple of (height, width) of original image
            patch_size: Size of patches
            
        Returns:
            Stitched image of shape (height, width, 3), float32 in [0, 1]
        """
        height, width = image_shape
        
        # Initialize output and weight map
        output = np.zeros((height, width, 3), dtype=np.float32)
        weight_map = np.zeros((height, width, 1), dtype=np.float32)
        
        # Place each patch
        for i, coords in enumerate(coordinates):
            x = coords['x']
            y = coords['y']
            
            # Get patch (convert from CHW to HWC)
            patch = patches[i]
            patch_hwc = np.transpose(patch, (1, 2, 0))
            
            # Accumulate patch and weights
            output[y:y + patch_size, x:x + patch_size, :] += patch_hwc
            weight_map[y:y + patch_size, x:x + patch_size, :] += 1.0
        
        # Average overlapping regions
        output = output / (weight_map + 1e-8)
        
        # Clip to valid range
        output = np.clip(output, 0.0, 1.0)
        
        return output
    
    @staticmethod
    def stitch_patches_weighted(
        patches: np.ndarray,
        coordinates: List[Dict],
        image_shape: Tuple[int, int],
        patch_size: int = 64,
        kernel: str = "gaussian"
    ) -> np.ndarray:
        """
        Stitch patches with weighted averaging using a kernel.
        
        Args:
            patches: Array of decoded patches, shape (num_patches, 3, 64, 64)
            coordinates: List of coordinate dicts
            image_shape: Tuple of (height, width) of original image
            patch_size: Size of patches
            kernel: Type of weighting kernel ("gaussian", "raised_cosine", "uniform")
            
        Returns:
            Stitched image of shape (height, width, 3), float32 in [0, 1]
        """
        height, width = image_shape
        
        # Get weight kernel
        weight_kernel = PatchStitcher._get_weight_kernel(patch_size, kernel)
        
        # Initialize output and weight map
        output = np.zeros((height, width, 3), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Place each patch
        for i, coords in enumerate(coordinates):
            x = coords['x']
            y = coords['y']
            
            # Get patch (convert from CHW to HWC)
            patch = patches[i]
            patch_hwc = np.transpose(patch, (1, 2, 0))
            
            # Get valid region (handle boundary)
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            
            valid_x = x_end - x
            valid_y = y_end - y
            
            # Apply weighting
            weight_valid = weight_kernel[:valid_y, :valid_x]
            patch_valid = patch_hwc[:valid_y, :valid_x, :]
            
            # Accumulate
            output[y:y_end, x:x_end, :] += patch_valid * weight_valid[:, :, np.newaxis]
            weight_map[y:y_end, x:x_end] += weight_valid
        
        # Normalize
        output = output / (weight_map[:, :, np.newaxis] + 1e-8)
        
        # Clip to valid range
        output = np.clip(output, 0.0, 1.0)
        
        return output
    
    @staticmethod
    def _get_weight_kernel(patch_size: int, kernel_type: str) -> np.ndarray:
        """
        Generate weighting kernel for overlap blending.
        
        Args:
            patch_size: Size of patch
            kernel_type: Type of kernel
            
        Returns:
            Weight kernel of shape (patch_size, patch_size)
        """
        if kernel_type == "gaussian":
            # Gaussian kernel
            x = np.linspace(-1, 1, patch_size)
            gaussian = np.exp(-4 * x ** 2)
            kernel = np.outer(gaussian, gaussian)
        elif kernel_type == "raised_cosine":
            # Raised cosine kernel
            x = np.linspace(0, np.pi, patch_size)
            cosine = (1 - np.cos(x)) / 2
            kernel = np.outer(cosine, cosine)
        elif kernel_type == "uniform":
            # Uniform weighting
            kernel = np.ones((patch_size, patch_size))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Normalize
        kernel = kernel / (kernel.max() + 1e-8)
        
        return kernel
    
    @staticmethod
    def save_image(
        image_array: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save image array to file.
        
        Args:
            image_array: Image array of shape (height, width, 3), float32 in [0, 1]
            output_path: Path to save image
        """
        # Convert to uint8
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL and save
        image = Image.fromarray(image_uint8, mode='RGB')
        image.save(output_path)
        
        print(f"Saved image to {output_path}")
