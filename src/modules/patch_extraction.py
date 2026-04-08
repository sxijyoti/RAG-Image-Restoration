"""
Patch Extraction Module for Image Restoration

This module handles extracting overlapping patches from images and reconstructing
images from patches with proper handling of overlapping regions.

Configuration:
- Patch size: 64×64
- Stride: 32 (50% overlap)
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path
from PIL import Image


class PatchExtractor:
    """
    Extracts overlapping patches from images and reconstructs images from patches.
    
    Attributes:
        patch_size (int): Width/height of square patches (default: 64)
        stride (int): Step size between patch starts (default: 32)
    """
    
    def __init__(self, patch_size: int = 64, stride: int = 32):
        """
        Initialize patch extractor.
        
        Args:
            patch_size: Size of square patches (default 64)
            stride: Step size between patches (default 32, means 50% overlap)
        """
        self.patch_size = patch_size
        self.stride = stride
        
        # Validate configuration
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if stride > patch_size:
            raise ValueError("stride should not exceed patch_size")
    
    def extract(
        self, 
        image: Union[np.ndarray, str, Path, Image.Image],
        return_coords: bool = True,
        debug: bool = False
    ) -> Union[Tuple[List[np.ndarray], List[Tuple[int, int]]], List[np.ndarray]]:
        """
        Extract overlapping patches from an image.
        
        Args:
            image: Input image as numpy array (H, W, 3), PIL Image, or path
            return_coords: If True, return (patches, coords); if False, return patches only
            debug: If True, print extraction statistics
        
        Returns:
            If return_coords=True: (patches, coords)
                - patches: List of numpy arrays, each shape (64, 64, 3)
                - coords: List of tuples (x, y) representing top-left corner in original image
            If return_coords=False: patches only
            
        Raises:
            ValueError: If image is smaller than patch_size in either dimension
        """
        # Load image if needed
        image_array = self._load_image(image)
        height, width, channels = image_array.shape
        
        # Validate image dimensions
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image size ({height}×{width}) must be at least {self.patch_size}×{self.patch_size}"
            )
        
        patches = []
        coords = []
        
        # Sliding window extraction
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image_array[
                    y:y + self.patch_size,
                    x:x + self.patch_size,
                    :
                ].copy()  # Copy to avoid memory sharing
                
                patches.append(patch)
                coords.append((x, y))
        
        # Handle bottom and right edges if not covered by regular grid
        # Bottom edge patches (if there's uncovered area)
        if (height - self.patch_size) % self.stride != 0:
            y_edge = height - self.patch_size
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image_array[
                    y_edge:y_edge + self.patch_size,
                    x:x + self.patch_size,
                    :
                ].copy()
                
                patches.append(patch)
                coords.append((x, y_edge))
        
        # Right edge patches (if there's uncovered area)
        if (width - self.patch_size) % self.stride != 0:
            x_edge = width - self.patch_size
            for y in range(0, height - self.patch_size + 1, self.stride):
                patch = image_array[
                    y:y + self.patch_size,
                    x_edge:x_edge + self.patch_size,
                    :
                ].copy()
                
                patches.append(patch)
                coords.append((x_edge, y))
        
        # Bottom-right corner patch (if there's uncovered area)
        if ((height - self.patch_size) % self.stride != 0 and 
            (width - self.patch_size) % self.stride != 0):
            y_edge = height - self.patch_size
            x_edge = width - self.patch_size
            patch = image_array[
                y_edge:y_edge + self.patch_size,
                x_edge:x_edge + self.patch_size,
                :
            ].copy()
            
            patches.append(patch)
            coords.append((x_edge, y_edge))
        
        if debug:
            print(f"Image size: {width}×{height}")
            print(f"Number of patches: {len(patches)}")
            print(f"Patch grid coverage:")
            print(f"  Regular grid: {len(range(0, height - self.patch_size + 1, self.stride))} × "
                  f"{len(range(0, width - self.patch_size + 1, self.stride))}")
        
        if return_coords:
            return patches, coords
        else:
            return patches
    
    def reconstruct(
        self,
        patches: List[np.ndarray],
        coords: List[Tuple[int, int]],
        image_shape: Tuple[int, int, int],
        blend_mode: str = "average"
    ) -> np.ndarray:
        """
        Reconstruct image from patches using averaging in overlapping regions.
        
        Args:
            patches: List of patch arrays, each shape (patch_size, patch_size, channels)
            coords: List of (x, y) coordinates for each patch
            image_shape: Target image shape (height, width, channels)
            blend_mode: How to handle overlapping regions ("average" only option for now)
        
        Returns:
            Reconstructed image as numpy array of shape image_shape
            
        Raises:
            ValueError: If patches and coords lengths don't match or shapes are inconsistent
        """
        if len(patches) != len(coords):
            raise ValueError(f"Number of patches ({len(patches)}) must match number of coords ({len(coords)})")
        
        if len(patches) == 0:
            raise ValueError("No patches provided")
        
        height, width, channels = image_shape
        
        # Validate patch shapes
        for i, patch in enumerate(patches):
            if patch.shape != (self.patch_size, self.patch_size, channels):
                raise ValueError(
                    f"Patch {i} has shape {patch.shape}, "
                    f"expected ({self.patch_size}, {self.patch_size}, {channels})"
                )
        
        # Initialize output image and count array
        # Count array tracks how many patches contributed to each pixel (for averaging)
        reconstructed = np.zeros(image_shape, dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        
        # Place each patch in the reconstructed image
        for patch, (x, y) in zip(patches, coords):
            y_end = min(y + self.patch_size, height)
            x_end = min(x + self.patch_size, width)
            
            patch_height = y_end - y
            patch_width = x_end - x
            
            # Add patch contribution (only use the part that fits)
            reconstructed[y:y_end, x:x_end, :] += patch[:patch_height, :patch_width, :].astype(np.float32)
            count[y:y_end, x:x_end] += 1
        
        # Average overlapping regions
        # Expand count to 3D for broadcasting
        count_3d = np.expand_dims(count, axis=2)
        
        # Avoid division by zero (shouldn't happen if patches cover image properly)
        count_3d = np.maximum(count_3d, 1)
        
        reconstructed = reconstructed / count_3d
        
        # Convert back to uint8 if input was likely uint8
        if reconstructed.max() <= 255:
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def get_patch_grid_info(self, image_shape: Tuple[int, int]) -> dict:
        """
        Get information about patch grid for a given image size.
        
        Args:
            image_shape: Image shape as (height, width)
            
        Returns:
            Dictionary with grid information
        """
        height, width = image_shape
        
        # Regular grid dimensions
        num_y_regular = len(range(0, height - self.patch_size + 1, self.stride))
        num_x_regular = len(range(0, width - self.patch_size + 1, self.stride))
        
        # Check for edge patches
        has_bottom_edge = (height - self.patch_size) % self.stride != 0
        has_right_edge = (width - self.patch_size) % self.stride != 0
        has_corner = has_bottom_edge and has_right_edge
        
        total_patches = num_y_regular * num_x_regular
        if has_bottom_edge:
            total_patches += num_x_regular
        if has_right_edge:
            total_patches += num_y_regular
        if has_corner:
            total_patches += 1
        
        return {
            "image_shape": image_shape,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "regular_grid": (num_y_regular, num_x_regular),
            "has_bottom_edge": has_bottom_edge,
            "has_right_edge": has_right_edge,
            "has_corner": has_corner,
            "total_patches": total_patches,
        }
    
    @staticmethod
    def _load_image(image: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        """
        Load image into numpy array format (H, W, 3).
        
        Args:
            image: Image input (numpy array, path, or PIL Image)
            
        Returns:
            Image as numpy array (height, width, 3)
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=2)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")
            return image
        
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img)
        
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.array(image)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


# Convenience functions for quick usage
def extract_patches(
    image: Union[np.ndarray, str, Path, Image.Image],
    patch_size: int = 64,
    stride: int = 32,
    debug: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Quick function to extract patches from an image.
    
    Args:
        image: Input image
        patch_size: Size of patches (default 64)
        stride: Stride between patches (default 32)
        debug: Print statistics if True
        
    Returns:
        (patches, coords) tuple
    """
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    return extractor.extract(image, return_coords=True, debug=debug)


def reconstruct_image(
    patches: List[np.ndarray],
    coords: List[Tuple[int, int]],
    image_shape: Tuple[int, int, int],
    patch_size: int = 64,
    stride: int = 32
) -> np.ndarray:
    """
    Quick function to reconstruct image from patches.
    
    Args:
        patches: List of patches
        coords: List of coordinates
        image_shape: Target image shape (height, width, channels)
        patch_size: Size of patches (default 64)
        stride: Stride used during extraction (default 32)
        
    Returns:
        Reconstructed image
    """
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    return extractor.reconstruct(patches, coords, image_shape)
