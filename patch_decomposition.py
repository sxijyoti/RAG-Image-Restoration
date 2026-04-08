"""
Patch decomposition and reconstruction for image restoration pipeline.

Provides efficient overlapping patch extraction and reconstruction with averaging.
"""

import numpy as np
from typing import List, Tuple, Union
from PIL import Image


def decompose_image_to_patches(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 32,
    stride: int = 16,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Decompose an image into overlapping patches.
    
    Args:
        image: Input image as PIL Image or numpy array (H, W) or (H, W, C).
        patch_size: Size of square patches (default: 32).
        stride: Step size between patches (default: 16, creates overlaps).
    
    Returns:
        patches: List of numpy arrays, each of shape (patch_size, patch_size) or 
                 (patch_size, patch_size, C).
        coordinates: List of (x, y) tuples indicating top-left corner of each patch.
    
    Notes:
        - Uses efficient numpy slicing to avoid redundant copies.
        - Image boundaries: patches that extend beyond image are padded with 0s.
        - Coordinate system: (x, y) where x is column (horizontal), y is row (vertical).
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is a numpy array
    image = np.asarray(image)
    
    # Handle 2D (grayscale) and 3D (color) images
    if image.ndim == 2:
        height, width = image.shape
        is_grayscale = True
    elif image.ndim == 3:
        height, width, channels = image.shape
        is_grayscale = False
    else:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
    
    patches = []
    coordinates = []
    
    # Compute patch grid positions
    # Calculate how many patches fit with stride
    num_patches_y = (height - patch_size) // stride + 1
    num_patches_x = (width - patch_size) // stride + 1
    
    # Pad image if patches extend beyond boundaries
    pad_bottom = max(0, (num_patches_y - 1) * stride + patch_size - height)
    pad_right = max(0, (num_patches_x - 1) * stride + patch_size - width)
    
    if pad_bottom > 0 or pad_right > 0:
        if is_grayscale:
            padded_image = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode='constant')
        else:
            padded_image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant')
    else:
        padded_image = image
    
    # Extract patches efficiently using stride tricks or slicing
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            y_start = y * stride
            x_start = x * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Extract patch (creates view, minimal copy overhead)
            if is_grayscale:
                patch = padded_image[y_start:y_end, x_start:x_end].copy()
            else:
                patch = padded_image[y_start:y_end, x_start:x_end, :].copy()
            
            patches.append(patch)
            coordinates.append((x_start, y_start))
    
    return patches, coordinates


def reconstruct_image_from_patches(
    patches: List[np.ndarray],
    coordinates: List[Tuple[int, int]],
    image_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    patch_size: int = 32,
) -> np.ndarray:
    """
    Reconstruct an image from overlapping patches using averaging.
    
    Args:
        patches: List of numpy arrays (patch_size, patch_size) or 
                 (patch_size, patch_size, C).
        coordinates: List of (x, y) tuples for each patch's top-left corner.
        image_shape: Target image shape as (height, width) or (height, width, channels).
        patch_size: Size of square patches (default: 32).
    
    Returns:
        reconstructed: Reconstructed image as numpy array of shape image_shape.
    
    Notes:
        - Overlapping regions are averaged.
        - Output dtype is inferred from patch dtype (converted to float for averaging).
    """
    if len(image_shape) == 2:
        height, width = image_shape
        channels = None
        is_grayscale = True
    elif len(image_shape) == 3:
        height, width, channels = image_shape
        is_grayscale = False
    else:
        raise ValueError(f"image_shape must be 2D or 3D, got {len(image_shape)}D")
    
    if is_grayscale:
        # Use float for accurate averaging
        reconstructed = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
    else:
        reconstructed = np.zeros((height, width, channels), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
    
    # Place each patch and accumulate weights
    for patch, (x, y) in zip(patches, coordinates):
        y_start = y
        x_start = x
        y_end = min(y_start + patch_size, height)
        x_end = min(x_start + patch_size, width)
        
        patch_height = y_end - y_start
        patch_width = x_end - x_start
        
        # Extract the relevant part of patch (in case of boundary patches)
        patch_slice = patch[:patch_height, :patch_width]
        
        # Add to reconstruction and track weights for averaging
        if is_grayscale:
            reconstructed[y_start:y_end, x_start:x_end] += patch_slice.astype(np.float32)
            weights[y_start:y_end, x_start:x_end] += 1.0
        else:
            reconstructed[y_start:y_end, x_start:x_end] += patch_slice.astype(np.float32)
            weights[y_start:y_end, x_start:x_end] += 1.0
    
    # Average overlapping regions
    if is_grayscale:
        valid = weights > 0
        reconstructed[valid] /= weights[valid]
    else:
        valid = weights > 0
        reconstructed[valid, :] /= weights[valid, np.newaxis]
    
    # Convert back to original dtype
    original_dtype = patches[0].dtype
    if original_dtype == np.uint8:
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    elif original_dtype in (np.float32, np.float64):
        reconstructed = np.clip(reconstructed, 0, 1).astype(original_dtype)
    else:
        reconstructed = reconstructed.astype(original_dtype)
    
    return reconstructed


def get_patch_grid_info(
    image_height: int,
    image_width: int,
    patch_size: int = 32,
    stride: int = 16,
) -> dict:
    """
    Get information about the patch grid for an image.
    
    Args:
        image_height: Height of image.
        image_width: Width of image.
        patch_size: Size of square patches (default: 32).
        stride: Step size between patches (default: 16).
    
    Returns:
        Dictionary containing:
            - num_patches_y: Number of patches in vertical direction.
            - num_patches_x: Number of patches in horizontal direction.
            - total_patches: Total number of patches.
            - coverage_pct: Percentage of image covered by patches.
            - overlap_ratio: Ratio of overlapping areas (0-1).
    """
    num_patches_y = (image_height - patch_size) // stride + 1
    num_patches_x = (image_width - patch_size) // stride + 1
    total_patches = num_patches_y * num_patches_x
    
    # Calculate coverage
    coverage = min(1.0, (patch_size ** 2) / ((stride) ** 2)) if stride > 0 else 1.0
    coverage_pct = coverage * 100
    
    # Calculate overlap ratio
    overlap_ratio = 1.0 - (stride / patch_size) if stride < patch_size else 0.0
    
    return {
        "num_patches_y": num_patches_y,
        "num_patches_x": num_patches_x,
        "total_patches": total_patches,
        "coverage_pct": coverage_pct,
        "overlap_ratio": overlap_ratio,
    }


if __name__ == "__main__":
    # Example usage
    # import matplotlib.pyplot as plt  # Optional for visualization
    
    # Create a simple test image
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    print("Original image shape:", test_image.shape)
    
    # Decompose
    patches, coords = decompose_image_to_patches(test_image, patch_size=32, stride=16)
    print(f"Number of patches: {len(patches)}")
    print(f"Patch shape: {patches[0].shape}")
    print(f"First 5 coordinates: {coords[:5]}")
    
    # Get grid info
    grid_info = get_patch_grid_info(512, 512, patch_size=32, stride=16)
    print(f"\nPatch grid info: {grid_info}")
    
    # Reconstruct
    reconstructed = reconstruct_image_from_patches(
        patches, coords, test_image.shape, patch_size=32
    )
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction error (MSE): {np.mean((test_image.astype(float) - reconstructed.astype(float)) ** 2):.6f}")
