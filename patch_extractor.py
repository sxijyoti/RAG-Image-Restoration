"""
Patch extractor for RAG-IR system.

Extracts patches with correct spatial tracking for later reconstruction.
- Patch size: 64x64
- Stride: 32 (50% overlap)
"""

import numpy as np
from typing import List, Tuple, Union
from PIL import Image


def load_image(image_path_or_pil: Union[str, Image.Image]) -> np.ndarray:
    """
    Load image from path or PIL Image.
    
    Args:
        image_path_or_pil: File path or PIL Image
    
    Returns:
        numpy array in range [0, 255] as uint8
    """
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert('RGB')
        return np.array(img)
    elif isinstance(image_path_or_pil, Image.Image):
        return np.array(image_path_or_pil.convert('RGB'))
    else:
        return np.asarray(image_path_or_pil)


def extract_patches(
    image: Union[str, Image.Image, np.ndarray],
    patch_size: int = 64,
    stride: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Extract overlapping patches from image.
    
    Args:
        image: Image path, PIL Image, or numpy array
        patch_size: Patch dimension (default 64x64)
        stride: Stride between patches (default 32, creates 50% overlap)
    
    Returns:
        patches: List of numpy arrays (patch_size, patch_size, 3)
        coordinates: List of (x, y) top-left corner positions
        original_shape: (height, width) of input image
    """
    # Load image
    image_array = load_image(image)
    
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {image_array.shape}")
    
    height, width = image_array.shape[:2]
    original_shape = (height, width)
    
    # Calculate patch grid
    num_patches_y = (height - patch_size + stride - 1) // stride + 1 if height > patch_size else 1
    num_patches_x = (width - patch_size + stride - 1) // stride + 1 if width > patch_size else 1
    
    # Pad image for full coverage
    pad_bottom = max(0, (num_patches_y - 1) * stride + patch_size - height)
    pad_right = max(0, (num_patches_x - 1) * stride + patch_size - width)
    
    if pad_bottom > 0 or pad_right > 0:
        padded_image = np.pad(
            image_array,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode='constant'
        )
    else:
        padded_image = image_array
    
    # Extract patches
    patches = []
    coordinates = []
    
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            y_start = y * stride
            x_start = x * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            patch = padded_image[y_start:y_end, x_start:x_end].copy()
            patches.append(patch)
            coordinates.append((x_start, y_start))
    
    return patches, coordinates, original_shape


def reconstruct_image(
    patches: List[np.ndarray],
    coordinates: List[Tuple[int, int]],
    original_shape: Tuple[int, int],
    patch_size: int = 64,
) -> np.ndarray:
    """
    Reconstruct image from overlapping patches using averaging.
    
    Args:
        patches: List of patch arrays
        coordinates: List of (x, y) patch positions
        original_shape: Target output shape (height, width)
        patch_size: Patch dimension
    
    Returns:
        Reconstructed image (height, width, 3) as uint8
    """
    height, width = original_shape
    
    # Reconstruction buffers (use float for accurate averaging)
    reconstructed = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    
    # Place each patch and accumulate
    for patch, (x, y) in zip(patches, coordinates):
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)
        
        patch_h = y_end - y
        patch_w = x_end - x
        
        # Use only the valid part of patch
        patch_slice = patch[:patch_h, :patch_w]
        
        reconstructed[y:y_end, x:x_end] += patch_slice.astype(np.float32)
        weights[y:y_end, x:x_end] += 1.0
    
    # Average overlapping regions
    valid = weights > 0
    reconstructed[valid] /= weights[valid, np.newaxis]
    
    # Handle any edge pixels that weren't covered (shouldn't happen with proper padding)
    reconstructed[~valid] = 0
    
    # Convert back to uint8
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed


def get_patch_stats(patches: List[np.ndarray]) -> dict:
    """Get statistics about extracted patches."""
    patch_data = np.stack(patches)  # (N, 64, 64, 3)
    
    return {
        "num_patches": len(patches),
        "patch_shape": patches[0].shape,
        "min_value": int(patch_data.min()),
        "max_value": int(patch_data.max()),
        "mean_value": float(patch_data.mean()),
        "std_value": float(patch_data.std()),
    }


if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("PATCH EXTRACTOR TEST")
    print("=" * 70)
    
    # Test with real image
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    print("\nTest 1: Arbitrary size image")
    patches, coords, orig_shape = extract_patches(test_image, patch_size=64, stride=32)
    print(f"  Original shape: {orig_shape}")
    print(f"  Number of patches: {len(patches)}")
    print(f"  Grid size: ~{len(patches)**0.5:.0f}x{len(patches)**0.5:.0f}")
    print(f"  Patch shape: {patches[0].shape}")
    
    # Test reconstruction
    reconstructed = reconstruct_image(patches, coords, orig_shape, patch_size=64)
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    mse = np.mean((test_image.astype(float) - reconstructed.astype(float)) ** 2)
    print(f"  MSE: {mse:.6f}")
    if mse < 1e-6:
        print("  ✓ Perfect reconstruction!")
    
    # Test with different size
    print("\nTest 2: Non-square image")
    test_image2 = np.random.randint(0, 256, (512, 384, 3), dtype=np.uint8)
    patches2, coords2, orig_shape2 = extract_patches(test_image2)
    reconstructed2 = reconstruct_image(patches2, coords2, orig_shape2)
    mse2 = np.mean((test_image2.astype(float) - reconstructed2.astype(float)) ** 2)
    print(f"  Image shape: {test_image2.shape}")
    print(f"  Patches: {len(patches2)}")
    print(f"  MSE: {mse2:.6f}")
    
    # Get stats
    print("\nTest 3: Patch statistics")
    stats = get_patch_stats(patches2)
    for key, val in stats.items():
        print(f"  {key}: {val}")
