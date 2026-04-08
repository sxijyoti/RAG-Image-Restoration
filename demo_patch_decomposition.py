"""
Demonstration of patch decomposition and reconstruction.

Shows how to use the patch decomposition module for image restoration pipelines.
"""

import numpy as np
from patch_decomposition import (
    decompose_image_to_patches,
    reconstruct_image_from_patches,
    get_patch_grid_info,
)


def demo_basic_usage():
    """Basic usage example with a random image."""
    print("=" * 60)
    print("DEMO 1: Basic Patch Decomposition and Reconstruction")
    print("=" * 60)
    
    # Create a test image (RGB)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    print(f"Original image shape: {image.shape}")
    print(f"Original image dtype: {image.dtype}")
    
    # Decompose into patches
    patches, coordinates = decompose_image_to_patches(
        image, patch_size=32, stride=16
    )
    print(f"\nDecomposition:")
    print(f"  Number of patches: {len(patches)}")
    print(f"  Patch shape: {patches[0].shape}")
    print(f"  Patch dtype: {patches[0].dtype}")
    
    # Get grid info
    grid_info = get_patch_grid_info(512, 512, patch_size=32, stride=16)
    print(f"\nPatch grid info:")
    for key, value in grid_info.items():
        print(f"  {key}: {value}")
    
    # Reconstruct from patches
    reconstructed = reconstruct_image_from_patches(
        patches, coordinates, image.shape, patch_size=32
    )
    print(f"\nReconstruction:")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstructed dtype: {reconstructed.dtype}")
    
    # Verify quality
    mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    return patches, coordinates, image, reconstructed


def demo_grayscale():
    """Example with grayscale image."""
    print("\n" + "=" * 60)
    print("DEMO 2: Grayscale Image")
    print("=" * 60)
    
    # Create grayscale test image
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    print(f"Original image shape: {image.shape}")
    
    patches, coordinates = decompose_image_to_patches(
        image, patch_size=32, stride=16
    )
    print(f"Number of patches: {len(patches)}")
    print(f"Patch shape: {patches[0].shape}")
    
    reconstructed = reconstruct_image_from_patches(
        patches, coordinates, image.shape, patch_size=32
    )
    
    mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")


def demo_patch_access():
    """Example showing how to process patches."""
    print("\n" + "=" * 60)
    print("DEMO 3: Accessing and Processing Patches")
    print("=" * 60)
    
    # Create a test image
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    patches, coordinates = decompose_image_to_patches(
        image, patch_size=32, stride=16
    )
    
    print(f"Total patches: {len(patches)}")
    print("\nFirst 3 patches info:")
    for i in range(3):
        patch = patches[i]
        coord = coordinates[i]
        mean_val = np.mean(patch)
        print(f"  Patch {i}: coord={coord}, mean={mean_val:.2f}, shape={patch.shape}")
    
    # Example: apply a simple operation to patches
    print("\nApplying brightness adjustment to all patches...")
    modified_patches = [np.clip(p * 0.9, 0, 255).astype(np.uint8) for p in patches]
    
    # Reconstruct from modified patches
    modified_image = reconstruct_image_from_patches(
        modified_patches, coordinates, image.shape, patch_size=32
    )
    print(f"Modified image shape: {modified_image.shape}")


def demo_different_sizes():
    """Example with different image sizes."""
    print("\n" + "=" * 60)
    print("DEMO 4: Different Image Sizes")
    print("=" * 60)
    
    sizes = [(256, 256), (512, 384), (1024, 768)]
    
    for height, width in sizes:
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        patches, coords = decompose_image_to_patches(image, patch_size=32, stride=16)
        reconstructed = reconstruct_image_from_patches(
            patches, coords, image.shape, patch_size=32
        )
        mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)
        
        print(
            f"Image {height}x{width}: {len(patches)} patches, "
            f"MSE: {mse:.6f}"
        )


def demo_coordinate_mapping():
    """Example showing coordinate mapping."""
    print("\n" + "=" * 60)
    print("DEMO 5: Coordinate Mapping for Reconstruction")
    print("=" * 60)
    
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    patches, coordinates = decompose_image_to_patches(
        image, patch_size=32, stride=16
    )
    
    print(f"Image size: {image.shape[:2]}")
    print(f"Patch size: 32x32, Stride: 16")
    print(f"Number of patches: {len(patches)}\n")
    
    print("All patch coordinates:")
    for i, (x, y) in enumerate(coordinates):
        print(f"  Patch {i:2d}: (x={x:3d}, y={y:3d})", end="")
        if (i + 1) % 4 == 0:
            print()
        else:
            print(" | ", end="")
    print()


if __name__ == "__main__":
    # Run all demonstrations
    demo_basic_usage()
    demo_grayscale()
    demo_patch_access()
    demo_different_sizes()
    demo_coordinate_mapping()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)
