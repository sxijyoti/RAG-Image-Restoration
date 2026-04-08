"""
Test patch decomposition with real images from the images/ folder.
"""

import numpy as np
from PIL import Image
import os
from patch_decomposition import (
    decompose_image_to_patches,
    reconstruct_image_from_patches,
    get_patch_grid_info,
)


def test_with_real_images():
    """Test decomposition and reconstruction with real images."""
    images_dir = "images"
    # Only process original images, not reconstructed ones
    image_files = [f for f in os.listdir(images_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('reconstructed_')]
    
    print("=" * 70)
    print("TESTING PATCH DECOMPOSITION WITH REAL IMAGES")
    print("=" * 70)
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        print(f"\n{'-' * 70}")
        print(f"Image: {image_file}")
        print(f"{'-' * 70}")
        
        # Load image
        pil_image = Image.open(image_path)
        image_array = np.array(pil_image)
        
        print(f"Original shape: {image_array.shape}")
        print(f"Original dtype: {image_array.dtype}")
        print(f"Image size: {pil_image.size[0]}x{pil_image.size[1]}")
        
        # Decompose
        patches, coordinates = decompose_image_to_patches(
            image_array, 
            patch_size=32, 
            stride=16
        )
        
        print(f"\nDecomposition:")
        print(f"  Number of patches: {len(patches)}")
        print(f"  Patch shape: {patches[0].shape}")
        print(f"  First 5 coordinates: {coordinates[:5]}")
        
        # Get grid info
        grid_info = get_patch_grid_info(
            image_array.shape[0], 
            image_array.shape[1], 
            patch_size=32, 
            stride=16
        )
        print(f"\nPatch grid:")
        print(f"  Grid: {grid_info['num_patches_y']}x{grid_info['num_patches_x']}")
        print(f"  Total patches: {grid_info['total_patches']}")
        print(f"  Overlap ratio: {grid_info['overlap_ratio']:.1%}")
        
        # Reconstruct
        reconstructed = reconstruct_image_from_patches(
            patches, 
            coordinates, 
            image_array.shape, 
            patch_size=32
        )
        
        print(f"\nReconstruction:")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Reconstructed dtype: {reconstructed.dtype}")
        
        # Calculate reconstruction error
        if image_array.dtype == np.uint8:
            mse = np.mean((image_array.astype(float) - reconstructed.astype(float)) ** 2)
        else:
            mse = np.mean((image_array - reconstructed) ** 2)
        
        print(f"  MSE: {mse:.6f}")
        
        # Max absolute difference
        if image_array.dtype == np.uint8:
            max_diff = np.max(np.abs(image_array.astype(float) - reconstructed.astype(float)))
        else:
            max_diff = np.max(np.abs(image_array - reconstructed))
        
        print(f"  Max absolute difference: {max_diff:.6f}")
        
        # Check if reconstruction is perfect
        if mse < 1e-6:
            print(f"  ✓ Perfect reconstruction!")
        else:
            print(f"  Warning: Reconstruction error detected")
        
        # Save reconstructed image for visual comparison
        if image_array.dtype == np.uint8:
            reconstructed_image = Image.fromarray(reconstructed.astype(np.uint8))
        else:
            # For float images, convert to 0-255 range
            reconstructed_uint8 = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
            reconstructed_image = Image.fromarray(reconstructed_uint8)
        
        output_path = os.path.join(images_dir, f"reconstructed_{image_file}")
        reconstructed_image.save(output_path)
        print(f"  Saved reconstructed image to: {output_path}")


def test_patch_statistics():
    """Test and show patch statistics."""
    print("\n" + "=" * 70)
    print("PATCH STATISTICS FOR EACH IMAGE")
    print("=" * 70)
    
    images_dir = "images"
    # Only process original images, not reconstructed ones
    image_files = [f for f in os.listdir(images_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('reconstructed_')]
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_array = np.array(Image.open(image_path))
        
        patches, _ = decompose_image_to_patches(image_array, patch_size=32, stride=16)
        
        print(f"\n{image_file}:")
        print(f"  Total patches: {len(patches)}")
        
        # Compute statistics across patches
        patch_means = [np.mean(p) for p in patches]
        patch_stds = [np.std(p) for p in patches]
        
        print(f"  Patch mean (across patches): {np.mean(patch_means):.2f}")
        print(f"  Patch std (across patches): {np.mean(patch_stds):.2f}")
        print(f"  Min patch mean: {np.min(patch_means):.2f}")
        print(f"  Max patch mean: {np.max(patch_means):.2f}")


if __name__ == "__main__":
    test_with_real_images()
    test_patch_statistics()
    
    print("\n" + "=" * 70)
    print("Testing completed!")
    print("=" * 70)
