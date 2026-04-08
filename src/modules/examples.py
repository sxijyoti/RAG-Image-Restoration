"""
Example usage and tests for the Patch Extraction module.

This demonstrates:
- Basic patch extraction
- Image reconstruction with overlap blending
- Edge case handling
- Debugging and visualization
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Warning: matplotlib not available. Skipping visualization examples.")

from pathlib import Path
from PIL import Image as PILImage

from patch_extraction import PatchExtractor, extract_patches, reconstruct_image


def create_test_image(width: int = 256, height: int = 256, pattern: str = "gradient") -> np.ndarray:
    """
    Create a synthetic test image.
    
    Args:
        width: Image width
        height: Image height
        pattern: "gradient", "checkerboard", or "random"
        
    Returns:
        Image as numpy array (height, width, 3)
    """
    if pattern == "gradient":
        # Create RGB gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        r = (xx * 255).astype(np.uint8)
        g = (yy * 255).astype(np.uint8)
        b = ((xx + yy) / 2 * 255).astype(np.uint8)
        
        return np.stack([r, g, b], axis=2)
    
    elif pattern == "checkerboard":
        # Create checkerboard pattern
        square_size = 32
        board = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    board[i:i+square_size, j:j+square_size, :] = 255
        return board
    
    elif pattern == "random":
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def visualize_patch_grid(
    image: np.ndarray,
    coords: list,
    patch_size: int = 64,
    save_path: str = None
):
    """
    Visualize patches on the original image.
    
    Args:
        image: Input image
        coords: List of patch coordinates
        patch_size: Size of patches
        save_path: If provided, save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Visualization requires matplotlib. Skipping.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Draw rectangles for each patch
    for x, y in coords:
        rect = Rectangle((x, y), patch_size, patch_size, 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(f"Patch Grid ({len(coords)} patches)")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def example_basic_extraction():
    """Example 1: Basic patch extraction and reconstruction."""
    print("=" * 60)
    print("Example 1: Basic Patch Extraction and Reconstruction")
    print("=" * 60)
    
    # Create test image
    image = create_test_image(256, 256, "gradient")
    print(f"Created test image: {image.shape}")
    
    # Extract patches
    extractor = PatchExtractor(patch_size=64, stride=32)
    patches, coords = extractor.extract(image, debug=True)
    
    print(f"\nExtracted {len(patches)} patches")
    print(f"First few coordinates: {coords[:5]}")
    print(f"Patch shape: {patches[0].shape}")
    
    # Reconstruct image
    reconstructed = extractor.reconstruct(patches, coords, image.shape)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Check reconstruction quality
    mse = np.mean((image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    print(f"Reconstruction MSE: {mse:.4f} (should be 0 for perfect reconstruction)")
    
    return image, patches, coords, reconstructed


def example_edge_cases():
    """Example 2: Handle edge cases."""
    print("\n" + "=" * 60)
    print("Example 2: Edge Cases")
    print("=" * 60)
    
    extractor = PatchExtractor(patch_size=64, stride=32)
    
    test_cases = [
        ("Small image (matches patch size)", (64, 64)),
        ("Image slightly larger than patch", (80, 80)),
        ("Non-aligned image", (100, 120)),
        ("Large rectangular image", (300, 500)),
    ]
    
    for name, (h, w) in test_cases:
        try:
            image = create_test_image(w, h, "checkerboard")
            info = extractor.get_patch_grid_info((h, w))
            print(f"\n{name} ({h}×{w}):")
            print(f"  Total patches: {info['total_patches']}")
            print(f"  Regular grid: {info['regular_grid']}")
            print(f"  Needs edge patches: bottom={info['has_bottom_edge']}, "
                  f"right={info['has_right_edge']}, corner={info['has_corner']}")
            
            # Actually extract to verify
            patches, coords = extractor.extract(image)
            print(f"  Verified: {len(patches)} patches extracted")
            
        except ValueError as e:
            print(f"\n{name}: {e}")


def example_reconstruction_quality():
    """Example 3: Test reconstruction quality with overlapping regions."""
    print("\n" + "=" * 60)
    print("Example 3: Reconstruction Quality with Overlapping Regions")
    print("=" * 60)
    
    extractor = PatchExtractor(patch_size=64, stride=32)
    
    # Test with different patterns
    patterns = ["gradient", "checkerboard", "random"]
    
    for pattern in patterns:
        image = create_test_image(256, 256, pattern)
        patches, coords = extractor.extract(image)
        reconstructed = extractor.reconstruct(patches, coords, image.shape)
        
        # Calculate metrics
        mse = np.mean((image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"\nPattern: {pattern}")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        
        # Check if reconstruction is lossless
        max_diff = np.max(np.abs(image.astype(np.int16) - reconstructed.astype(np.int16)))
        print(f"  Max pixel difference: {max_diff}")


def example_patch_inspection():
    """Example 4: Inspect individual patches."""
    print("\n" + "=" * 60)
    print("Example 4: Patch Inspection")
    print("=" * 60)
    
    extractor = PatchExtractor(patch_size=64, stride=32)
    image = create_test_image(256, 256, "gradient")
    
    patches, coords = extractor.extract(image, debug=True)
    
    print("\nFirst 5 patches:")
    for i in range(min(5, len(patches))):
        x, y = coords[i]
        patch = patches[i]
        print(f"\nPatch {i}:")
        print(f"  Coordinates: ({x}, {y})")
        print(f"  Shape: {patch.shape}")
        print(f"  Data range: [{patch.min()}, {patch.max()}]")
        print(f"  Mean: {patch.mean():.2f}")


def example_quick_api():
    """Example 5: Using convenience functions."""
    print("\n" + "=" * 60)
    print("Example 5: Quick API Usage")
    print("=" * 60)
    
    image = create_test_image(200, 200, "checkerboard")
    
    # Quick extraction
    patches, coords = extract_patches(image, debug=True)
    print(f"\nExtracted {len(patches)} patches using quick API")
    
    # Quick reconstruction
    reconstructed = reconstruct_image(patches, coords, image.shape)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    mse = np.mean((image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    print(f"MSE: {mse:.6f}")


def example_real_image():
    """Example 6: Load and process a real image (if available)."""
    print("\n" + "=" * 60)
    print("Example 6: Real Image Processing")
    print("=" * 60)
    
    extractor = PatchExtractor(patch_size=64, stride=32)
    
    # Try to find an image in the repository
    image_paths = list(Path("../../images").glob("*.png")) + \
                  list(Path("../../images").glob("*.jpg"))
    
    if image_paths:
        image_path = image_paths[0]
        print(f"Found image: {image_path}")
        
        try:
            image = PILImage.open(image_path)
            if image.size[0] >= 64 and image.size[1] >= 64:
                patches, coords = extractor.extract(image, debug=True)
                print(f"Successfully extracted {len(patches)} patches from real image")
            else:
                print(f"Image too small: {image.size}")
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("No images found in ../../images/")


def run_all_examples():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PATCH EXTRACTION MODULE EXAMPLES" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        example_basic_extraction()
        example_edge_cases()
        example_reconstruction_quality()
        example_patch_inspection()
        example_quick_api()
        example_real_image()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
