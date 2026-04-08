"""
Integration example: Using patch decomposition in an image restoration pipeline.

This demonstrates a complete workflow for patch-based restoration.
"""

import numpy as np
from patch_decomposition import (
    decompose_image_to_patches,
    reconstruct_image_from_patches,
    get_patch_grid_info,
)


class PatchBasedRestorationPipeline:
    """
    A template for integrating patch decomposition into restoration pipelines.
    """
    
    def __init__(self, patch_size=32, stride=16):
        """Initialize with patch configuration."""
        self.patch_size = patch_size
        self.stride = stride
    
    def restore(self, input_image, patch_processor):
        """
        Restore an image using patch-based processing.
        
        Args:
            input_image: Input image (PIL Image or numpy array)
            patch_processor: Function that takes a patch and returns a processed patch
        
        Returns:
            restored_image: Restored image
        """
        # Step 1: Decompose image into patches
        patches, coordinates = decompose_image_to_patches(
            input_image,
            patch_size=self.patch_size,
            stride=self.stride
        )
        
        print(f"Decomposed image into {len(patches)} patches")
        
        # Step 2: Process each patch
        processed_patches = []
        for i, patch in enumerate(patches):
            if i % 100 == 0:
                print(f"  Processing patch {i}/{len(patches)}")
            
            processed_patch = patch_processor(patch)
            processed_patches.append(processed_patch)
        
        # Step 3: Reconstruct from processed patches
        restored_image = reconstruct_image_from_patches(
            processed_patches,
            coordinates,
            input_image.shape if isinstance(input_image, np.ndarray) else np.array(input_image).shape,
            patch_size=self.patch_size
        )
        
        print(f"Reconstructed image from patches")
        
        return restored_image, patches, processed_patches, coordinates
    
    def get_pipeline_info(self, image_shape):
        """Get information about patch processing for this image."""
        if len(image_shape) == 3:
            h, w, c = image_shape
        else:
            h, w = image_shape
            c = 1
        
        info = get_patch_grid_info(h, w, self.patch_size, self.stride)
        
        return {
            **info,
            "image_shape": image_shape,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "channels": c,
        }


# Example processing functions
def simple_blur_filter(patch):
    """Example: Apply simple Gaussian blur to a patch."""
    from scipy import ndimage
    if patch.ndim == 3:
        # Process each channel
        return np.stack([
            ndimage.gaussian_filter(patch[:, :, i], sigma=1.0)
            for i in range(patch.shape[2])
        ], axis=2).astype(patch.dtype)
    else:
        return ndimage.gaussian_filter(patch, sigma=1.0).astype(patch.dtype)


def normalization_filter(patch):
    """Example: Normalize patch to [0, 1] range."""
    p_min = patch.min()
    p_max = patch.max()
    if p_max > p_min:
        normalized = (patch.astype(float) - p_min) / (p_max - p_min)
    else:
        normalized = np.zeros_like(patch, dtype=float)
    
    return normalized.astype(patch.dtype)


def identity_filter(patch):
    """Example: No-op filter for testing."""
    return patch


def denoise_filter(patch, strength=0.5):
    """
    Example: Simple denoising using bilateral-like filtering.
    In a real pipeline, this would be a neural network.
    """
    # This is a placeholder - in reality you'd call your restoration model
    # Example: reduce noise by 50%
    if patch.dtype == np.uint8:
        float_patch = patch.astype(np.float32) / 255.0
        denoised = float_patch * (1 - strength) + strength * np.mean(float_patch)
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)
    else:
        return patch * (1 - strength) + strength * np.mean(patch)


def demo_restoration_workflow():
    """Demonstrate a complete restoration workflow."""
    print("=" * 70)
    print("PATCH-BASED IMAGE RESTORATION PIPELINE DEMO")
    print("=" * 70)
    
    # Create example image (in practice, this would be loaded from file)
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    print(f"\nInput image shape: {image.shape}")
    print(f"Input image dtype: {image.dtype}")
    
    # Initialize pipeline
    pipeline = PatchBasedRestorationPipeline(patch_size=32, stride=16)
    
    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info(image.shape)
    print(f"\nPipeline configuration:")
    for key, value in pipeline_info.items():
        print(f"  {key}: {value}")
    
    # Run restoration with identity filter (for testing)
    print("\n" + "-" * 70)
    print("TEST 1: Restoration with identity filter (no changes)")
    print("-" * 70)
    
    restored, patches, processed, coords = pipeline.restore(image, identity_filter)
    
    # Check reconstruction quality
    mse = np.mean((image.astype(float) - restored.astype(float)) ** 2)
    print(f"MSE (should be ~0): {mse:.6f}")
    
    # Test with normalization
    print("\n" + "-" * 70)
    print("TEST 2: Restoration with normalization filter")
    print("-" * 70)
    
    restored_norm, _, _, _ = pipeline.restore(image, normalization_filter)
    print(f"Restoration completed")
    print(f"Output shape: {restored_norm.shape}")
    print(f"Output dtype: {restored_norm.dtype}")
    
    # Test with denoising
    print("\n" + "-" * 70)
    print("TEST 3: Restoration with denoising filter")
    print("-" * 70)
    
    def denoise_strength_5(patch):
        return denoise_filter(patch, strength=0.5)
    
    restored_denoised, _, _, _ = pipeline.restore(image, denoise_strength_5)
    print(f"Restoration completed")
    mean_diff = np.mean(np.abs(image.astype(float) - restored_denoised.astype(float)))
    print(f"Mean absolute difference: {mean_diff:.2f}")


def demo_batch_processing():
    """Demonstrate batch processing multiple images."""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING DEMO")
    print("=" * 70)
    
    # Create pipeline
    pipeline = PatchBasedRestorationPipeline(patch_size=32, stride=16)
    
    # Simulate processing multiple images
    image_sizes = [(256, 256, 3), (512, 512, 3), (128, 256, 3)]
    
    print("\nProcessing multiple images...")
    for size in image_sizes:
        image = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # Get expected patch count
        info = pipeline.get_pipeline_info(image.shape)
        
        # Restore
        restored, _, _, _ = pipeline.restore(image, identity_filter)
        
        # Verify
        mse = np.mean((image.astype(float) - restored.astype(float)) ** 2)
        
        print(f"  Image {size}: {info['total_patches']} patches, MSE={mse:.6f}")


def demo_integration_template():
    """Template showing how to integrate with your restoration model."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEMPLATE")
    print("=" * 70)
    
    print("""
To integrate with your restoration model:

1. Define your processor function:
    
    def my_restoration_model(patch):
        # Convert to model input format
        model_input = prepare_input(patch)
        
        # Run through your model
        model_output = your_model(model_input)
        
        # Convert back to image format
        restored_patch = postprocess_output(model_output)
        
        return restored_patch

2. Use with pipeline:
    
    pipeline = PatchBasedRestorationPipeline(patch_size=32, stride=16)
    restored_image, patches, processed, coords = pipeline.restore(
        input_image,
        my_restoration_model
    )

3. For batch processing:
    
    def batch_processor(patches_batch):
        # Run entire batch through model at once
        outputs = your_model_batch(patches_batch)
        return outputs
    
    # Decompose
    patches, coords = decompose_image_to_patches(image)
    
    # Batch process
    batch = np.stack(patches)
    processed_batch = batch_processor(batch)
    processed_patches = list(processed_batch)
    
    # Reconstruct
    restored = reconstruct_image_from_patches(
        processed_patches, 
        coords, 
        image.shape
    )
""")


if __name__ == "__main__":
    demo_restoration_workflow()
    demo_batch_processing()
    demo_integration_template()
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
