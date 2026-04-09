"""
Example usage of the RAG-Image-Restoration pipeline.

This script demonstrates different ways to use the restoration system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import run_pipeline, RestorationPipeline


def example_basic():
    """Basic single-image restoration."""
    print("Example 1: Basic Restoration")
    print("-" * 50)
    
    restored = run_pipeline(
        image_path="input.png",
        index_path="indexes/clean_patches.index",
        patch_map_path="indexes/patch_map.json",
        output_path="restored.png",
        config_path="configs/config.yaml"
    )
    
    print(f"Restored image shape: {restored.shape}")
    print("✓ Saved to restored.png\n")


def example_pipeline_object():
    """Using RestorationPipeline object for multiple images."""
    print("Example 2: Using Pipeline Object")
    print("-" * 50)
    
    # Initialize pipeline once
    pipeline = RestorationPipeline(
        index_path="indexes/clean_patches.index",
        patch_map_path="indexes/patch_map.json",
        config_path="configs/config.yaml"
    )
    
    # Restore multiple images
    images = ["image1.png", "image2.png", "image3.png"]
    for img_path in images:
        if Path(img_path).exists():
            output_path = f"{Path(img_path).stem}_restored.png"
            pipeline.run(img_path, output_path)
    
    print("✓ Pipeline processing complete\n")


def example_batch():
    """Batch processing of images."""
    print("Example 3: Batch Processing")
    print("-" * 50)
    
    pipeline = RestorationPipeline(
        index_path="indexes/clean_patches.index",
        patch_map_path="indexes/patch_map.json"
    )
    
    pipeline.run_batch(
        image_dir="degraded_images/",
        output_dir="restored_images/",
        pattern="*.png"
    )
    
    print("✓ Batch processing complete\n")


def example_custom_config():
    """Using custom configuration."""
    print("Example 4: Custom Configuration")
    print("-" * 50)
    
    # Create custom config for different settings
    custom_config = {
        'patch_size': 64,
        'stride': 32,
        'top_k': 10,  # Increased k
        'fusion_method': 'mean',
        'device': 'cuda',
    }
    
    # Save custom config
    import yaml
    with open("custom_config.yaml", "w") as f:
        yaml.dump(custom_config, f)
    
    restored = run_pipeline(
        image_path="input.png",
        index_path="indexes/clean_patches.index",
        patch_map_path="indexes/patch_map.json",
        output_path="restored_custom.png",
        config_path="custom_config.yaml"
    )
    
    print("✓ Custom config processing complete\n")


def example_with_decoder():
    """Using trained decoder checkpoint."""
    print("Example 5: With Decoder Checkpoint")
    print("-" * 50)
    
    restored = run_pipeline(
        image_path="input.png",
        index_path="indexes/clean_patches.index",
        patch_map_path="indexes/patch_map.json",
        output_path="restored_trained.png",
        decoder_checkpoint="weights/decoder_epoch_50.pt"
    )
    
    print("✓ Trained decoder restoration complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RAG-Image-Restoration Examples")
    print("=" * 50 + "\n")
    
    # Uncomment the examples you want to run:
    
    # example_basic()
    # example_pipeline_object()
    # example_batch()
    # example_custom_config()
    # example_with_decoder()
    
    print("\nNote: Uncomment examples in the script to run them.")
    print("Each example requires appropriate input files.")
