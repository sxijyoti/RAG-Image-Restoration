"""
Dataset embedding generation.

Generates CLIP embeddings for clean dataset patches.
MUST use exact same encoding as query pipeline for consistency.

This script is designed to run on Kaggle GPU (CUDA).
Results saved as numpy arrays for later FAISS indexing.
"""

import numpy as np
import os
from typing import List
from pathlib import Path


def get_clean_patches_from_dataset(dataset_dir: str) -> List[np.ndarray]:
    """
    Load clean patches from dataset directory.
    
    Expected structure:
    dataset_dir/
        clean_1.jpg
        clean_2.jpg
        ...
    
    Each image is converted to patches of 64x64 with stride 32.
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        List of patches (each 64x64x3 uint8 arrays)
    """
    from patch_extractor import extract_patches
    
    image_files = sorted(Path(dataset_dir).glob("*.jpg")) + \
                  sorted(Path(dataset_dir).glob("*.png"))
    
    all_patches = []
    
    print(f"Loading clean patches from {dataset_dir}...")
    print(f"Found {len(image_files)} images")
    
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(image_files)}...")
        
        # Extract patches (uses same 64x64, stride 32 as query pipeline)
        patches, _, _ = extract_patches(str(image_path), patch_size=64, stride=32)
        all_patches.extend(patches)
    
    print(f"Total patches extracted: {len(all_patches)}")
    return all_patches


def encode_dataset_patches(
    patches: List[np.ndarray],
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Encode clean patches using CLIP.
    
    CRITICAL: Must use exact same CLIP model as query pipeline.
    - Model: openai/clip-vit-base-patch32
    - Returns: Token embeddings (NOT pooled)
    - Output shape: (N, 50, 768)
    
    Args:
        patches: List of (64, 64, 3) patch arrays
        batch_size: Batch size for encoding
        device: Device ("cuda" for Kaggle GPU)
    
    Returns:
        embeddings: (N, 50, 768) token embeddings
    """
    from clip_encoder import CLIPPatchEncoder
    
    print(f"\nEncoding {len(patches)} patches with CLIP...")
    print(f"Batch size: {batch_size}, Device: {device}")
    
    encoder = CLIPPatchEncoder(device=device)
    
    # Process in batches
    all_embeddings = []
    
    for batch_start in range(0, len(patches), batch_size):
        batch_end = min(batch_start + batch_size, len(patches))
        batch = patches[batch_start:batch_end]
        
        # Encode batch
        batch_embeddings = encoder.encode_patches(batch, return_numpy=True)
        all_embeddings.append(batch_embeddings)
        
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  Encoded {batch_end}/{len(patches)} patches")
    
    # Concatenate all
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected: ({len(patches)}, 50, 768)")
    
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str,
) -> None:
    """
    Save embeddings to compressed numpy file.
    
    Args:
        embeddings: (N, 50, 768) embeddings
        output_path: Path to save .npz file
    """
    print(f"\nSaving embeddings to {output_path}...")
    
    # Use compressed storage to save space
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {file_size_mb:.1f} MB")


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load embeddings from saved file."""
    data = np.load(embeddings_path)
    return data["embeddings"]


def generate_dataset_embeddings(
    dataset_dir: str,
    output_path: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Complete pipeline: extract, encode, and save dataset embeddings.
    
    Args:
        dataset_dir: Path to clean image dataset
        output_path: Path to save embeddings
        device: Device for encoding ("cuda" on Kaggle)
    
    Returns:
        embeddings: (N, 50, 768) embeddings
    """
    print("=" * 70)
    print("DATASET EMBEDDING GENERATION")
    print("=" * 70)
    
    # Step 1: Extract patches
    patches = get_clean_patches_from_dataset(dataset_dir)
    
    # Step 2: Encode
    embeddings = encode_dataset_patches(patches, device=device)
    
    # Step 3: Save
    save_embeddings(embeddings, output_path)
    
    print("\n" + "=" * 70)
    print("✓ Dataset embeddings generation complete!")
    print("=" * 70)
    
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate dataset embeddings for RAG-IR system"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./clean_dataset",
        help="Path to clean image dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./clean_embeddings.npz",
        help="Path to save embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for encoding (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    
    # Run generation
    embeddings = generate_dataset_embeddings(
        args.dataset_dir,
        args.output,
        device=args.device,
    )
    
    print(f"\nDataset embeddings saved to: {args.output}")
    print(f"Shape: {embeddings.shape}")
    print("\nNext step: Load in rag_restoration_pipeline.py for inference")
