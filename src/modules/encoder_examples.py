"""
DA-CLIP Encoder examples and integration with Phase 1 patch extraction.

Demonstrates:
- Loading and initializing encoder
- Encoding single patches
- Batch encoding
- Integration with patch extraction
- Embeddings for FAISS retrieval
"""

import numpy as np
import torch
from pathlib import Path

from patch_extraction import PatchExtractor
from da_clip_encoder import DACLIPEncoder, load_encoder


def example_basic_encoding():
    """Example 1: Basic patch encoding."""
    print("=" * 60)
    print("Example 1: Basic Patch Encoding")
    print("=" * 60)
    
    # Load encoder
    print("\n[1] Loading DA-CLIP encoder...")
    encoder = load_encoder(debug=True)
    
    # Create synthetic patch
    print("\n[2] Creating test patch...")
    patch = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    print(f"   Patch shape: {patch.shape}")
    
    # Encode
    print("\n[3] Encoding patch...")
    embedding = encoder.encode_patch(patch, debug=True)
    print(f"   Embedding type: {type(embedding)}")
    print(f"   Embedding shape: {embedding.shape}")
    
    # Check normalization
    norm = torch.norm(embedding, p=2).item()
    print(f"   L2 norm: {norm:.6f} (should be ~1.0 if normalized)")


def example_batch_encoding():
    """Example 2: Batch encoding for efficiency."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Encoding")
    print("=" * 60)
    
    encoder = load_encoder(debug=False)
    
    # Create batch of patches
    print("\n[1] Creating batch of 10 patches...")
    patches = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    print(f"   Number of patches: {len(patches)}")
    print(f"   Each patch shape: {patches[0].shape}")
    
    # Encode batch
    print("\n[2] Encoding batch...")
    embeddings = encoder.encode_batch(patches, batch_size=4, debug=True)
    print(f"   Output shape: {embeddings.shape}")
    
    # Convert to numpy for downstream processing
    print("\n[3] Converting to numpy...")
    embeddings_np = encoder.to_numpy(embeddings)
    print(f"   Numpy shape: {embeddings_np.shape}")
    print(f"   Numpy dtype: {embeddings_np.dtype}")


def example_consistency_validation():
    """Example 3: Validate consistency across runs."""
    print("\n" + "=" * 60)
    print("Example 3: Consistency Validation")
    print("=" * 60)
    
    encoder = load_encoder(debug=False)
    
    # Create uniform patch for testing
    print("\n[1] Creating uniform test patch...")
    patch = np.ones((64, 64, 3), dtype=np.uint8) * 128
    
    # Run consistency check
    print("\n[2] Running consistency validation...")
    is_consistent = encoder.validate_consistency(patch, num_trials=3)
    
    print(f"\n   Result: {'✅ PASS' if is_consistent else '❌ FAIL'}")


def example_integration_with_phase1():
    """Example 4: Integration with Phase 1 patch extraction."""
    print("\n" + "=" * 60)
    print("Example 4: Integration with Phase 1 Patch Extraction")
    print("=" * 60)
    
    # Create synthetic image
    print("\n[1] Creating synthetic degraded image...")
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    print(f"   Image shape: {image.shape}")
    
    # Phase 1: Extract patches
    print("\n[2] Phase 1: Extracting patches...")
    extractor = PatchExtractor(patch_size=64, stride=32)
    patches, coords = extractor.extract(image, debug=True)
    print(f"   Extracted {len(patches)} patches")
    
    # Phase 2: Encode patches
    print("\n[3] Phase 2: Encoding patches...")
    encoder = load_encoder(debug=False)
    embeddings = encoder.encode_batch(patches, batch_size=16, debug=True)
    
    # Show results
    print("\n[4] Results Summary:")
    print(f"   Input image: {image.shape}")
    print(f"   Number of patches: {len(patches)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[-1]}")
    print(f"   Coordinates: {len(coords)} (first 3: {coords[:3]})")
    
    # Ready for Phase 3 (FAISS retrieval)
    print("\n[5] Next: These embeddings are ready for:")
    print("   - FAISS index building (Kaggle)")
    print("   - Similarity search (retrieval)")
    print("   - Fusion with retrieved embeddings")


def example_comparison_similar_vs_different():
    """Example 5: Compare embeddings of similar vs different patches."""
    print("\n" + "=" * 60)
    print("Example 5: Embedding Similarity Analysis")
    print("=" * 60)
    
    encoder = load_encoder(debug=False)
    
    # Create different patches
    print("\n[1] Creating test patches...")
    
    # Very similar patches (small perturbation)
    patch1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    patch1_perturbed = patch1.copy()
    patch1_perturbed[0:5, 0:5] = 130  # Small change
    
    # Very different patch
    patch2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    print("   Created: uniform patch, slightly perturbed patch, random patch")
    
    # Encode
    print("\n[2] Encoding patches...")
    emb1 = encoder.encode_patch(patch1)
    emb1_perturbed = encoder.encode_patch(patch1_perturbed)
    emb2 = encoder.encode_patch(patch2)
    
    # Calculate similarities (cosine)
    print("\n[3] Calculating similarities (cosine)...")
    sim_same = torch.nn.functional.cosine_similarity(emb1, emb1_perturbed, dim=-1).item()
    sim_diff = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).item()
    
    print(f"   Similarity (same patch vs perturbed): {sim_same:.4f}")
    print(f"   Similarity (uniform vs random): {sim_diff:.4f}")
    print(f"   Difference: {sim_same - sim_diff:.4f}")
    
    if sim_same > sim_diff:
        print("\n   ✅ Embeddings capture semantic similarity correctly")
    else:
        print("\n   ⚠️  Unexpected: random patches more similar")


def example_memory_efficiency():
    """Example 6: Memory efficiency with different batch sizes."""
    print("\n" + "=" * 60)
    print("Example 6: Memory Efficiency Investigation")
    print("=" * 60)
    
    encoder = load_encoder(debug=False)
    
    # Create large batch
    print("\n[1] Creating large batch of 100 patches...")
    patches = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(100)]
    
    # Test different batch sizes
    print("\n[2] Encoding with different batch sizes...")
    batch_sizes = [1, 16, 32, 64, 100]
    
    for bs in batch_sizes:
        try:
            embeddings = encoder.encode_batch(patches, batch_size=bs)
            print(f"   Batch size {bs:3d}: ✓ (shape: {embeddings.shape})")
        except Exception as e:
            print(f"   Batch size {bs:3d}: ✗ ({e})")


def example_dataset_embedding_pipeline():
    """Example 7: Simulating FAISS index building pipeline."""
    print("\n" + "=" * 60)
    print("Example 7: Dataset Embedding for FAISS Index (Kaggle Phase)")
    print("=" * 60)
    
    print("\n[Context] On Kaggle GPU, you would:")
    print("  1. Load large dataset of clean images")
    print("  2. Extract patches from ALL clean images")
    print("  3. Encode patches with THIS encoder")
    print("  4. Save embeddings to file")
    print("  5. Build FAISS index")
    
    print("\n[Simulation] Encoding synthetic clean image patches...")
    encoder = load_encoder(debug=False)
    
    # Simulate clean dataset
    print("\n  Creating 10 'clean' patches...")
    clean_patches = [np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    
    # Encode
    embeddings = encoder.encode_batch(clean_patches, debug=True)
    embeddings_np = encoder.to_numpy(embeddings)
    
    print("\n[Output for Kaggle]")
    print(f"  Embeddings shape: {embeddings_np.shape}")
    print(f"  Ready to save: embeddings.npy")
    print(f"  Then build FAISS index and sync to local")
    
    print("\n[Local inference]")
    print("  1. Load downloaded FAISS index")
    print("  2. Encode degraded patches (here)")
    print("  3. Search index for similar clean patches")
    print("  4. Use for fusion/decoding")


def run_all_examples():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "DA-CLIP ENCODER EXAMPLES" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        example_basic_encoding()
        example_batch_encoding()
        example_consistency_validation()
        example_integration_with_phase1()
        example_comparison_similar_vs_different()
        example_memory_efficiency()
        example_dataset_embedding_pipeline()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
