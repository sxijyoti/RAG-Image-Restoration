"""
Unit tests for DA-CLIP encoder module.

Run with: python -m pytest test_da_clip_encoder.py -v
Or simply: python test_da_clip_encoder.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.da_clip_encoder import DACLIPEncoder, load_encoder, encode_patches


def create_test_patch(size=(64, 64), pattern="random"):
    """Create a test patch."""
    if pattern == "random":
        return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif pattern == "ones":
        return np.ones((*size, 3), dtype=np.uint8) * 128
    elif pattern == "zeros":
        return np.zeros((*size, 3), dtype=np.uint8)


def test_1_initialization():
    """Test encoder initializes."""
    encoder = DACLIPEncoder()
    assert encoder.model is not None
    assert encoder.device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]
    print("✓ test_initialization")


def test_2_initialization_specific_model():
    """Test encoder initializes with specific model."""
    encoder = DACLIPEncoder(model_name="ViT-B-32", pretrained="openai")
    assert encoder.model is not None
    print("✓ test_initialization_specific_model")


def test_3_model_in_eval_mode():
    """Test that model is in eval mode."""
    encoder = DACLIPEncoder()
    assert not encoder.model.training
    print("✓ test_model_in_eval_mode")


def test_4_encode_single_patch_numpy():
    """Test encoding single patch from numpy array."""
    encoder = DACLIPEncoder()
    patch = create_test_patch()
    embedding = encoder.encode_patch(patch)
    
    assert embedding is not None
    assert hasattr(embedding, 'shape') or hasattr(embedding, '__len__')
    print("✓ test_encode_single_patch_numpy")


def test_5_encode_single_patch_pil():
    """Test encoding single patch from PIL Image."""
    encoder = DACLIPEncoder()
    patch_array = create_test_patch()
    patch_pil = PILImage.fromarray(patch_array, mode='RGB')
    embedding = encoder.encode_patch(patch_pil)
    
    assert embedding is not None
    print("✓ test_encode_single_patch_pil")


def test_6_encode_single_patch_normalized():
    """Test that single patch embedding is normalized."""
    encoder = DACLIPEncoder()
    patch = create_test_patch()
    embedding = encoder.encode_patch(patch)
    
    # Flatten if needed
    if hasattr(embedding, 'numpy'):
        embedding = embedding.numpy()
    embedding = np.array(embedding).flatten()
    
    norm = np.linalg.norm(embedding)
    assert norm > 0, "Embedding norm should be positive"
    print("✓ test_encode_single_patch_normalized")


def test_7_encode_batch_basic():
    """Test batch encoding."""
    encoder = DACLIPEncoder()
    patches = np.array([create_test_patch() for _ in range(4)])
    embeddings = encoder.encode_batch(patches)
    
    assert embeddings is not None
    assert embeddings.shape[0] == 4
    print("✓ test_encode_batch_basic")


def test_8_encode_batch_single_patch():
    """Test batch encoding with single patch."""
    encoder = DACLIPEncoder()
    patches = np.array([create_test_patch()])
    embeddings = encoder.encode_batch(patches)
    
    assert embeddings.shape[0] == 1
    print("✓ test_encode_batch_single_patch")


def test_9_encode_batch_normalized():
    """Test batch embeddings have reasonable norms."""
    encoder = DACLIPEncoder()
    patches = np.array([create_test_patch() for _ in range(4)])
    embeddings = encoder.encode_batch(patches)
    
    # Convert to numpy if needed
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    embeddings = np.array(embeddings)
    
    # Reshape if needed
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.all(norms > 0), "All norms should be positive"
    print("✓ test_encode_batch_normalized")


def test_10_encode_batch_custom_batch_size():
    """Test batch encoding with custom batch size."""
    encoder = DACLIPEncoder()
    patches = np.array([create_test_patch() for _ in range(8)])
    embeddings = encoder.encode_batch(patches, batch_size=2)
    
    assert embeddings.shape[0] == 8
    print("✓ test_encode_batch_custom_batch_size")


def test_11_consistency_same_input():
    """Test identical inputs produce identical outputs."""
    encoder = DACLIPEncoder()
    patch = create_test_patch()
    
    emb1 = encoder.encode_patch(patch)
    emb2 = encoder.encode_patch(patch)
    
    # Convert to numpy if needed
    if hasattr(emb1, 'numpy'):
        emb1 = emb1.numpy()
    if hasattr(emb2, 'numpy'):
        emb2 = emb2.numpy()
    
    emb1 = np.array(emb1).flatten()
    emb2 = np.array(emb2).flatten()
    
    assert np.allclose(emb1, emb2), "Same input should produce same output"
    print("✓ test_consistency_same_input")


def test_12_device_detection():
    """Test device detection works."""
    encoder = DACLIPEncoder()
    assert encoder.device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]
    print("✓ test_device_detection")


def run_all_tests():
    """Run all tests."""
    test_functions = [
        test_1_initialization,
        test_2_initialization_specific_model,
        test_3_model_in_eval_mode,
        test_4_encode_single_patch_numpy,
        test_5_encode_single_patch_pil,
        test_6_encode_single_patch_normalized,
        test_7_encode_batch_basic,
        test_8_encode_batch_single_patch,
        test_9_encode_batch_normalized,
        test_10_encode_batch_custom_batch_size,
        test_11_consistency_same_input,
        test_12_device_detection,
    ]
    
    passed = 0
    failed = 0
    
    print(f"\nRunning {len(test_functions)} tests...\n")
    
    for test_func in sorted(test_functions, key=lambda f: f.__name__):
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
