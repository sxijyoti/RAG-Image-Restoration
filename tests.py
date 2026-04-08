"""
Test Suite for RAG Image Restoration Pipeline
Verifies each module works correctly
"""

import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from configs.config import DEVICE, DACLIP_OUTPUT_SHAPE, TENSORS_DIR
from src.patch_segmentation import PatchSegmentor
from src.clip_encoder import DAClipEncoder
from src.retrieval import RetrieverFAISS
from src.context_fusion import ContextFusionPipeline
from src.run_pipeline import RAGRestorationPipeline, verify_output


def create_dummy_image(size=(256, 256)):
    """Create a random test image"""
    return Image.new("RGB", size, color=(128, 128, 128))


def test_patch_segmentation():
    """Test patch segmentation module"""
    print("\n" + "="*70)
    print("TEST 1: Patch Segmentation")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        img_path = Path(tmpdir) / "test.png"
        test_img = create_dummy_image((256, 256))
        test_img.save(img_path)
        
        # Segment
        segmentor = PatchSegmentor(patch_size=64, stride=32)
        patches = segmentor.segment(str(img_path))
        
        # Verify
        assert len(patches) > 0, "No patches extracted"
        assert patches[0][0].shape == (3, 64, 64), f"Wrong patch shape: {patches[0][0].shape}"
        assert patches[0][1] >= 0 and patches[0][2] >= 0, "Wrong coordinates"
        
        print(f"✓ Segmentation test passed!")
        print(f"  Extracted {len(patches)} patches")
        print(f"  Patch shape: {patches[0][0].shape}")
        print(f"  First patch coords: ({patches[0][1]}, {patches[0][2]})")
        
        return True


def test_clip_encoder():
    """Test CLIP encoder module"""
    print("\n" + "="*70)
    print("TEST 2: CLIP Encoder")
    print("="*70)
    
    encoder = DAClipEncoder(pretrained=False)
    
    # Create dummy patch
    dummy_patch = torch.rand(3, 64, 64)
    
    # Encode
    embedding = encoder.encode(dummy_patch)
    
    # Verify
    assert embedding.shape == torch.Size(DACLIP_OUTPUT_SHAPE), \
        f"Wrong shape: {embedding.shape} vs {DACLIP_OUTPUT_SHAPE}"
    assert embedding.device.type == DEVICE.type, \
        f"Wrong device: {embedding.device} vs {DEVICE}"
    assert not torch.isnan(embedding).any(), "NaN values in embedding"
    
    print(f"✓ CLIP encoder test passed!")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Output device: {embedding.device}")
    print(f"  Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    return True


def test_retrieval():
    """Test retrieval module"""
    print("\n" + "="*70)
    print("TEST 3: Retrieval")
    print("="*70)
    
    retriever = RetrieverFAISS(top_k=5)
    
    # Create dummy query
    dummy_query = torch.randn(DACLIP_OUTPUT_SHAPE)
    
    # Retrieve
    retrieved = retriever.retrieve(dummy_query)
    
    # Verify
    assert len(retrieved) == 5, f"Wrong number of results: {len(retrieved)}"
    for i, emb in enumerate(retrieved):
        assert emb.shape == torch.Size(DACLIP_OUTPUT_SHAPE), \
            f"Retrieved {i}: wrong shape {emb.shape}"
    
    print(f"✓ Retrieval test passed!")
    print(f"  Retrieved {len(retrieved)} embeddings")
    for i, emb in enumerate(retrieved):
        print(f"  Embedding {i}: shape={emb.shape}, device={emb.device}")
    
    return True


def test_context_fusion():
    """Test context fusion module"""
    print("\n" + "="*70)
    print("TEST 4: Context Fusion")
    print("="*70)
    
    fusion = ContextFusionPipeline()
    
    # Create dummy embeddings
    degraded = torch.randn(DACLIP_OUTPUT_SHAPE)
    retrieved = [torch.randn(DACLIP_OUTPUT_SHAPE) for _ in range(5)]
    
    # Fuse
    fused = fusion.fuse(degraded, retrieved)
    
    # Verify
    assert fused.shape == torch.Size(DACLIP_OUTPUT_SHAPE), \
        f"Wrong fused shape: {fused.shape}"
    assert fused.device.type == DEVICE.type, \
        f"Wrong device: {fused.device}"
    
    print(f"✓ Fusion test passed!")
    print(f"  Fused shape: {fused.shape}")
    print(f"  Fused device: {fused.device}")
    print(f"  Value range: [{fused.min():.4f}, {fused.max():.4f}]")
    
    return True


def test_full_pipeline():
    """Test full pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Full Pipeline")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        img_path = Path(tmpdir) / "test_degraded.png"
        test_img = create_dummy_image((384, 384))  # Use larger size
        test_img.save(img_path)
        
        # Run pipeline
        pipeline = RAGRestorationPipeline()
        result = pipeline.run(str(img_path))
        
        # Verify
        assert result["num_patches"] > 0, "No patches processed"
        assert result["fused_tensors_saved"] > 0, "No tensors saved"
        
        # Check output files
        tensor_dir = Path(result["tensor_dir"])
        coords_file = Path(result["coordinates_file"])
        
        assert tensor_dir.exists(), f"Tensor directory doesn't exist: {tensor_dir}"
        assert coords_file.exists(), f"Coordinates file doesn't exist: {coords_file}"
        
        # Verify first tensor
        first_tensor = tensor_dir / "fused_patch_0000.pt"
        assert first_tensor.exists(), f"First tensor not found: {first_tensor}"
        
        tensor = torch.load(first_tensor, map_location='cpu')
        assert tensor.shape == torch.Size(DACLIP_OUTPUT_SHAPE), \
            f"Wrong tensor shape: {tensor.shape}"
        
        print(f"✓ Full pipeline test passed!")
        print(f"  Processed image: {result['image_path']}")
        print(f"  Patches: {result['num_patches']}")
        print(f"  Saved tensors: {result['fused_tensors_saved']}")
        print(f"  Output directory: {result['tensor_dir']}")
        
        return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RAG IMAGE RESTORATION - TEST SUITE")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    tests = [
        ("Patch Segmentation", test_patch_segmentation),
        ("CLIP Encoder", test_clip_encoder),
        ("Retrieval", test_retrieval),
        ("Context Fusion", test_context_fusion),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED!")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
