"""
Unit tests for patch extraction module.

Run with: python -m pytest test_patch_extraction.py -v
Or simply: python test_patch_extraction.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modules.patch_extraction import PatchExtractor, extract_patches, reconstruct_image


class TestPatchExtractor:
    """Test suite for PatchExtractor class."""
    
    def setup_method(self):
        """Setup before each test."""
        self.extractor = PatchExtractor(patch_size=64, stride=32)
    
    def create_test_image(self, h=256, w=256, pattern="random"):
        """Create a simple test image."""
        if pattern == "random":
            return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        elif pattern == "ones":
            return np.ones((h, w, 3), dtype=np.uint8) * 128
        elif pattern == "gradient":
            x = np.linspace(0, 255, w, dtype=np.uint8)
            y = np.linspace(0, 255, h, dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            return np.stack([xx, yy, (xx.astype(int) + yy.astype(int))//2], axis=2).astype(np.uint8)
    
    # ===== Basic Functionality Tests =====
    
    def test_extract_returns_patches_and_coords(self):
        """Test that extract returns both patches and coordinates."""
        img = self.create_test_image()
        patches, coords = self.extractor.extract(img)
        
        assert isinstance(patches, list)
        assert isinstance(coords, list)
        assert len(patches) == len(coords)
        assert len(patches) > 0
    
    def test_extract_patch_shape(self):
        """Test that patches have correct shape."""
        img = self.create_test_image()
        patches, _ = self.extractor.extract(img)
        
        for patch in patches:
            assert patch.shape == (64, 64, 3)
            assert patch.dtype == np.uint8
    
    def test_extract_coordinates_range(self):
        """Test that coordinates are within image bounds."""
        img = self.create_test_image(256, 256)
        _, coords = self.extractor.extract(img)
        
        for x, y in coords:
            assert 0 <= x <= 256 - 64
            assert 0 <= y <= 256 - 64
    
    def test_extract_no_out_of_bounds(self):
        """Test that extraction doesn't go out of bounds."""
        img = self.create_test_image(100, 100)
        patches, coords = self.extractor.extract(img)
        
        h, w = img.shape[:2]
        for patch, (x, y) in zip(patches, coords):
            # All patches should be valid
            assert patch.shape == (64, 64, 3)
            assert x + 64 <= w
            assert y + 64 <= h
    
    # ===== Reconstruction Tests =====
    
    def test_reconstruction_lossless_unmodified(self):
        """Test that unmodified patches reconstruct exactly."""
        img = self.create_test_image(256, 256, "ones")
        patches, coords = self.extractor.extract(img)
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        
        mse = np.mean((img.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        assert mse < 1e-6, f"MSE too high: {mse}"
    
    def test_reconstruction_shape(self):
        """Test that reconstruction has correct output shape."""
        img = self.create_test_image(200, 300)
        patches, coords = self.extractor.extract(img)
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        
        assert reconstructed.shape == img.shape
        assert reconstructed.dtype == np.uint8
    
    def test_reconstruction_value_range(self):
        """Test that reconstructed image has valid value range."""
        img = self.create_test_image()
        patches, coords = self.extractor.extract(img)
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        
        assert reconstructed.min() >= 0
        assert reconstructed.max() <= 255
    
    def test_reconstruction_gradient(self):
        """Test reconstruction with gradient image."""
        img = self.create_test_image(200, 200, "gradient")
        patches, coords = self.extractor.extract(img)
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        
        # Gradient should reconstruct perfectly
        diff = np.abs(img.astype(int) - reconstructed.astype(int))
        max_error = diff.max()
        assert max_error <= 1, f"Gradient reconstruction error: {max_error}"
    
    # ===== Edge Case Tests =====
    
    def test_exact_patch_size_image(self):
        """Test image exactly matching patch size."""
        img = self.create_test_image(64, 64)
        patches, coords = self.extractor.extract(img)
        
        assert len(patches) == 1
        assert coords[0] == (0, 0)
        assert np.array_equal(patches[0], img)
    
    def test_slightly_larger_than_patch(self):
        """Test image slightly larger than patch size."""
        img = self.create_test_image(80, 80)
        patches, coords = self.extractor.extract(img)
        
        # Should have multiple patches due to edge patches
        assert len(patches) >= 2
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        assert reconstructed.shape == img.shape
    
    def test_non_aligned_image(self):
        """Test image where height/width not divisible by stride."""
        img = self.create_test_image(200, 200)  # Non-aligned
        patches, coords = self.extractor.extract(img)
        
        # Should still work
        assert len(patches) > 0
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        assert reconstructed.shape == img.shape
    
    def test_rectangular_image(self):
        """Test non-square image."""
        img = self.create_test_image(300, 500)
        patches, coords = self.extractor.extract(img)
        
        assert len(patches) > 0
        reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
        
        mse = np.mean((img.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
        assert mse < 1e-5, f"MSE too high for rectangular: {mse}"
    
    def test_image_too_small_raises_error(self):
        """Test that too-small images raise ValueError."""
        img = self.create_test_image(32, 32)
        
        try:
            self.extractor.extract(img)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
    
    # ===== Grid Info Tests =====
    
    def test_grid_info_basic(self):
        """Test grid info calculation."""
        info = self.extractor.get_patch_grid_info((256, 256))
        
        assert info['total_patches'] > 0
        assert info['regular_grid'][0] > 0
        assert info['regular_grid'][1] > 0
        assert info['patch_size'] == 64
        assert info['stride'] == 32
    
    def test_grid_info_exact_size(self):
        """Test grid for exact patch size."""
        info = self.extractor.get_patch_grid_info((64, 64))
        
        assert info['total_patches'] == 1
        assert info['regular_grid'] == (1, 1)
        assert not info['has_bottom_edge']
        assert not info['has_right_edge']
        assert not info['has_corner']
    
    def test_grid_info_matches_extraction(self):
        """Test that grid info matches actual extraction."""
        img = self.create_test_image(300, 400)
        patches, _ = self.extractor.extract(img)
        
        info = self.extractor.get_patch_grid_info(img.shape[:2])
        assert info['total_patches'] == len(patches)
    
    # ===== Input Format Tests =====
    
    def test_extract_from_numpy_array(self):
        """Test extraction from numpy array."""
        img = self.create_test_image()
        patches, _ = self.extractor.extract(img)
        assert len(patches) > 0
    
    def test_extract_from_pil_image(self):
        """Test extraction from PIL Image."""
        img_array = self.create_test_image()
        img_pil = PILImage.fromarray(img_array, mode='RGB')
        
        patches, _ = self.extractor.extract(img_pil)
        assert len(patches) > 0
    
    def test_extract_from_grayscale_converts_to_rgb(self):
        """Test that grayscale images are converted to RGB."""
        img_gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        patches, _ = self.extractor.extract(img_gray)
        assert len(patches) > 0
        assert patches[0].shape == (64, 64, 3)
    
    # ===== Error Handling Tests =====
    
    def test_reconstruct_mismatched_lengths_raises_error(self):
        """Test that mismatched patches/coords raises error."""
        img = self.create_test_image()
        patches, coords = self.extractor.extract(img)
        
        try:
            self.extractor.reconstruct(patches[:-1], coords, img.shape)
            assert False, "Should raise ValueError"
        except ValueError:
            pass  # Expected
    
    def test_reconstruct_wrong_patch_shape_raises_error(self):
        """Test that wrong patch shape raises error."""
        img = self.create_test_image()
        _, coords = self.extractor.extract(img)
        
        # Create patches with wrong shape
        wrong_patches = [np.ones((65, 65, 3), dtype=np.uint8) for _ in coords]
        
        try:
            self.extractor.reconstruct(wrong_patches, coords, img.shape)
            assert False, "Should raise ValueError"
        except ValueError:
            pass  # Expected
    
    # ===== Convenience Function Tests =====
    
    def test_convenience_extract_function(self):
        """Test quick extract function."""
        img = self.create_test_image()
        patches, coords = extract_patches(img)
        
        assert len(patches) > 0
        assert len(coords) > 0
    
    def test_convenience_reconstruct_function(self):
        """Test quick reconstruct function."""
        img = self.create_test_image()
        patches, coords = extract_patches(img)
        reconstructed = reconstruct_image(patches, coords, img.shape)
        
        assert reconstructed.shape == img.shape
    
    # ===== Parameterized Tests =====
    
    def test_multiple_image_sizes(self):
        """Test extraction and reconstruction with various sizes."""
        test_sizes = [
            (64, 64),
            (100, 100),
            (128, 128),
            (200, 200),
            (256, 256),
            (200, 300),
            (512, 512),
        ]
        
        for h, w in test_sizes:
            img = self.create_test_image(h, w)
            patches, coords = self.extractor.extract(img)
            reconstructed = self.extractor.reconstruct(patches, coords, img.shape)
            
            mse = np.mean((img.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
            assert mse < 1e-5, f"Failed for size {h}×{w}: MSE={mse}"


def run_all_tests():
    """Run all tests."""
    test_obj = TestPatchExtractor()
    test_methods = [m for m in dir(test_obj) if m.startswith('test_')]
    
    passed = 0
    failed = 0
    
    print(f"\nRunning {len(test_methods)} tests...\n")
    
    for method_name in sorted(test_methods):
        try:
            test_obj.setup_method()
            method = getattr(test_obj, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
