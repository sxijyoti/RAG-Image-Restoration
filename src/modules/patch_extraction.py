"""
Patch Extraction Module for Image Restoration

Fixes applied:
1. Gaussian-weighted blending replaces flat average → eliminates seam/tiling artifacts
2. reconstruct() range check fixed → no longer silently skips *255 conversion
3. Feathered patch borders ensure smooth transitions
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path
from PIL import Image


def _make_gaussian_window(patch_size: int, sigma_ratio: float = 0.35) -> np.ndarray:
    """
    Create a 2D Gaussian weight window for a patch.
    Peaks at 1.0 in the center, falls toward ~0.1 at corners.
    This ensures overlapping patches blend smoothly with no visible seams.
    """
    sigma = patch_size * sigma_ratio
    ax = np.arange(patch_size) - (patch_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    window = np.outer(gauss_1d, gauss_1d)  # (patch_size, patch_size)
    window = window / window.max()          # normalize peak to 1.0
    return window.astype(np.float32)


class PatchExtractor:
    """
    Extracts overlapping patches from images and reconstructs images from patches.

    Attributes:
        patch_size (int): Width/height of square patches (default: 64)
        stride (int): Step size between patch starts (default: 32)
    """

    def __init__(self, patch_size: int = 64, stride: int = 32):
        self.patch_size = patch_size
        self.stride = stride

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if stride > patch_size:
            raise ValueError("stride should not exceed patch_size")

        # Pre-compute the Gaussian window once
        self._gauss_window = _make_gaussian_window(patch_size)  # (P, P)

    # ------------------------------------------------------------------
    def extract(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        return_coords: bool = True,
        debug: bool = False
    ) -> Union[Tuple[List[np.ndarray], List[Tuple[int, int]]], List[np.ndarray]]:
        """
        Extract overlapping patches from an image.

        Args:
            image: Input image as numpy array (H, W, 3), PIL Image, or path
            return_coords: If True, return (patches, coords)
            debug: Print extraction statistics

        Returns:
            If return_coords=True: (patches, coords)
                patches: List of numpy arrays shape (64, 64, 3)
                coords:  List of (x, y) top-left corners
        """
        image_array = self._load_image(image)
        height, width, channels = image_array.shape

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image size ({height}×{width}) must be >= {self.patch_size}×{self.patch_size}"
            )

        patches, coords = [], []

        # Regular sliding-window grid
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patches.append(image_array[y:y + self.patch_size, x:x + self.patch_size, :].copy())
                coords.append((x, y))

        # Bottom-edge strip
        if (height - self.patch_size) % self.stride != 0:
            y_edge = height - self.patch_size
            for x in range(0, width - self.patch_size + 1, self.stride):
                patches.append(image_array[y_edge:y_edge + self.patch_size, x:x + self.patch_size, :].copy())
                coords.append((x, y_edge))

        # Right-edge strip
        if (width - self.patch_size) % self.stride != 0:
            x_edge = width - self.patch_size
            for y in range(0, height - self.patch_size + 1, self.stride):
                patches.append(image_array[y:y + self.patch_size, x_edge:x_edge + self.patch_size, :].copy())
                coords.append((x_edge, y))

        # Bottom-right corner
        if ((height - self.patch_size) % self.stride != 0 and
                (width - self.patch_size) % self.stride != 0):
            y_edge = height - self.patch_size
            x_edge = width - self.patch_size
            patches.append(image_array[y_edge:y_edge + self.patch_size, x_edge:x_edge + self.patch_size, :].copy())
            coords.append((x_edge, y_edge))

        if debug:
            print(f"Image size: {width}×{height}")
            print(f"Total patches: {len(patches)}")

        return (patches, coords) if return_coords else patches

    # ------------------------------------------------------------------
    def reconstruct(
        self,
        patches: List[np.ndarray],
        coords: List[Tuple[int, int]],
        image_shape: Tuple[int, int, int],
        blend_mode: str = "gaussian"   # default changed to gaussian
    ) -> np.ndarray:
        """
        Reconstruct image from patches using Gaussian-weighted blending.

        Gaussian weighting gives higher influence to the center of each patch
        and lower influence to its edges, completely eliminating seam artifacts
        that appear with flat (average) blending.

        Args:
            patches: List of patch arrays (patch_size, patch_size, channels)
            coords:  List of (x, y) for each patch
            image_shape: Target (height, width, channels)
            blend_mode: "gaussian" (recommended) or "average"

        Returns:
            Reconstructed image as uint8 numpy array
        """
        if len(patches) != len(coords):
            raise ValueError(f"patches ({len(patches)}) and coords ({len(coords)}) must match")
        if len(patches) == 0:
            raise ValueError("No patches provided")

        height, width, channels = image_shape

        # Accumulate weighted patch contributions
        reconstructed = np.zeros((height, width, channels), dtype=np.float64)
        weight_sum   = np.zeros((height, width),            dtype=np.float64)

        if blend_mode == "gaussian":
            window = self._gauss_window.astype(np.float64)   # (P, P)
        else:
            window = np.ones((self.patch_size, self.patch_size), dtype=np.float64)

        for patch, (x, y) in zip(patches, coords):
            y_end = min(y + self.patch_size, height)
            x_end = min(x + self.patch_size, width)
            ph = y_end - y
            pw = x_end - x

            # patch may be float [0,1] or uint8 [0,255] — normalise to float
            p = patch[:ph, :pw, :].astype(np.float64)

            w = window[:ph, :pw]                              # (ph, pw)
            reconstructed[y:y_end, x:x_end, :] += p * w[:, :, np.newaxis]
            weight_sum[y:y_end, x:x_end]        += w

        # Avoid div-by-zero on any uncovered pixel
        weight_sum = np.maximum(weight_sum, 1e-8)
        reconstructed /= weight_sum[:, :, np.newaxis]

        # ----------------------------------------------------------------
        # FIX: explicit range detection instead of the broken max() check.
        # Decoded patches are always in [0, 1] (after tanh + normalise).
        # Training targets are also stored as [0, 1] tensors.
        # Only raw uint8 numpy patches from PIL would be in [0, 255].
        # ----------------------------------------------------------------
        sample_patch = patches[0]
        if sample_patch.dtype in (np.float32, np.float64) and sample_patch.max() <= 1.0 + 1e-6:
            # Float [0, 1] → scale to [0, 255]
            reconstructed = np.clip(reconstructed * 255.0, 0, 255)
        else:
            # Already in [0, 255] range (uint8 inputs)
            reconstructed = np.clip(reconstructed, 0, 255)

        return reconstructed.astype(np.uint8)

    # ------------------------------------------------------------------
    def get_patch_grid_info(self, image_shape: Tuple[int, int]) -> dict:
        height, width = image_shape
        num_y = len(range(0, height - self.patch_size + 1, self.stride))
        num_x = len(range(0, width - self.patch_size + 1, self.stride))
        has_bottom = (height - self.patch_size) % self.stride != 0
        has_right  = (width  - self.patch_size) % self.stride != 0
        has_corner = has_bottom and has_right

        total = num_y * num_x
        if has_bottom: total += num_x
        if has_right:  total += num_y
        if has_corner: total += 1

        return {
            "image_shape": image_shape,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "regular_grid": (num_y, num_x),
            "has_bottom_edge": has_bottom,
            "has_right_edge": has_right,
            "has_corner": has_corner,
            "total_patches": total,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(image: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=2)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")
            return image
        elif isinstance(image, (str, Path)):
            img = Image.open(image)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img)
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.array(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def extract_patches(
    image: Union[np.ndarray, str, Path, Image.Image],
    patch_size: int = 64,
    stride: int = 32,
    debug: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    return extractor.extract(image, return_coords=True, debug=debug)


def reconstruct_image(
    patches: List[np.ndarray],
    coords: List[Tuple[int, int]],
    image_shape: Tuple[int, int, int],
    patch_size: int = 64,
    stride: int = 32,
    blend_mode: str = "gaussian"
) -> np.ndarray:
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    return extractor.reconstruct(patches, coords, image_shape, blend_mode=blend_mode)
