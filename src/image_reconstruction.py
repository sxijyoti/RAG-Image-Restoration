"""
Image Reconstruction Module for RAG-based Image Restoration

Reconstructs full-size restored images from decoded patches using coordinate
information and proper blending in overlapping regions.

This is Phase 7 of the pipeline: Patches → Full Restored Image
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from pathlib import Path
import logging

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageReconstructor:
    """
    Reconstructs full images from decoded patches with proper blending.
    
    Handles:
    - Placing patches at correct coordinates
    - Blending overlapping regions (averaging)
    - Converting tensor patches to image format
    - Optional visualization of patch grid
    """
    
    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        debug: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Image Reconstructor.
        
        Args:
            patch_size: Size of patches (default 64)
            stride: Stride used during extraction (default 32)
            debug: Print debug information
            logger: Optional logger instance
        """
        self.patch_size = patch_size
        self.stride = stride
        self.debug = debug
        self.logger = logger or self._setup_default_logger()
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup basic logger if none provided."""
        logger = logging.getLogger("ImageReconstructor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def reconstruct(
        self,
        decoded_patches: Union[torch.Tensor, np.ndarray, List],
        coords: List[Tuple[int, int]],
        original_image_shape: Tuple[int, int, int],
        blend_mode: str = "average"
    ) -> np.ndarray:
        """
        Reconstruct full image from decoded patches.
        
        Args:
            decoded_patches: Decoded patches, shape (N, 3, 64, 64) or list of (3, 64, 64)
            coords: List of (x, y) coordinates for each patch
            original_image_shape: Target shape (height, width, 3)
            blend_mode: How to blend overlaps ("average" or "max")
            
        Returns:
            Reconstructed image as numpy array, shape (H, W, 3), values [0, 1]
        """
        height, width, channels = original_image_shape
        
        # Convert patches to numpy if needed
        if isinstance(decoded_patches, torch.Tensor):
            patches_np = self._tensor_to_patches(decoded_patches)
        elif isinstance(decoded_patches, list):
            patches_np = self._list_to_patches(decoded_patches)
        else:
            patches_np = decoded_patches
        
        self.logger.info(f"Reconstructing image: {width}×{height}")
        self.logger.info(f"Number of patches: {len(patches_np)}")
        self.logger.info(f"Blend mode: {blend_mode}")
        
        if len(patches_np) != len(coords):
            raise ValueError(
                f"Number of patches ({len(patches_np)}) must match "
                f"number of coordinates ({len(coords)})"
            )
        
        # Initialize output and count arrays for blending
        if blend_mode == "average":
            reconstructed = np.zeros((height, width, channels), dtype=np.float32)
            count = np.zeros((height, width), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported blend_mode: {blend_mode}")
        
        # Place each patch at its coordinate
        for patch, (x, y) in zip(patches_np, coords):
            # Handle patches that go beyond image boundaries (shouldn't happen normally)
            y_end = min(y + self.patch_size, height)
            x_end = min(x + self.patch_size, width)
            
            patch_height = y_end - y
            patch_width = x_end - x
            
            # Add patch contribution
            reconstructed[y:y_end, x:x_end, :] += patch[:patch_height, :patch_width, :].astype(
                np.float32
            )
            count[y:y_end, x:x_end] += 1
        
        # Blend overlapping regions by averaging
        count_3d = np.expand_dims(count, axis=2)
        count_3d = np.maximum(count_3d, 1)  # Avoid division by zero
        reconstructed = reconstructed / count_3d
        
        # Ensure output is in [0, 1] range
        reconstructed = np.clip(reconstructed, 0, 1).astype(np.float32)
        
        self.logger.info(f"Reconstruction complete")
        self.logger.info(f"Output range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
        
        return reconstructed
    
    def _tensor_to_patches(self, patches: torch.Tensor) -> List[np.ndarray]:
        """
        Convert tensor patches (N, 3, 64, 64) to list of numpy arrays (H, W, 3).
        
        Args:
            patches: Tensor of shape (N, 3, 64, 64)
            
        Returns:
            List of N numpy arrays, each shape (64, 64, 3)
        """
        patches_list = []
        for i in range(patches.shape[0]):
            # Convert (3, H, W) to (H, W, 3)
            patch = patches[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            patches_list.append(patch)
        return patches_list
    
    def _list_to_patches(self, patches: List) -> List[np.ndarray]:
        """
        Convert list of tensor/numpy patches to list of numpy arrays (H, W, 3).
        
        Args:
            patches: List of patch tensors/arrays
            
        Returns:
            List of numpy arrays, each shape (64, 64, 3)
        """
        patches_list = []
        for patch in patches:
            if isinstance(patch, torch.Tensor):
                # Convert (C, H, W) to (H, W, C)
                if patch.dim() == 3:
                    patch = patch.permute(1, 2, 0).cpu().numpy()
                else:
                    patch = patch.cpu().numpy()
            elif isinstance(patch, np.ndarray):
                # If (C, H, W), convert to (H, W, C)
                if patch.ndim == 3 and patch.shape[0] == 3:
                    patch = np.transpose(patch, (1, 2, 0))
            
            patches_list.append(patch)
        
        return patches_list
    
    def save_reconstructed_image(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        format: str = "png"
    ) -> Path:
        """
        Save reconstructed image to disk.
        
        Args:
            image: Image array, shape (H, W, 3), values [0, 1]
            output_path: Path to save image
            format: Output format ("png" or "jpg")
            
        Returns:
            Path to saved image
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available. Install with: pip install pillow")
        
        output_path = Path(output_path)
        
        # Convert to 8-bit
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Save using PIL
        pil_image = Image.fromarray(image_uint8, mode="RGB")
        pil_image.save(str(output_path), format=format.upper())
        
        self.logger.info(f"Saved reconstructed image to: {output_path}")
        self.logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def visualize_patch_grid(
        self,
        image_shape: Tuple[int, int],
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[np.ndarray]:
        """
        Visualize patch grid overlay on image.
        
        Args:
            image_shape: Image shape (height, width)
            output_path: Optional path to save visualization
            
        Returns:
            Visualization array if save_image=True, else None
        """
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available, skipping visualization")
            return None
        
        height, width = image_shape
        
        # Create white background
        viz = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(viz)
        
        # Draw patch grid
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                rect = [x, y, x + self.patch_size, y + self.patch_size]
                draw.rectangle(rect, outline="blue", width=1)
        
        # Handle edge patches
        if (height - self.patch_size) % self.stride != 0:
            y_edge = height - self.patch_size
            for x in range(0, width - self.patch_size + 1, self.stride):
                rect = [x, y_edge, x + self.patch_size, y_edge + self.patch_size]
                draw.rectangle(rect, outline="red", width=1)
        
        if (width - self.patch_size) % self.stride != 0:
            x_edge = width - self.patch_size
            for y in range(0, height - self.patch_size + 1, self.stride):
                rect = [x_edge, y, x_edge + self.patch_size, y + self.patch_size]
                draw.rectangle(rect, outline="red", width=1)
        
        if (height - self.patch_size) % self.stride != 0 and (width - self.patch_size) % self.stride != 0:
            y_edge = height - self.patch_size
            x_edge = width - self.patch_size
            rect = [x_edge, y_edge, x_edge + self.patch_size, y_edge + self.patch_size]
            draw.rectangle(rect, outline="green", width=2)
        
        if output_path:
            viz.save(str(output_path))
            self.logger.info(f"Saved patch grid visualization to: {output_path}")
        
        return np.array(viz)


def reconstruct_image_from_patches(
    decoded_patches: Union[torch.Tensor, np.ndarray],
    coords: List[Tuple[int, int]],
    image_shape: Tuple[int, int, int],
    patch_size: int = 64,
    stride: int = 32,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Convenience function to reconstruct image from patches.
    
    Args:
        decoded_patches: Decoded patches tensor/array
        coords: Patch coordinates
        image_shape: Target image shape (H, W, 3)
        patch_size: Patch size
        stride: Stride
        logger: Optional logger
        
    Returns:
        Reconstructed image as numpy array
    """
    reconstructor = ImageReconstructor(
        patch_size=patch_size,
        stride=stride,
        logger=logger
    )
    return reconstructor.reconstruct(decoded_patches, coords, image_shape)
