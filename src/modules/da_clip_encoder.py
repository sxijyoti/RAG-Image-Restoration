"""
DA-CLIP Patch Encoder Module for Image Restoration

Encodes image patches into degradation-aware embeddings using DA-CLIP (ViT-B/32).

CRITICAL: This module MUST use:
  - model.encode_image(image, control=True)
  - Extract: degra_features (NOT image_features)
  - Consistency is essential for FAISS retrieval
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
from PIL import Image as PILImage

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("⚠️  Warning: open_clip not installed. Install with: pip install open-clip-torch")


class DACLIPEncoder:
    """
    Encodes image patches into degradation-aware embeddings using DA-CLIP.
    
    DA-CLIP is a variant of CLIP designed to extract degradation-aware features
    that are more suitable for image restoration tasks.
    
    Attributes:
        model: Loaded DA-CLIP model
        preprocess: Preprocessing pipeline for the model
        device: torch device (cpu or mps)
        embedding_dim: Dimension of output embeddings
        normalize: Whether to L2-normalize embeddings
    """
    
    def __init__(
        self, 
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        normalize: bool = True,
        device: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize DA-CLIP encoder.
        
        Args:
            model_name: Model architecture (default: ViT-B-32)
            pretrained: Pretrained weights (default: laion2b_s34b_b79k)
            normalize: L2-normalize embeddings (default: True)
            device: Device to use ("cpu", "mps", or None for auto-detect)
            debug: Print initialization details
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip not installed. "
                "Install with: pip install open-clip-torch"
            )
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.debug = debug
        self.is_daclip = False  # Will be set during model loading
        
        # Auto-detect device
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = torch.device(device)
        
        if debug:
            print(f"🔧 DACLIPEncoder initializing...")
            print(f"   Model: {model_name}, Pretrained: {pretrained}")
            print(f"   Device: {self.device}")
            print(f"   Normalization: {normalize}")
        
        # Load model
        self._load_model()
        
        if debug:
            print(f"✅ DACLIPEncoder ready")
            print(f"   Embedding dimension: {self.embedding_dim}")
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect available device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        """Load DA-CLIP model and preprocessing pipeline."""
        try:
            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension and check for DA-CLIP support
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Try to use control=True for DA-CLIP, fall back to standard CLIP
                try:
                    output = self.model.encode_image(dummy, control=True)
                    self.is_daclip = True
                except TypeError:
                    # Standard CLIP doesn't support control parameter
                    output = self.model.encode_image(dummy)
                    self.is_daclip = False
                    if self.debug:
                        print("   Note: Using standard CLIP features (not DA-CLIP)")
                
                # Extract embedding dimension
                if isinstance(output, dict):
                    self.embedding_dim = output.get('degra_features', output.get('image_features')).shape[-1]
                else:
                    self.embedding_dim = output.shape[-1]
            
            if self.debug:
                print(f"   Model loaded: {self.model_name}")
                print(f"   Embedding dim: {self.embedding_dim}")
                print(f"   Mode: {'DA-CLIP' if self.is_daclip else 'Standard CLIP'}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load DA-CLIP model: {e}")
    
    def _prepare_image(self, image: Union[np.ndarray, PILImage.Image]) -> torch.Tensor:
        """
        Prepare single image for encoding.
        
        Args:
            image: numpy array (H, W, 3) or PIL Image
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224) or appropriate size
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = PILImage.fromarray(image)
            elif image.dtype == np.float32 and image.max() <= 1.0:
                image = PILImage.fromarray((image * 255).astype(np.uint8))
            else:
                image = PILImage.fromarray(image.astype(np.uint8))
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dim
        
        return image_tensor
    
    def _prepare_batch(self, images: List[Union[np.ndarray, PILImage.Image]]) -> torch.Tensor:
        """
        Prepare batch of images for encoding.
        
        Args:
            images: List of image arrays or PIL Images
            
        Returns:
            Preprocessed tensor batch (batch_size, 3, 224, 224)
        """
        tensors = [self._prepare_image(img).squeeze(0) for img in images]
        batch = torch.stack(tensors, dim=0)
        return batch
    
    def encode_patch(
        self,
        patch: Union[np.ndarray, PILImage.Image],
        normalize: Optional[bool] = None,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Encode a single patch into degradation-aware embedding.
        
        Args:
            patch: Single patch (H, W, 3) numpy array or PIL Image
            normalize: Whether to L2-normalize (overrides default)
            debug: Print debug info
            
        Returns:
            Embedding tensor, shape (1, embedding_dim) or (embedding_dim,)
        """
        image_tensor = self._prepare_image(patch)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            # Encode with control=True for DA-CLIP, or standard for regular CLIP
            if self.is_daclip:
                output = self.model.encode_image(image_tensor, control=True)
            else:
                output = self.model.encode_image(image_tensor)
            
            # Extract degradation-aware features or fallback to image features
            if isinstance(output, dict):
                embedding = output.get('degra_features')
                if embedding is None:
                    embedding = output.get('image_features')
                    if debug:
                        print("⚠️  Warning: using image_features instead of degra_features")
            else:
                embedding = output
            
            # Normalize if requested
            use_norm = normalize if normalize is not None else self.normalize
            if use_norm:
                embedding = F.normalize(embedding, p=2, dim=-1)
            
            # Move to CPU and detach
            embedding = embedding.cpu().detach()
        
        if debug:
            print(f"Encoded patch: {embedding.shape}")
            print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"  Norm: {embedding.norm(p=2, dim=-1).item():.4f}")
        
        return embedding
    
    def encode_batch(
        self,
        patches: List[Union[np.ndarray, PILImage.Image]],
        batch_size: int = 32,
        normalize: Optional[bool] = None,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Encode a batch of patches.
        
        Args:
            patches: List of patches
            batch_size: Processing batch size (for memory efficiency)
            normalize: Whether to L2-normalize
            debug: Print debug info
            
        Returns:
            Embeddings tensor, shape (len(patches), embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            batch_tensor = self._prepare_batch(batch_patches)
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                # Encode with control or standard mode
                if self.is_daclip:
                    output = self.model.encode_image(batch_tensor, control=True)
                else:
                    output = self.model.encode_image(batch_tensor)
                
                # Extract degradation-aware features or fallback
                if isinstance(output, dict):
                    embeddings = output.get('degra_features')
                    if embeddings is None:
                        embeddings = output.get('image_features')
                else:
                    embeddings = output
                
                # Normalize if requested
                use_norm = normalize if normalize is not None else self.normalize
                if use_norm:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                
                embeddings = embeddings.cpu().detach()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        result = torch.cat(all_embeddings, dim=0)
        
        if debug:
            print(f"Encoded batch of {len(patches)} patches:")
            print(f"  Output shape: {result.shape}")
            print(f"  Range: [{result.min():.4f}, {result.max():.4f}]")
            print(f"  L2 norms: [{result.norm(p=2, dim=-1).min():.4f}, {result.norm(p=2, dim=-1).max():.4f}]")
        
        return result
    
    def validate_consistency(
        self,
        patch: Union[np.ndarray, PILImage.Image],
        num_trials: int = 3
    ) -> bool:
        """
        Validate that same patch produces identical embeddings.
        
        Args:
            patch: Test patch
            num_trials: Number of encoding trials
            
        Returns:
            True if all embeddings match, False otherwise
        """
        embeddings = []
        for _ in range(num_trials):
            emb = self.encode_patch(patch)
            embeddings.append(emb)
        
        # Check if all are identical
        differences = []
        for i in range(1, len(embeddings)):
            diff = torch.norm(embeddings[0] - embeddings[i]).item()
            differences.append(diff)
        
        max_diff = max(differences) if differences else 0
        is_consistent = max_diff < 1e-5
        
        print(f"✓ Consistency check: {num_trials} runs")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Result: {'PASS' if is_consistent else 'FAIL'}")
        
        return is_consistent
    
    @staticmethod
    def to_numpy(
        embedding: torch.Tensor,
        dtype: str = "float32"
    ) -> np.ndarray:
        """
        Convert embedding tensor to numpy array.
        
        Args:
            embedding: torch tensor embedding
            dtype: numpy dtype
            
        Returns:
            numpy array
        """
        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy().astype(dtype)
        return embedding.astype(dtype)
    
    @staticmethod
    def from_numpy(embedding_array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to embedding tensor.
        
        Args:
            embedding_array: numpy array
            
        Returns:
            torch tensor
        """
        return torch.from_numpy(embedding_array).float()


# Convenience functions

def load_encoder(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    normalize: bool = True,
    debug: bool = True
) -> DACLIPEncoder:
    """
    Quick function to load DA-CLIP encoder.
    
    Args:
        model_name: Model architecture
        pretrained: Pretrained weights
        normalize: L2-normalize embeddings
        debug: Print debug info
        
    Returns:
        Initialized DACLIPEncoder
    """
    return DACLIPEncoder(
        model_name=model_name,
        pretrained=pretrained,
        normalize=normalize,
        debug=debug
    )


def encode_patches(
    patches: List[Union[np.ndarray, PILImage.Image]],
    encoder: Optional[DACLIPEncoder] = None,
    batch_size: int = 32,
    normalize: bool = True,
    debug: bool = False
) -> torch.Tensor:
    """
    Quick function to encode patches.
    
    Args:
        patches: List of patches
        encoder: Encoder instance (loads if None)
        batch_size: Batch size
        normalize: L2-normalize
        debug: Print debug info
        
    Returns:
        Embeddings tensor
    """
    if encoder is None:
        encoder = load_encoder(normalize=normalize, debug=debug)
    
    return encoder.encode_batch(patches, batch_size=batch_size, debug=debug)
