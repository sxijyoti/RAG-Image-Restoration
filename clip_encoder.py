"""
CLIP encoder for patch embeddings.

Uses openai/clip-vit-base-patch32 to extract token embeddings from patches.
Critical: Extracts token embeddings (NOT pooled) for later fusion.
"""

import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModel
from typing import Union, List, Tuple


class CLIPPatchEncoder:
    """
    CLIP-based encoder for patch embeddings.
    
    Uses token embeddings to preserve spatial information:
    - Input: (1, 3, 224, 224) - CLIP vision expects 224x224
    - Patches: 64x64, resized to 224x224 for CLIP
    - Output tokens: (1, 50, 768) where 50 = 7x7 vision tokens + class token
    - For (1, 512, 16, 16) reshape to match downstream: 256 tokens
    
    Key: Token embeddings preserve spatial grid structure needed for fusion.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to use (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Device: {device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # CLIP ViT-B/32 uses 224x224 input
        self.input_size = 224
        
        # Expected token shape for ViT-B/32
        # 224/32 = 7, so 7x7=49 patch tokens + 1 class token = 50 tokens
        self.num_tokens = 50
        self.embed_dim = 768
    
    def encode_patches(
        self,
        patches: Union[List[np.ndarray], np.ndarray],
        batch_size: int = 8,
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode patches to token embeddings.
        
        Args:
            patches: List of (64, 64, 3) arrays or (N, 64, 64, 3) array
            batch_size: Number of patches per batch for efficiency
            return_numpy: If True, return numpy arrays; else torch tensors
        
        Returns:
            embeddings: (N, 50, 768) token embeddings
                - 50 tokens = 1 class token + 49 patch tokens (7x7 grid)
                - 768 = embedding dimension for ViT-B/32
        
        Token Structure (ViT-B/32):
            [CLS, patch_0_0, patch_0_1, ..., patch_6_6]
            One class token captures global context
            49 patch tokens form 7x7 spatial grid (224/32 = 7)
        """
        # Convert to list if needed
        if isinstance(patches, np.ndarray) and patches.ndim == 4:
            patches = [patches[i] for i in range(patches.shape[0])]
        
        num_patches = len(patches)
        embeddings_list = []
        
        with torch.no_grad():
            # Process in batches for efficiency
            for batch_start in range(0, num_patches, batch_size):
                batch_end = min(batch_start + batch_size, num_patches)
                batch_patches = patches[batch_start:batch_end]
                
                # Preprocess batch (handles resizing 64x64 → 224x224)
                inputs = self.processor(
                    images=batch_patches,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to device
                pixel_values = inputs["pixel_values"].to(self.device)
                
                # Forward pass - VISION MODEL ONLY (no text encoder, no pooling)
                # outputs.last_hidden_state has shape (batch_size, 50, 768)
                outputs = self.model(pixel_values)
                
                # Extract token embeddings (NOT pooled)
                # last_hidden_state preserves full token sequence for spatial info
                token_embeddings = outputs.last_hidden_state  # (batch, 50, 768)
                
                embeddings_list.append(token_embeddings.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)  # (N, 50, 768)
        
        # Validate output shape
        assert embeddings.shape == (num_patches, 50, 768), \
            f"Expected shape ({num_patches}, 50, 768), got {embeddings.shape}"
        
        if return_numpy:
            return embeddings.numpy()
        else:
            return embeddings
    
    def extract_spatial_tokens(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spatial tokens from full token sequence.
        
        Token structure breakdown:
            Token 0: [CLS] - class token (global context)
            Tokens 1-49: Patch tokens in row-major order over 7x7 grid
            
        Args:
            embeddings: (N, 50, 768) full token embeddings
        
        Returns:
            class_token: (N, 1, 768)
            spatial_tokens: (N, 49, 768) patch tokens only
        """
        N, num_tokens, embed_dim = embeddings.shape
        
        assert num_tokens == 50, f"Expected 50 tokens, got {num_tokens}"
        assert embed_dim == 768, f"Expected 768 embedding dim, got {embed_dim}"
        
        # Split class and spatial tokens
        class_token = embeddings[:, 0:1, :]        # (N, 1, 768)
        spatial_tokens = embeddings[:, 1:50, :]    # (N, 49, 768)
        
        return class_token, spatial_tokens
    
    def reshape_for_fusion(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reshape spatial tokens to 2D grid for fusion.
        
        Token arrangement in row-major order:
            7x7 grid = 49 tokens
            [patch_0_0, patch_0_1, ..., patch_0_6,
             patch_1_0, patch_1_1, ..., patch_1_6,
             ...
             patch_6_0, patch_6_1, ..., patch_6_6]
        
        Args:
            embeddings: (N, 50, 768) full token embeddings
        
        Returns:
            spatial_grid: (N, 7, 7, 768) spatially organized tokens
        """
        N, num_tokens, embed_dim = embeddings.shape
        
        # Extract spatial tokens (skip class token at index 0)
        _, spatial_tokens = self.extract_spatial_tokens(embeddings)
        
        # Reshape to 2D spatial grid: (N, 7, 7, 768)
        spatial_grid = spatial_tokens.reshape(N, 7, 7, embed_dim)
        
        return spatial_grid
    
    def get_config(self) -> dict:
        """Get encoder configuration."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "input_size": self.input_size,
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
        }


def create_encoder(device: str = None) -> CLIPPatchEncoder:
    """Factory function to create encoder."""
    return CLIPPatchEncoder(device=device)


if __name__ == "__main__":
    print("=" * 70)
    print("CLIP ENCODER TEST")
    print("=" * 70)
    
    try:
        encoder = CLIPPatchEncoder()
        print("\n✓ Model loaded successfully")
        
        # Test 1: Batch encoding
        print("\nTest 1: Batch Encoding")
        test_patches = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        embeddings = encoder.encode_patches(test_patches, batch_size=4, return_numpy=True)
        print(f"  Input: {len(test_patches)} patches (64, 64, 3)")
        print(f"  Batch size: 4")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected: (10, 50, 768)")
        assert embeddings.shape == (10, 50, 768), f"Shape mismatch!"
        print("  ✓ Shape validated")
        
        # Test 2: Token structure
        print("\nTest 2: Token Structure")
        print(f"  Total tokens per patch: {embeddings.shape[1]}")
        print(f"  Token breakdown:")
        print(f"    - Token 0: [CLS] (class/global token)")
        print(f"    - Tokens 1-49: Spatial tokens (7x7 grid)")
        print(f"  Embedding dimension: {embeddings.shape[2]}")
        
        # Test 3: Extract spatial tokens
        print("\nTest 3: Extract Spatial Tokens")
        class_token, spatial_tokens = encoder.extract_spatial_tokens(embeddings)
        print(f"  Class token shape: {class_token.shape} (global context)")
        print(f"  Spatial tokens shape: {spatial_tokens.shape} (patch grid)")
        assert class_token.shape == (10, 1, 768), "Class token shape wrong"
        assert spatial_tokens.shape == (10, 49, 768), "Spatial token shape wrong"
        print("  ✓ Token extraction validated")
        
        # Test 4: Reshape for fusion
        print("\nTest 4: Reshape for Fusion")
        spatial = encoder.reshape_for_fusion(embeddings)
        print(f"  Spatial grid shape: {spatial.shape}")
        print(f"  Expected: (10, 7, 7, 768)")
        assert spatial.shape == (10, 7, 7, 768), "Spatial reshape failed"
        print("  ✓ Spatial reshape validated")
        
        # Test 5: Config
        print("\nTest 5: Encoder Config")
        config = encoder.get_config()
        for key, val in config.items():
            print(f"  {key}: {val}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
