"""
RAG-Image-Restoration Modules Package

Includes:
- patch_extraction: Phase 1 - Extract overlapping patches
- da_clip_encoder: Phase 2 - Encode patches with DA-CLIP
"""

from .patch_extraction import (
    PatchExtractor,
    extract_patches,
    reconstruct_image,
)

# DA-CLIP encoder is optional (requires open_clip)
try:
    from .da_clip_encoder import (
        DACLIPEncoder,
        load_encoder,
        encode_patches,
    )
    __all__ = [
        "PatchExtractor",
        "extract_patches",
        "reconstruct_image",
        "DACLIPEncoder",
        "load_encoder",
        "encode_patches",
    ]
except ImportError:
    __all__ = [
        "PatchExtractor",
        "extract_patches",
        "reconstruct_image",
    ]
