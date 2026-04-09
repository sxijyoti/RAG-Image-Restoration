"""
Main pipeline module for image restoration.

Orchestrates the full retrieval-augmented restoration process.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import yaml

from patching import extract_patches, load_image, get_image_shape
from encoder import DAClipEncoder
from retrieval import FAISSRetriever
from fusion import EmbeddingFuser
from decoder import load_decoder
from stitching import PatchStitcher


class RestorationPipeline:
    """Full image restoration pipeline."""
    
    def __init__(
        self,
        index_path: str,
        patch_map_path: str,
        config_path: Optional[str] = None,
        decoder_checkpoint: Optional[str] = None
    ):
        """
        Initialize restoration pipeline.
        
        Args:
            index_path: Path to FAISS index
            patch_map_path: Path to patch map JSON
            config_path: Path to config YAML (optional)
            decoder_checkpoint: Path to decoder checkpoint (optional)
        """
        # Load config
        if config_path is None:
            self.config = self._get_default_config()
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Set device
        self.device = self.config.get('device', self._detect_device())
        
        # Initialize components
        print("Initializing pipeline components...")
        
        self.encoder = DAClipEncoder(
            model_name=self.config.get('model_name', 'ViT-B-32'),
            pretrained=self.config.get('pretrained', 'openai'),
            device=self.device
        )
        
        self.retriever = FAISSRetriever(index_path, patch_map_path)
        
        self.fuser = EmbeddingFuser(
            method=self.config.get('fusion_method', 'mean')
        )
        
        self.decoder = load_decoder(
            checkpoint_path=decoder_checkpoint,
            embedding_dim=self.encoder.embedding_dim,
            patch_size=self.config.get('patch_size', 64),
            device=self.device
        )
        
        self.stitcher = PatchStitcher()
        
        self.patch_size = self.config.get('patch_size', 64)
        self.stride = self.config.get('stride', 32)
        self.top_k = self.config.get('top_k', 5)
        
        print("Pipeline initialized successfully!")
    
    @staticmethod
    def _detect_device() -> str:
        """Auto-detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def _get_default_config() -> Dict:
        """Get default configuration."""
        return {
            'patch_size': 64,
            'stride': 32,
            'top_k': 5,
            'model_name': 'ViT-B-32',
            'pretrained': 'openai',
            'fusion_method': 'mean',
            'device': RestorationPipeline._detect_device()
        }
    
    def run(
        self,
        image_path: str,
        output_path: str,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Run full restoration pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Path to save restored image
            verbose: Whether to print progress
            
        Returns:
            Restored image array of shape (height, width, 3), float32 in [0, 1]
        """
        print(f"\n{'='*60}")
        print(f"Image Restoration Pipeline")
        print(f"{'='*60}")
        print(f"Input image: {image_path}")
        print(f"Output path: {output_path}")
        
        # Step 1: Extract patches
        if verbose:
            print("\n[1/7] Extracting patches...")
        patches, coordinates = extract_patches(
            image_path,
            patch_size=self.patch_size,
            stride=self.stride
        )
        num_patches = len(coordinates)
        if verbose:
            print(f"  Extracted {num_patches} patches")
        
        # Step 2: Encode patches with DA-CLIP
        if verbose:
            print("\n[2/7] Encoding patches with DA-CLIP...")
        query_embeddings = self.encoder.encode_batch(patches)
        if verbose:
            print(f"  Encoded to embeddings of shape {query_embeddings.shape}")
        
        # Step 3: Retrieve top-k patches for each query
        if verbose:
            print(f"\n[3/7] Retrieving top-{self.top_k} similar patches...")
        distances, indices = self.retriever.search_batch(query_embeddings, k=self.top_k)
        if verbose:
            print(f"  Retrieval complete - indices shape: {indices.shape}")
        
        # Step 4: Fuse query embeddings with retrieved embeddings
        if verbose:
            print("\n[4/7] Fusing embeddings...")
        fused_embeddings = []
        for i in tqdm(range(num_patches), disable=not verbose, desc="Fusing"):
            query_emb = query_embeddings[i]
            retrieved_idx = indices[i]
            
            # Load retrieved embeddings
            retrieved_embs = []
            for idx in retrieved_idx:
                metadata = self.retriever.get_patch_metadata(int(idx))
                # For now, we use the indices - embeddings would need to be pre-computed
                # This is a design choice: we can either store embeddings or reload images
                retrieved_embs.append(query_emb)  # Placeholder
            
            retrieved_embs = torch.stack(retrieved_embs)
            fused_emb = self.fuser.fuse(query_emb, retrieved_embs)
            fused_embeddings.append(fused_emb)
        
        fused_embeddings = torch.stack(fused_embeddings)
        if verbose:
            print(f"  Fused embeddings shape: {fused_embeddings.shape}")
        
        # Step 5: Decode patches
        if verbose:
            print("\n[5/7] Decoding patches...")
        with torch.no_grad():
            decoded_patches = self.decoder(fused_embeddings.to(self.device))
        decoded_patches_np = decoded_patches.cpu().numpy()
        if verbose:
            print(f"  Decoded patches shape: {decoded_patches_np.shape}")
        
        # Step 6: Stitch patches
        if verbose:
            print("\n[6/7] Stitching patches...")
        image_shape = get_image_shape(image_path)
        restored_image = self.stitcher.stitch_patches(
            decoded_patches_np,
            coordinates,
            image_shape,
            patch_size=self.patch_size
        )
        if verbose:
            print(f"  Stitched image shape: {restored_image.shape}")
        
        # Step 7: Save image
        if verbose:
            print("\n[7/7] Saving restored image...")
        self.stitcher.save_image(restored_image, output_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Restoration complete!")
        print(f"{'='*60}\n")
        
        return restored_image
    
    def run_batch(
        self,
        image_dir: str,
        output_dir: str,
        pattern: str = "*.png",
        verbose: bool = True
    ) -> None:
        """
        Run restoration on a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save restored images
            pattern: File pattern to match (default: *.png)
            verbose: Whether to print progress
        """
        from pathlib import Path
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(image_dir.glob(pattern))
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Batch processing"):
            output_path = output_dir / image_path.name
            try:
                self.run(str(image_path), str(output_path), verbose=False)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")


def run_pipeline(
    image_path: str,
    index_path: str,
    patch_map_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    decoder_checkpoint: Optional[str] = None
) -> np.ndarray:
    """
    Public API for running the restoration pipeline.
    
    Args:
        image_path: Path to input image
        index_path: Path to FAISS index
        patch_map_path: Path to patch map JSON
        output_path: Path to save restored image (optional)
        config_path: Path to config YAML (optional)
        decoder_checkpoint: Path to decoder checkpoint (optional)
        
    Returns:
        Restored image array of shape (height, width, 3), float32 in [0, 1]
    """
    # Initialize pipeline
    pipeline = RestorationPipeline(
        index_path=index_path,
        patch_map_path=patch_map_path,
        config_path=config_path,
        decoder_checkpoint=decoder_checkpoint
    )
    
    # Run restoration
    if output_path is None:
        from pathlib import Path
        image_stem = Path(image_path).stem
        output_path = f"{image_stem}_restored.png"
    
    restored_image = pipeline.run(image_path, output_path)
    
    return restored_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Image Restoration using Retrieval-Augmented Generation"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("index_path", help="Path to FAISS index")
    parser.add_argument("patch_map_path", help="Path to patch map JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save restored image"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default=None,
        help="Path to decoder checkpoint"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        image_path=args.image_path,
        index_path=args.index_path,
        patch_map_path=args.patch_map_path,
        output_path=args.output,
        config_path=args.config,
        decoder_checkpoint=args.decoder_checkpoint
    )
