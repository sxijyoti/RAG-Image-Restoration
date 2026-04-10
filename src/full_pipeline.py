"""
Complete RAG-based Image Restoration Pipeline

Full workflow:
1. Patch Extraction - Extract overlapping patches from image
2. DA-CLIP Encoding - Encode patches with degradation-aware features
3. FAISS Retrieval - Find similar clean patches from index
4. Context Fusion - Combine retrieved patches with query
5. Save Tensors - Store fused embeddings for decoder
6. Decoding - Transform embeddings to restored patches

Modular, logging, progress tracking, error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from datetime import datetime
from tqdm import tqdm
import traceback

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from modules.da_clip_encoder import DACLIPEncoder
    from retrieval import FAISSIndexLoader, PatchRetriever, PatchLoader
    from context_fusion import ContextFusionPipeline
    from modules.patch_extraction import PatchExtractor
    from image_reconstruction import ImageReconstructor
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")


# ============================================================================
# Phase 6: Decoder Architecture
# ============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block with upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_conv_transpose: bool = False
    ):
        super().__init__()
        
        self.use_conv_transpose = use_conv_transpose
        
        if use_conv_transpose:
            self.upsample = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2 * scale_factor,
                stride=scale_factor,
                padding=scale_factor // 2
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for skip connections."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + identity


class UNetDecoder(nn.Module):
    """UNet-style decoder: (B, 512) → (B, 3, 64, 64)"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        output_channels: int = 3,
        use_residual: bool = True,
        use_conv_transpose: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        # Step 1: Latent Projection (512 → 256*8*8)
        self.projection = nn.Linear(embedding_dim, 256 * 8 * 8)
        self.norm_proj = nn.LayerNorm(256 * 8 * 8)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Step 2: UNet Decoder Blocks
        self.block1 = DecoderBlock(256, 128, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block1 = ResidualBlock(128) if use_residual else None
        
        self.block2 = DecoderBlock(128, 64, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block2 = ResidualBlock(64) if use_residual else None
        
        self.block3 = DecoderBlock(64, 32, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block3 = ResidualBlock(32) if use_residual else None
        
        # Step 3: Final Output
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 512) → (B, 3, 64, 64)"""
        batch_size = x.shape[0]
        
        # Project and reshape
        x = self.projection(x)
        x = self.norm_proj(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = x.view(batch_size, 256, 8, 8)
        
        # Decoder blocks
        x = self.block1(x)
        if self.res_block1 is not None:
            x = self.res_block1(x)
        
        x = self.block2(x)
        if self.res_block2 is not None:
            x = self.res_block2(x)
        
        x = self.block3(x)
        if self.res_block3 is not None:
            x = self.res_block3(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.activation(x)
        
        return x


def normalize_output(
    tensor: torch.Tensor,
    from_range: Tuple[float, float] = (-1, 1),
    to_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    """Normalize tensor from one range to another."""
    from_min, from_max = from_range
    to_min, to_max = to_range
    
    tensor = torch.clamp(tensor, from_min, from_max)
    normalized = (tensor - from_min) / (from_max - from_min)
    return normalized * (to_max - to_min) + to_min


# Setup logging
def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    """Setup logging with both file and console handlers."""
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logger = logging.getLogger("RAGPipeline")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class RAGImageRestorationPipeline:
    """
    Complete RAG-based Image Restoration Pipeline
    
    Workflow:
    1. Extract patches from degraded image
    2. Encode patches with DA-CLIP
    3. Retrieve similar clean patches from FAISS index
    4. Fuse retrieved context with query embeddings
    5. Save fused tensors for decoder
    """
    
    def __init__(
        self,
        config_path: Union[str, Path] = "config.json",
        dataset_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        fusion_strategy: str = "attention",
        debug: bool = True
    ):
        """
        Initialize RAG Pipeline.
        
        Args:
            config_path: Path to config.json
            dataset_root: Root directory of image dataset
            device: "cpu", "cuda", "mps", or None (auto-detect)
            fusion_strategy: "mean", "concat", or "attention"
            debug: Print debug information
        """
        self.logger = setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("Initializing RAG Image Restoration Pipeline")
        self.logger.info("=" * 80)
        
        # Device setup
        self.device = self._setup_device(device)
        self.logger.info(f"Device: {self.device}")
        
        # Load config
        self.config = self._load_config(config_path)
        self.logger.info(f"Config loaded from: {config_path}")
        
        # Initialize components
        self.patch_extractor = None
        self.encoder = None
        self.faiss_loader = None
        self.retriever = None
        self.patch_loader = None
        self.fusion_pipeline = None
        self.decoder = None
        self.reconstructor = None
        
        self.fusion_strategy = fusion_strategy
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.debug = debug
        
        # Initialize all components
        self._initialize_components()
        self.logger.info("=" * 80)
        self.logger.info("Pipeline initialization complete")
        self.logger.info("=" * 80)
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Auto-detect or set device."""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info("Auto-detected CUDA device")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("Auto-detected MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        else:
            device = torch.device(device)
            self.logger.info(f"Using requested device: {device}")
        
        return device
    
    def _load_config(self, config_path: Union[str, Path]) -> dict:
        """Load configuration from JSON."""
        config_path = Path(config_path)
        default_config = {
            "patch_size": 64,
            "stride": 32,
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "num_retrieved": 5,
            "embedding_dim": 512
        }
        
        if not config_path.exists():
            self.logger.warning(f"Config not found: {config_path}. Using defaults.")
            return default_config
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Merge with defaults
            config = {**default_config, **config}
            self.logger.info(f"Config: {json.dumps(config, indent=2)}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using defaults.")
            return default_config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # 1. Patch Extractor
            self.logger.info("\n[1/5] Initializing Patch Extractor...")
            self.patch_extractor = PatchExtractor(
                patch_size=self.config.get("patch_size", 64),
                stride=self.config.get("stride", 32)
            )
            self.logger.info("Patch Extractor initialized successfully")
            
            # 2. DA-CLIP Encoder
            self.logger.info("\n[2/5] Initializing DA-CLIP Encoder...")
            self.encoder = DACLIPEncoder(
                model_name=self.config.get("model_name", "ViT-B-32"),
                pretrained=self.config.get("pretrained", "laion2b_s34b_b79k"),
                normalize=True,
                device=str(self.device),
                debug=self.debug
            )
            self.logger.info("DA-CLIP Encoder initialized successfully")
            
            # 3. FAISS Index & Retriever
            self.logger.info("\n[3/5] Initializing FAISS Index...")
            self.faiss_loader = FAISSIndexLoader(debug=self.debug)
            
            index_path = Path("indexes/clean_patches.index")
            patch_map_path = Path("indexes/patch_map.json")
            
            if index_path.exists() and patch_map_path.exists():
                self.logger.info(f"Loading FAISS index from: {index_path}")
                self.index = self.faiss_loader.load_index(index_path)
                self.patch_map = self.faiss_loader.load_patch_map(patch_map_path)
                
                self.retriever = PatchRetriever(
                    self.index,
                    self.patch_map,
                    normalize_query=True,
                    debug=self.debug
                )
                self.logger.info("FAISS Retriever initialized successfully")
            else:
                self.logger.warning(f"FAISS index not found at {index_path}. Retrieval will be disabled.")
                self.retriever = None
            
            # 4. Patch Loader
            self.logger.info("\n[4/5] Initializing Patch Loader...")
            if self.dataset_root:
                self.logger.info(f"Using dataset root: {self.dataset_root}")
                self.patch_loader = PatchLoader(
                    dataset_root=self.dataset_root,
                    debug=self.debug
                )
                self.logger.info("Patch Loader initialized successfully")
            else:
                self.logger.warning("Dataset root not set. Patch loading will be disabled.")
                self.patch_loader = None
            
            # 5. Context Fusion
            self.logger.info("\n[5/5] Initializing Context Fusion...")
            num_retrieved = self.config.get("num_retrieved", 5)
            embedding_dim = self.config.get("embedding_dim", 512)
            
            self.logger.info(f"Fusion strategy: {self.fusion_strategy}")
            self.logger.info(f"Number of retrieved patches: {num_retrieved}")
            
            self.fusion_pipeline = ContextFusionPipeline(
                strategy=self.fusion_strategy,
                embedding_dim=embedding_dim,
                num_retrieved=num_retrieved,
                output_spatial=True,
                spatial_size=16,
                debug=self.debug
            )
            self.fusion_pipeline = self.fusion_pipeline.to(self.device)
            self.logger.info("Context Fusion initialized successfully")
            
            # 6. UNet Decoder (Phase 6)
            self.logger.info("\n[6/7] Initializing UNet Decoder...")
            self.decoder = UNetDecoder(
                embedding_dim=embedding_dim,
                output_channels=3,
                use_residual=True,
                use_conv_transpose=False,
                dropout_rate=0.0
            )
            self.decoder = self.decoder.to(self.device)
            
            # Load pretrained weights
            decoder_checkpoint = Path("checkpoints/decoder_pretrained.pt")
            if decoder_checkpoint.exists():
                checkpoint = torch.load(decoder_checkpoint, map_location=self.device)
                self.decoder.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info(f"✓ Loaded pretrained decoder from {decoder_checkpoint}")
            else:
                self.logger.warning(f"Pretrained decoder not found at {decoder_checkpoint}. Using random initialization.")
            
            self.logger.info(f"Decoder initialized with {sum(p.numel() for p in self.decoder.parameters()):,} parameters")
            
            # 7. Image Reconstructor (Phase 7)
            self.logger.info("\n[7/7] Initializing Image Reconstructor...")
            self.reconstructor = ImageReconstructor(
                patch_size=self.config.get("patch_size", 64),
                stride=self.config.get("stride", 32),
                debug=self.debug,
                logger=self.logger
            )
            self.logger.info("Image Reconstructor initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Path = Path("outputs"),
        k: int = 5,
        save_intermediate: bool = False
    ) -> Dict:
        """
        Process single image through full pipeline.
        
        Args:
            image_path: Path to degraded image
            output_dir: Directory to save results
            k: Number of patches to retrieve
            save_intermediate: Save intermediate results
            
        Returns:
            Dictionary with pipeline results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {
            "image": str(image_path),
            "status": "processing",
            "steps": {}
        }
        
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"Processing image: {image_path}")
            self.logger.info("=" * 80)
            
            # Step 1: Extract patches
            self.logger.info("\nStep 1: Extracting patches from image...")
            patches, coords = self.patch_extractor.extract(image_path, return_coords=True)
            self.logger.info(f"Successfully extracted {len(patches)} patches")
            self.logger.info(f"Patch size: {self.config.get('patch_size', 64)}x{self.config.get('patch_size', 64)}")
            results["steps"]["extraction"] = {
                "num_patches": len(patches),
                "patch_size": self.config.get('patch_size', 64)
            }
            
            # Step 2: Encode patches
            self.logger.info("\nStep 2: Encoding patches with DA-CLIP...")
            embeddings = []
            for i, patch in enumerate(tqdm(patches, desc="Encoding", disable=not self.debug)):
                emb = self.encoder.encode_patch(patch).cpu().detach()
                embeddings.append(emb)
            
            embeddings = torch.cat(embeddings, dim=0)  # (num_patches, 512)
            self.logger.info(f"Encoded embeddings shape: {embeddings.shape}")
            self.logger.info(f"Embedding dimension: {embeddings.shape[-1]}")
            results["steps"]["encoding"] = {
                "embeddings_shape": str(embeddings.shape),
                "embedding_dim": int(embeddings.shape[-1])
            }
            
            if save_intermediate:
                emb_path = output_dir / f"{Path(image_path).stem}_embeddings.pt"
                torch.save(embeddings, emb_path)
                self.logger.info(f"Saved embeddings to: {emb_path}")
            
            # Step 3 & 4: Retrieve and Fuse
            if self.retriever is None:
                self.logger.warning("Skipping retrieval (index not available)")
                fused_embeddings = embeddings
                results["steps"]["retrieval"] = {"status": "skipped", "reason": "index_not_available"}
            else:
                self.logger.info(f"\nStep 3: Retrieving {k} similar patches from FAISS index...")
                self.logger.info("Step 4: Fusing retrieved context with query embeddings...")
                
                fused_embeddings = []
                retrieval_stats = {"total": len(embeddings), "successful": 0, "failed": 0}
                
                for i, emb in enumerate(tqdm(embeddings, desc="Retrieve & Fuse", disable=not self.debug)):
                    try:
                        # Search FAISS
                        indices, distances, metadata = self.retriever.search(
                            emb.numpy().reshape(1, -1),
                            k=k
                        )
                        
                        # Encode retrieved patches
                        if self.patch_loader:
                            retrieved_patches = self.patch_loader.load_patches_from_metadata(metadata)
                            if len(retrieved_patches) > 0:
                                retrieved_embs = []
                                for p in retrieved_patches:
                                    r_emb = self.encoder.encode_patch(p).cpu().detach()
                                    retrieved_embs.append(r_emb)
                                
                                retrieved_embs = torch.cat(retrieved_embs, dim=0)  # (k, 512)
                                
                                # Fuse
                                query_emb = emb.unsqueeze(0).to(self.device)
                                retrieved_embs = retrieved_embs.to(self.device)
                                
                                if self.fusion_strategy == "attention":
                                    fused, attn_weights = self.fusion_pipeline(query_emb, retrieved_embs)
                                else:
                                    fused = self.fusion_pipeline(query_emb, retrieved_embs)
                                
                                fused_embeddings.append(fused.cpu().detach())
                                retrieval_stats["successful"] += 1
                            else:
                                self.logger.warning(f"No retrieved patches for patch {i}")
                                fused_embeddings.append(emb.unsqueeze(0))
                                retrieval_stats["failed"] += 1
                        else:
                            self.logger.warning("Patch loader not available, skipping fusion")
                            fused_embeddings.append(emb.unsqueeze(0))
                            retrieval_stats["failed"] += 1
                    
                    except Exception as e:
                        self.logger.error(f"Error processing patch {i}: {e}")
                        fused_embeddings.append(emb.unsqueeze(0))
                        retrieval_stats["failed"] += 1
                
                if fused_embeddings:
                    fused_embeddings = torch.cat(fused_embeddings, dim=0)
                else:
                    fused_embeddings = embeddings
                
                self.logger.info(f"Retrieval & Fusion complete: {retrieval_stats['successful']}/{retrieval_stats['total']} successful")
                self.logger.info(f"Fused embeddings shape: {fused_embeddings.shape}")
                results["steps"]["retrieval"] = retrieval_stats
                results["steps"]["fusion"] = {
                    "strategy": self.fusion_strategy,
                    "fused_embeddings_shape": str(fused_embeddings.shape)
                }
            
            # Step 5: Save tensors
            self.logger.info("\nStep 5: Saving fused tensors to disk...")
            output_path = output_dir / f"{Path(image_path).stem}_fused.pt"
            
            torch.save({
                "fused_embeddings": fused_embeddings,
                "spatial": fused_embeddings.shape[-1] == 3,
                "num_patches": len(patches),
                "patch_coords": coords,
                "image": str(image_path),
                "config": self.config
            }, output_path)
            
            self.logger.info(f"Saved fused tensors to: {output_path}")
            self.logger.info(f"Output file size: {output_path.stat().st_size / 1024:.2f} KB")
            results["steps"]["saving"] = {
                "output_path": str(output_path),
                "file_size_kb": output_path.stat().st_size / 1024
            }
            
            # Step 6: Reconstruct patches from fused embeddings
            self.logger.info("\nStep 6: Reconstructing patches from fused embeddings...")
            try:
                # Use the UNet decoder with fused spatial embeddings
                # Fused embeddings are (B, 512, 16, 16) from attention fusion
                
                if fused_embeddings.dim() == 4:
                    # Reduce spatial dimensions: (B, 512, 16, 16) → (B, 512)
                    # Use adaptive average pooling
                    batch_size, channels, h, w = fused_embeddings.shape
                    fused_reduced = F.adaptive_avg_pool2d(fused_embeddings, 1)  # (B, 512, 1, 1)
                    fused_reduced = fused_reduced.squeeze(-1).squeeze(-1)  # (B, 512)
                    fused_reduced = fused_reduced.to(self.device)
                else:
                    fused_reduced = fused_embeddings.to(self.device)
                
                # Decode with trained neural network
                with torch.no_grad():
                    decoded_patches = self.decoder(fused_reduced).cpu()  # (B, 3, 64, 64)
                
                # Decoder uses Tanh activation → output is [-1, 1]
                # Convert to [0, 1] range
                decoded_patches = (decoded_patches + 1) / 2
                decoded_patches = torch.clamp(decoded_patches, 0, 1)
                
                # Log output stats
                self.logger.info(f"Decoder output range: [{decoded_patches.min():.6f}, {decoded_patches.max():.6f}]")
                self.logger.info(f"Decoder output mean: {decoded_patches.mean():.6f}")
                
                # Save decoded patches
                from torchvision.utils import save_image
                decoded_path = output_dir / f"{Path(image_path).stem}_decoded.pt"
                torch.save(decoded_patches, decoded_path)
                
                self.logger.info(f"Reconstructed patches shape: {decoded_patches.shape}")
                self.logger.info(f"Saved reconstructed patches to: {decoded_path}")
                self.logger.info(f"Decoded file size: {decoded_path.stat().st_size / 1024:.2f} KB")
                
                # Save first patch as PNG for visualization
                png_path = output_dir / f"{Path(image_path).stem}_decoded_sample.png"
                save_image(decoded_patches[0:1], str(png_path))
                self.logger.info(f"Saved sample reconstructed patch to: {png_path}")
                
                results["steps"]["decoding"] = {
                    "output_shape": str(decoded_patches.shape),
                    "output_range": f"[{decoded_patches.min():.4f}, {decoded_patches.max():.4f}]",
                    "method": "spatial_upsampling_from_fused_embeddings",
                    "decoded_path": str(decoded_path),
                    "sample_png": str(png_path)
                }
                
                # Step 7: Reconstruct full image from patches
                self.logger.info("\nStep 7: Reconstructing full image from decoded patches...")
                try:
                    # Load original image to get dimensions
                    from PIL import Image as PILImage
                    original_image = PILImage.open(image_path)
                    original_image_array = np.array(original_image.convert("RGB"))
                    original_height, original_width, _ = original_image_array.shape
                    
                    self.logger.info(f"Original image size: {original_width}×{original_height}")
                    
                    # Reconstruct using ImageReconstructor
                    restored_image = self.reconstructor.reconstruct(
                        decoded_patches,
                        coords,
                        (original_height, original_width, 3),
                        blend_mode="average"
                    )
                    
                    # Save restored image
                    restored_image_path = output_dir / f"{Path(image_path).stem}_restored.png"
                    self.reconstructor.save_reconstructed_image(
                        restored_image,
                        restored_image_path,
                        format="png"
                    )
                    
                    self.logger.info(f"Restored image shape: {restored_image.shape}")
                    self.logger.info(f"Restored image range: [{restored_image.min():.6f}, {restored_image.max():.6f}]")
                    
                    # Optional: Save reconstruction statistics
                    stats_path = output_dir / f"{Path(image_path).stem}_reconstruction_stats.json"
                    stats = {
                        "original_size": f"{original_width}×{original_height}",
                        "num_patches": len(patches),
                        "patch_size": self.config.get("patch_size", 64),
                        "stride": self.config.get("stride", 32),
                        "restored_image_range": f"[{float(restored_image.min()):.6f}, {float(restored_image.max()):.6f}]",
                        "restored_image_mean": float(restored_image.mean()),
                        "output_path": str(restored_image_path)
                    }
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                    results["steps"]["reconstruction"] = {
                        "status": "success",
                        "restored_image_path": str(restored_image_path),
                        "restored_size": f"{original_width}×{original_height}",
                        "output_range": f"[{float(restored_image.min()):.6f}, {float(restored_image.max()):.6f}]",
                        "stats_file": str(stats_path)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Image reconstruction error: {e}")
                    self.logger.error(traceback.format_exc())
                    results["steps"]["reconstruction"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            except Exception as e:
                self.logger.error(f"Reconstruction error: {e}")
                self.logger.error(traceback.format_exc())
                results["steps"]["decoding"] = {"status": "failed", "error": str(e)}
            
            results["status"] = "success"
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Image processing completed successfully")
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.error(f"\n{'=' * 80}")
            self.logger.error(f"Pipeline error: {e}")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 80)
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def process_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Path = Path("outputs"),
        k: int = 5,
        pattern: str = "*.png"
    ) -> List[Dict]:
        """
        Process multiple images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            k: Number of patches to retrieve
            pattern: File pattern (e.g., "*.png")
            
        Returns:
            List of results for each image
        """
        image_dir = Path(image_dir)
        images = sorted(list(image_dir.glob(pattern)))
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Batch Processing: {len(images)} images from {image_dir}")
        self.logger.info("=" * 80)
        
        all_results = []
        for image_path in tqdm(images, desc="Processing images"):
            result = self.process_image(image_path, output_dir, k)
            all_results.append(result)
        
        # Summary
        successful = sum(1 for r in all_results if r["status"] == "success")
        failed = len(all_results) - successful
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Batch Processing Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Total images: {len(all_results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Success rate: {100*successful/len(all_results):.1f}%")
        self.logger.info("=" * 80)
        
        return all_results


def main():
    """Example usage of the full pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Image Restoration Pipeline")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--batch", type=str, help="Path to directory with images")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--dataset", type=str, help="Dataset root directory")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps")
    parser.add_argument("--fusion", type=str, default="attention", help="Fusion strategy")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved patches")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGImageRestorationPipeline(
        config_path=args.config,
        dataset_root=args.dataset,
        device=args.device,
        fusion_strategy=args.fusion,
        debug=args.debug
    )
    
    # Process image or batch
    if args.image:
        result = pipeline.process_image(args.image, Path(args.output), k=args.k)
        print(f"\nResult: {json.dumps(result, indent=2)}")
    
    elif args.batch:
        results = pipeline.process_batch(args.batch, Path(args.output), k=args.k)
        print(f"\nProcessed {len(results)} images")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
