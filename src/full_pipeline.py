"""
Complete RAG-based Image Restoration Pipeline

Full workflow:
1. Patch Extraction - Extract overlapping patches from image
2. DA-CLIP Encoding - Encode patches with degradation-aware features
3. FAISS Retrieval - Find similar clean patches from index
4. Context Fusion - Combine retrieved patches with query
5. Save Tensors - Store fused embeddings for decoder

Modular, logging, progress tracking, error handling.
"""

import torch
import torch.nn as nn
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
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")


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

