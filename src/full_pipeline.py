"""
Complete RAG-based Image Restoration Pipeline

Full workflow:
1. Patch Extraction - Extract overlapping patches from image
2. DA-CLIP Encoding - Encode patches with degradation-aware features
3. FAISS Retrieval - Find similar clean patches from index
4. [FUSION REMOVED] - Query embedding used directly
5. Save Tensors - Store embeddings for decoder
6. Decoding - Transform embeddings to restored patches

Key fixes applied:
- Fusion step completely removed (was homogenizing embeddings → tiled output)
- Restored patches now use retrieved clean patch PIXELS when decoder is untrained
- Reconstruction uses Hann-window feathered blending (patch_reconstruction.py)
  instead of flat averaging — eliminates visible grid seams
- image_shape correctly ordered as (H, W, C) not swapped
- generate_training_data() saves "query_embeddings" key for train_decoder.py
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
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from modules.da_clip_encoder import DACLIPEncoder
    from retrieval import FAISSIndexLoader, PatchRetriever, PatchLoader
    from modules.patch_extraction import PatchExtractor
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")




# ============================================================================
# Phase 6: Decoder Architecture
# ============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block with upsampling."""

    def __init__(self, in_channels, out_channels, scale_factor=2, use_conv_transpose=False):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2 * scale_factor, stride=scale_factor, padding=scale_factor // 2
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

    def forward(self, x):
        x = self.upsample(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for skip connections."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + identity


class UNetDecoder(nn.Module):
    """UNet-style decoder: (B, 512) → (B, 3, 64, 64)"""

    def __init__(self, embedding_dim=512, output_channels=3,
                 use_residual=True, use_conv_transpose=False, dropout_rate=0.0):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, 256 * 8 * 8)
        self.norm_proj = nn.LayerNorm(256 * 8 * 8)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.block1 = DecoderBlock(256, 128, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block1 = ResidualBlock(128) if use_residual else None

        self.block2 = DecoderBlock(128, 64, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block2 = ResidualBlock(64) if use_residual else None

        self.block3 = DecoderBlock(64, 32, scale_factor=2, use_conv_transpose=use_conv_transpose)
        self.res_block3 = ResidualBlock(32) if use_residual else None

        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.projection(x)
        x = self.norm_proj(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(batch_size, 256, 8, 8)

        x = self.block1(x)
        if self.res_block1 is not None:
            x = self.res_block1(x)

        x = self.block2(x)
        if self.res_block2 is not None:
            x = self.res_block2(x)

        x = self.block3(x)
        if self.res_block3 is not None:
            x = self.res_block3(x)

        x = self.final_conv(x)
        x = self.activation(x)
        return x


def normalize_output(tensor, from_range=(-1, 1), to_range=(0, 1)):
    from_min, from_max = from_range
    to_min, to_max = to_range
    tensor = torch.clamp(tensor, from_min, from_max)
    normalized = (tensor - from_min) / (from_max - from_min)
    return normalized * (to_max - to_min) + to_min


def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logger = logging.getLogger("RAGPipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

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
    Complete RAG-based Image Restoration Pipeline (Fusion-Free).

    Workflow:
    1. Extract patches from degraded image
    2. Encode patches with DA-CLIP
    3. Retrieve similar clean patches from FAISS index
    4. [NO FUSION] Use query embedding directly, or use retrieved pixel patches
    5. Decode with trained decoder OR fall back to retrieved clean patch pixels
    6. Reconstruct full image via weighted blending
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "config.json",
        dataset_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        fusion_strategy: str = "none",   # kept for CLI compat, ignored internally
        debug: bool = True
    ):
        self.logger = setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("Initializing RAG Image Restoration Pipeline (Fusion-Free)")
        self.logger.info("=" * 80)

        self.device = self._setup_device(device)
        self.logger.info(f"Device: {self.device}")

        self.config = self._load_config(config_path)
        self.logger.info(f"Config loaded from: {config_path}")

        self.patch_extractor = None
        self.encoder = None
        self.faiss_loader = None
        self.retriever = None
        self.patch_loader = None
        # NOTE: fusion_pipeline intentionally removed

        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.debug = debug
        self.decoder_is_trained = False  # set during init

        self._initialize_components()
        self.logger.info("=" * 80)
        self.logger.info("Pipeline initialization complete")
        self.logger.info("=" * 80)

    def _setup_device(self, device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_config(self, config_path):
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
            return {**default_config, **config}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using defaults.")
            return default_config

    def _initialize_components(self):
        try:
            # 1. Patch Extractor
            self.logger.info("\n[1/5] Initializing Patch Extractor...")
            self.patch_extractor = PatchExtractor(
                patch_size=self.config.get("patch_size", 64),
                stride=self.config.get("stride", 32)
            )

            # 2. DA-CLIP Encoder
            self.logger.info("\n[2/5] Initializing DA-CLIP Encoder...")
            self.encoder = DACLIPEncoder(
                model_name=self.config.get("model_name", "ViT-B-32"),
                pretrained=self.config.get("pretrained", "laion2b_s34b_b79k"),
                normalize=True,
                device=str(self.device),
                debug=self.debug
            )

            # 3. FAISS Index & Retriever
            self.logger.info("\n[3/5] Initializing FAISS Index...")
            self.faiss_loader = FAISSIndexLoader(debug=self.debug)
            index_path = Path("indexes/clean_patches.index")
            patch_map_path = Path("indexes/patch_map.json")

            if index_path.exists() and patch_map_path.exists():
                self.index = self.faiss_loader.load_index(index_path)
                self.patch_map = self.faiss_loader.load_patch_map(patch_map_path)
                self.retriever = PatchRetriever(
                    self.index, self.patch_map,
                    normalize_query=True, debug=self.debug
                )
                self.logger.info("FAISS Retriever initialized successfully")
            else:
                self.logger.warning(f"FAISS index not found at {index_path}. Retrieval disabled.")
                self.retriever = None

            # 4. Patch Loader
            self.logger.info("\n[4/5] Initializing Patch Loader...")
            if self.dataset_root:
                self.patch_loader = PatchLoader(dataset_root=self.dataset_root, debug=self.debug)
            else:
                self.logger.warning("Dataset root not set. Patch pixel loading disabled.")
                self.patch_loader = None

            # 5. UNet Decoder
            self.logger.info("\n[5/5] Initializing UNet Decoder...")
            embedding_dim = self.config.get("embedding_dim", 512)
            self.decoder = UNetDecoder(
                embedding_dim=embedding_dim,
                output_channels=3,
                use_residual=True,
                use_conv_transpose=False,
                dropout_rate=0.0
            ).to(self.device)

            decoder_checkpoint = Path("checkpoints/decoder_pretrained.pt")
            if decoder_checkpoint.exists():
                checkpoint = torch.load(decoder_checkpoint, map_location=self.device, weights_only=False)
                self.decoder.load_state_dict(checkpoint["model_state_dict"])
                self.decoder_is_trained = True
                self.logger.info(f"✓ Loaded pretrained decoder from {decoder_checkpoint}")
            else:
                self.decoder_is_trained = False
                self.logger.warning(
                    "Pretrained decoder NOT found. Will use retrieved clean patch pixels as "
                    "restoration (pixel-copy fallback). Train the decoder first for learned restoration."
                )

            self.logger.info(
                f"Decoder params: {sum(p.numel() for p in self.decoder.parameters()):,}"
            )

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Core restoration logic
    # ------------------------------------------------------------------

    def _restore_patch(
        self,
        patch_idx: int,
        patch: np.ndarray,        # (64, 64, 3) uint8
        x: int,
        y: int,
        k: int
    ) -> np.ndarray:
        """
        Restore a single patch.

        Strategy (in priority order):
          A. Trained decoder  → encode degraded patch → decoder → pixels
          B. Retrieved pixels → take the top-1 clean match pixel crop directly
          C. Pass-through     → return the degraded patch as-is (last resort)

        Returns restored patch as float32 in [0, 1], shape (64, 64, 3).
        """
        patch_size = self.config.get("patch_size", 64)

        # --- Strategy A: trained decoder ---
        if self.decoder_is_trained:
            query_emb = self.encoder.encode_patch(patch).cpu().detach()  # (1, 512)
            # No fusion: pass embedding directly to decoder
            with torch.no_grad():
                decoded = self.decoder(query_emb.to(self.device)).cpu()  # (1, 3, 64, 64)
            decoded = (decoded + 1) / 2          # tanh → [0,1]
            decoded = torch.clamp(decoded, 0, 1)
            # (1, 3, 64, 64) → (64, 64, 3)
            return decoded.squeeze(0).permute(1, 2, 0).numpy()

        # --- Strategy B: retrieved clean pixel patch ---
        if self.retriever is not None and self.patch_loader is not None:
            query_emb = self.encoder.encode_patch(patch).cpu().detach()
            query_np = query_emb.numpy()  # (1, 512)

            indices, distances, metadata = self.retriever.search(query_np, k=k)

            if metadata:
                retrieved_patches = self.patch_loader.load_patches_from_metadata(
                    metadata, patch_size=patch_size
                )
                if retrieved_patches:
                    # Use top-1 retrieved clean patch pixel crop directly
                    top_patch = retrieved_patches[0]  # (64, 64, 3) uint8
                    return top_patch.astype(np.float32) / 255.0

        # --- Strategy C: pass-through (degraded patch) ---
        if patch_idx == 0:
            self.logger.warning(
                "No decoder and no retrieval available — using degraded patches as output. "
                "Results will not be restored, just reconstructed from input."
            )
        return patch.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Path = Path("outputs"),
        k: int = 5,
        save_intermediate: bool = False
    ) -> Dict:
        """
        Process a single image through the fusion-free RAG pipeline.

        Each patch is independently: encoded → retrieved → restored.
        Restored patches are blended back into the full image.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        results = {"image": str(image_path), "status": "processing", "steps": {}}

        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"Processing image: {image_path}")
            self.logger.info(f"Mode: FUSION-FREE per-patch restoration")
            self.logger.info(f"Decoder trained: {self.decoder_is_trained}")
            self.logger.info("=" * 80)

            # ── Step 1: Extract patches ──────────────────────────────────
            self.logger.info("\nStep 1: Extracting patches...")
            patches, coords = self.patch_extractor.extract(image_path, return_coords=True)
            self.logger.info(f"✓ Extracted {len(patches)} patches")

            # Get original image dimensions (H, W) — PIL gives (W, H) from .size
            original_image = Image.open(image_path).convert("RGB")
            img_w, img_h = original_image.size      # PIL: (width, height)
            image_shape = (img_h, img_w, 3)          # numpy convention: (H, W, C)
            self.logger.info(f"  Image shape (H, W, C): {image_shape}")
            self.logger.info(f"  Coordinates (x, y) top-left: {len(coords)} entries")

            results["steps"]["extraction"] = {
                "num_patches": len(patches),
                "patch_size": self.config.get("patch_size", 64),
                "image_shape": list(image_shape)
            }

            # ── Step 2-4: Restore each patch independently ───────────────
            self.logger.info(f"\nStep 2-4: Restoring {len(patches)} patches (no fusion)...")

            patches_folder = output_dir / f"{Path(image_path).stem}_decoded_patches"
            if save_intermediate:
                patches_folder.mkdir(exist_ok=True)

            metadata_list = []
            restored_patches = []   # list of (64, 64, 3) float32 in [0,1]
            patch_stats = {"total": len(patches), "successful": 0, "failed": 0}

            for patch_idx, (patch, (x, y)) in enumerate(
                tqdm(zip(patches, coords), total=len(patches), desc="Restoring patches")
            ):
                try:
                    # patch may be PIL Image or numpy array — normalise to numpy uint8
                    if hasattr(patch, 'numpy') or isinstance(patch, np.ndarray):
                        patch_np = np.array(patch, dtype=np.uint8)
                    else:
                        patch_np = np.array(patch, dtype=np.uint8)  # PIL Image

                    restored = self._restore_patch(patch_idx, patch_np, x, y, k)  # (64,64,3) float32
                    restored_patches.append(restored)

                    if save_intermediate:
                        from torchvision.utils import save_image
                        restored_t = torch.tensor(restored).permute(2, 0, 1).unsqueeze(0)
                        patch_filename = f"patch_{patch_idx:04d}_xy_{x:04d}_{y:04d}.png"
                        save_image(restored_t, str(patches_folder / patch_filename))
                        metadata_list.append({"patch_idx": patch_idx, "coord": [x, y],
                                               "filename": patch_filename, "status": "success"})

                    patch_stats["successful"] += 1

                except Exception as e:
                    self.logger.error(f"Patch {patch_idx} failed: {e}")
                    patch_stats["failed"] += 1
                    # Black patch fallback
                    restored_patches.append(np.zeros((64, 64, 3), dtype=np.float32))
                    if save_intermediate:
                        metadata_list.append({"patch_idx": patch_idx, "coord": [x, y],
                                               "status": "failed", "error": str(e)})

            self.logger.info(
                f"✓ Restoration complete: {patch_stats['successful']}/{patch_stats['total']} successful"
            )

            # ── Step 5: Reconstruct full image ───────────────────────────
            self.logger.info("\nStep 5: Reconstructing full image from restored patches...")

            # Validate: coords must be (x, y) top-left; reconstruct expects (H, W, C) shape
            if len(restored_patches) != len(coords):
                raise RuntimeError(
                    f"Patch count mismatch: {len(restored_patches)} restored vs {len(coords)} coords"
                )

            # Hann-window feathered blending is now built into patch_extractor.reconstruct()
            full_image = self.patch_extractor.reconstruct(
                patches=restored_patches,        # list of (64, 64, 3) float32 [0,1]
                coords=coords,                   # list of (x, y) top-left corners
                image_shape=image_shape,         # (H, W, C)
            )

            self.logger.info(f"✓ Reconstructed image shape: {full_image.shape}")
            self.logger.info(f"  Range: [{full_image.min():.4f}, {full_image.max():.4f}]")

            # ── Step 6: Save output ───────────────────────────────────────
            self.logger.info("\nStep 6: Saving restored image...")
            # reconstruct() already returns uint8
            full_image_pil = Image.fromarray(full_image)
            out_path = output_dir / f"{Path(image_path).stem}_restored.png"
            full_image_pil.save(str(out_path))
            self.logger.info(f"✓ Saved to: {out_path}")

            if save_intermediate and metadata_list:
                meta_file = output_dir / f"{Path(image_path).stem}_patches_metadata.json"
                with open(meta_file, "w") as f:
                    json.dump({"patches": metadata_list, **patch_stats}, f, indent=2)

            results["steps"]["reconstruction"] = {
                "full_image_path": str(out_path),
                "full_image_shape": list(full_image.shape),
                "patch_stats": patch_stats,
                "decoder_trained": self.decoder_is_trained
            }
            results["status"] = "success"

        except Exception as e:
            self.logger.error(f"\nPipeline error: {e}")
            self.logger.error(traceback.format_exc())
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def generate_training_data(
        self,
        degraded_image_path: Union[str, Path],
        output_path: Path = Path("training_data.pt"),
    ) -> Path:
        """
        Encode all patches from the degraded image and save embeddings as
        training_data.pt with key "query_embeddings" (N, 512).

        Run this ONCE before training the decoder:
            python full_pipeline.py --generate-training-data images/image2.jpeg

        Then train:
            python train_decoder.py --data training_data.pt --clean-image images/0021.png
        """
        degraded_image_path = Path(degraded_image_path)
        self.logger.info(f"\nGenerating training data from: {degraded_image_path}")

        patches, coords = self.patch_extractor.extract(degraded_image_path, return_coords=True)
        self.logger.info(f"  Extracted {len(patches)} patches")

        embeddings = []
        for patch in tqdm(patches, desc="Encoding patches"):
            patch_np = np.array(patch, dtype=np.uint8)
            emb = self.encoder.encode_patch(patch_np).cpu().detach()  # (1, 512)
            embeddings.append(emb)

        embeddings_tensor = torch.cat(embeddings, dim=0)  # (N, 512)
        self.logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")

        torch.save({
            "query_embeddings": embeddings_tensor,
            "coords": coords,
            "image_path": str(degraded_image_path),
        }, output_path)

        self.logger.info(f"✓ Saved training data to: {output_path}")
        return output_path

    def process_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Path = Path("outputs"),
        k: int = 5,
        pattern: str = "*.png"
    ) -> List[Dict]:
        image_dir = Path(image_dir)
        images = sorted(list(image_dir.glob(pattern)))
        self.logger.info(f"Batch Processing: {len(images)} images")

        all_results = []
        for image_path in tqdm(images, desc="Processing images"):
            result = self.process_image(image_path, output_dir, k)
            all_results.append(result)

        successful = sum(1 for r in all_results if r["status"] == "success")
        self.logger.info(f"Done: {successful}/{len(all_results)} successful")
        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Image Restoration Pipeline (Fusion-Free)")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--batch", type=str, help="Path to directory with images")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--dataset", type=str, help="Dataset root directory")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--generate-training-data", type=str, metavar="DEGRADED_IMAGE",
                        help="Encode patches from this degraded image and save training_data.pt")
    parser.add_argument("--training-data-output", type=str, default="training_data.pt")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Save individual decoded patch images")

    args = parser.parse_args()

    pipeline = RAGImageRestorationPipeline(
        config_path=args.config,
        dataset_root=args.dataset,
        device=args.device,
        debug=args.debug
    )

    if args.generate_training_data:
        pipeline.generate_training_data(
            args.generate_training_data,
            output_path=Path(args.training_data_output)
        )
    elif args.image:
        result = pipeline.process_image(
            args.image, Path(args.output), k=args.k,
            save_intermediate=args.save_intermediate
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    elif args.batch:
        results = pipeline.process_batch(args.batch, Path(args.output), k=args.k)
        print(f"\nProcessed {len(results)} images")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
