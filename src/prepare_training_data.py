"""
Prepare training data for decoder: (fused_embeddings, clean_patches) pairs

Since we can't separate degraded/clean, we'll use:
- Degraded patches → encoder → fused embeddings → TARGET
- Clean retrieved patches → TARGET

This assumes retrieved patches ARE the clean targets.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from modules.da_clip_encoder import DACLIPEncoder
    from retrieval import FAISSIndexLoader, PatchRetriever, PatchLoader
    from context_fusion import ContextFusionPipeline
    from modules.patch_extraction import PatchExtractor
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


def prepare_training_data(
    image_path: Path,
    dataset_root: Path,
    output_path: Path = Path("training_data.pt"),
    device: str = "cuda",
    k: int = 5,
    max_samples: int = None
) -> Dict:
    """
    Prepare training data pairs: (fused_embeddings, clean_patch_tensors)
    
    Args:
        image_path: Path to degraded image
        dataset_root: Root of dataset with clean patches
        output_path: Where to save training data
        device: Device to use
        k: Number of retrieved patches
        max_samples: Limit number of samples (for testing)
    
    Returns:
        Dictionary with training data statistics
    """
    
    device = torch.device(device)
    print(f"\n{'='*80}")
    print(f"Preparing Training Data from: {image_path}")
    print(f"{'='*80}")
    
    # Initialize components
    print("\nInitializing components...")
    patch_extractor = PatchExtractor(patch_size=64, stride=32)
    encoder = DACLIPEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        normalize=True,
        device=str(device),
        debug=False
    )
    
    # Load FAISS
    faiss_loader = FAISSIndexLoader(debug=False)
    index = faiss_loader.load_index(Path("indexes/clean_patches.index"))
    patch_map = faiss_loader.load_patch_map(Path("indexes/patch_map.json"))
    
    retriever = PatchRetriever(index, patch_map, normalize_query=True, debug=False)
    patch_loader = PatchLoader(dataset_root=dataset_root, debug=False)
    
    # Initialize fusion (important: must match full_pipeline!)
    fusion_pipeline = ContextFusionPipeline(
        strategy="attention",
        embedding_dim=512,
        num_retrieved=k,
        output_spatial=True,
        spatial_size=16,
        debug=False
    )
    fusion_pipeline = fusion_pipeline.to(device)
    
    print("✓ Components initialized")
    
    # Extract patches from degraded image
    print("\nExtracting patches from degraded image...")
    patches, coords = patch_extractor.extract(image_path, return_coords=True)
    print(f"✓ Extracted {len(patches)} patches")
    
    if max_samples:
        patches = patches[:max_samples]
        print(f"  Limiting to {max_samples} samples")
    
    # Prepare training data
    training_data = {
        "fused_embeddings": [],
        "clean_patches": [],
        "stats": []
    }
    
    print("\nProcessing patches...")
    failed_count = 0
    
    for i, patch in enumerate(tqdm(patches, desc="Processing")):
        try:
            # 1. Encode degraded patch
            query_emb = encoder.encode_patch(patch).to(device)  # (1, 512)
            
            # 2. Retrieve similar clean patches
            indices, distances, metadata = retriever.search(
                query_emb.cpu().numpy().reshape(1, -1),
                k=k
            )
            
            # 3. Load clean patches
            retrieved_patches = patch_loader.load_patches_from_metadata(metadata)
            
            if len(retrieved_patches) == 0:
                failed_count += 1
                continue
            
            # 4. Encode retrieved clean patches
            retrieved_embs = []
            retrieved_tensors = []
            
            for clean_patch in retrieved_patches:
                r_emb = encoder.encode_patch(clean_patch).to(device)  # (1, 512)
                retrieved_embs.append(r_emb)
                
                # Convert PIL to tensor [0, 1]
                clean_tensor = torch.tensor(
                    np.array(clean_patch, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)  # (3, 64, 64)
                retrieved_tensors.append(clean_tensor)
            
            retrieved_embs = torch.cat(retrieved_embs, dim=0)  # (k, 512)
            
            # 5. Fuse query with retrieved
            fused, attn_weights = fusion_pipeline(query_emb, retrieved_embs)  # (1, 512, 16, 16)
            
            # 6. Average clean patches as target (ensemble of retrieved)
            clean_target = torch.stack(retrieved_tensors, dim=0).mean(dim=0)  # (3, 64, 64)
            clean_target = torch.clamp(clean_target, 0, 1)
            
            # Store
            training_data["fused_embeddings"].append(fused.cpu().detach())
            training_data["clean_patches"].append(clean_target)
            training_data["stats"].append({
                "patch_idx": i,
                "num_retrieved": len(retrieved_patches),
                "attn_weights": attn_weights.cpu().detach().numpy() if hasattr(attn_weights, 'cpu') else None
            })
        
        except Exception as e:
            failed_count += 1
            print(f"\n⚠ Error processing patch {i}: {e}")
    
    # Stack tensors
    if training_data["fused_embeddings"]:
        training_data["fused_embeddings"] = torch.cat(
            training_data["fused_embeddings"], dim=0
        )
        training_data["clean_patches"] = torch.stack(
            training_data["clean_patches"], dim=0
        )
    else:
        print("❌ No valid training samples generated!")
        return {}
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Data Summary:")
    print(f"{'='*80}")
    print(f"Total samples: {len(training_data['clean_patches'])}")
    print(f"Failed: {failed_count}")
    print(f"Fused embeddings shape: {training_data['fused_embeddings'].shape}")
    print(f"Clean patches shape: {training_data['clean_patches'].shape}")
    print(f"Clean patch range: [{training_data['clean_patches'].min():.4f}, {training_data['clean_patches'].max():.4f}]")
    
    # Save
    torch.save(training_data, output_path)
    print(f"\n✓ Training data saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return {
        "num_samples": len(training_data['clean_patches']),
        "failed": failed_count,
        "fused_shape": str(training_data['fused_embeddings'].shape),
        "clean_shape": str(training_data['clean_patches'].shape),
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="images/image1.png")
    parser.add_argument("--dataset", type=str, default="images")
    parser.add_argument("--output", type=str, default="training_data.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    result = prepare_training_data(
        image_path=Path(args.image),
        dataset_root=Path(args.dataset),
        output_path=Path(args.output),
        device=args.device,
        k=args.k,
        max_samples=args.max_samples
    )
    
    print(f"\n{json.dumps(result, indent=2)}")
