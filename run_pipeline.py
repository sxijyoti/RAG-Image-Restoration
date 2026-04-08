"""
RAG Image Restoration Pipeline - Lightweight CLI
Minimal, machine-independent execution interface
"""

import sys
import argparse
import torch
from pathlib import Path
import json
from tqdm import tqdm

from src.patch_segmentation import PatchSegmentor
from src.clip_encoder import DAClipEncoder
from src.retrieval import RetrieverFAISS
from src.context_fusion import ContextFusionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Image Restoration Pipeline"
    )
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Setup
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Initialize modules
    segmentor = PatchSegmentor()
    encoder = DAClipEncoder(pretrained=False)
    retriever = RetrieverFAISS(top_k=5)
    fusion = ContextFusionPipeline()
    
    # Pipeline execution
    print(f"\nProcessing: {input_path.name}")
    
    # Step 1: Segment
    patches = segmentor.segment(str(input_path))
    print(f"Extracted {len(patches)} patches")
    
    # Step 2-4: Process each patch
    coords = {
        "image": str(input_path),
        "patches": []
    }
    
    for patch_idx, (patch_tensor, x, y) in enumerate(tqdm(patches, desc="Processing")):
        patch_tensor = patch_tensor.to(device)
        
        # Encode
        embedding = encoder.encode(patch_tensor)
        
        # Retrieve
        retrieved = retriever.retrieve(embedding)
        
        # Fuse
        fused = fusion.fuse(embedding, retrieved)
        
        # Save
        tensor_file = output_dir / f"fused_patch_{patch_idx:04d}.pt"
        torch.save(fused.cpu(), tensor_file)
        
        coords["patches"].append({
            "id": f"{patch_idx:04d}",
            "x": int(x),
            "y": int(y),
            "file": f"fused_patch_{patch_idx:04d}.pt"
        })
    
    # Save coordinates
    coords_file = output_dir / "patch_coords.json"
    with open(coords_file, 'w') as f:
        json.dump(coords, f)
    
    print(f"\n✓ Completed: {len(patches)} tensors saved to {output_dir}")
    print(f"✓ Coordinates: {coords_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
