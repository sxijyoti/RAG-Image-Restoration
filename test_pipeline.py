#!/usr/bin/env python
"""Quick test for complete RAG restoration pipeline."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from full_pipeline import RAGImageRestorationPipeline

print("\n" + "="*80)
print("RAG-based Image Restoration Pipeline - Complete End-to-End Test")
print("="*80)
    
    # Green gradient
    img_array[:, :, 1] = np.linspace(0, 255, height)[:, np.newaxis]
    
    # Blue pattern
    img_array[:, :, 2] = (np.sin(np.linspace(0, 4*np.pi, height))[:, np.newaxis] * 127 + 128).astype(np.uint8)
    
    # Add some geometric shapes
    for i in range(0, width, 64):
        img_array[i:i+32, :, 0] = np.clip(img_array[i:i+32, :, 0] + 50, 0, 255)
    
    img = Image.fromarray(img_array, mode="RGB")
    img.save(output_path)
    
    print(f"✓ Created test image: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def test_pipeline(
    image_path: Path,
    config_path: Path = Path("config.json"),
    dataset_root: Path = None,
    device: str = None,
    output_dir: Path = Path("test_outputs")
):
    """
    Test the complete RAG Image Restoration Pipeline.
    
    Args:
        image_path: Path to input image
        config_path: Path to config.json
        dataset_root: Optional dataset root for patch loading
        device: Device to use (auto-detect if None)
        output_dir: Output directory for results
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 80)
    print("RAG Image Restoration Pipeline - End-to-End Test")
    print("=" * 80)
    
    try:
        # Initialize pipeline
        print("\n[INIT] Initializing RAG Pipeline...")
        pipeline = RAGImageRestorationPipeline(
            config_path=str(config_path),
            dataset_root=dataset_root,
            device=device,
            fusion_strategy="attention",
            debug=True
        )
        print("✓ Pipeline initialized successfully\n")
        
        # Process image
        print("[PROCESS] Processing image through full 7-phase pipeline...")
        result = pipeline.process_image(
            image_path,
            output_dir=output_dir,
            k=5,
            save_intermediate=True
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("Pipeline Execution Results")
        print("=" * 80)
        
        import json
        print(json.dumps(result, indent=2))
        
        # Save results summary
        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Full results saved to: {summary_path}")
        
        # Check for output files
        print("\n" + "=" * 80)
        print("Generated Output Files")
        print("=" * 80)
        
        output_files = list(output_dir.glob(f"{image_path.stem}_*"))
        for file_path in sorted(output_files):
            size_kb = file_path.stat().st_size / 1024
            print(f"  - {file_path.name} ({size_kb:.1f} KB)")
        
        # Verify key outputs exist
        print("\n" + "=" * 80)
        print("Pipeline Completion Checklist")
        print("=" * 80)
        
        checks = {
            "Extracted patches": result["steps"].get("extraction", {}).get("num_patches", 0) > 0,
            "Encoded to embeddings": "encoding" in result["steps"],
            "Fused embeddings": "fusion" in result["steps"] or "retrieval" in result["steps"],
            "Decoded patches": "decoding" in result["steps"],
            "Reconstructed full image": "reconstruction" in result["steps"],
            "Status: Success": result["status"] == "success"
        }
        
        for check, status in checks.items():
            symbol = "✓" if status else "✗"
            print(f"  {symbol} {check}")
        
        all_passed = all(checks.values())
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✓ ALL PIPELINE PHASES COMPLETED SUCCESSFULLY")
            print("✓ Image restored from patches to full-size image")
        else:
            print("✗ Some pipeline phases failed or did not complete")
            if "reconstruction" in result["steps"] and result["steps"]["reconstruction"].get("status") == "success":
                restored_image_path = result["steps"]["reconstruction"].get("restored_image_path")
                print(f"\n✓ Restored image saved to: {restored_image_path}")
        print("=" * 80)
        
        return result, all_passed
    
    except Exception as e:
        print(f"\n✗ Pipeline test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return None, False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test RAG Image Restoration Pipeline with Image Reconstruction"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (creates dummy if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config file path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create a dummy test image"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of dummy test image (default 512x512)"
    )
    
    args = parser.parse_args()
    
    # Get or create test image
    image_path = Path(args.image) if args.image else None
    
    if not image_path or not image_path.exists():
        if args.create_dummy:
            image_path = create_dummy_image(args.size, args.size)
        else:
            # Try to find a test image
            images = list(Path("images").glob("*.png")) + list(Path("images").glob("*.jpg"))
            if images:
                image_path = images[0]
                print(f"Using image from workspace: {image_path}")
            else:
                print("No image found. Use --create-dummy flag or provide --image path")
                return
    
    # Run test
    result, success = test_pipeline(
        image_path,
        config_path=Path(args.config),
        dataset_root=Path(args.dataset) if args.dataset else None,
        device=args.device,
        output_dir=Path(args.output)
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
