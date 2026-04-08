# Quick Start Guide

## Installation (One-time setup)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**For GPU (with CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall faiss-cpu -y && pip install faiss-gpu
```

## Run the Pipeline

```bash
# Basic command
python run_pipeline.py --input dataset/sample_degraded.jpg --output ./results

# With custom image
python run_pipeline.py --input /path/to/your/image.jpg --output ./my_results

# Force CPU (even on GPU machine)
python run_pipeline.py --input image.jpg --output ./results --no-gpu
```

## What You'll Get

After running, check `./results/`:
- `fused_patch_0000.pt` - PyTorch tensor (1, 512, 16, 16)
- `fused_patch_0001.pt` - Another patch
- ...
- `patch_coords.json` - Metadata with patch coordinates

## Sample Dataset Included

We created a small test dataset:

```
dataset/
├── sample_degraded.jpg ............ Test image (256×256)
├── reference_patch_0000.png ....... Reference patches (10 total)
├── reference_patch_0001.png
├── ...
└── patch_map.json ................ Index→path mapping

indexes/
└── clean_patches.index ........... FAISS vector database
```

This is perfect for testing. On GPU later, replace with your large dataset.

## Pipeline Flow

```
Input Image (256×256)
    ↓
[1] Segment patches (64×64, stride 32)
    ↓
[2] Encode with DA-CLIP (1, 512, 16, 16)
    ↓
[3] Retrieve top-5 similar patches from FAISS
    ↓
[4] Fuse with context (cross-attention)
    ↓
Output tensors + coordinates
```

## Output Format

`patch_coords.json` contains:
```json
{
  "image": "dataset/sample_degraded.jpg",
  "patches": [
    {"id": "0000", "x": 0, "y": 0, "file": "fused_patch_0000.pt"},
    {"id": "0001", "x": 32, "y": 0, "file": "fused_patch_0001.pt"}
  ]
}
```

## For GPU Server Deployment

1. Clone and setup same as above
2. Replace `dataset/` with large dataset
3. Rebuild FAISS index with real embeddings
4. Run same command: `python run_pipeline.py --input image.jpg`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Activate venv: `source venv/bin/activate` |
| FAISS import error | OK for testing. Later: `pip install faiss-cpu` |
| Image file not found | Provide correct path to `--input` |
| GPU not detected | Check CUDA installation or use `--no-gpu` |
