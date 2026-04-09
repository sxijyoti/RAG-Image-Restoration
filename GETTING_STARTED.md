# Getting Started Guide

## Installation (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/RAG-Image-Restoration.git
cd RAG-Image-Restoration
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag-ir python=3.10
conda activate rag-ir
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; import open_clip; import faiss; print('✓ All dependencies installed')"
```

---

## Quick Test (Minimal Example)

### Step 1: Prepare Test Data

Ensure you have:
- `indexes/clean_patches.index` (FAISS index file)
- `indexes/patch_map.json` (patch metadata)
- A test degraded image: `test_image.png`

### Step 2: Run Basic Restoration

```python
from src.pipeline import run_pipeline

# Run restoration
restored = run_pipeline(
    image_path="test_image.png",
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    output_path="test_restored.png"
)

print(f"✓ Restored image saved: {restored.shape}")
```

### Step 3: Check Output

```bash
# View the restored image
open test_restored.png  # macOS
# or
xdg-open test_restored.png  # Linux
# or use any image viewer
```

---

## Key Module Usage Patterns

### Pattern 1: Single Image Restoration

```python
from src.pipeline import run_pipeline

restored = run_pipeline(
    image_path="input.png",
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    output_path="output.png",
    config_path="configs/config.yaml"
)
```

### Pattern 2: Reuse Pipeline for Multiple Images

```python
from src.pipeline import RestorationPipeline

# Initialize once
pipeline = RestorationPipeline(
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    config_path="configs/config.yaml"
)

# Use multiple times (faster than run_pipeline)
for image_path in ["image1.png", "image2.png", "image3.png"]:
    output_path = f"restored_{image_path}"
    pipeline.run(image_path, output_path)
```

### Pattern 3: Manual Control Over Each Step

```python
import torch
from src.patching import extract_patches
from src.encoder import DAClipEncoder
from src.retrieval import FAISSRetriever
from src.fusion import EmbeddingFuser
from src.decoder import load_decoder
from src.stitching import PatchStitcher

# Step 1: Extract patches
patches, coords = extract_patches("image.png", 64, 32)

# Step 2: Encode
encoder = DAClipEncoder("ViT-B-32", device="cuda")
embeddings = encoder.encode_batch(patches)

# Step 3: Retrieve
retriever = FAISSRetriever("indexes/clean_patches.index", 
                           "indexes/patch_map.json")
distances, indices = retriever.search_batch(embeddings, k=5)

# Step 4: Fuse
fuser = EmbeddingFuser("mean")
fused_list = []
for i in range(len(embeddings)):
    query = embeddings[i]
    retrieved = embeddings[indices[i]]  # placeholder
    fused = fuser.fuse(query, retrieved)
    fused_list.append(fused)
fused_embeddings = torch.stack(fused_list)

# Step 5: Decode
decoder = load_decoder(device="cuda")
with torch.no_grad():
    restored_patches = decoder(fused_embeddings)
restored_patches_np = restored_patches.cpu().numpy()

# Step 6: Stitch
stitcher = PatchStitcher()
restored_image = stitcher.stitch_patches(
    restored_patches_np, coords, (1024, 1024), 64
)

# Step 7: Save
stitcher.save_image(restored_image, "output.png")
```

---

## Configuration Management

### Using Default Config

```python
from src.pipeline import run_pipeline

# Uses configs/config.yaml automatically
restored = run_pipeline(
    image_path="input.png",
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json"
)
```

### Using Custom Config

```yaml
# custom_config.yaml
patch_size: 64
stride: 32
top_k: 10  # Retrieve more patches
fusion_method: "mean"
device: "cuda"
stitching_kernel: "gaussian"
```

```python
from src.pipeline import run_pipeline

restored = run_pipeline(
    image_path="input.png",
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    config_path="custom_config.yaml"
)
```

### Modifying Config at Runtime

```python
import yaml
from src.pipeline import RestorationPipeline

# Load and modify config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

config['top_k'] = 10
config['device'] = 'cuda'

# Save modified config
with open("temp_config.yaml", "w") as f:
    yaml.dump(config, f)

# Use it
pipeline = RestorationPipeline(
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    config_path="temp_config.yaml"
)
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'open_clip'"

**Solution**: 
```bash
pip install open-clip-torch
```

### Issue: "FAISS index not found"

**Solution**:
1. Verify index file exists: `ls -lh indexes/clean_patches.index`
2. Check path is correct (absolute or relative to script location)
3. Ensure FAISS is installed: `pip install faiss-cpu`

### Issue: "Out of Memory (OOM)"

**Solutions**:
- Reduce `batch_size` in config
- Increase `stride` (fewer patches)
- Use CPU instead of GPU (slower but less memory)
- Process images in tiles

```python
# Example: Process large image in tiles
def process_large_image(image_path, tile_size=512, overlap=64):
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_path)
    tiles = extract_tiles(np.array(img), tile_size, overlap)
    
    restored_tiles = []
    for tile in tiles:
        # Process each tile
        restored_tile = run_pipeline(..., image=tile)
        restored_tiles.append(restored_tile)
    
    return stitch_tiles(restored_tiles, overlap)
```

### Issue: "Slow inference on CPU"

**Solutions**:
- Use GPU if available: `device: cuda` in config
- Increase `stride` parameter (fewer patches to process)
- Use batch processing instead of per-image
- Pre-compute embeddings (not yet supported, future optimization)

### Issue: "Poor restoration quality"

**Solutions**:
- Train decoder on your degradation type (decoder is untrained by default)
- Increase `top_k` (retrieve more similar patches)
- Use `stitching_kernel: gaussian` for better blending
- Verify FAISS index contains relevant clean patches

---

## Performance Optimization

### For Speed

```yaml
# configs/fast_config.yaml
patch_size: 64
stride: 64          # Larger stride = fewer patches
top_k: 3            # Fewer retrievals
device: cuda        # GPU acceleration
batch_size: 32
stitching_kernel: uniform  # Faster stitching
```

### For Quality

```yaml
# configs/quality_config.yaml
patch_size: 64
stride: 16          # Small stride = more patches, better coverage
top_k: 10           # More retrievals for better context
device: cuda
batch_size: 16
stitching_kernel: gaussian  # Smooth blending
```

### For Memory Efficiency

```yaml
# configs/memory_config.yaml
patch_size: 64
stride: 48          # Balance speed and quality
top_k: 3
device: cpu         # Use CPU if GPU memory limited
batch_size: 4       # Small batches
```

---

## Batch Processing

### Process Directory of Images

```python
from src.pipeline import RestorationPipeline

pipeline = RestorationPipeline(
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json"
)

# Process all PNG files
pipeline.run_batch(
    image_dir="degraded_images/",
    output_dir="restored_images/",
    pattern="*.png",
    verbose=True
)
```

### Process with Progress Tracking

```python
from pathlib import Path
from tqdm import tqdm
from src.pipeline import RestorationPipeline

pipeline = RestorationPipeline(...)

image_files = list(Path("degraded_images/").glob("*.png"))

for image_path in tqdm(image_files, desc="Restoring"):
    output_path = f"restored/{image_path.name}"
    try:
        pipeline.run(str(image_path), output_path, verbose=False)
    except Exception as e:
        print(f"Error: {image_path}: {e}")
```

---

## Using with Jupyter Notebooks

```python
# In Jupyter notebook
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from pipeline import run_pipeline
import matplotlib.pyplot as plt

# Restore image
restored = run_pipeline(
    image_path="input.png",
    index_path="indexes/clean_patches.index",
    patch_map_path="indexes/patch_map.json",
    output_path="output.png"
)

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(restored)
plt.axis('off')
plt.title('Restored Image')
plt.tight_layout()
plt.show()
```

---

## Next Steps

1. **Understand Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Review Modules**: Check individual module docstrings
3. **Try Examples**: Run [example_usage.py](example_usage.py)
4. **Train Decoder**: Prepare training script for your use case
5. **Optimize Config**: Tune parameters for your hardware

---

## Support & Troubleshooting

- Check [README.md](README.md) for detailed documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Review module docstrings: `python -c "from src.pipeline import run_pipeline; help(run_pipeline)"`
- Check GitHub issues for known problems

---

**Good Luck!** 🚀
