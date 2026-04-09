# System Architecture

## Overview

The Retrieval-Augmented Image Restoration system is a modular pipeline that restores degraded images by retrieving similar clean patches and using them to guide reconstruction.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│            Input Degraded Image                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [1] Patch Extraction (patching.py)                     │
│  • Split image into overlapping 64×64 patches           │
│  • Stride: 32 pixels                                    │
│  • Returns: patch array + coordinate metadata           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [2] DA-CLIP Encoding (encoder.py)                      │
│  • Load DA-CLIP ViT-B/32 model                          │
│  • Call: model.encode_image(image, control=True)        │
│  • Output: 512-dim degradation-aware embeddings         │
└──────────────────┬──────────────────────────────────────┘
                   │
          Batch Processing
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [3] FAISS Retrieval (retrieval.py)                     │
│  • Load FAISS index of clean patch embeddings           │
│  • k-NN search: distances, indices = search(query, k=5) │
│  • Load patch metadata from patch_map.json              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [4] Embedding Fusion (fusion.py)                       │
│  • Combine query embedding with k retrieved embeddings  │
│  • Method 1: Mean fusion (simple average)               │
│  • Method 2: Concat + learnable weights                 │
│  • Output: fused embedding (512-dim)                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [5] CNN Decoder (decoder.py)                           │
│  • Input: fused embedding (512)                         │
│  • FC layer: 512 → 256×16×16                            │
│  • Transposed convolution layers (4 blocks)             │
│  • Output: restored patch (3×64×64)                     │
└──────────────────┬──────────────────────────────────────┘
                   │
         Batch Decoding
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  [6] Overlap Stitching (stitching.py)                   │
│  • Accumulate decoded patches on output grid            │
│  • Handle overlaps with averaging/weighting             │
│  • Kernel options: gaussian, raised_cosine, uniform     │
│  • Output: full restored image                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│        Output Restored Image                            │
└─────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Patching (patching.py)

**Purpose**: Extract overlapping patches from images for processing

**Configuration**:
- Patch size: 64×64 pixels
- Stride: 32 pixels
- Format: float32 in [0, 1]
- Channel order: (C, H, W) for torch compatibility

**Functions**:
```python
extract_patches(image_path, patch_size=64, stride=32)
    → (patches: np.ndarray[(N, 3, 64, 64)], 
       coordinates: List[Dict])
```

**Key Features**:
- Handles overlapping regions correctly
- Stores precise coordinates for reconstruction
- Lazy loading (reads image from disk once)

---

### 2. Encoder (encoder.py)

**Purpose**: Generate degradation-aware embeddings using DA-CLIP

**Model**: 
- Architecture: ViT-B/32 (Vision Transformer)
- Variant: openai (DA-CLIP fine-tuned for degradation)
- Output dimension: 512

**Critical Implementation**:
```python
# MUST be called with control=True
degra_features = model.encode_image(image, control=True)
# NOT: image_features = model.encode_image(image)
```

**Device Support**:
- Auto-detection: CUDA → MPS → CPU
- Manual override via `device` parameter

---

### 3. Retrieval (retrieval.py)

**Purpose**: Find similar clean patches for each degraded patch

**Components**:
1. **FAISS Index**
   - Contains embeddings of clean patches
   - L2 distance metric
   - Pre-computed offline (Phase 3)

2. **Patch Map (JSON)**
   ```json
   {
     "0": {"image_path": "...", "x": 0, "y": 0, "size": 64},
     "1": {...},
     ...
   }
   ```

**Search**:
```python
distances, indices = retriever.search_batch(embeddings, k=5)
# distances: (batch, k) 
# indices: (batch, k) - FAISS index IDs
```

---

### 4. Fusion (fusion.py)

**Purpose**: Combine query and retrieved embeddings

**Methods**:

#### Method 1: Mean Fusion (Baseline)
```
fused = (query + retrieved[0] + ... + retrieved[k-1]) / (k+1)
```
- Simple, no parameters
- Effective baseline

#### Method 2: Advanced Fusion
```
weights = softmax(learned_weights)
gate = sigmoid(Linear(concat(query, weighted_retrieved)))
fused = gate * query + (1-gate) * weighted_retrieved
```
- Learnable
- Requires training

---

### 5. Decoder (decoder.py)

**Purpose**: Reconstruct image patches from embeddings

**Architecture**:
```
Embedding (512)
    ↓
FC Layer (512 → 256×16×16)
    ↓
ConvTranspose2d [256 → 128] (stride=1)
    ↓
ConvTranspose2d [128 → 128] (stride=2) ← 16×16 → 32×32
    ↓
ConvTranspose2d [128 → 64] (stride=1)
    ↓
ConvTranspose2d [64 → 64] (stride=2) ← 32×32 → 64×64
    ↓
ConvTranspose2d [64 → 64] (stride=1)
    ↓
Conv2d [64 → 3] + Sigmoid
    ↓
Output (3×64×64) in [0, 1]
```

**Features**:
- BatchNorm for stability
- ReLU activations
- Sigmoid output (normalized)
- Pre-trained weights optional

---

### 6. Stitching (stitching.py)

**Purpose**: Reconstruct full image from overlapping patches

**Algorithm**:
1. Initialize output grid + weight map
2. For each patch:
   - Accumulate patch values weighted by kernel
   - Accumulate weights
3. Normalize by weights (handles overlaps)

**Blending Methods**:

#### Simple Averaging
```
output = sum(patches) / count
```

#### Gaussian Kernel
```
kernel = exp(-4 * x²) where x ∈ [-1, 1]
output = sum(patches * kernel) / sum(kernel)
```
Smooth blending at patch boundaries

#### Raised Cosine
```
kernel = (1 - cos(π*x)) / 2 where x ∈ [0, π]
```
Smooth but faster rise

---

### 7. Pipeline (pipeline.py)

**Purpose**: Orchestrate the full restoration process

**Main Class**:
```python
class RestorationPipeline:
    def __init__(self, index_path, patch_map_path, 
                 config_path, decoder_checkpoint):
        # Initialize all components
    
    def run(self, image_path, output_path):
        # Execute full pipeline
```

**Public API**:
```python
def run_pipeline(image_path, index_path, patch_map_path, 
                 output_path, config_path, decoder_checkpoint):
    # Convenience function
```

**Processing Steps**:
1. Load and validate inputs
2. Extract patches
3. Batch encode (with progress bar)
4. Batch retrieve
5. Fuse embeddings
6. Batch decode
7. Stitch image
8. Save output

---

## Data Flow Examples

### Single Image Processing
```
image.png (1024×1024)
    ↓
extract_patches() 
    → 961 patches (64×64 with stride 32)
    → coordinates list
    ↓
encoder.encode_batch()
    → embeddings (961, 512)
    ↓
retriever.search_batch()
    → distances (961, 5)
    → indices (961, 5)
    ↓
fusion loop:
    for each query + 5 retrieved → fused (961, 512)
    ↓
decoder()
    → patches (961, 3, 64, 64)
    ↓
stitcher.stitch_patches()
    → image (1024, 1024, 3)
    ↓
save_image()
    → restored.png
```

---

## Configuration Management

All settings in `configs/config.yaml`:

```yaml
# Patch extraction
patch_size: 64        # Patch size
stride: 32            # Stride

# Retrieval
top_k: 5              # Nearest neighbors

# Model
model_name: "ViT-B-32"
pretrained: "openai"

# Fusion
fusion_method: "mean"

# Device
device: "auto"        # cuda, cpu, mps, or auto

# Stitching
stitching_method: "average"
stitching_kernel: "gaussian"

# Batch
batch_size: 16
```

---

## External Dependencies

### Pre-computed Offline
- **clean_patches.index**: FAISS index (Phase 3)
- **patch_map.json**: Patch metadata (Phase 3)

These are passed at runtime - not included in repository.

### Runtime Files
- Input degraded image
- Config YAML (optional, uses defaults)
- Decoder checkpoint (optional, uses untrained)

---

## Performance Characteristics

### Memory Usage
- Embedding: 512 dims × 4 bytes = 2 KB per patch
- Batch of 100: ~200 KB
- Typical image: <100 MB

### Computation Time
- Patching: O(image_size) - fast
- Encoding: 100ms/patch (GPU), 1s/patch (CPU)
- Retrieval: <1ms/patch (FAISS)
- Decoding: 10ms/patch (GPU), 100ms/patch (CPU)
- Stitching: fast, O(image_size)

### Recommended Settings
- CPU: batch_size=1, stride=64
- GPU: batch_size=32, stride=32

---

## Extension Points

### Custom Decoder
Implement `nn.Module` and load:
```python
pipeline = RestorationPipeline(
    ...,
    decoder_checkpoint="custom.pt"
)
```

### Custom Fusion
Modify `EmbeddingFuser.fuse()` method

### Custom Stitching Kernel
Add to `PatchStitcher._get_weight_kernel()`

### Multi-scale Processing
Modify `pipeline.run()` to process at multiple scales

---

## Quality Assurance

### Invariants
- Patches always 64×64
- Embeddings always 512-dim
- Output image normalized [0, 1]
- No hardcoded paths
- Graceful error handling

### Testing Checklist
- [ ] Extract patches correctly
- [ ] Embeddings have correct shape
- [ ] Retrieval returns valid indices
- [ ] Fusion preserves embedding dim
- [ ] Decoder output in [0, 1]
- [ ] Stitching preserves image dims
- [ ] No memory leaks on large images

---

## Version History

- **v1.0.0** (Apr 2026): Initial release
  - Core pipeline
  - DA-CLIP integration
  - FAISS retrieval
  - CNN decoder

---

**Note**: The decoder is untrained. For production use, train on your specific degradation type.
