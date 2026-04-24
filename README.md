# A Retrieval-Augmented Generative Framework for Context-Aware Image Enhancement

A **Retrieval-Augmented Generation (RAG)** pipeline for image restoration using **DA-CLIP** (Degradation-Aware CLIP) embeddings and **FAISS** similarity search. Degraded image patches are encoded into degradation-aware embeddings, matched against a pre-built index of clean patches, and stitched back together, optionally refined by a learned UNet.

---
## Results
### Input Image
<img width="534" height="211" alt="image" src="https://github.com/user-attachments/assets/ec7f0673-9f7b-4cfa-b615-f184b06bffa5" />

### Output Image 
<img width="534" height="196" alt="image" src="https://github.com/user-attachments/assets/4d537bcc-3257-4c26-9aff-9a2036a08966" />

---

## Architecture
<img width="553" height="450" alt="image" src="https://github.com/user-attachments/assets/1fdda60e-ef13-4d6b-9f82-4467be89b0a6" />

---

## Workflow
<img width="358" height="299" alt="image" src="https://github.com/user-attachments/assets/56ea430f-778e-4e59-97a6-5acf265b37a0" />

---

## Project Structure

```
sxijyoti-rag-image-restoration/
├── README.md
├── check_dependencies.py          # Verify all required packages are installed
├── config.json                    # Index config (patch size, model, dataset info)
├── daclip-faiss-indexer.ipynb     # Kaggle notebook: builds the FAISS index
├── src/
│   ├── __init__.py
│   ├── restore_retrieval_only.py  # Main inference pipeline (retrieval + optional UNet)
│   ├── retrieval.py               # FAISS index loader, patch retriever, patch loader
│   ├── train_refinement.py        # Train the RefinementUNet (retrieved→clean)
│   └── modules/
│       ├── __init__.py
│       ├── da_clip_encoder.py     # DA-CLIP encoder wrapper (encode_image control=True)
│       ├── patch_extraction.py    # Sliding-window extraction + Gaussian reconstruction
│       ├── encoder_examples.py    # DA-CLIP usage examples
│       └── examples.py            # Patch extraction usage examples
└── tests/
    ├── test_da_clip_encoder.py    # 12 unit tests for the encoder
    └── test_patch_extraction.py   # 20+ unit tests for patch extraction
```

---

## Installation

**Check what you have:**
```bash
python check_dependencies.py
```
---

## Quick Start

### 1. Restore an image 
```bash
python src/restore_retrieval_only.py \
    --image path/to/degraded.jpg \
    --dataset path/to/clean/images/ \
    --index indexes/clean_patches.index \
    --patch-map indexes/patch_map.json \
    --output outputs/ \
    --stride 16 \
    --k-coarse 20
```

### 2. Train the refinement UNet, then restore

```bash
# Train
python src/train_refinement.py \
    --degraded images/degraded.jpg \
    --clean-image images/clean.png \
    --device cuda \
    --epochs 200 \
    --stride 16 \
    --augment-retrieved

# Restore using the trained checkpoint
python src/restore_retrieval_only.py \
    --image images/degraded.jpg \
    --dataset images/ \
    --output outputs/ \
    --stride 8 \
    --k-coarse 20 \
    --refinement-ckpt checkpoints/refinement_pretrained.pt
```

---

## Key Components

- `src/modules/patch_extraction.py` — PatchExtractor

Extracts overlapping patches from any image and reconstructs the full image using **Gaussian-weighted blending** to eliminate seam artifacts.

- `src/modules/da_clip_encoder.py` — DACLIPEncoder

Wraps DA-CLIP (`ViT-B/32`) to extract 512-dim **degradation-aware** embeddings. Automatically falls back to standard CLIP features if DA-CLIP weights aren't loaded. Supports single patch and batch encoding.

DA-CLIP weights are available on HuggingFace:
```
repo: weblzw/daclip-uir-ViT-B-32-irsde
file: daclip_ViT-B-32.pt
```

- `src/retrieval.py` — FAISSIndexLoader / PatchRetriever

Loads a pre-built FAISS `IndexFlatIP` index and searches it with L2-normalised query embeddings (cosine similarity via inner product).

- `src/train_refinement.py` — RefinementUNet

A lightweight encoder-decoder UNet trained on the task `retrieved_patch → clean_patch`. Uses residual learning (`output = input + tanh(correction) × 0.5`) with a combined loss of L1 (0.45) + SSIM (0.35) + VGG perceptual (0.20), warmup + cosine LR schedule, and early stopping.

- `daclip-faiss-indexer.ipynb` — Index Building

Kaggle notebook that builds the FAISS index offline:
- Loads DA-CLIP weights from HuggingFace
- Extracts 64×64 patches (stride 32) from DIV2K training images
- Encodes ~2M patches in batches of 256 on Tesla T4 GPU
- Saves `clean_patches.index` (~4.2 GB), `embeddings.npy`, and `patch_map.json`

---

## Configuration

`config.json` records index-build settings and is used automatically at inference time:

```json
{
  "patch_size": 64,
  "stride": 32,
  "model_name": "ViT-B-32",
  "pretrained": "laion2b_s34b_b79k",
  "faiss_metric": "cosine",
  "faiss_index_type": "IndexFlatIP",
  "normalised": true,
  "num_patches": 2050526,
  "num_images": 800
}
```

---

## CLI Reference

### `restore_retrieval_only.py`

| Argument | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to degraded input image |
| `--reference` | `None` | Optional clean reference image |
| `--output` | `outputs/` | Output directory |
| `--k` | `5` | Top-k patches for with-reference blending |
| `--k-coarse` | `20` | Coarse FAISS candidates before sharpness re-ranking |
| `--patch-size` | `64` | Patch size in pixels |
| `--stride` | `16` | Extraction stride (smaller = more patches, slower) |
| `--ref-weight` | `0.4` | Blend weight for reference path (0–1) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dataset` | `None` | Root directory of clean image dataset |
| `--index` | `indexes/clean_patches.index` | FAISS index path |
| `--patch-map` | `indexes/patch_map.json` | Patch map JSON path |
| `--refinement-ckpt` | `checkpoints/refinement_pretrained.pt` | UNet checkpoint |
| `--debug` | `False` | Verbose logging |

### `train_refinement.py`

| Argument | Default | Description |
|---|---|---|
| `--degraded` | `images/image2.jpeg` | Degraded image (sets target dimensions) |
| `--clean-image` | `images/0021.png` | Clean reference image |
| `--output` | `checkpoints/` | Checkpoint directory |
| `--epochs` | `200` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `2e-4` | Learning rate |
| `--stride` | `16` | Patch extraction stride |
| `--patience` | `25` | Early stopping patience |
| `--w-l1` | `0.45` | L1 loss weight |
| `--w-ssim` | `0.35` | SSIM loss weight |
| `--w-perceptual` | `0.20` | VGG perceptual loss weight |
| `--no-augment` | `False` | Disable retrieval imperfection simulation |

---

## Testing

```bash
# Patch extraction (no GPU required)
python tests/test_patch_extraction.py

# DA-CLIP encoder (requires open-clip-torch)
python tests/test_da_clip_encoder.py

# With pytest
pytest tests/ -v
```

---

## Design Notes

**Why sharpness-ranked top-1 instead of blending?**
Averaging multiple retrieved patches washes out fine edges and texture. FAISS cosine distance already handles semantic relevance — among semantically similar candidates, the sharpest one (highest Laplacian variance) is the best restoration target.

**Why Gaussian-weighted reconstruction?**
Flat (average) blending of overlapping patches creates visible grid seams. A 2D Gaussian window that peaks at 1.0 in each patch center and falls toward 0.1 at corners eliminates this completely.

**Why train on retrieved→clean instead of degraded→clean?**
The UNet only needs to correct small imperfections in already-similar retrieved patches, not learn the full degradation mapping. This is a much easier task, converges faster, and generalises better with limited training data.

---

## Requirements Summary

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Model inference and training |
| `numpy` | ≥ 1.24 | Array operations |
| `Pillow` | ≥ 9.0 | Image I/O |
| `open-clip-torch` | ≥ 2.20 | DA-CLIP / CLIP encoder |
| `faiss-cpu` or `faiss-gpu-cu12` | ≥ 1.7 | Vector similarity search |
| `torchvision` | ≥ 0.15 | VGG perceptual loss (optional) |
| `tqdm` | any | Progress bars |
| `huggingface_hub` | any | DA-CLIP weight download |
