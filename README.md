# RAG-based Image Restoration Pipeline

Lightweight, machine-independent image restoration using Retrieval-Augmented Generation (RAG) with DA-CLIP embeddings and FAISS vector search.

## Quick Setup

### Linux - CPU
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --input image.jpg --output ./results
```

### Linux - GPU
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall faiss-cpu -y && pip install faiss-gpu
python run_pipeline.py --input image.jpg --output ./results
```

### macOS - CPU
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --input image.jpg --output ./results
```

### Windows - CPU
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --input image.jpg --output ./results
```

### Windows - GPU (NVIDIA)
```cmd
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall faiss-cpu -y && pip install faiss-gpu
python run_pipeline.py --input image.jpg --output ./results
```

### Remote GPU via SSH
```bash
ssh user@gpu_server
cd RAG-Image-Restoration
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install faiss-gpu
python run_pipeline.py --input image.jpg --output ./results
```

## Pipeline Architecture

```
Input Image
    ↓
[1. Patch Segmentation] (64×64, stride 32)
    ↓
[2. DA-CLIP Encoding] (1, 512, 16, 16)
    ↓
[3. FAISS Retrieval] (top-5 similar)
    ↓
[4. Context Fusion] (cross-attention)
    ↓
Output: Fused tensors + coordinates
```

## Usage

Basic:
```bash
python run_pipeline.py --input image.jpg --output ./results
```

Force CPU:
```bash
python run_pipeline.py --input image.jpg --output ./results --no-gpu
```

## Output Format

- Tensors: `fused_patch_XXXX.pt` - PyTorch tensors (1, 512, 16, 16)
- Coordinates: `patch_coords.json` - JSON metadata

## Project Structure

```
├── configs/config.py          # Minimal configuration
├── src/
│   ├── patch_segmentation.py
│   ├── clip_encoder.py
│   ├── retrieval.py
│   └── context_fusion.py
├── run_pipeline.py
├── tests.py
└── README.md
```

## Device Configuration

Auto-detects GPU/CPU. Use `--no-gpu` to force CPU.

## Dependencies

- PyTorch 2.0+
- FAISS (faiss-cpu or faiss-gpu)
- Pillow, NumPy, tqdm

## Testing

```bash
python tests.py
```

## Performance

- Minimal codebase (521 lines)
- Device-agnostic (CPU/GPU transparent)
- SSH-ready for remote GPU access
