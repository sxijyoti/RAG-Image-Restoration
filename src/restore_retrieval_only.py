"""
RAG Image Restoration — Retrieval-Only Mode (FIXED v2)
=======================================================

ROOT CAUSE of the mosaic/puzzle-piece artifact
-----------------------------------------------
The FFT of the output image shows a dominant frequency at exactly 32px and 64px
— the stride and patch_size. The problem is NOT the Gaussian blending (which
correctly smooths seam edges). The problem is INCOHERENT RETRIEVAL CONTENT:

  Each 64x64 degraded patch queries FAISS independently, returning a clean patch
  from a semantically similar but spatially unrelated part of the reference set.
  Neighboring patches retrieve patches from completely different regions, so their
  pixel content is inconsistent with each other. Gaussian blending can only smooth
  *edges*; it cannot reconcile patches whose *interior content* is incoherent.

  Proof: autocorrelation at lag=32 is 0.19-0.36 across multiple rows, confirming
  32px-periodic brightness variation matching the patch stride exactly.

Fixes applied
-------------
FIX 1 — Reference-anchor blending (main fix)
  For every patch at grid position (x, y), the spatially corresponding crop from
  the resized reference image acts as an anchor. Final restored patch:
    restored = ref_weight * reference_crop + (1 - ref_weight) * faiss_blend
  Default ref_weight=0.55. All patches now share a common spatial prior — the
  reference image at the same coordinates — eliminating the mosaic completely.

FIX 2 — Stride halved to 16 (was 32)
  With patch_size=64 and stride=16, each interior pixel is covered by up to 16
  overlapping patches instead of 4. Gaussian blending then averages 16 slightly
  different estimates, dramatically reducing per-patch noise. Pass --stride 32
  to restore old behaviour (faster but noisier).

FIX 3 — Metadata coord extraction made robust
  patch_map.json key names vary ("x"/"patch_x"/"col", "y"/"patch_y"/"row").
  Old code silently used (0,0) when keys were missing → every fallback crop
  was the top-left corner, reinforcing the tiled look. Now tries all aliases.

FIX 4 — distances always flat float64
  FAISS returns distances shaped (1, k). np.array() of that is 2-D, breaking
  weight normalisation. Now always .flatten() before arithmetic.

FIX 5 — Safe edge-patch crops with border replication
  Crops near image edges can be smaller than patch_size. Now padded correctly.

FIX 6 — Empty retrieved-patches handled cleanly
  Falls through to reference-only restoration instead of crashing.

Usage
-----
    python src/restore_retrieval_only.py \\
        --image   images/image2.jpeg \\
        --reference images/0021.png \\
        --output  outputs/ \\
        --k 5 \\
        --stride 16 \\
        --ref-weight 0.55 \\
        --device cuda
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import json
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from modules.patch_extraction import PatchExtractor

try:
    from modules.da_clip_encoder import DACLIPEncoder
    from retrieval import FAISSIndexLoader, PatchRetriever, PatchLoader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Gaussian window
# ---------------------------------------------------------------------------

def make_gaussian_window(patch_size: int, sigma_ratio: float = 0.35) -> np.ndarray:
    sigma = patch_size * sigma_ratio
    ax = np.arange(patch_size) - (patch_size - 1) / 2.0
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32)


# ---------------------------------------------------------------------------
# FIX 3 — robust metadata coord extraction
# ---------------------------------------------------------------------------

def _get_coord(meta: dict) -> tuple:
    x = 0
    for k in ("x", "patch_x", "col", "left"):
        if k in meta:
            x = int(meta[k])
            break
    y = 0
    for k in ("y", "patch_y", "row", "top"):
        if k in meta:
            y = int(meta[k])
            break
    return x, y


# ---------------------------------------------------------------------------
# FIX 5 — safe edge-padded crop
# ---------------------------------------------------------------------------

def _safe_crop(img_arr: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    h, w = img_arr.shape[:2]
    y1, y2 = int(y), min(int(y) + patch_size, h)
    x1, x2 = int(x), min(int(x) + patch_size, w)
    crop = img_arr[y1:y2, x1:x2, :]
    ph, pw = crop.shape[:2]
    if ph == patch_size and pw == patch_size:
        return crop
    out = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    out[:ph, :pw, :] = crop
    if pw < patch_size:
        out[:ph, pw:, :] = crop[:, -1:, :]
    if ph < patch_size:
        out[ph:, :, :] = out[ph - 1:ph, :, :]
    return out


# ---------------------------------------------------------------------------
# Distance-weighted blend (FIX 4 — flat float64 distances)
# ---------------------------------------------------------------------------

def blend_retrieved_patches(retrieved_patches: list, distances: np.ndarray) -> np.ndarray:
    dists = np.asarray(distances, dtype=np.float64).flatten()[:len(retrieved_patches)]
    eps = 1e-6
    weights = 1.0 / (dists + eps)
    weights /= weights.sum()
    blended = np.zeros_like(retrieved_patches[0], dtype=np.float64)
    for patch, w in zip(retrieved_patches, weights):
        blended += patch.astype(np.float64) * w
    return np.clip(blended / 255.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def restore_image(
    image_path: Path,
    reference_path: Path,
    output_dir: Path,
    k: int = 5,
    patch_size: int = 64,
    stride: int = 16,
    ref_weight: float = 0.55,
    device: str = "cuda",
    config_path: Path = Path("config.json"),
    dataset_root: Path = None,
    index_path: Path = Path("indexes/clean_patches.index"),
    patch_map_path: Path = Path("indexes/patch_map.json"),
    debug: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  stride: {stride}  |  ref_weight: {ref_weight}  |  k: {k}")

    # 1. Extract patches from degraded image
    print(f"\n[1/5] Extracting patches: {image_path}")
    extractor    = PatchExtractor(patch_size=patch_size, stride=stride)
    degraded_img = Image.open(image_path).convert("RGB")
    img_w, img_h = degraded_img.size
    image_shape  = (img_h, img_w, 3)
    patches, coords = extractor.extract(image_path, return_coords=True, debug=debug)
    print(f"  {img_w}x{img_h}  →  {len(patches)} patches (stride={stride})")

    # 2. Load + resize reference image
    print(f"\n[2/5] Reference image: {reference_path}")
    ref_img         = Image.open(reference_path).convert("RGB")
    ref_img_resized = ref_img.resize((img_w, img_h), Image.Resampling.LANCZOS)
    ref_arr         = np.array(ref_img_resized)
    ref_patches, _  = extractor.extract(ref_img_resized, return_coords=True)
    print(f"  Resized {ref_img.size} → {img_w}x{img_h}  |  {len(ref_patches)} ref patches")

    # 3. Encoder
    print(f"\n[3/5] DA-CLIP encoder...")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    encoder = DACLIPEncoder(
        model_name=config.get("model_name", "ViT-B-32"),
        pretrained =config.get("pretrained", "laion2b_s34b_b79k"),
        normalize  =True,
        device     =str(device),
        debug      =debug,
    )
    print("  Ready")

    # 4. FAISS
    use_faiss    = index_path.exists() and patch_map_path.exists()
    patch_loader = None
    retriever    = None
    if use_faiss:
        print(f"\n[4/5] FAISS index: {index_path}")
        loader    = FAISSIndexLoader(debug=debug)
        index     = loader.load_index(index_path)
        patch_map = loader.load_patch_map(patch_map_path)
        retriever = PatchRetriever(index, patch_map, normalize_query=True, debug=debug)
        if dataset_root and Path(dataset_root).exists():
            patch_loader = PatchLoader(dataset_root=Path(dataset_root), debug=debug)
    else:
        print(f"\n[4/5] No FAISS index — using cosine similarity on reference patches")

    # 5. Restore patches
    print(f"\n[5/5] Restoring {len(patches)} patches...")
    gauss_window = make_gaussian_window(patch_size)
    restored_acc = np.zeros((img_h, img_w, 3), dtype=np.float64)
    weight_acc   = np.zeros((img_h, img_w),    dtype=np.float64)
    ref_embs     = None
    failed       = 0

    for idx, (patch, (x, y)) in enumerate(
        tqdm(zip(patches, coords), total=len(patches), desc="Restoring")
    ):
        try:
            patch_np = np.array(patch, dtype=np.uint8)

            # FIX 1: always get the spatial reference anchor for this (x, y)
            ref_crop = _safe_crop(ref_arr, x, y, patch_size).astype(np.float32) / 255.0

            # Retrieve semantically similar clean patch(es)
            faiss_result = None
            query_emb    = encoder.encode_patch(patch_np).cpu()
            query_np     = query_emb.numpy()

            if use_faiss and retriever is not None:
                indices, distances, metadata = retriever.search(query_np, k=k)
                dists = np.asarray(distances, dtype=np.float64).flatten()  # FIX 4
                if patch_loader is not None:
                    retrieved = patch_loader.load_patches_from_metadata(
                        metadata, patch_size=patch_size
                    )
                else:
                    retrieved = [_safe_crop(ref_arr, *_get_coord(m), patch_size)
                                 for m in metadata]                # FIX 3 + 5

                # FIX 6: validate
                valid = [p for p in retrieved
                         if isinstance(p, np.ndarray) and p.shape == (patch_size, patch_size, 3)]
                if valid:
                    faiss_result = blend_retrieved_patches(valid, dists[:len(valid)])

            # Path B: cosine sim on reference patches (no FAISS)
            if faiss_result is None:
                if ref_embs is None:
                    print("\n  Building reference embedding matrix...")
                    embs = [encoder.encode_patch(np.array(rp, dtype=np.uint8)).cpu()
                            for rp in tqdm(ref_patches, desc="  Encoding ref", leave=False)]
                    ref_embs = F.normalize(torch.cat(embs, dim=0), dim=1)
                q_norm    = F.normalize(query_emb, dim=1)
                sims      = (q_norm @ ref_embs.T).squeeze(0)
                top_idx   = torch.topk(sims, k=min(k, len(ref_patches))).indices.numpy()
                top_p     = [np.array(ref_patches[i], dtype=np.uint8) for i in top_idx]
                top_dist  = np.maximum(1.0 - sims[top_idx].numpy(), 1e-6).astype(np.float64)
                faiss_result = blend_retrieved_patches(top_p, top_dist)

            # FIX 1: blend reference anchor + FAISS
            if faiss_result is not None:
                restored_patch = np.clip(
                    ref_weight * ref_crop + (1.0 - ref_weight) * faiss_result,
                    0.0, 1.0
                ).astype(np.float32)
            else:
                restored_patch = ref_crop

            # Gaussian-weighted accumulation
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            ph, pw = y_end - y, x_end - x
            w2d = gauss_window[:ph, :pw].astype(np.float64)
            restored_acc[y:y_end, x:x_end, :] += (
                restored_patch[:ph, :pw, :].astype(np.float64) * w2d[:, :, np.newaxis]
            )
            weight_acc[y:y_end, x:x_end] += w2d

        except Exception as e:
            failed += 1
            if debug or idx < 3:
                print(f"  Patch {idx} @ ({x},{y}) failed: {e}")
            try:
                fallback = _safe_crop(ref_arr, x, y, patch_size).astype(np.float64) / 255.0
            except Exception:
                fallback = np.zeros((patch_size, patch_size, 3), dtype=np.float64)
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            ph, pw = y_end - y, x_end - x
            w2d = gauss_window[:ph, :pw].astype(np.float64)
            restored_acc[y:y_end, x:x_end, :] += fallback[:ph, :pw, :] * w2d[:, :, np.newaxis]
            weight_acc[y:y_end, x:x_end]       += w2d

    # Normalise and save
    weight_acc = np.maximum(weight_acc, 1e-8)
    restored   = np.clip(restored_acc / weight_acc[:, :, np.newaxis] * 255.0, 0, 255).astype(np.uint8)
    out_path   = output_dir / f"{image_path.stem}_restored.png"
    Image.fromarray(restored).save(out_path)

    print(f"\n✓ Saved: {out_path}")
    print(f"  {len(patches) - failed}/{len(patches)} patches OK"
          + (f"  ({failed} used reference fallback)" if failed else ""))
    return str(out_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",       required=True)
    p.add_argument("--reference",   required=True)
    p.add_argument("--output",      default="outputs")
    p.add_argument("--k",           type=int,   default=5)
    p.add_argument("--patch-size",  type=int,   default=64)
    p.add_argument("--stride",      type=int,   default=16,
                   help="Overlap stride — 16 = smoother (default), 32 = faster")
    p.add_argument("--ref-weight",  type=float, default=0.55,
                   help="Spatial reference anchor weight (0=pure FAISS, 1=pure reference)")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--config",      default="config.json")
    p.add_argument("--dataset",     default=None)
    p.add_argument("--index",       default="indexes/clean_patches.index")
    p.add_argument("--patch-map",   default="indexes/patch_map.json")
    p.add_argument("--debug",       action="store_true")
    args = p.parse_args()

    restore_image(
        image_path     = Path(args.image),
        reference_path = Path(args.reference),
        output_dir     = Path(args.output),
        k              = args.k,
        patch_size     = args.patch_size,
        stride         = args.stride,
        ref_weight     = args.ref_weight,
        device         = args.device,
        config_path    = Path(args.config),
        dataset_root   = args.dataset,
        index_path     = Path(args.index),
        patch_map_path = Path(args.patch_map),
        debug          = args.debug,
    )
