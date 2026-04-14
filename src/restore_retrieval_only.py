"""
RAG Image Restoration — Retrieval + Refinement Pipeline (FIXED)
================================================================

BUGS FIXED IN NO-REFERENCE PATH
---------------------------------

BUG 1 — Geometric re-ranking was INVERTED (main cause of blur)
  OLD: score = 0.6*SSIM(clean_candidate, degraded_patch)
       SSIM between a CLEAN patch and a DEGRADED patch is LOW for sharp
       structural regions — the degradation artifacts (crosshatch, blur)
       actively lower SSIM against clean patches with sharp edges.
       Result: re-ranker preferred blurry/low-detail candidates → blurry output.
  FIX: Remove SSIM-vs-degraded entirely. Instead rank by SHARPNESS of the
       candidate itself (Laplacian variance). Among semantically correct
       FAISS results, the sharpest candidate is the best restoration.

BUG 2 — Same-source constraint forced over-smoothing
  OLD: Counter(sources).most_common(1) → keep ONLY patches from the most
       common source image. Then blend all k of them. Since they're from
       the same image at nearby coords, blending averages out fine detail.
  FIX: Remove same-source constraint. FAISS embedding distance already
       ensures semantic relevance. Allow diverse sources.

BUG 3 — Blending k patches destroys sharpness
  OLD: blend all k=5 top patches with distance weights → averaging washes
       out edges, veins, windows, any fine structure.
  FIX: Use top-1 sharpest patch only (no blending) for the no-reference
       path. Sharp edges cannot survive averaging across multiple patches.

BUG 4 — Edge correlation scored against degraded patch
  OLD: edge_corr = correlation(sobel(candidate), sobel(degraded))
       Degraded patch edges are attenuated/noisy → correlation is low
       for correctly-sharp candidates, high for blurry ones.
  FIX: Dropped entirely — replaced by candidate sharpness ranking.

With-reference path: UNCHANGED. ref_weight blend still works correctly.
"""

import torch
import torch.nn as nn
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


# ── Refinement UNet (unchanged — must match train_refinement.py) ──────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class RefinementUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch,     base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 1)

    def forward(self, x):
        inp = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.clamp(inp + torch.tanh(self.out_conv(d1)) * 0.5, 0, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_gaussian_window(patch_size: int, sigma_ratio: float = 0.35) -> np.ndarray:
    sigma = patch_size * sigma_ratio
    ax = np.arange(patch_size) - (patch_size - 1) / 2.0
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32)


def _get_coord(meta: dict) -> tuple:
    x = 0
    for k in ("x", "patch_x", "col", "left"):
        if k in meta: x = int(meta[k]); break
    y = 0
    for k in ("y", "patch_y", "row", "top"):
        if k in meta: y = int(meta[k]); break
    return x, y


def _resolve_path(raw_path: str, dataset_root=None) -> str:
    if not raw_path: return raw_path
    p = Path(raw_path)
    if p.exists(): return str(p)
    if dataset_root:
        c = Path(dataset_root) / p.name
        if c.exists(): return str(c)
    c2 = Path("images") / p.name
    if c2.exists(): return str(c2)
    return raw_path


def _safe_crop(img_arr: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    h, w = img_arr.shape[:2]
    y1, y2 = int(y), min(int(y) + patch_size, h)
    x1, x2 = int(x), min(int(x) + patch_size, w)
    crop = img_arr[y1:y2, x1:x2, :]
    ph, pw = crop.shape[:2]
    if ph == patch_size and pw == patch_size: return crop
    out = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    out[:ph, :pw, :] = crop
    if pw < patch_size: out[:ph, pw:, :] = crop[:, -1:, :]
    if ph < patch_size: out[ph:, :, :]   = out[ph-1:ph, :, :]
    return out


def _load_patch_from_metadata(meta: dict, patch_size: int,
                               dataset_root=None, debug=False):
    raw_path = ""
    for k in ("image_path", "path", "file", "filename", "img_path", "img", "image"):
        if k in meta and meta[k]: raw_path = str(meta[k]); break
    if not raw_path: return None
    resolved = _resolve_path(raw_path, dataset_root)
    try:
        arr = np.array(Image.open(resolved).convert("RGB"))
        mx, my = _get_coord(meta)
        return _safe_crop(arr, mx, my, patch_size)
    except Exception as e:
        if debug: print(f"  [warn] {resolved}: {e}")
        return None


# ── FIX: sharpness-based top-1 selection ─────────────────────────────────────

def _laplacian_sharpness(patch: np.ndarray) -> float:
    """
    Laplacian variance of grayscale patch.
    Higher = sharper = more fine detail preserved.
    This is the correct criterion for selecting among FAISS candidates:
    FAISS already ensures semantic similarity; among semantically similar
    patches, the sharpest one is the best restoration target.
    """
    gray = (0.299 * patch[:,:,0].astype(np.float32) +
            0.587 * patch[:,:,1].astype(np.float32) +
            0.114 * patch[:,:,2].astype(np.float32))
    # Simple 3x3 Laplacian
    gp = np.pad(gray, 1, mode='reflect')
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    h, w = gray.shape
    lap = sum(kernel[i,j] * gp[i:i+h, j:j+w] for i in range(3) for j in range(3))
    return float(lap.var())


def _retrieve_best_patch(
    retriever,
    query_np: np.ndarray,
    k_coarse: int,
    patch_size: int,
    patch_loader=None,
    dataset_root=None,
    ref_arr=None,
    debug=False,
) -> np.ndarray | None:
    """
    FIX: Two-stage retrieval — coarse FAISS then sharpness-ranked top-1.

    Stage 1: fetch k_coarse candidates via FAISS embedding distance.
             FAISS ensures semantic/content similarity.
    Stage 2: among those candidates, return the SINGLE sharpest one.
             No blending — blending averages away fine detail.

    Removes same-source constraint (BUG 2): diverse sources are fine,
    FAISS distance already handles relevance.
    """
    indices, distances, metadata = retriever.search(query_np, k=k_coarse)

    # Load candidate pixels from all sources (no same-source filtering)
    candidates = []
    if patch_loader is not None:
        loaded = patch_loader.load_patches_from_metadata(metadata, patch_size=patch_size)
        for p in loaded:
            if isinstance(p, np.ndarray) and p.shape == (patch_size, patch_size, 3):
                candidates.append(p)
    elif ref_arr is not None:
        for m in metadata:
            p = _safe_crop(ref_arr, *_get_coord(m), patch_size)
            candidates.append(p)
    else:
        for m in metadata:
            p = _load_patch_from_metadata(m, patch_size, dataset_root, debug)
            if p is not None:
                candidates.append(p)

    if not candidates:
        return None

    # FIX BUG 1+3: select top-1 by sharpness, no blending
    if len(candidates) == 1:
        best = candidates[0]
    else:
        sharpness = [_laplacian_sharpness(c) for c in candidates]
        best = candidates[int(np.argmax(sharpness))]

    return best.astype(np.float32) / 255.0


# ── Main ──────────────────────────────────────────────────────────────────────

def restore_image(
    image_path: Path,
    reference_path: Path   = None,
    output_dir: Path       = Path("outputs"),
    k: int                 = 5,
    k_coarse: int          = 20,
    patch_size: int        = 64,
    stride: int            = 16,
    ref_weight: float      = 0.4,
    device: str            = "cuda",
    config_path: Path      = Path("config.json"),
    dataset_root: Path     = None,
    index_path: Path       = Path("indexes/clean_patches.index"),
    patch_map_path: Path   = Path("indexes/patch_map.json"),
    refinement_ckpt: Path  = Path("checkpoints/refinement_pretrained.pt"),
    debug: bool            = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | stride: {stride} | k_coarse: {k_coarse}")

    # 1. Extract degraded patches
    print(f"\n[1/6] Extracting patches: {image_path}")
    extractor    = PatchExtractor(patch_size=patch_size, stride=stride)
    deg_img      = Image.open(image_path).convert("RGB")
    img_w, img_h = deg_img.size
    patches, coords = extractor.extract(image_path, return_coords=True)
    print(f"  {img_w}x{img_h} -> {len(patches)} patches")

    # 2. Reference image (optional)
    ref_arr = None
    if reference_path is not None:
        print(f"\n[2/6] Reference: {reference_path}")
        ref_img = Image.open(reference_path).convert("RGB").resize(
            (img_w, img_h), Image.Resampling.LANCZOS
        )
        ref_arr = np.array(ref_img)
        print(f"  Resized to {img_w}x{img_h}")
    else:
        ref_weight = 0.0
        print("\n[2/6] No reference — pure FAISS + sharpness-ranked top-1 retrieval")

    # 3. Encoder
    print("\n[3/6] DA-CLIP encoder...")
    config = {}
    if config_path.exists():
        with open(config_path) as f: config = json.load(f)
    encoder = DACLIPEncoder(
        model_name=config.get("model_name", "ViT-B-32"),
        pretrained =config.get("pretrained", "laion2b_s34b_b79k"),
        normalize=True, device=str(device), debug=debug,
    )
    print("  Ready")

    # 4. FAISS
    retriever    = None
    patch_loader = None
    if index_path.exists() and patch_map_path.exists():
        print(f"\n[4/6] FAISS: {index_path}")
        loader    = FAISSIndexLoader(debug=debug)
        index     = loader.load_index(index_path)
        patch_map = loader.load_patch_map(patch_map_path)
        retriever = PatchRetriever(index, patch_map, normalize_query=True, debug=debug)
        if dataset_root and Path(dataset_root).exists():
            patch_loader = PatchLoader(dataset_root=Path(dataset_root), debug=debug)
            print(f"  PatchLoader active: {dataset_root}")
        else:
            print("  Loading patches from metadata paths directly")
    else:
        print(f"\n[4/6] No FAISS index found")
        if ref_arr is None:
            print("  WARNING: no FAISS and no reference — output will equal input")

    # 5. Refinement UNet (optional)
    refine_net = None
    if refinement_ckpt.exists():
        print(f"\n[5/6] Loading RefinementUNet: {refinement_ckpt}")
        ckpt = torch.load(refinement_ckpt, map_location=device, weights_only=False)
        refine_net = RefinementUNet(in_channels=3, base_ch=64).to(device)
        refine_net.load_state_dict(ckpt["model_state_dict"])
        refine_net.eval()
        print(f"  Loaded (epoch {ckpt.get('epoch','?')}, val={ckpt.get('val_loss',0):.5f})")
    else:
        print(f"\n[5/6] No refinement checkpoint — retrieval-only mode")

    # 6. Restore
    print(f"\n[6/6] Restoring {len(patches)} patches...")
    gauss  = make_gaussian_window(patch_size)
    acc    = np.zeros((img_h, img_w, 3), dtype=np.float64)
    wt     = np.zeros((img_h, img_w),    dtype=np.float64)
    failed = 0
    ds_root = str(dataset_root) if dataset_root else None

    for idx, (patch, (x, y)) in enumerate(
        tqdm(zip(patches, coords), total=len(patches), desc="Restoring")
    ):
        try:
            patch_np = np.array(patch, dtype=np.uint8)

            # Spatial reference crop (with-reference path — unchanged)
            ref_crop = (_safe_crop(ref_arr, x, y, patch_size).astype(np.float32) / 255.0
                        if ref_arr is not None else None)

            # FAISS retrieval
            faiss_result = None
            if retriever is not None:
                q_np = encoder.encode_patch(patch_np).cpu().numpy()

                if ref_arr is not None:
                    # WITH-REFERENCE: original distance-weighted blend (unchanged)
                    indices_, distances_, metadata_ = retriever.search(q_np, k=k)
                    dists_ = np.asarray(distances_, dtype=np.float64).flatten()
                    retrieved_ = []
                    if patch_loader is not None:
                        retrieved_ = patch_loader.load_patches_from_metadata(
                            metadata_, patch_size=patch_size)
                    else:
                        for m in metadata_:
                            p_ = _load_patch_from_metadata(m, patch_size, ds_root, debug)
                            if p_ is not None: retrieved_.append(p_)
                    valid_ = [p_ for p_ in retrieved_
                              if isinstance(p_, np.ndarray) and p_.shape == (patch_size, patch_size, 3)]
                    if valid_:
                        d_ = dists_[:len(valid_)]
                        w_ = 1.0 / (d_ + 1e-6); w_ /= w_.sum()
                        blended_ = np.zeros((patch_size, patch_size, 3), dtype=np.float64)
                        for p_, wi_ in zip(valid_, w_):
                            blended_ += p_.astype(np.float64) * wi_
                        faiss_result = np.clip(blended_ / 255.0, 0, 1).astype(np.float32)
                else:
                    # NO-REFERENCE: FIX — sharpness-ranked top-1, no blending
                    faiss_result = _retrieve_best_patch(
                        retriever, q_np,
                        k_coarse=k_coarse,
                        patch_size=patch_size,
                        patch_loader=patch_loader,
                        dataset_root=ds_root,
                        ref_arr=None,
                        debug=(debug and idx == 0),
                    )

            # Combine
            if faiss_result is not None and ref_crop is not None:
                blended = ref_weight * ref_crop + (1 - ref_weight) * faiss_result
            elif faiss_result is not None:
                blended = faiss_result
            elif ref_crop is not None:
                blended = ref_crop
            else:
                if failed == 0:
                    print(f"\n  [warn] no retrieval and no reference — using degraded input")
                blended = patch_np.astype(np.float32) / 255.0
                failed += 1

            # Optional refinement UNet
            if refine_net is not None:
                with torch.no_grad():
                    inp_t = torch.tensor(blended).permute(2,0,1).unsqueeze(0).to(device)
                    out_t = refine_net(inp_t).squeeze(0).permute(1,2,0).cpu().numpy()
                restored = out_t
            else:
                restored = blended

            # Gaussian-weighted accumulation
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            ph, pw = y_end - y, x_end - x
            g = gauss[:ph, :pw].astype(np.float64)
            acc[y:y_end, x:x_end, :] += restored[:ph,:pw,:].astype(np.float64) * g[:,:,np.newaxis]
            wt[y:y_end, x:x_end]     += g

        except Exception as e:
            failed += 1
            if debug or idx < 3: print(f"  Patch {idx} @ ({x},{y}) failed: {e}")
            try:
                fb = (_safe_crop(ref_arr, x, y, patch_size).astype(np.float64) / 255.0
                      if ref_arr is not None
                      else np.array(patch, dtype=np.float64) / 255.0)
            except Exception:
                fb = np.zeros((patch_size, patch_size, 3), dtype=np.float64)
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            ph, pw = y_end - y, x_end - x
            g = gauss[:ph, :pw].astype(np.float64)
            acc[y:y_end, x:x_end, :] += fb[:ph,:pw,:] * g[:,:,np.newaxis]
            wt[y:y_end, x:x_end]     += g

    wt       = np.maximum(wt, 1e-8)
    restored = np.clip(acc / wt[:,:,np.newaxis] * 255, 0, 255).astype(np.uint8)
    out_path = output_dir / f"{image_path.stem}_restored.png"
    Image.fromarray(restored).save(out_path)

    print(f"\n✓ Saved: {out_path}")
    print(f"  {len(patches)-failed}/{len(patches)} patches OK")
    if refine_net is None:
        print("  Tip: train train_refinement.py then re-run for sharper output")
    return str(out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",           required=True)
    p.add_argument("--reference",       default=None)
    p.add_argument("--output",          default="outputs")
    p.add_argument("--k",               type=int,   default=5)
    p.add_argument("--k-coarse",        type=int,   default=20,
                   help="Coarse FAISS retrieval count before sharpness ranking")
    p.add_argument("--patch-size",      type=int,   default=64)
    p.add_argument("--stride",          type=int,   default=16)
    p.add_argument("--ref-weight",      type=float, default=0.4)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--config",          default="config.json")
    p.add_argument("--dataset",         default=None)
    p.add_argument("--index",           default="indexes/clean_patches.index")
    p.add_argument("--patch-map",       default="indexes/patch_map.json")
    p.add_argument("--refinement-ckpt", default="checkpoints/refinement_pretrained.pt")
    p.add_argument("--debug",           action="store_true")
    args = p.parse_args()

    restore_image(
        image_path       = Path(args.image),
        reference_path   = Path(args.reference) if args.reference else None,
        output_dir       = Path(args.output),
        k                = args.k,
        k_coarse         = args.k_coarse,
        patch_size       = args.patch_size,
        stride           = args.stride,
        ref_weight       = args.ref_weight,
        device           = args.device,
        config_path      = Path(args.config),
        dataset_root     = Path(args.dataset) if args.dataset else None,
        index_path       = Path(args.index),
        patch_map_path   = Path(args.patch_map),
        refinement_ckpt  = Path(args.refinement_ckpt),
        debug            = args.debug,
    )
