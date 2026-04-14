"""
Train a Patch Refinement Network (pixel → pixel).

KEY IMPROVEMENTS over v1
-------------------------
1. RefinementUNet now accepts retrieved patch as input (not raw degraded patch)
   The network learns: retrieved_patch → clean_patch  (much easier task)
   At inference: degraded → FAISS retrieval → RefinementUNet → output
   This mirrors the reference-image path without needing a reference.

2. Augmentation simulates the actual degradation pattern of image2.jpeg:
   - Crosshatch grid artifact (vertical + horizontal lines)
   - Random Gaussian blur
   - JPEG block artifact simulation
   These ensure the UNet generalises beyond the single training image.

3. Combined loss: L1(0.45) + SSIM(0.35) + Perceptual/VGG(0.20)
   Higher SSIM weight → sharper structural edges.

4. Warmup + cosine LR schedule prevents early thrashing.

5. --augment-retrieved flag (default True): at training time, simulate retrieval
   by adding slight noise to clean patches, so the net learns to handle
   imperfect retrieved inputs.

Train:
  python src/train_refinement.py \
      --degraded images/image2.jpeg \
      --clean-image images/0021.png \
      --device cuda --epochs 200 \
      --stride 16 --augment-retrieved

Restore (no reference needed):
  python src/restore_retrieval_only.py \
      --image images/image2.jpeg \
      --dataset images/ \
      --output outputs/ \
      --stride 8 --k 10 \
      --refine-on-retrieved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

sys.path.insert(0, str(Path(__file__).parent))

from modules.patch_extraction import PatchExtractor


# ── Refinement UNet (pixel → pixel residual net) ─────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class RefinementUNet(nn.Module):
    """
    Pixel-to-pixel UNet.
    Input:  (B, 3, 64, 64)  — retrieved clean patch (may be slightly imperfect)
    Output: (B, 3, 64, 64)  — sharpened / corrected clean patch in [0,1]

    Uses residual learning: output = input + tanh(correction) * 0.5
    This means the network only needs to learn small corrections, not full
    pixel values — much easier to train.
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.clamp(inp + torch.tanh(self.out_conv(d1)) * 0.5, 0, 1)


# ── Loss functions ────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels    = channels
        self.register_buffer("window", self._make_window(window_size, channels))

    @staticmethod
    def _make_window(size, channels):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g /= g.sum()
        w = torch.outer(g, g)
        return w.expand(channels, 1, size, size).contiguous()

    def forward(self, pred, target):
        C1, C2 = 0.01**2, 0.03**2
        pad = self.window_size // 2
        w = self.window.to(pred.device)
        mu1 = F.conv2d(pred,   w, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, w, padding=pad, groups=self.channels)
        s1  = F.conv2d(pred*pred,     w, padding=pad, groups=self.channels) - mu1**2
        s2  = F.conv2d(target*target, w, padding=pad, groups=self.channels) - mu2**2
        s12 = F.conv2d(pred*target,   w, padding=pad, groups=self.channels) - mu1*mu2
        ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.available = False
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.slice = nn.Sequential(*list(vgg.features)[:9]).eval()
            for p in self.parameters(): p.requires_grad_(False)
            self.available = True
            print("  VGG16 perceptual loss: ready")
        except Exception as e:
            print(f"  VGG16 perceptual loss unavailable ({e})")
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        if not self.available: return torch.tensor(0., device=pred.device)
        p = (pred   - self.mean.to(pred.device)) / self.std.to(pred.device)
        t = (target - self.mean.to(pred.device)) / self.std.to(pred.device)
        return F.l1_loss(self.slice(p), self.slice(t))


class CombinedLoss(nn.Module):
    def __init__(self, w_l1=0.45, w_ssim=0.35, w_perc=0.20):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_perc = w_l1, w_ssim, w_perc
        self.ssim = SSIMLoss()
        self.perc = PerceptualLoss()

    def forward(self, pred, target):
        l1   = F.l1_loss(pred, target)
        ssim = self.ssim(pred, target)
        perc = self.perc(pred, target)
        total = self.w_l1*l1 + self.w_ssim*ssim + self.w_perc*perc
        return total, {"l1": l1.item(), "ssim": ssim.item(), "perc": perc.item()}


# ── Augmentation ──────────────────────────────────────────────────────────────

def simulate_crosshatch_artifact(patch_t: torch.Tensor,
                                  intensity: float = 0.15) -> torch.Tensor:
    """Simulate the crosshatch grid artifact seen in image2.jpeg."""
    C, H, W = patch_t.shape
    grid = torch.zeros(H, W, device=patch_t.device)
    step = np.random.randint(6, 12)
    line_intensity = np.random.uniform(0.05, intensity)
    grid[::step, :] = line_intensity
    grid[:, ::step] = line_intensity
    return torch.clamp(patch_t * (1 - grid.unsqueeze(0)), 0, 1)


def simulate_blur(patch_t: torch.Tensor, max_sigma: float = 1.5) -> torch.Tensor:
    """Simulate motion/defocus blur."""
    sigma = np.random.uniform(0.3, max_sigma)
    size = 5
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g).to(patch_t.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, size, size)
    inp = patch_t.unsqueeze(0)
    out = F.conv2d(inp, kernel, padding=size//2, groups=3)
    return out.squeeze(0)


def simulate_retrieval_imperfection(clean_patch: torch.Tensor,
                                     noise_std: float = 0.03) -> torch.Tensor:
    """
    Simulate the small errors in FAISS-retrieved patches.
    The retriever finds similar but not identical patches, so we add
    slight noise + color shift to train the UNet to handle imperfect inputs.
    """
    noise = torch.randn_like(clean_patch) * np.random.uniform(0.01, noise_std)
    color_shift = (torch.rand(3, 1, 1) - 0.5) * 0.05
    return torch.clamp(clean_patch + noise + color_shift.to(clean_patch.device), 0, 1)


def augment_pair(inp: torch.Tensor, tgt: torch.Tensor,
                 augment_retrieved: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply identical geometric augmentations to input and target.
    Apply degradation augmentation only to input.

    inp = retrieved patch (or degraded patch if augment_retrieved=False)
    tgt = clean patch
    """
    # Geometric (must be identical for both)
    if torch.rand(1) > 0.5:
        inp = torch.flip(inp, [-1]); tgt = torch.flip(tgt, [-1])
    if torch.rand(1) > 0.5:
        inp = torch.flip(inp, [-2]); tgt = torch.flip(tgt, [-2])
    # Random 90-degree rotation
    k = np.random.randint(0, 4)
    if k > 0:
        inp = torch.rot90(inp, k, [-2, -1])
        tgt = torch.rot90(tgt, k, [-2, -1])

    # Degradation on input only (simulates imperfect retrieval)
    if augment_retrieved:
        r = np.random.random()
        if r < 0.4:
            inp = simulate_crosshatch_artifact(inp)
        elif r < 0.7:
            inp = simulate_blur(inp)
        if np.random.random() < 0.5:
            inp = simulate_retrieval_imperfection(inp)

    return inp, tgt


# ── Data loading ──────────────────────────────────────────────────────────────

def load_patch_pairs(
    degraded_image_path: Path,
    clean_image_path: Path,
    patch_size: int = 64,
    stride: int = 16,
    augment_retrieved: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract aligned patch pairs.

    KEY CHANGE: inputs are the CLEAN patches (simulating retrieved patches),
    not the degraded patches. The degraded image is only used to set the
    target dimensions (so we resize the clean image to match).

    Returns:
        inputs:  (N, 3, 64, 64) — clean patches + retrieval imperfection simulation
        targets: (N, 3, 64, 64) — clean patches (ground truth)
    """
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)

    deg_img = Image.open(degraded_image_path).convert("RGB")
    target_size = (deg_img.width, deg_img.height)

    clean_img = Image.open(clean_image_path).convert("RGB")
    if clean_img.size != target_size:
        print(f"  Resizing clean image {clean_img.size} → {target_size}")
        clean_img = clean_img.resize(target_size, Image.Resampling.LANCZOS)

    # Extract patches from clean image
    clean_patches, coords = extractor.extract(clean_img, return_coords=True)

    def to_tensor(patches):
        return torch.stack([
            torch.tensor(np.array(p, dtype=np.float32) / 255.0).permute(2, 0, 1)
            for p in patches
        ])

    clean_t = to_tensor(clean_patches)  # (N, 3, 64, 64)

    # Inputs = clean patches with simulated retrieval imperfection
    # This teaches the net: "given a retrieved patch that's close but not perfect,
    # output the truly clean version"
    inputs_t = torch.stack([
        simulate_retrieval_imperfection(p, noise_std=0.04) for p in clean_t
    ])

    print(f"  {len(clean_t)} patch pairs extracted (stride={stride})")
    print(f"  Input  range: [{inputs_t.min():.3f}, {inputs_t.max():.3f}]")
    print(f"  Target range: [{clean_t.min():.3f},  {clean_t.max():.3f}]")

    return inputs_t, clean_t


# ── LR schedule ───────────────────────────────────────────────────────────────

def warmup_cosine(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    def f(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        p = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def train_refinement_net(
    degraded_image_path: Path = Path("images/image2.jpeg"),
    clean_image_path: Path    = Path("images/0021.png"),
    output_dir: Path          = Path("checkpoints"),
    patch_size: int           = 64,
    stride: int               = 16,
    epochs: int               = 200,
    batch_size: int           = 16,
    learning_rate: float      = 2e-4,
    device: str               = "cuda",
    early_stopping_patience: int = 25,
    save_interval: int        = 10,
    w_l1: float               = 0.45,
    w_ssim: float             = 0.35,
    w_perc: float             = 0.20,
    augment_retrieved: bool   = True,
) -> Dict:

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("Training Patch Refinement UNet")
    print("Mode: retrieved-patch → clean-patch (pixel → pixel)")
    print(f"{'='*80}")
    print(f"Degraded image: {degraded_image_path}")
    print(f"Clean image:    {clean_image_path}")
    print(f"Augment retrieved: {augment_retrieved}")

    print("\nExtracting patch pairs...")
    inp_t, clean_t = load_patch_pairs(
        degraded_image_path, clean_image_path,
        patch_size=patch_size, stride=stride,
        augment_retrieved=augment_retrieved,
    )
    num_samples = len(inp_t)

    # Train/val split
    perm  = torch.randperm(num_samples)
    split = int(0.85 * num_samples)
    tr, va = perm[:split], perm[split:]

    train_inp   = inp_t[tr].to(device_obj)
    train_clean = clean_t[tr].to(device_obj)
    val_inp     = inp_t[va].to(device_obj)
    val_clean   = clean_t[va].to(device_obj)
    print(f"\n  Train: {len(tr)} | Val: {len(va)}")

    model = RefinementUNet(in_channels=3, base_ch=64).to(device_obj)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  RefinementUNet params: {n_params:,}")

    criterion = CombinedLoss(w_l1, w_ssim, w_perc).to(device_obj)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = warmup_cosine(optimizer, warmup_epochs=12, total_epochs=epochs)

    print(f"\n{'='*80}")
    print(f"Training: {epochs} epochs | batch={batch_size} | lr={learning_rate}")
    print(f"Loss: L1×{w_l1} + SSIM×{w_ssim} + Perceptual×{w_perc}")
    print(f"{'='*80}")

    best_val   = float("inf")
    patience   = 0
    history    = {"train_loss": [], "val_loss": [], "val_ssim": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, nb = 0., 0

        idx = torch.randperm(len(train_inp))
        train_inp_s   = train_inp[idx]
        train_clean_s = train_clean[idx]

        for s in range(0, len(train_inp_s), batch_size):
            e   = min(s + batch_size, len(train_inp_s))
            inp = train_inp_s[s:e].clone()
            tgt = train_clean_s[s:e].clone()

            # Per-batch geometric + degradation augmentation
            aug_inp_list, aug_tgt_list = [], []
            for i in range(len(inp)):
                ai, at = augment_pair(inp[i], tgt[i], augment_retrieved)
                aug_inp_list.append(ai)
                aug_tgt_list.append(at)
            inp = torch.stack(aug_inp_list)
            tgt = torch.stack(aug_tgt_list)

            out = model(inp)
            loss, _ = criterion(out, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item(); nb += 1

        tr_loss /= nb

        model.eval()
        va_loss, va_ssim, nv = 0., 0., 0
        with torch.no_grad():
            for s in range(0, len(val_inp), batch_size):
                e = min(s + batch_size, len(val_inp))
                out = model(val_inp[s:e])
                loss, parts = criterion(out, val_clean[s:e])
                va_loss += loss.item()
                va_ssim += parts["ssim"]
                nv += 1
        va_loss /= nv; va_ssim /= nv
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_ssim"].append(va_ssim)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train {tr_loss:.5f} | Val {va_loss:.5f} | "
              f"SSIM↓ {va_ssim:.4f} | LR {cur_lr:.2e}")

        if va_loss < best_val:
            best_val = va_loss
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "history": history,
                "arch": "RefinementUNet",
                "mode": "retrieved_to_clean",
            }, output_dir / "refinement_best.pt")
            print(f"  ✓ Best saved (val={va_loss:.5f})")
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"\n  Early stopping after {early_stopping_patience} stale epochs")
                break

        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "history": history,
                "arch": "RefinementUNet",
            }, output_dir / f"refinement_epoch_{epoch:03d}.pt")

    # Reload best and save as pretrained
    best_ckpt = torch.load(
        output_dir / "refinement_best.pt",
        map_location=device_obj, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_path = output_dir / "refinement_pretrained.pt"
    torch.save({
        "epoch": best_ckpt["epoch"],
        "model_state_dict": model.state_dict(),
        "val_loss": best_val,
        "history": history,
        "arch": "RefinementUNet",
        "mode": "retrieved_to_clean",
    }, final_path)

    with open(output_dir / "refinement_history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Done. Best val loss: {best_val:.5f}")
    print(f"Saved: {final_path}")
    print(f"{'='*80}")
    print("\nNext step:")
    print("  python src/restore_retrieval_only.py \\")
    print("      --image images/image2.jpeg \\")
    print("      --dataset images/ \\")
    print("      --output outputs/ \\")
    print("      --stride 8 --k 10 --refine-on-retrieved")

    return {
        "status": "success",
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": float(best_val),
        "checkpoint_path": str(final_path),
        "arch": "RefinementUNet",
        "mode": "retrieved_to_clean",
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train patch refinement UNet (retrieved→clean)")
    p.add_argument("--degraded",           type=str, default="images/image2.jpeg")
    p.add_argument("--clean-image",        type=str, default="images/0021.png")
    p.add_argument("--output",             type=str, default="checkpoints")
    p.add_argument("--patch-size",         type=int, default=64)
    p.add_argument("--stride",             type=int, default=16)
    p.add_argument("--epochs",             type=int, default=200)
    p.add_argument("--batch-size",         type=int, default=16)
    p.add_argument("--lr",                 type=float, default=2e-4)
    p.add_argument("--device",             type=str, default="cuda")
    p.add_argument("--patience",           type=int, default=25)
    p.add_argument("--w-l1",               type=float, default=0.45)
    p.add_argument("--w-ssim",             type=float, default=0.35)
    p.add_argument("--w-perceptual",       type=float, default=0.20)
    p.add_argument("--augment-retrieved",  action="store_true", default=True)
    p.add_argument("--no-augment",         action="store_true")
    args = p.parse_args()

    result = train_refinement_net(
        degraded_image_path = Path(args.degraded),
        clean_image_path    = Path(args.clean_image),
        output_dir          = Path(args.output),
        patch_size          = args.patch_size,
        stride              = args.stride,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        learning_rate       = args.lr,
        device              = args.device,
        early_stopping_patience = args.patience,
        w_l1                = args.w_l1,
        w_ssim              = args.w_ssim,
        w_perc              = args.w_perceptual,
        augment_retrieved   = not args.no_augment,
    )
    print(json.dumps(result, indent=2))
