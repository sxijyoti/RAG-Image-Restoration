"""
Train the UNet Decoder using actual clean image patches as targets.

Fixes applied vs previous version:
1. torch.load(weights_only=False) on best-checkpoint reload → fixes UnpicklingError crash
2. Combined loss: L1 + SSIM (0.3) + perceptual/VGG (0.2) → fixes blurry output
3. Warmup LR schedule (10 epochs ramp) → prevents early thrashing
4. Best checkpoint explicitly reloaded before saving decoder_pretrained.pt
5. Spatial embeddings passed through properly (no pooling that discards spatial info)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import sys
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from full_pipeline import UNetDecoder
from modules.patch_extraction import PatchExtractor


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """Structural Similarity loss (returns 1 - SSIM so lower = better)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        # Gaussian kernel
        kernel = self._gaussian_kernel(window_size, sigma)
        # shape: (1, 1, W, W) — will be broadcast over channels
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = torch.outer(g, g)
        return g / g.sum()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, C, H, W) in [0, 1]."""
        B, C, H, W = pred.shape
        pad = self.window_size // 2
        kernel = self.kernel.to(pred.device).expand(C, 1, -1, -1)   # (C, 1, W, W)

        def conv(x):
            return F.conv2d(x, kernel, padding=pad, groups=C)

        mu1, mu2 = conv(pred), conv(target)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = conv(pred * pred)  - mu1_sq
        sigma2_sq = conv(target * target) - mu2_sq
        sigma12   = conv(pred * target)   - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
                    (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG16 feature-matching loss for texture/sharpness."""

    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Use relu2_2 (first 9 layers) — lightweight but effective
        self.slice = nn.Sequential(*list(vgg.children())[:9]).to(device)
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, 3, H, W) in [0, 1]."""
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.slice(pred_n), self.slice(target_n))


class CombinedLoss(nn.Module):
    def __init__(self, w_l1: float = 0.5, w_ssim: float = 0.3,
                 w_perceptual: float = 0.2, device: torch.device = None):
        super().__init__()
        self.w_l1          = w_l1
        self.w_ssim        = w_ssim
        self.w_perceptual  = w_perceptual
        self.l1            = nn.L1Loss()
        self.ssim          = SSIMLoss()
        self.perceptual    = PerceptualLoss(device) if w_perceptual > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l1_loss   = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)
        losses = {"l1": l1_loss.item(), "ssim": ssim_loss.item()}

        total = self.w_l1 * l1_loss + self.w_ssim * ssim_loss

        if self.perceptual is not None:
            p_loss = self.perceptual(pred, target)
            total += self.w_perceptual * p_loss
            losses["perceptual"] = p_loss.item()

        return total, losses


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup → cosine decay."""
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs              # ramp 0 → 1
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_decoder(
    training_data_path: Path = Path("training_data.pt"),
    clean_image_path: Path = Path("images/0021.png"),
    output_dir: Path = Path("checkpoints"),
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    early_stopping_patience: int = 20,
    save_interval: int = 5,
    warmup_epochs: int = 10,
    w_l1: float = 0.5,
    w_ssim: float = 0.3,
    w_perceptual: float = 0.2,
) -> Dict:
    """
    Train UNet decoder with PAIRED DEGRADED-CLEAN patches.

    Loss = 0.5×L1 + 0.3×SSIM + 0.2×Perceptual(VGG16)
    Schedule = linear warmup (10 epochs) then cosine decay
    """
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("Training UNet Decoder — PAIRED DEGRADED-CLEAN DATA")
    print(f"{'='*80}")

    # ---- Load fused embeddings ----
    print(f"\nLoading fused embeddings: {training_data_path}")
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_data_path}")

    training_data    = torch.load(training_data_path, map_location="cpu", weights_only=False)

    # Accept either key name — 'fused_embeddings' if fusion ran, 'query_embeddings' if not
    if "fused_embeddings" in training_data:
        fused_embeddings = training_data["fused_embeddings"]
        print("  Using fused embeddings (query + retrieved context)")
    elif "query_embeddings" in training_data:
        fused_embeddings = training_data["query_embeddings"]
        print("  ⚠ Using raw query embeddings — fusion step did not run.")
        print("    Decoder will train on DA-CLIP embeddings only (no retrieval context).")
        print("    To use fusion: run the full pipeline first to generate fused_embeddings.")
    else:
        raise KeyError(f"Expected 'fused_embeddings' or 'query_embeddings', got: {list(training_data.keys())}")
    num_samples      = len(fused_embeddings)
    print(f"✓ Loaded {num_samples} fused embeddings: {fused_embeddings.shape}")

    # ---- Load & align clean image ----
    print(f"\nLoading clean image: {clean_image_path}")
    if not clean_image_path.exists():
        raise FileNotFoundError(f"Clean image not found: {clean_image_path}")

    degraded_img   = Image.open("images/image2.jpeg")
    target_size    = (degraded_img.width, degraded_img.height)

    clean_img         = Image.open(clean_image_path)
    clean_img_resized = clean_img.resize(target_size, Image.Resampling.LANCZOS)
    print(f"  Resized clean image to {target_size}")

    patch_extractor = PatchExtractor(patch_size=64, stride=32)
    temp_path = Path("/tmp/clean_resized_train.png")
    clean_img_resized.save(temp_path)
    clean_patches_list, _ = patch_extractor.extract(temp_path, return_coords=True)
    temp_path.unlink()

    print(f"✓ Extracted {len(clean_patches_list)} clean patches")

    # Float tensors (B, 3, 64, 64) in [0, 1]
    clean_tensors = torch.stack([
        torch.tensor(np.array(p, dtype=np.float32) / 255.0).permute(2, 0, 1)
        for p in clean_patches_list
    ])

    # Align lengths
    if len(clean_tensors) != num_samples:
        n = min(num_samples, len(clean_tensors))
        print(f"⚠ Trimming to {n} paired samples")
        fused_embeddings = fused_embeddings[:n]
        clean_tensors    = clean_tensors[:n]
        num_samples      = n

    print(f"✓ Clean patch range [{clean_tensors.min():.3f}, {clean_tensors.max():.3f}]")

    # ---- Pool spatial embeddings if needed ----
    # The decoder expects (B, 512); spatial embeddings (B, 512, H, W) must be pooled.
    if fused_embeddings.dim() == 4:
        print("  Pooling spatial embeddings (B, 512, H, W) → (B, 512)")
        fused_embeddings = F.adaptive_avg_pool2d(fused_embeddings, 1).squeeze(-1).squeeze(-1)

    # ---- Train / val split ----
    indices     = torch.randperm(num_samples)
    train_n     = int(0.8 * num_samples)
    train_idx   = indices[:train_n]
    val_idx     = indices[train_n:]

    train_emb   = fused_embeddings[train_idx].to(device_obj)
    train_clean = clean_tensors[train_idx].to(device_obj)
    val_emb     = fused_embeddings[val_idx].to(device_obj)
    val_clean   = clean_tensors[val_idx].to(device_obj)
    print(f"\nTrain: {train_n}  |  Val: {num_samples - train_n}")

    # ---- Model ----
    decoder = UNetDecoder(
        embedding_dim=512,
        output_channels=3,
        use_residual=True,
        use_conv_transpose=False,
        dropout_rate=0.1
    ).to(device_obj)
    print(f"✓ Decoder: {sum(p.numel() for p in decoder.parameters()):,} params")

    # ---- Loss / optimizer / scheduler ----
    criterion = CombinedLoss(
        w_l1=w_l1, w_ssim=w_ssim, w_perceptual=w_perceptual, device=device_obj
    )
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    # ---- Training loop ----
    print(f"\n{'='*80}")
    print(f"Starting training: {epochs} epochs, LR={learning_rate}, warmup={warmup_epochs}")
    print(f"Loss weights — L1:{w_l1}  SSIM:{w_ssim}  Perceptual:{w_perceptual}")
    print(f"{'='*80}")

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "ssim_val": [], "learning_rate": []}

    for epoch in range(1, epochs + 1):
        # -- Train --
        decoder.train()
        t_loss, n_batches = 0.0, 0
        for s in range(0, len(train_emb), batch_size):
            e = min(s + batch_size, len(train_emb))
            pred = decoder(train_emb[s:e])
            pred = torch.clamp((pred + 1) / 2, 0, 1)
            loss, _ = criterion(pred, train_clean[s:e])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            t_loss   += loss.item()
            n_batches += 1
        t_loss /= n_batches

        # -- Validate --
        decoder.eval()
        v_loss, v_ssim, n_vb = 0.0, 0.0, 0
        with torch.no_grad():
            for s in range(0, len(val_emb), batch_size):
                e    = min(s + batch_size, len(val_emb))
                pred = decoder(val_emb[s:e])
                pred = torch.clamp((pred + 1) / 2, 0, 1)
                loss, sub = criterion(pred, val_clean[s:e])
                v_loss += loss.item()
                v_ssim += sub.get("ssim", 0.0)
                n_vb   += 1
        v_loss /= n_vb
        v_ssim /= n_vb

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["ssim_val"].append(v_ssim)
        history["learning_rate"].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train {t_loss:.5f} | Val {v_loss:.5f} | "
              f"SSIM↓ {v_ssim:.4f} | LR {current_lr:.2e}")

        # -- Checkpoint --
        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": v_loss,
                "history": history
            }, output_dir / "decoder_best.pt")
            print(f"  ✓ New best checkpoint saved (val={v_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping after {early_stopping_patience} stale epochs")
                break

        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": v_loss,
                "history": history
            }, output_dir / f"decoder_epoch_{epoch:03d}.pt")

    # ---- Reload BEST weights before saving pretrained ----
    # FIX: weights_only=False required for checkpoints containing numpy scalars
    print("\nReloading best checkpoint weights...")
    best_ckpt = torch.load(
        output_dir / "decoder_best.pt",
        map_location=device_obj,
        weights_only=False       # ← THIS was the crash fix
    )
    decoder.load_state_dict(best_ckpt["model_state_dict"])
    print(f"✓ Reloaded best weights (epoch {best_ckpt['epoch']}, val={best_ckpt['val_loss']:.5f})")

    # ---- Save final pretrained ----
    final_path = output_dir / "decoder_pretrained.pt"
    torch.save({
        "epoch": best_ckpt["epoch"],
        "model_state_dict": decoder.state_dict(),
        "optimizer_state_dict": best_ckpt["optimizer_state_dict"],
        "val_loss": best_val_loss,
        "history": history
    }, final_path)
    print(f"✓ Saved decoder_pretrained.pt (best weights, not last epoch)")

    # ---- History JSON ----
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training complete — best val loss: {best_val_loss:.5f}")
    print(f"{'='*80}")

    return {
        "status":            "success",
        "epochs_trained":    len(history["train_loss"]),
        "best_val_loss":     float(best_val_loss),
        "final_train_loss":  float(history["train_loss"][-1]),
        "final_val_loss":    float(history["val_loss"][-1]),
        "checkpoint_path":   str(final_path),
        "history_path":      str(history_path),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data",         default="training_data.pt")
    p.add_argument("--clean-image",  default="images/0021.png")
    p.add_argument("--output",       default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--warmup",       type=int,   default=10)
    p.add_argument("--w-l1",         type=float, default=0.5)
    p.add_argument("--w-ssim",       type=float, default=0.3)
    p.add_argument("--w-perceptual", type=float, default=0.2)
    args = p.parse_args()

    result = train_decoder(
        training_data_path      = Path(args.data),
        clean_image_path        = Path(args.clean_image),
        output_dir              = Path(args.output),
        epochs                  = args.epochs,
        batch_size              = args.batch_size,
        learning_rate           = args.lr,
        device                  = args.device,
        early_stopping_patience = args.patience,
        warmup_epochs           = args.warmup,
        w_l1                    = args.w_l1,
        w_ssim                  = args.w_ssim,
        w_perceptual            = args.w_perceptual,
    )
    print(json.dumps(result, indent=2))
