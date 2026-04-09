"""
Train the UNet Decoder using prepared training data

Simple, clean training loop with:
- L1 loss (good for pixel prediction)
- Adam optimizer with cosine annealing
- Early stopping
- Checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from full_pipeline import UNetDecoder


def train_decoder(
    training_data_path: Path = Path("training_data.pt"),
    output_dir: Path = Path("checkpoints"),
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    early_stopping_patience: int = 5,
    save_interval: int = 1
) -> Dict:
    """
    Train UNet decoder on prepared training data.
    
    Args:
        training_data_path: Path to training_data.pt
        output_dir: Where to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to use
        early_stopping_patience: Stop if val loss doesn't improve
        save_interval: Save checkpoint every N epochs
    
    Returns:
        Dictionary with training statistics
    """
    
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training UNet Decoder")
    print(f"{'='*80}")
    
    # Load training data
    print(f"\nLoading training data from: {training_data_path}")
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_data_path}")
    
    training_data = torch.load(training_data_path, map_location="cpu", weights_only=False)
    fused_embeddings = training_data["fused_embeddings"]  # (N, 512, 16, 16)
    clean_patches = training_data["clean_patches"]  # (N, 3, 64, 64)
    
    print(f"✓ Loaded {len(clean_patches)} training samples")
    print(f"  Fused embeddings: {fused_embeddings.shape}")
    print(f"  Clean patches: {clean_patches.shape}")
    print(f"  Clean patch range: [{clean_patches.min():.4f}, {clean_patches.max():.4f}]")
    
    # Split: 80% train, 20% val
    total_samples = len(clean_patches)
    train_size = int(0.8 * total_samples)
    
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_fused = fused_embeddings[train_indices].to(device)
    train_clean = clean_patches[train_indices].to(device)
    val_fused = fused_embeddings[val_indices].to(device)
    val_clean = clean_patches[val_indices].to(device)
    
    print(f"\nTrain samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Initialize decoder
    print("\nInitializing decoder...")
    decoder = UNetDecoder(
        embedding_dim=512,
        output_channels=3,
        use_residual=True,
        use_conv_transpose=False,
        dropout_rate=0.1
    )
    decoder = decoder.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ Decoder initialized with {num_params:,} parameters")
    
    # Loss, optimizer, scheduler
    criterion = nn.L1Loss()  # L1 is better for pixel prediction
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*80}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }
    
    for epoch in range(1, epochs + 1):
        # Training phase
        decoder.train()
        train_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for batch_start in range(0, len(train_fused), batch_size):
            batch_end = min(batch_start + batch_size, len(train_fused))
            
            batch_fused = train_fused[batch_start:batch_end]  # (B, 512, 16, 16)
            batch_clean = train_clean[batch_start:batch_end]  # (B, 3, 64, 64)
            
            # Forward pass
            # For spatial embeddings (B, 512, 16, 16), need to reduce to (B, 512)
            # Average pooling: (B, 512, 16, 16) → (B, 512)
            batch_fused_reduced = torch.nn.functional.adaptive_avg_pool2d(batch_fused, 1)
            batch_fused_reduced = batch_fused_reduced.squeeze(-1).squeeze(-1)  # (B, 512)
            
            output = decoder(batch_fused_reduced)  # (B, 3, 64, 64)
            
            # Convert tanh output [-1, 1] to [0, 1]
            output_normalized = (output + 1) / 2
            output_normalized = torch.clamp(output_normalized, 0, 1)
            
            # Loss
            loss = criterion(output_normalized, batch_clean)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation phase
        decoder.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_start in range(0, len(val_fused), batch_size):
                batch_end = min(batch_start + batch_size, len(val_fused))
                
                batch_fused = val_fused[batch_start:batch_end]
                batch_clean = val_clean[batch_start:batch_end]
                
                batch_fused_reduced = torch.nn.functional.adaptive_avg_pool2d(batch_fused, 1)
                batch_fused_reduced = batch_fused_reduced.squeeze(-1).squeeze(-1)
                
                output = decoder(batch_fused_reduced)
                output_normalized = (output + 1) / 2
                output_normalized = torch.clamp(output_normalized, 0, 1)
                
                loss = criterion(output_normalized, batch_clean)
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss /= num_val_batches
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        
        # Print
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best checkpoint
            checkpoint_path = output_dir / f"decoder_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "history": history
            }, checkpoint_path)
            print(f"  ✓ Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping: validation loss did not improve for {early_stopping_patience} epochs")
                break
        
        # Save periodic checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = output_dir / f"decoder_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "history": history
            }, checkpoint_path)
    
    # Final summary
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"Total Epochs: {len(history['train_loss'])}")
    
    # Save final model as pretrained
    final_path = output_dir / "decoder_pretrained.pt"
    torch.save({
        "epoch": len(history['train_loss']),
        "model_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
        "history": history
    }, final_path)
    print(f"✓ Saved final model to: {final_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    history_json = {
        "train_loss": [float(x) for x in history["train_loss"]],
        "val_loss": [float(x) for x in history["val_loss"]],
        "learning_rate": history["learning_rate"]
    }
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"✓ Saved training history to: {history_path}")
    
    return {
        "status": "success",
        "epochs_trained": len(history['train_loss']),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history['train_loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
        "checkpoint_path": str(final_path),
        "history_path": str(history_path)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="training_data.pt")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patience", type=int, default=5)
    
    args = parser.parse_args()
    
    result = train_decoder(
        training_data_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        early_stopping_patience=args.patience
    )
    
    print(f"\n{json.dumps(result, indent=2)}")
