"""
PlatePal — CVAE Training Script
=================================
Usage:
    python train_cvae.py --fp16

Trains a Conditional Variational Autoencoder on the combined Food-101 + CAFD
dataset. Uses CLIP embeddings for class conditioning.

Key differences from DCGAN:
  - Single model (no adversarial game)
  - Loss: MSE reconstruction + KL divergence
  - Converges steadily in ~50 epochs
  - No mode collapse possible

Output: checkpoints/cvae/cvae_final.pth
"""

import os
import sys
import math
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Project imports
from app.models.image_gen import CVAE, cvae_loss
from app.utils.clip_embedder import get_embedder

# ── Reuse data loading from train_image_model.py ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from train_image_model import build_dataloader, build_class_embedding_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CVAE for PlatePal food image generation")
    p.add_argument("--food101-h5",  type=str, default="data/raw/food_128.h5")
    p.add_argument("--cafd-dir",    type=str, default="data/raw/CAFD")
    p.add_argument("--output",      type=str, default="checkpoints/cvae")
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--latent-dim",  type=int, default=256)
    p.add_argument("--embed-dim",   type=int, default=512)
    p.add_argument("--kl-weight",   type=float, default=0.01,
                   help="Weight for KL divergence term. Start low to prioritise reconstruction.")
    p.add_argument("--fp16",        action="store_true", help="Enable mixed-precision training")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    print(f"\n{'='*60}")
    print(f" PlatePal CVAE Training")
    print(f" Device: {DEVICE}  |  FP16: {args.fp16}")
    print(f" Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────────────
    dataloader, class_names = build_dataloader(args)
    n_classes = len(class_names)

    # ── CLIP Embedder ──────────────────────────────────────────────
    embedder = get_embedder(use_clip=True, device=str(DEVICE))
    class_embeddings = build_class_embedding_cache(class_names, embedder, DEVICE)
    # shape: (n_classes, 512)

    # ── Model ──────────────────────────────────────────────────────
    model = CVAE(latent_dim=args.latent_dim, embed_dim=args.embed_dim).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[CVAE] Model parameters: {total_params:,}")

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # Cosine LR schedule: gently decays LR over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Mixed Precision ────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and DEVICE == "cuda" else None

    # ── Output Dirs ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    samples_dir = os.path.join(args.output, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # ── Fixed samples for visual tracking ─────────────────────────
    # Always use the same 16 class embeddings to track visual progress
    fixed_indices = [i % n_classes for i in range(16)]
    fixed_embed   = class_embeddings[fixed_indices]  # (16, 512)

    # ── Training History ───────────────────────────────────────────
    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    print(f"[CVAE] Starting training: {args.epochs} epochs, {len(dataloader)} batches/epoch\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0

        # Anneal KL weight: start at 10% of target, reach full by epoch 20
        # This is called "KL Annealing" — prevents posterior collapse early in training
        kl_anneal = min(1.0, epoch / 20.0) * args.kl_weight

        progress = tqdm(dataloader, desc=f"Epoch {epoch:>3}/{args.epochs}", unit="batch")
        for real_imgs, labels in progress:
            real_imgs = real_imgs.to(DEVICE)
            labels    = labels.clamp(0, n_classes - 1).long()

            # Look up CLIP embedding for each image's class
            cond_embed = class_embeddings[labels]  # (B, 512)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    recon, mu, logvar = model(real_imgs, cond_embed)
                    loss, rl, kl = cvae_loss(recon, real_imgs, mu, logvar, kl_weight=kl_anneal)
                scaler.scale(loss).backward()
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, mu, logvar = model(real_imgs, cond_embed)
                loss, rl, kl = cvae_loss(recon, real_imgs, mu, logvar, kl_weight=kl_anneal)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_total += loss.item()
            epoch_recon += rl.item()
            epoch_kl    += kl.item()

            progress.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "recon": f"{rl.item():.4f}",
                "kl":    f"{kl.item():.4f}",
                "kl_w":  f"{kl_anneal:.2f}",
            })

        scheduler.step()

        # ── Epoch Summary ──────────────────────────────────────────
        n = len(dataloader)
        avg_total = epoch_total / n
        avg_recon = epoch_recon / n
        avg_kl    = epoch_kl    / n
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        print(f"  Epoch [{epoch:>3}/{args.epochs}] Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        # ── Save Sample Grid ───────────────────────────────────────
        if epoch % args.save_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Generate 16 samples for the fixed classes
                samples = model.generate(fixed_embed, temperature=0.8)
                samples = (samples + 1) / 2  # [-1,1] -> [0,1]
                save_path = os.path.join(samples_dir, f"sample_epoch_{epoch}.png")
                # Grid of 4x4
                save_image(samples, save_path, nrow=4)
                print(f"  -> Saved 128x128 sample grid: {save_path}")
            model.train()

    # ── Save Final Model ───────────────────────────────────────────
    final_path = os.path.join(args.output, "cvae_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[CVAE] Training complete. Model saved to {final_path}")

    # ── Save Loss History ──────────────────────────────────────────
    hist_path = os.path.join(args.output, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[CVAE] Loss history saved to {hist_path}")
    print(f"\n[CVAE] Next step:")
    print(f"  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    args = parse_args()

    # ── GPU Info ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / (1024**3)
        print(f"[GPU] {props.name} -- {vram:.1f} GB VRAM")

    train(args)
