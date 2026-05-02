"""
PlatePal — Conditional DCGAN Training Script
==============================================

Trains a Conditional DCGAN where:
  - The Generator receives a 100-dim noise vector + 512-dim CLIP embedding.
  - The Discriminator classifies 64x64x3 images as real or fake.

Includes:
  - Mixed-precision training (FP16)
  - Mode collapse detection via periodic image saving
  - Loss logging for stability monitoring

Usage:
    python train_image_model.py \
        --food101-h5  data/raw/food_c101_n10099_r64x64x3.h5 \
        --cafd-dir    data/raw/CAFD \
        --output      checkpoints/dcgan \
        --epochs      200 \
        --batch-size  128
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.models.image_gen import Generator, Discriminator
from app.utils.data_preprocessing import (
    load_food101_h5,
    load_cafd_images,
    FoodImageDataset,
)
from app.utils.clip_embedder import get_embedder


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional DCGAN for food images")
    p.add_argument("--food101-h5",  type=str, help="Path to Food-101 HDF5 file")
    p.add_argument("--cafd-dir",    type=str, help="Path to CAFD root directory")
    p.add_argument("--output",      type=str, default="checkpoints/dcgan")
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=128, help="128 recommended for 16 GB GPU")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--beta1",       type=float, default=0.5, help="Adam beta1")
    p.add_argument("--z-dim",       type=int, default=100, help="Noise vector dimension")
    p.add_argument("--embed-dim",   type=int, default=512, help="CLIP embedding dimension")
    p.add_argument("--fp16",        action="store_true", help="Enable mixed-precision")
    p.add_argument("--save-interval", type=int, default=5, help="Save sample images every N epochs")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Dataset Preparation
# ─────────────────────────────────────────────────────────────────────

def build_dataloader(args):
    """Load and combine all available image datasets."""
    datasets = []

    if args.food101_h5 and os.path.exists(args.food101_h5):
        images, labels = load_food101_h5(args.food101_h5)
        datasets.append(FoodImageDataset(images, labels))
        print(f"[Data] Food-101: {len(images):,} images")

    if args.cafd_dir and os.path.isdir(args.cafd_dir):
        images, labels = load_cafd_images(args.cafd_dir)
        datasets.append(FoodImageDataset(images))
        print(f"[Data] CAFD: {len(images):,} images")

    if not datasets:
        # Generate a small synthetic dataset for testing the pipeline
        print("[Data] WARNING: No dataset provided. Creating synthetic data for pipeline test.")
        synthetic = np.random.randint(0, 255, (500, 64, 64, 3), dtype=np.uint8)
        datasets.append(FoodImageDataset(synthetic))

    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Data] Total images: {len(combined):,}  |  Batches: {len(loader)}")
    return loader


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Models ─────────────────────────────────────────────────────
    generator     = Generator(z_dim=args.z_dim, embed_dim=args.embed_dim).to(device)
    discriminator = Discriminator().to(device)

    # ── Optimizers ─────────────────────────────────────────────────
    opt_G = torch.optim.Adam(generator.parameters(),     lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # ── Loss ───────────────────────────────────────────────────────
    adversarial_loss = nn.BCELoss()

    # ── Data ───────────────────────────────────────────────────────
    dataloader = build_dataloader(args)

    # ── CLIP Embedder (for conditioning) ───────────────────────────
    embedder = get_embedder(use_clip=True, device=str(device))

    # ── Mixed Precision ────────────────────────────────────────────
    scaler_G = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None
    scaler_D = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    # ── Fixed noise for visual tracking ────────────────────────────
    fixed_z     = torch.randn(16, args.z_dim).to(device)
    fixed_embed = torch.randn(16, args.embed_dim).to(device)  # Will be replaced with real embeddings

    # ── Output dirs ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    samples_dir = os.path.join(args.output, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # ── Loss history for stability monitoring ──────────────────────
    history = {"d_loss": [], "g_loss": [], "d_real_acc": [], "d_fake_acc": []}

    print(f"\n{'='*60}")
    print(f" Starting DCGAN Training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0

        for step, (real_imgs, _) in enumerate(dataloader, 1):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels
            valid = torch.ones(batch_size, 1, device=device) * 0.9   # label smoothing
            fake  = torch.zeros(batch_size, 1, device=device)

            # Random conditioning embeddings (in production, use CLIP)
            cond_embed = torch.randn(batch_size, args.embed_dim, device=device)

            # ─── Train Discriminator ───────────────────────────────
            opt_D.zero_grad()

            if scaler_D:
                with torch.amp.autocast("cuda"):
                    d_real = discriminator(real_imgs)
                    loss_real = adversarial_loss(d_real, valid)

                    z = torch.randn(batch_size, args.z_dim, device=device)
                    gen_imgs = generator(z, cond_embed).detach()
                    d_fake = discriminator(gen_imgs)
                    loss_fake = adversarial_loss(d_fake, fake)

                    d_loss = (loss_real + loss_fake) / 2

                scaler_D.scale(d_loss).backward()
                scaler_D.step(opt_D)
                scaler_D.update()
            else:
                d_real = discriminator(real_imgs)
                loss_real = adversarial_loss(d_real, valid)

                z = torch.randn(batch_size, args.z_dim, device=device)
                gen_imgs = generator(z, cond_embed).detach()
                d_fake = discriminator(gen_imgs)
                loss_fake = adversarial_loss(d_fake, fake)

                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                opt_D.step()

            # ─── Train Generator ───────────────────────────────────
            opt_G.zero_grad()

            if scaler_G:
                with torch.amp.autocast("cuda"):
                    z = torch.randn(batch_size, args.z_dim, device=device)
                    gen_imgs = generator(z, cond_embed)
                    g_pred = discriminator(gen_imgs)
                    g_loss = adversarial_loss(g_pred, valid)

                scaler_G.scale(g_loss).backward()
                scaler_G.step(opt_G)
                scaler_G.update()
            else:
                z = torch.randn(batch_size, args.z_dim, device=device)
                gen_imgs = generator(z, cond_embed)
                g_pred = discriminator(gen_imgs)
                g_loss = adversarial_loss(g_pred, valid)
                g_loss.backward()
                opt_G.step()

            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

        # ── Epoch Summary ──────────────────────────────────────────
        avg_d = d_epoch_loss / len(dataloader)
        avg_g = g_epoch_loss / len(dataloader)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)

        print(f"Epoch [{epoch:>3}/{args.epochs}]  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

        # ── MODE COLLAPSE CHECK: Save samples every N epochs ──────
        if epoch % args.save_interval == 0 or epoch == 1:
            generator.eval()
            with torch.no_grad():
                sample_imgs = generator(fixed_z, fixed_embed)
                sample_imgs = (sample_imgs + 1) / 2  # [-1,1] → [0,1]
                save_path = os.path.join(samples_dir, f"sample_epoch_{epoch}.png")
                save_image(sample_imgs, save_path, nrow=4)
                print(f"  → Saved sample grid: {save_path}")
            generator.train()

        # ── LOSS STABILITY CHECK ──────────────────────────────────
        if avg_d < 0.01:
            print("  ⚠ WARNING: D_loss near zero — Discriminator may be too strong!")
        if avg_g > 10.0:
            print("  ⚠ WARNING: G_loss very high — Generator struggling to learn!")

        # ── Checkpoint every 50 epochs ─────────────────────────────
        if epoch % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output, f"generator_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.output, f"discriminator_epoch_{epoch}.pth"))
            print(f"  ★ Checkpoint saved at epoch {epoch}")

    # ── Save final models ──────────────────────────────────────────
    torch.save(generator.state_dict(), os.path.join(args.output, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.output, "discriminator_final.pth"))

    # ── Save loss history ──────────────────────────────────────────
    import json
    with open(os.path.join(args.output, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Complete. Weights saved to {args.output}/")


if __name__ == "__main__":
    args = parse_args()
    train(args)
