"""
PlatePal — Conditional DCGAN Training Script
==============================================

Trains a Conditional DCGAN where:
  - The Generator receives a 100-dim noise vector + 512-dim CLIP embedding.
  - The Discriminator classifies 64x64x3 images as real or fake.

Conditioning Strategy:
  Each image class label (e.g., "pizza", "biryani") is encoded once via CLIP
  into a 512-dim embedding and cached. During training, the Generator receives
  the class-specific CLIP embedding — NOT random noise — so the generated
  images are semantically tied to the food class.

Includes:
  - Mixed-precision training (FP16)
  - Mode collapse detection via periodic image saving
  - Loss logging for stability monitoring
  - GPU-only enforcement

Usage:
    python train_image_model.py \
        --food101-h5  data/raw/food_c101_n10099_r64x64x3.h5 \
        --cafd-dir    data/raw/CAFD \
        --output      checkpoints/dcgan \
        --epochs      100 \
        --batch-size  256 \
        --fp16
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
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.models.image_gen import CVAE as Generator  # Alias for backward compat in this script
from app.utils.data_preprocessing import (
    load_food101_h5,
    load_cafd_images,
    FoodImageDataset,
)
from app.utils.clip_embedder import get_embedder


# ═══════════════════════════════════════════════════════════════════
# GPU-ONLY GUARD — Training on CPU is NOT permitted
# ═══════════════════════════════════════════════════════════════════
if not torch.cuda.is_available():
    raise RuntimeError(
        "╔════════════════════════════════════════════════════════════╗\n"
        "║  FATAL: No CUDA GPU detected.                            ║\n"
        "║  PlatePal DCGAN training requires an NVIDIA GPU.         ║\n"
        "║  CPU training is disabled by design.                     ║\n"
        "╚════════════════════════════════════════════════════════════╝"
    )

DEVICE = torch.device("cuda")
print(f"[GPU] {torch.cuda.get_device_name(0)} — "
      f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM")


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional DCGAN for food images")
    p.add_argument("--food101-h5",  type=str, help="Path to Food-101 HDF5 file")
    p.add_argument("--cafd-dir",    type=str, help="Path to CAFD root directory")
    p.add_argument("--output",      type=str, default="checkpoints/dcgan")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch-size",  type=int, default=128, help="128 recommended for 16 GB GPU")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--beta1",       type=float, default=0.5, help="Adam beta1")
    p.add_argument("--z-dim",       type=int, default=100, help="Noise vector dimension")
    p.add_argument("--embed-dim",   type=int, default=512, help="CLIP embedding dimension")
    p.add_argument("--fp16",        action="store_true", help="Enable mixed-precision")
    p.add_argument("--save-interval", type=int, default=10, help="Save sample images every N epochs")
    p.add_argument("--resume",      type=int, default=None, help="Resume from checkpoint at this epoch number")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# CLIP Class Embedding Cache
# ─────────────────────────────────────────────────────────────────────

def build_class_embedding_cache(class_names: list, embedder, device: str) -> torch.Tensor:
    """
    Pre-compute CLIP embeddings for each class label.

    Args:
        class_names: List of unique class name strings (e.g., ['pizza', 'sushi', ...]).
        embedder: CLIPEmbedder or DummyCLIPEmbedder instance.
        device: Target device.

    Returns:
        Tensor of shape (num_classes, 512) — one embedding per class.
    """
    print(f"[CLIP Cache] Building embeddings for {len(class_names)} classes...")
    embeddings = []
    for name in class_names:
        # Format as a descriptive prompt for better CLIP encoding
        prompt = f"A photo of {name.replace('_', ' ')} food dish"
        emb = embedder.encode_text(prompt)  # (1, 512)
        embeddings.append(emb)
    cache = torch.cat(embeddings, dim=0).to(device)  # (num_classes, 512)
    print(f"[CLIP Cache] Built cache: {cache.shape}")
    return cache


# ─────────────────────────────────────────────────────────────────────
# Dataset Preparation
# ─────────────────────────────────────────────────────────────────────

def build_dataloader(args, target_size=(128, 128)):
    """Load and combine all available image datasets. Returns (dataloader, class_names)."""
    datasets = []
    all_class_names = []

    if args.food101_h5 and os.path.exists(args.food101_h5):
        images, labels = load_food101_h5(args.food101_h5)
        datasets.append(FoodImageDataset(images, labels))
        # Build class names from unique labels
        unique_labels = sorted(set(labels.tolist()))
        # Food-101 uses integer labels — map them to descriptive names
        food101_classes = [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
            "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
            "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
            "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
            "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
            "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters",
            "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib",
            "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
            "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
            "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
            "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
            "waffles",
        ]
        for idx in unique_labels:
            if idx < len(food101_classes):
                all_class_names.append(food101_classes[idx])
            else:
                all_class_names.append(f"food_class_{idx}")
        print(f"[Data] Food-101: {len(images):,} images, {len(unique_labels)} classes")

    if args.cafd_dir and os.path.isdir(args.cafd_dir):
        images, labels = load_cafd_images(args.cafd_dir, target_size=target_size)
        # Offset CAFD labels so they don't overlap with Food-101
        offset = len(all_class_names)
        labels_offset = labels + offset
        datasets.append(FoodImageDataset(images, labels_offset))

        # Get CAFD class names from directory structure
        cafd_dirs = sorted([d.name for d in Path(args.cafd_dir).iterdir() if d.is_dir()])
        all_class_names.extend(cafd_dirs)
        print(f"[Data] CAFD: {len(images):,} images, {len(cafd_dirs)} classes (offset: +{offset})")

    if not datasets:
        # Generate a small synthetic dataset for testing the pipeline
        print("[Data] WARNING: No dataset provided. Creating synthetic data for pipeline test.")
        synthetic = np.random.randint(0, 255, (500, 64, 64, 3), dtype=np.uint8)
        synthetic_labels = np.zeros(500, dtype=np.int64)
        datasets.append(FoodImageDataset(synthetic, synthetic_labels))
        all_class_names = ["synthetic_food"]

    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Data] Total images: {len(combined):,}  |  Batches: {len(loader)}  |  Classes: {len(all_class_names)}")
    return loader, all_class_names


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = DEVICE
    print(f"[Train] Device: {device}")

    # ── Models ─────────────────────────────────────────────────────
    generator     = Generator(z_dim=args.z_dim, embed_dim=args.embed_dim).to(device)
    discriminator = Discriminator().to(device)

    # ── Resume from checkpoint ─────────────────────────────────────
    start_epoch = 1
    if args.resume:
        g_ckpt = os.path.join(args.output, f"generator_epoch_{args.resume}.pth")
        d_ckpt = os.path.join(args.output, f"discriminator_epoch_{args.resume}.pth")
        if os.path.isfile(g_ckpt) and os.path.isfile(d_ckpt):
            generator.load_state_dict(torch.load(g_ckpt, map_location=device))
            discriminator.load_state_dict(torch.load(d_ckpt, map_location=device))
            start_epoch = args.resume + 1
            print(f"[Resume] Loaded checkpoint from epoch {args.resume}")
        else:
            print(f"[Resume] WARNING: Checkpoint files not found at epoch {args.resume}, starting fresh")

    # ── Optimizers (asymmetric LR: D learns faster to provide useful gradients) ──
    opt_G = torch.optim.Adam(generator.parameters(),     lr=args.lr,        betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr * 4,    betas=(args.beta1, 0.999))

    # ── Loss ───────────────────────────────────────────────────────
    adversarial_loss = nn.BCEWithLogitsLoss()

    # ── Data ───────────────────────────────────────────────────────
    dataloader, class_names = build_dataloader(args)

    # ── CLIP Embedder + Class Embedding Cache ──────────────────────
    embedder = get_embedder(use_clip=True, device=str(device))
    class_embeddings = build_class_embedding_cache(class_names, embedder, device)
    # class_embeddings shape: (num_classes, 512)

    # ── Mixed Precision ────────────────────────────────────────────
    scaler_G = torch.amp.GradScaler("cuda") if args.fp16 else None
    scaler_D = torch.amp.GradScaler("cuda") if args.fp16 else None

    # ── Fixed noise + fixed embeddings for visual tracking ─────────
    fixed_z     = torch.randn(16, args.z_dim).to(device)
    # Use the first 16 class embeddings (cycling if needed) for consistent visual tracking
    n_classes = len(class_names)
    fixed_indices = [i % n_classes for i in range(16)]
    fixed_embed = class_embeddings[fixed_indices]  # (16, 512)

    # ── Output dirs ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    samples_dir = os.path.join(args.output, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # ── Loss history for stability monitoring ──────────────────────
    history = {"d_loss": [], "g_loss": []}

    print(f"\n{'='*60}")
    print(f" Starting DCGAN Training: epochs {start_epoch}→{args.epochs}")
    print(f" Classes: {len(class_names)}  |  Embedding cache: {class_embeddings.shape}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        generator.train()
        discriminator.train()

        progress = tqdm(dataloader, desc=f"Epoch {epoch:>3}/{args.epochs}", unit="batch")
        for step, (real_imgs, labels) in enumerate(progress, 1):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # Label smoothing + label noise for regularization
            valid = torch.ones(batch_size, 1, device=device) * 0.9
            fake  = torch.ones(batch_size, 1, device=device) * 0.1

            # Instance noise: inject Gaussian noise into D inputs (decays over training)
            noise_std = max(0.0, 0.5 * (1.0 - epoch / args.epochs))

            # Look up CLIP embeddings for this batch's class labels
            labels_clamped = labels.clamp(0, len(class_names) - 1).long()
            cond_embed = class_embeddings[labels_clamped]  # (batch_size, 512)

            # ─── Train Discriminator (2 steps per G step) ─────────
            for d_step in range(2):
                opt_D.zero_grad()

                # Add instance noise to real images
                real_noisy = real_imgs + noise_std * torch.randn_like(real_imgs)

                if scaler_D:
                    with torch.amp.autocast("cuda"):
                        d_real = discriminator(real_noisy)
                        loss_real = adversarial_loss(d_real, valid)

                        z = torch.randn(batch_size, args.z_dim, device=device)
                        gen_imgs = generator(z, cond_embed).detach()
                        # Add instance noise to fake images too
                        gen_noisy = gen_imgs + noise_std * torch.randn_like(gen_imgs)
                        d_fake = discriminator(gen_noisy)
                        loss_fake = adversarial_loss(d_fake, fake)

                        d_loss = (loss_real + loss_fake) / 2

                    scaler_D.scale(d_loss).backward()
                    scaler_D.step(opt_D)
                    scaler_D.update()
                else:
                    d_real = discriminator(real_noisy)
                    loss_real = adversarial_loss(d_real, valid)

                    z = torch.randn(batch_size, args.z_dim, device=device)
                    gen_imgs = generator(z, cond_embed).detach()
                    gen_noisy = gen_imgs + noise_std * torch.randn_like(gen_imgs)
                    d_fake = discriminator(gen_noisy)
                    loss_fake = adversarial_loss(d_fake, fake)

                    d_loss = (loss_real + loss_fake) / 2
                    d_loss.backward()
                    opt_D.step()

            # ─── Train Generator (1 step) ─────────────────────────
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

            progress.set_postfix({
                "D": f"{d_loss.item():.4f}",
                "G": f"{g_loss.item():.4f}",
                "noise": f"{noise_std:.2f}",
            })

        # ── Epoch Summary ──────────────────────────────────────────
        avg_d = d_epoch_loss / len(dataloader)
        avg_g = g_epoch_loss / len(dataloader)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)

        print(f"  Epoch [{epoch:>3}/{args.epochs}]  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

        # ── MODE COLLAPSE CHECK: Save samples every N epochs ──────
        if epoch % args.save_interval == 0 or epoch == 1:
            generator.eval()
            with torch.no_grad():
                sample_imgs = generator(fixed_z, fixed_embed)
                sample_imgs = (sample_imgs + 1) / 2  # [-1,1] -> [0,1]
                save_path = os.path.join(samples_dir, f"sample_epoch_{epoch}.png")
                save_image(sample_imgs, save_path, nrow=4)
                print(f"  -> Saved sample grid: {save_path}")
            generator.train()

        # ── LOSS STABILITY CHECK ──────────────────────────────────
        if avg_d < 0.01:
            print("  [WARN] WARNING: D_loss near zero -- Discriminator may be too strong!")
        if avg_g > 10.0:
            print("  [WARN] WARNING: G_loss very high -- Generator struggling to learn!")

        # ── Checkpoint every 50 epochs ─────────────────────────────
        if epoch % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output, f"generator_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.output, f"discriminator_epoch_{epoch}.pth"))
            print(f"  [CKPT] Checkpoint saved at epoch {epoch}")

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
