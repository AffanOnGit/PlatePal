"""
PlatePal — GPT-2 / DistilGPT-2 Fine-Tuning Script
===================================================

Fine-tunes a pre-trained language model on the cleaned RecipeNLG corpus
using the autoregressive next-token prediction objective.

Usage:
    python train_text_model.py \
        --corpus  data/processed/recipes_train.csv \
        --output  checkpoints/text_model \
        --epochs  5 \
        --batch-size 32 \
        --lr 2e-4
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.utils.data_preprocessing import RecipeTextDataset


# ─────────────────────────────────────────────────────────────────────
# Hyperparameters & Config
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 for recipe generation")
    p.add_argument("--model-name",  type=str, default="distilgpt2")
    p.add_argument("--corpus",      type=str, required=True, help="Path to preprocessed recipe corpus")
    p.add_argument("--output",      type=str, default="checkpoints/text_model")
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--batch-size",  type=int, default=32, help="Per-device batch size (32 recommended for 16 GB GPU)")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup-pct",  type=float, default=0.1, help="Warm-up fraction of total steps")
    p.add_argument("--max-length",  type=int, default=512)
    p.add_argument("--val-split",   type=float, default=0.05, help="Fraction held out for validation")
    p.add_argument("--fp16",        action="store_true", help="Enable mixed-precision training")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Tokenizer ──────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    special_tokens = {
        "bos_token": "<RECIPE_START>",
        "eos_token": "<RECIPE_END>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<INPUT_START>", "<INSTR_START>", "<TITLE>",
        ],
    }
    tokenizer.add_special_tokens(special_tokens)

    # ── Model ──────────────────────────────────────────────────────
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # ── Dataset ────────────────────────────────────────────────────
    full_dataset = RecipeTextDataset(args.corpus, tokenizer, max_length=args.max_length)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"[Train] Train size: {train_size:,}  |  Val size: {val_size:,}")

    # ── Optimizer & Scheduler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Mixed Precision ────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    # ── Training ───────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)

            # Mask padding tokens in loss
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if step % 50 == 0:
                avg = epoch_loss / step
                ppl = math.exp(min(avg, 100))
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f} | PPL {ppl:.2f}")

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask  = batch["attention_mask"].to(device)
                labels          = batch["labels"].to(device)
                labels[labels == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        val_ppl = math.exp(min(avg_val, 100))
        print(f"  ✓ Epoch {epoch} complete | Val Loss {avg_val:.4f} | Val PPL {val_ppl:.2f}")

        # ── Checkpointing ──────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(os.path.join(args.output, "best"))
            tokenizer.save_pretrained(os.path.join(args.output, "best"))
            print(f"  ★ Best model saved (Val Loss {avg_val:.4f})")

    # Save final model
    model.save_pretrained(os.path.join(args.output, "final"))
    tokenizer.save_pretrained(os.path.join(args.output, "final"))
    print(f"[Train] Done. Models saved to {args.output}/")


# ─────────────────────────────────────────────────────────────────────
# Quality Checks (post-training)
# ─────────────────────────────────────────────────────────────────────

def quality_check(model_dir: str, test_prompts: list = None):
    """
    Run the Logic Check from the training quality checklist:
    - Generate 10 recipes
    - Check for repetition and ingredient coverage
    """
    from app.models.text_gen import RecipeGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = RecipeGenerator(model_name=model_dir, device=device)

    if test_prompts is None:
        test_prompts = [
            "chicken, rice, garlic, onion",
            "pasta, tomato, basil, mozzarella",
            "beef, potato, carrot, onion",
            "eggs, flour, butter, sugar, vanilla",
            "fish, lemon, dill, olive oil",
            "chickpeas, tahini, garlic, lemon",
            "lamb, yogurt, cumin, coriander, rice",
            "shrimp, coconut milk, curry paste, basil",
            "tofu, soy sauce, ginger, sesame oil",
            "mushroom, cream, thyme, garlic, pasta",
        ]

    print("\n" + "=" * 60)
    print(" QUALITY CHECK — Logic & Coverage")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        recipe = gen.generate_recipe(prompt, max_length=256)
        ingredients_set = set(ing.strip().lower() for ing in prompt.split(","))

        # Coverage: what fraction of input ingredients appear in the output?
        mentioned = sum(1 for ing in ingredients_set if ing in recipe.lower())
        coverage = mentioned / len(ingredients_set)

        # Repetition: count duplicate sentences
        sentences = [s.strip() for s in recipe.split(".") if s.strip()]
        unique = set(sentences)
        repetition = 1.0 - (len(unique) / max(len(sentences), 1))

        status = "✓" if coverage >= 0.8 and repetition < 0.3 else "✗"
        print(f"\n[{status}] Recipe #{i}: {prompt}")
        print(f"    Coverage:   {coverage:.0%}  ({mentioned}/{len(ingredients_set)} ingredients)")
        print(f"    Repetition: {repetition:.0%}")
        print(f"    Preview:    {recipe[:120]}...")


if __name__ == "__main__":
    args = parse_args()
    train(args)
