"""
PlatePal — Full Evaluation Runner
===================================

Runs all evaluation metrics from the implementation plan:
  - Text: Perplexity, BLEU-4, ROUGE-L, Ingredient Coverage
  - Image: FID (Fréchet Inception Distance)

Usage:
    python run_evaluation.py \
        --text-model  checkpoints/text_model/best \
        --image-model checkpoints/dcgan/generator_final.pth \
        --food101-h5  data/raw/food_c101_n10099_r64x64x3.h5
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.models.text_gen import RecipeGenerator
from app.models.image_gen import Generator
from app.utils.evaluation import (
    evaluate_text_quality,
    evaluate_image_quality,
    ingredient_coverage,
    generate_report,
)
from app.utils.clip_embedder import get_embedder


# ─────────────────────────────────────────────────────────────────────
# Test Prompts
# ─────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "chicken, rice, garlic, onion, cumin",
    "pasta, tomato, basil, mozzarella, olive oil",
    "beef, potato, carrot, onion, thyme",
    "eggs, flour, butter, sugar, vanilla",
    "fish, lemon, dill, olive oil, capers",
    "chickpeas, tahini, garlic, lemon, paprika",
    "lamb, yogurt, cumin, coriander, rice",
    "shrimp, coconut milk, curry paste, basil, rice",
    "tofu, soy sauce, ginger, sesame oil, broccoli",
    "mushroom, cream, thyme, garlic, pasta",
]


def evaluate_text(args):
    """Run text model evaluation."""
    print("\n" + "=" * 60)
    print(" TEXT MODEL EVALUATION")
    print("=" * 60)

    ckpt = args.text_model if os.path.isdir(args.text_model) else None
    gen = RecipeGenerator(checkpoint_dir=ckpt, device=args.device)

    # Generate recipes
    generated = []
    coverages = []
    for prompt in TEST_PROMPTS:
        recipe = gen.generate_recipe(prompt, max_length=256)
        generated.append(recipe)
        cov = ingredient_coverage(prompt, recipe)
        coverages.append(cov)
        print(f"  [{cov:.0%}] {prompt[:40]}...")

    avg_coverage = np.mean(coverages)
    print(f"\n  Average Ingredient Coverage: {avg_coverage:.2%}")
    if avg_coverage < 0.8:
        print("  ⚠ Coverage < 80% — model may need more fine-tuning!")

    # Evaluate text quality
    results = evaluate_text_quality(
        gen.model, gen.tokenizer, generated, device=args.device
    )
    results["avg_ingredient_coverage"] = float(avg_coverage)

    print(f"\n  Perplexity: {results['perplexity']:.2f}")
    return results


def evaluate_images(args):
    """Run image model evaluation."""
    print("\n" + "=" * 60)
    print(" IMAGE MODEL EVALUATION")
    print("=" * 60)

    device = args.device

    # Load generator
    gen = Generator(z_dim=100, embed_dim=512).to(device)
    if os.path.isfile(args.image_model):
        gen.load_state_dict(torch.load(args.image_model, map_location=device))
        print(f"  Loaded weights: {args.image_model}")
    else:
        print(f"  ⚠ No weights found at {args.image_model}, using random init")
    gen.eval()

    # Generate fake images
    embedder = get_embedder(use_clip=True, device=device)
    n_samples = 100
    fake_images = []

    print(f"  Generating {n_samples} sample images...")
    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, 100, device=device)
            embed = torch.randn(1, 512, device=device)  # random conditioning
            img = gen(z, embed)
            img = (img + 1) / 2  # [-1,1] → [0,1]
            fake_images.append(img)

    fake_batch = torch.cat(fake_images, dim=0)

    # Load real images for FID comparison
    if args.food101_h5 and os.path.isfile(args.food101_h5):
        from app.utils.data_preprocessing import load_food101_h5
        real_imgs, _ = load_food101_h5(args.food101_h5)
        # Subsample and normalize
        indices = np.random.choice(len(real_imgs), min(n_samples, len(real_imgs)), replace=False)
        real_subset = real_imgs[indices]
        real_tensor = torch.from_numpy(real_subset).permute(0, 3, 1, 2).float() / 255.0

        results = evaluate_image_quality(real_tensor, fake_batch, device)
    else:
        print("  ⚠ No real image dataset provided — skipping FID")
        results = {"fid": "N/A (no reference dataset)"}

    return results


def main():
    parser = argparse.ArgumentParser(description="PlatePal Evaluation Runner")
    parser.add_argument("--text-model",  type=str, default="checkpoints/text_model/best")
    parser.add_argument("--image-model", type=str, default="checkpoints/dcgan/generator_final.pth")
    parser.add_argument("--food101-h5",  type=str, default="data/raw/food_c101_n10099_r64x64x3.h5")
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",      type=str, default="evaluation_report.json")
    args = parser.parse_args()

    # Run evaluations
    text_results = evaluate_text(args)
    image_results = evaluate_images(args)

    # Print report
    report = generate_report(text_results, image_results)
    print(report)

    # Save results to JSON
    all_results = {"text": text_results, "image": image_results}
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
