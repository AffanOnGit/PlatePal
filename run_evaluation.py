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

Quick evaluation (50 samples):
    python run_evaluation.py --quick
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.text_model if os.path.isdir(args.text_model) else None
    gen = RecipeGenerator(checkpoint_dir=ckpt, device=device)

    prompts = TEST_PROMPTS[:args.num_samples] if args.quick else TEST_PROMPTS

    # Generate recipes
    generated = []
    coverages = []
    for prompt in prompts:
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
        gen.model, gen.tokenizer, generated, device=device
    )
    results["avg_ingredient_coverage"] = float(avg_coverage)

    print(f"\n  Perplexity: {results['perplexity']:.2f}")
    return results


def evaluate_images(args):
    """Run image model evaluation."""
    print("\n" + "=" * 60)
    print(" IMAGE MODEL EVALUATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    n_samples = args.num_samples if args.quick else 100
    fake_images = []

    print(f"  Generating {n_samples} sample images...")
    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, 100, device=device)
            # Use a real CLIP embedding from a food description
            prompt_idx = i % len(TEST_PROMPTS)
            embed = embedder.embed_recipe(
                title=TEST_PROMPTS[prompt_idx].split(",")[0].strip(),
                ingredients=TEST_PROMPTS[prompt_idx],
            )
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
    parser.add_argument("--quick",       action="store_true", help="Run with fewer samples for fast iteration")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples when --quick is set")
    parser.add_argument("--output-dir",  type=str, default="evaluation")
    args = parser.parse_args()

    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluations
    text_results = evaluate_text(args)
    image_results = evaluate_images(args)

    # Print report
    report = generate_report(text_results, image_results)
    print(report)

    # Save results to JSON with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"report_{timestamp}.json")
    all_results = {"text": text_results, "image": image_results, "timestamp": timestamp}
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Eval] Results saved to {output_path}")


if __name__ == "__main__":
    main()
