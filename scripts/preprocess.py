"""
PlatePal — Consolidated Data Preprocessing Script
====================================================

A single entry point to prepare all datasets for training:
  1. RecipeNLG CSV → filtered + formatted training corpus (.txt)
  2. Food-101 HDF5 validation
  3. CAFD directory validation
  4. Smoke test of RecipeTextDataset with tokenizer
  5. Summary report

Usage:
    python scripts/preprocess.py \
        --recipe-csv  data/raw/RecipeNLG_dataset.csv \
        --food101-h5  data/raw/food_c101_n10099_r64x64x3.h5 \
        --cafd-dir    data/raw/CAFD \
        --output-dir  data/processed \
        --subset-size 20000
"""

import os
import sys
import argparse
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="PlatePal — Data Preprocessing Pipeline")
    p.add_argument("--recipe-csv",  type=str, default="data/raw/RecipeNLG_dataset.csv",
                   help="Path to RecipeNLG CSV file")
    p.add_argument("--food101-h5",  type=str, default="data/raw/food_c101_n10099_r64x64x3.h5",
                   help="Path to Food-101 HDF5 file")
    p.add_argument("--cafd-dir",    type=str, default="data/raw/CAFD",
                   help="Path to CAFD root directory")
    p.add_argument("--output-dir",  type=str, default="data/processed")
    p.add_argument("--subset-size", type=int, default=20000,
                   help="Max recipes to keep after filtering")
    return p.parse_args()


def step_1_recipes(args):
    """Process RecipeNLG CSV into training corpus."""
    print("\n" + "=" * 60)
    print(" STEP 1: RecipeNLG Text Corpus")
    print("=" * 60)

    if not os.path.isfile(args.recipe_csv):
        print(f"  ✗ File not found: {args.recipe_csv}")
        print(f"    Download from Kaggle and place at: {os.path.abspath(args.recipe_csv)}")
        return False

    from app.utils.data_preprocessing import prepare_text_dataset

    output_path = os.path.join(args.output_dir, "recipes_train.txt")
    df = prepare_text_dataset(args.recipe_csv, output_path, args.subset_size)

    # Print corpus stats
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lengths = [len(line.strip()) for line in lines if line.strip()]
    print(f"\n  Corpus Statistics:")
    print(f"    Total recipes:    {len(lengths):,}")
    print(f"    Avg length:       {np.mean(lengths):.0f} chars")
    print(f"    Min length:       {min(lengths):,} chars")
    print(f"    Max length:       {max(lengths):,} chars")
    print(f"    Output file:      {output_path}")
    print(f"  ✓ Text corpus ready")
    return True


def step_2_food101(args):
    """Validate the Food-101 HDF5 dataset."""
    print("\n" + "=" * 60)
    print(" STEP 2: Food-101 HDF5 Validation")
    print("=" * 60)

    if not os.path.isfile(args.food101_h5):
        print(f"  ✗ File not found: {args.food101_h5}")
        print(f"    Option A: Download pre-built HDF5")
        print(f"    Option B: Run 'python download_datasets.py --food101 --convert'")
        return False

    from app.utils.data_preprocessing import load_food101_h5

    images, labels = load_food101_h5(args.food101_h5)
    unique_labels = np.unique(labels)

    print(f"\n  Dataset Statistics:")
    print(f"    Images:           {len(images):,}")
    print(f"    Shape:            {images.shape}")
    print(f"    Dtype:            {images.dtype}")
    print(f"    Pixel range:      [{images.min()}, {images.max()}]")
    print(f"    Unique classes:   {len(unique_labels)}")
    print(f"  ✓ Food-101 HDF5 validated")
    return True


def step_3_cafd(args):
    """Validate the CAFD directory."""
    print("\n" + "=" * 60)
    print(" STEP 3: CAFD Directory Validation")
    print("=" * 60)

    if not os.path.isdir(args.cafd_dir):
        print(f"  ✗ Directory not found: {args.cafd_dir}")
        print(f"    Download from Kaggle and extract into: {os.path.abspath(args.cafd_dir)}")
        return False

    from app.utils.data_preprocessing import load_cafd_images

    images, labels = load_cafd_images(args.cafd_dir)
    unique_labels = np.unique(labels)

    print(f"\n  Dataset Statistics:")
    print(f"    Images:           {len(images):,}")
    print(f"    Shape:            {images.shape}")
    print(f"    Pixel range:      [{images.min()}, {images.max()}]")
    print(f"    Unique classes:   {len(unique_labels)}")
    print(f"  ✓ CAFD validated")
    return True


def step_4_smoke_test(args):
    """Smoke test the RecipeTextDataset with the tokenizer."""
    print("\n" + "=" * 60)
    print(" STEP 4: RecipeTextDataset Smoke Test")
    print("=" * 60)

    corpus_path = os.path.join(args.output_dir, "recipes_train.txt")
    if not os.path.isfile(corpus_path):
        print(f"  ✗ Corpus not found: {corpus_path}")
        print(f"    Run Step 1 first.")
        return False

    from transformers import GPT2Tokenizer
    from app.utils.data_preprocessing import RecipeTextDataset

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({
        "bos_token": "<RECIPE_START>",
        "eos_token": "<RECIPE_END>",
        "pad_token": "<PAD>",
        "additional_special_tokens": ["<INPUT_START>", "<INSTR_START>", "<TITLE>"],
    })

    dataset = RecipeTextDataset(corpus_path, tokenizer, max_length=512, max_samples=10)

    print(f"\n  Checking 10 samples:")
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        input_ids = item["input_ids"]
        attn = item["attention_mask"]
        non_pad = attn.sum().item()
        print(f"    Sample {i+1}: shape={input_ids.shape}, non-pad tokens={non_pad}")

    print(f"  ✓ RecipeTextDataset smoke test passed")
    return True


def step_5_summary(results):
    """Print final summary."""
    print("\n" + "=" * 60)
    print(" PREPROCESSING SUMMARY")
    print("=" * 60)

    for step_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  [{status}] {step_name}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n  ✅ All steps passed! Ready for training.")
        print(f"\n  Next commands:")
        print(f"    python train_text_model.py --corpus data/processed/recipes_train.txt --output checkpoints/text_model --fp16")
        print(f"    python train_image_model.py --food101-h5 data/raw/food_c101_n10099_r64x64x3.h5 --cafd-dir data/raw/CAFD --output checkpoints/dcgan --fp16")
    else:
        print(f"\n  ⚠ Some steps failed. Fix the issues above before training.")

    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    results["RecipeNLG Text Corpus"] = step_1_recipes(args)
    results["Food-101 HDF5"] = step_2_food101(args)
    results["CAFD Images"] = step_3_cafd(args)

    if results["RecipeNLG Text Corpus"]:
        results["RecipeTextDataset Smoke Test"] = step_4_smoke_test(args)
    else:
        results["RecipeTextDataset Smoke Test"] = False

    step_5_summary(results)
