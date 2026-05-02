"""
PlatePal - Dataset Preprocessing Utilities

Handles loading, cleaning, and formatting of:
  1. RecipeNLG (text corpus)
  2. Food-101 HDF5 (image dataset)
  3. CAFD (regional food images)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# 1. RecipeNLG Text Preprocessing
# ─────────────────────────────────────────────────────────────────────

# Phrases that indicate a "trivial" recipe to be filtered out
TRIVIAL_PHRASES = [
    "mix all", "combine everything", "stir together",
    "just add", "microwave", "open can",
]
MIN_INSTRUCTION_LENGTH = 50  # characters
MIN_INGREDIENT_COUNT = 3


def load_recipenlg(csv_path: str, subset_size: int = 20000) -> pd.DataFrame:
    """
    Load and filter the RecipeNLG CSV into high-quality structured recipes.

    Args:
        csv_path: Path to the RecipeNLG CSV file.
        subset_size: Maximum number of recipes to keep after filtering.

    Returns:
        Cleaned DataFrame with columns: title, ingredients, directions, NER.
    """
    print(f"[RecipeNLG] Loading from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    required_cols = {"title", "ingredients", "directions", "NER"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

    original_count = len(df)
    print(f"[RecipeNLG] Raw dataset size: {original_count:,}")

    # ── Filter 1: Remove rows with missing data ──
    df = df.dropna(subset=["title", "ingredients", "directions"])

    # ── Filter 2: Remove trivially short instructions ──
    df = df[df["directions"].str.len() >= MIN_INSTRUCTION_LENGTH]

    # ── Filter 3: Remove recipes with trivial phrases ──
    trivial_pattern = "|".join(TRIVIAL_PHRASES)
    df = df[~df["directions"].str.lower().str.contains(trivial_pattern, na=False)]

    # ── Filter 4: Require minimum ingredient count ──
    def count_ingredients(ingredients_str):
        try:
            items = json.loads(ingredients_str.replace("'", '"'))
            return len(items)
        except (json.JSONDecodeError, AttributeError):
            return 0

    df["_ing_count"] = df["ingredients"].apply(count_ingredients)
    df = df[df["_ing_count"] >= MIN_INGREDIENT_COUNT]
    df = df.drop(columns=["_ing_count"])

    # ── Subsample ──
    if len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=42)

    df = df.reset_index(drop=True)
    print(f"[RecipeNLG] After filtering: {len(df):,} recipes (from {original_count:,})")
    return df


def format_recipe_for_training(row: pd.Series) -> str:
    """
    Format a single recipe row into the special-token delimited string
    expected by the GPT-2 fine-tuning loop.

    Format:
        <RECIPE_START><TITLE>title<INPUT_START>ingredient1, ingredient2, ...
        <INSTR_START>Step 1. ... Step 2. ...<RECIPE_END>
    """
    title = str(row["title"]).strip()

    # Parse ingredient list
    try:
        ingredients = json.loads(str(row["ingredients"]).replace("'", '"'))
        if isinstance(ingredients, list):
            ingredients_str = ", ".join(ingredients)
        else:
            ingredients_str = str(ingredients)
    except (json.JSONDecodeError, AttributeError):
        ingredients_str = str(row["ingredients"])

    # Parse directions
    try:
        directions = json.loads(str(row["directions"]).replace("'", '"'))
        if isinstance(directions, list):
            directions_str = " ".join(
                f"Step {i+1}. {step.strip()}" for i, step in enumerate(directions)
            )
        else:
            directions_str = str(directions)
    except (json.JSONDecodeError, AttributeError):
        directions_str = str(row["directions"])

    return (
        f"<RECIPE_START>"
        f"<TITLE>{title}"
        f"<INPUT_START>{ingredients_str}"
        f"<INSTR_START>{directions_str}"
        f"<RECIPE_END>"
    )


def prepare_text_dataset(csv_path: str, output_path: str, subset_size: int = 20000):
    """
    Full pipeline: load CSV → filter → format → write training corpus.

    Writes one formatted recipe per line in a plain .txt file.
    Each line is a self-contained record delimited by <RECIPE_END>.
    """
    df = load_recipenlg(csv_path, subset_size)
    formatted = df.apply(format_recipe_for_training, axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for recipe_str in formatted:
            # Collapse internal newlines so each recipe is exactly one line
            clean = recipe_str.replace("\n", " ").replace("\r", " ").strip()
            f.write(clean + "\n")
    print(f"[RecipeNLG] Training corpus written to {output_path}  ({len(formatted):,} recipes)")
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. Food-101 HDF5 Image Preprocessing
# ─────────────────────────────────────────────────────────────────────

def load_food101_h5(h5_path: str):
    """
    Load the pre-processed Food-101 HDF5 dataset.

    Returns:
        images: np.ndarray of shape (N, 64, 64, 3) in [0, 255]
        labels: np.ndarray of shape (N,) with integer class labels
    """
    import h5py

    print(f"[Food-101] Loading HDF5 from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        # Common key patterns in this dataset
        keys = list(f.keys())
        print(f"[Food-101] Available keys: {keys}")

        if "images" in keys and "labels" in keys:
            images = np.array(f["images"])
            labels = np.array(f["labels"])
        elif "X" in keys and "y" in keys:
            images = np.array(f["X"])
            labels = np.array(f["y"])
        else:
            # Fall back to first two datasets
            images = np.array(f[keys[0]])
            labels = np.array(f[keys[1]]) if len(keys) > 1 else np.zeros(len(images))

    print(f"[Food-101] Loaded {len(images):,} images, shape: {images.shape}")
    return images, labels


def normalize_images_for_gan(images: np.ndarray) -> np.ndarray:
    """Normalize uint8 images [0,255] → float32 [-1,1] for Tanh-based GAN."""
    return (images.astype(np.float32) / 127.5) - 1.0


# ─────────────────────────────────────────────────────────────────────
# 3. CAFD (Central Asian Food Dataset) Preprocessing
# ─────────────────────────────────────────────────────────────────────

def load_cafd_images(root_dir: str, target_size: tuple = (128, 128)):
    """
    Load images from a folder-per-class directory structure.

    Args:
        root_dir: Path to the dataset root (contains subdirs per class).
        target_size: Resize all images to this (H, W).

    Returns:
        images: np.ndarray (N, H, W, 3)
        labels: np.ndarray of integer class indices (DataLoader-safe)
    """
    from PIL import Image as PILImage

    images, labels = [], []
    class_dirs = sorted([
        d for d in Path(root_dir).iterdir() if d.is_dir()
    ])
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    print(f"[CAFD] Found {len(class_dirs)} classes in {root_dir}")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx = class_to_idx[cls_name]
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = PILImage.open(img_path).convert("RGB").resize(target_size)
                images.append(np.array(img))
                labels.append(cls_idx)
            except Exception:
                continue

    images = np.array(images)
    labels = np.array(labels, dtype=np.int64)
    print(f"[CAFD] Loaded {len(images):,} images across {len(class_dirs)} classes")
    print(f"[CAFD] Class mapping: { {v: k for k, v in list(class_to_idx.items())[:5]} }...")
    return images, labels


# ─────────────────────────────────────────────────────────────────────
# 4. PyTorch Dataset Wrappers
# ─────────────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset


class FoodImageDataset(Dataset):
    """
    Generic dataset for food images from any source (Food-101, CAFD).
    Images are expected as np.ndarray (N, H, W, 3) in [0, 255].
    """

    def __init__(self, images: np.ndarray, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3) uint8

        if self.transform:
            from PIL import Image as PILImage
            img = PILImage.fromarray(img)
            img = self.transform(img)
        else:
            # Default: normalize to [-1, 1] and convert to CHW tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

        label = self.labels[idx] if self.labels is not None else -1
        return img, label


class RecipeTextDataset(Dataset):
    """
    Dataset for loading pre-formatted recipe strings for GPT-2 fine-tuning.

    Expects a .txt file with one formatted recipe per line.
    Each line contains special tokens: <RECIPE_START>...<RECIPE_END>
    """

    def __init__(self, corpus_path: str, tokenizer, max_length: int = 512, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"[RecipeTextDataset] Loading corpus from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.recipes = [line.strip() for line in f if line.strip()]

        if max_samples and len(self.recipes) > max_samples:
            self.recipes = self.recipes[:max_samples]
            print(f"[RecipeTextDataset] Capped to {max_samples:,} samples")

        print(f"[RecipeTextDataset] Loaded {len(self.recipes):,} recipes")

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        encoding = self.tokenizer(
            recipe,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # autoregressive target
        }


# ─────────────────────────────────────────────────────────────────────
# CLI Entry Point — quick dataset inspection
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlatePal Dataset Preprocessor")
    parser.add_argument("--recipe-csv", type=str, help="Path to RecipeNLG CSV")
    parser.add_argument("--food101-h5", type=str, help="Path to Food-101 HDF5")
    parser.add_argument("--cafd-dir", type=str, help="Path to CAFD root directory")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    if args.recipe_csv:
        prepare_text_dataset(
            args.recipe_csv,
            os.path.join(args.output_dir, "recipes_train.csv"),
        )

    if args.food101_h5:
        imgs, labels = load_food101_h5(args.food101_h5)
        print(f"  Image range: [{imgs.min()}, {imgs.max()}]")

    if args.cafd_dir:
        imgs, labels = load_cafd_images(args.cafd_dir)
        print(f"  Image range: [{imgs.min()}, {imgs.max()}]")
