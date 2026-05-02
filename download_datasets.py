"""
PlatePal — Dataset Download Helper
====================================

Automates downloading the required datasets:
  1. RecipeNLG (text) — from Kaggle or a mirror
  2. Food-101 HDF5 (images)
  3. CAFD (Central Asian Food Dataset)

Usage:
    python download_datasets.py --all
    python download_datasets.py --recipe
    python download_datasets.py --food101
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with a progress indicator."""
    if os.path.exists(dest):
        print(f"  [Skip] {desc or dest} already exists.")
        return

    print(f"  [Download] {desc or url}")
    print(f"  → {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r    {pct:.1f}% ({mb:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook)
        print(f"\n  ✓ Downloaded: {dest}")
    except Exception as e:
        print(f"\n  ✗ Failed to download: {e}")
        print(f"    Please download manually from: {url}")
        print(f"    and place it at: {dest}")


def setup_recipenlg():
    """
    RecipeNLG dataset setup instructions.
    The dataset must be downloaded manually from Kaggle due to auth requirements.
    """
    dest = RAW_DIR / "RecipeNLG_dataset.csv"
    if dest.exists():
        print("[RecipeNLG] ✓ Dataset found.")
        return

    print("\n" + "=" * 60)
    print(" RecipeNLG Dataset — Manual Download Required")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/recipenlg")
    print("   OR:    https://recipenlg.cs.put.poznan.pl/dataset")
    print()
    print(f"2. Download the CSV and place it at:")
    print(f"   {dest.resolve()}")
    print()
    print("3. Re-run this script to verify.")
    print()

    # Alternative: If kaggle CLI is available
    print("   [Alternative] Using Kaggle CLI:")
    print("   kaggle datasets download -d paultimothymooney/recipenlg -p data/raw/ --unzip")
    print()


def setup_food101():
    """
    Download the Food-101 HDF5 subset.
    """
    dest = RAW_DIR / "food_c101_n10099_r64x64x3.h5"
    if dest.exists():
        print("[Food-101] ✓ HDF5 dataset found.")
        return

    print("\n" + "=" * 60)
    print(" Food-101 HDF5 Subset — Download")
    print("=" * 60)
    print()
    print("The Food-101 HDF5 subset needs to be placed at:")
    print(f"  {dest.resolve()}")
    print()
    print("Sources:")
    print("  1. Kaggle: https://www.kaggle.com/datasets/dansbecker/food-101")
    print("  2. Direct: https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz")
    print()
    print("After downloading the full Food-101:")
    print("  - Convert to HDF5 using the provided notebook, OR")
    print("  - Use the folder-based loader in data_preprocessing.py")
    print()

    # Try downloading the full Food-101 tar.gz (2.5 GB)
    print("[Food-101] Attempting full Food-101 download (this is ~2.5 GB)...")
    tar_dest = RAW_DIR / "food-101.tar.gz"
    download_file(
        "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
        str(tar_dest),
        "Food-101 archive"
    )

    if tar_dest.exists():
        print("[Food-101] Extracting archive...")
        try:
            with tarfile.open(tar_dest, "r:gz") as tar:
                tar.extractall(path=RAW_DIR)
            print("[Food-101] ✓ Extracted to data/raw/food-101/")
        except Exception as e:
            print(f"[Food-101] ✗ Extraction failed: {e}")


def setup_cafd():
    """
    Central Asian Food Dataset (CAFD) setup.
    """
    dest = RAW_DIR / "CAFD"
    if dest.exists() and dest.is_dir():
        print("[CAFD] ✓ Dataset directory found.")
        return

    print("\n" + "=" * 60)
    print(" CAFD (Central Asian Food Dataset) — Download")
    print("=" * 60)
    print()
    print("Download the dataset from:")
    print("  https://www.kaggle.com/datasets/...")
    print("  (Search for 'Central Asian Food Dataset')")
    print()
    print(f"Extract into: {dest.resolve()}")
    print("Expected structure: CAFD/class_name/image.jpg")
    print()


def convert_food101_to_h5():
    """
    Convert folder-based Food-101 images to HDF5 for fast loading.
    """
    import numpy as np

    food101_dir = RAW_DIR / "food-101" / "images"
    h5_dest = RAW_DIR / "food_c101_n10099_r64x64x3.h5"

    if not food101_dir.exists():
        print("[Convert] Food-101 image directory not found. Skipping.")
        return

    if h5_dest.exists():
        print("[Convert] HDF5 already exists. Skipping.")
        return

    print("[Convert] Converting Food-101 images to HDF5 (64x64)...")

    try:
        import h5py
        from PIL import Image

        all_images = []
        all_labels = []
        classes = sorted([d.name for d in food101_dir.iterdir() if d.is_dir()])

        for cls_idx, cls_name in enumerate(classes):
            cls_dir = food101_dir / cls_name
            count = 0
            for img_path in sorted(cls_dir.glob("*.jpg"))[:100]:  # limit per class
                try:
                    img = Image.open(img_path).convert("RGB").resize((64, 64))
                    all_images.append(np.array(img))
                    all_labels.append(cls_idx)
                    count += 1
                except Exception:
                    continue
            print(f"  {cls_name}: {count} images")

        images_arr = np.array(all_images)
        labels_arr = np.array(all_labels)

        with h5py.File(h5_dest, "w") as f:
            f.create_dataset("images", data=images_arr, compression="gzip")
            f.create_dataset("labels", data=labels_arr)

        print(f"[Convert] ✓ Saved {len(images_arr)} images to {h5_dest}")

    except ImportError as e:
        print(f"[Convert] Missing dependency: {e}")
        print("  Install with: pip install h5py Pillow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PlatePal datasets")
    parser.add_argument("--all",     action="store_true", help="Setup all datasets")
    parser.add_argument("--recipe",  action="store_true", help="Setup RecipeNLG")
    parser.add_argument("--food101", action="store_true", help="Setup Food-101")
    parser.add_argument("--cafd",    action="store_true", help="Setup CAFD")
    parser.add_argument("--convert", action="store_true", help="Convert Food-101 folder to HDF5")
    args = parser.parse_args()

    ensure_dirs()

    if args.all or args.recipe:
        setup_recipenlg()
    if args.all or args.food101:
        setup_food101()
    if args.all or args.cafd:
        setup_cafd()
    if args.convert:
        convert_food101_to_h5()

    if not any(vars(args).values()):
        print("Usage: python download_datasets.py --all")
        print("       python download_datasets.py --recipe --food101 --cafd")
        print("       python download_datasets.py --convert")
