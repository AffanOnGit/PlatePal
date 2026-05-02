import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import os

RAW_DIR = Path("data/raw")
FOOD101_DIR = RAW_DIR / "food-101" / "images"
CAFD_DIR = RAW_DIR / "CAFD"
H5_DEST = RAW_DIR / "food_128.h5"

def rebuild():
    print("--- Starting Dataset Rebuild (128x128) ---")
    
    images = []
    labels = []
    
    # 1. Process Food-101 (limit 100 per class for speed)
    classes = sorted([d.name for d in FOOD101_DIR.iterdir() if d.is_dir()])
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = FOOD101_DIR / cls_name
        count = 0
        for img_path in list(cls_dir.glob("*.jpg"))[:100]:
            try:
                img = Image.open(img_path).convert("RGB").resize((128, 128), Image.LANCZOS)
                images.append(np.array(img))
                labels.append(cls_idx)
                count += 1
            except: continue
        print(f"  [Food-101] {cls_name}: {count} images")

    # 2. Process CAFD
    cafd_classes = sorted([d.name for d in CAFD_DIR.iterdir() if d.is_dir()])
    offset = len(classes)
    for cls_idx, cls_name in enumerate(cafd_classes):
        cls_dir = CAFD_DIR / cls_name
        count = 0
        for img_path in list(cls_dir.glob("*"))[:100]:
            try:
                img = Image.open(img_path).convert("RGB").resize((128, 128), Image.LANCZOS)
                images.append(np.array(img))
                labels.append(cls_idx + offset)
                count += 1
            except: continue
        print(f"  [CAFD] {cls_name}: {count} images")

    print(f"Saving {len(images)} images to {H5_DEST}...")
    with h5py.File(H5_DEST, "w") as f:
        f.create_dataset("images", data=np.array(images), compression="gzip")
        f.create_dataset("labels", data=np.array(labels))
    print("Dataset build complete!")

if __name__ == "__main__":
    rebuild()
