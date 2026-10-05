"""
Merge all cleaned datasets, standardize labels, and split into train/val/test.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

from config import DATASETS, DATA_CLEANED, DATA_MERGED, DATA_FINAL, CLASS_NAMES, SPLIT_RATIOS, ROOT


def collect_image_label_pairs(data_dir):
    """Walk through cleaned data and collect (image_path, label_path) pairs."""
    pairs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                img_path = os.path.join(root, f)
                # Find corresponding label
                label_path = img_path.replace("images", "labels")
                label_path = os.path.splitext(label_path)[0] + ".txt"
                if not os.path.exists(label_path):
                    label_path = None
                pairs.append((img_path, label_path))
    return pairs


def standardize_label(label_path, is_camouflage_dataset=False):
    """
    Read a YOLO label file and standardize class IDs.
    - Camouflage dataset: keep original classes 0-3
    - Other datasets: all class 0 → keep as is (no camouflage)
    Returns list of [class_id, x, y, w, h]
    """
    if label_path is None or not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            # Clamp class to 0-3 range
            cls = max(0, min(3, cls))
            boxes.append([cls] + [float(p) for p in parts[1:5]])
    return boxes


def write_label(label_path, boxes):
    """Write YOLO format label file."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def split_data(pairs, ratios):
    """Split pairs into train/val/test."""
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def generate_data_yaml(output_dir, class_names):
    """Generate data.yaml for YOLO training."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    config = {
        "path": output_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": [class_names[i] for i in sorted(class_names.keys())],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def main():
    print("=" * 60)
    print("Data Preparation: Merge & Split")
    print("=" * 60)

    # Step 1: Collect all data
    all_pairs = collect_image_label_pairs(DATA_CLEANED)
    if not all_pairs:
        # Fallback: collect from original datasets
        print("No cleaned data found, using original datasets...")
        all_pairs = []
        for name, src_dir in DATASETS.items():
            if os.path.exists(src_dir):
                pairs = collect_image_label_pairs(src_dir)
                # Tag camouflage dataset pairs
                for img, lbl in pairs:
                    all_pairs.append((img, lbl, name == "camouflage"))
        if not all_pairs:
            print("ERROR: No data found!")
            return

    print(f"\nTotal image-label pairs collected: {len(all_pairs)}")

    # Step 2: Count class distribution
    class_counts = defaultdict(int)
    is_camo_pairs = []

    standardized_pairs = []
    for item in all_pairs:
        if len(item) == 3:
            img, lbl, is_camo = item
        else:
            img, lbl = item
            is_camo = "伪装坦克" in img or "camouflage" in img.lower()

        boxes = standardize_label(lbl, is_camo)
        for box in boxes:
            class_counts[box[0]] += 1
        standardized_pairs.append((img, boxes, os.path.splitext(os.path.basename(img))[0]))

    print("\nClass distribution after standardization:")
    for cls_id in sorted(class_counts.keys()):
        print(f"  Class {cls_id} ({CLASS_NAMES.get(cls_id, 'unknown')}): {class_counts[cls_id]}")

    # Step 3: Split
    splits = split_data(standardized_pairs, SPLIT_RATIOS)
    print(f"\nSplit: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Step 4: Write to final directory
    for split_name, pairs in splits.items():
        img_dir = os.path.join(DATA_FINAL, split_name, "images")
        lbl_dir = os.path.join(DATA_FINAL, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path, boxes, basename in pairs:
            # Copy image
            ext = os.path.splitext(img_path)[1]
            dst_img = os.path.join(img_dir, f"{basename}{ext}")
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            # Write standardized label
            dst_lbl = os.path.join(lbl_dir, f"{basename}.txt")
            write_label(dst_lbl, boxes)

        print(f"  {split_name}: {len(pairs)} images written")

    # Step 5: Generate data.yaml
    yaml_path = generate_data_yaml(DATA_FINAL, CLASS_NAMES)
    print(f"\ndata.yaml saved to: {yaml_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Final dataset ready at: {DATA_FINAL}")


if __name__ == "__main__":
    random.seed(42)
    main()
