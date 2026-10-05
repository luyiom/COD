"""
Dataset builder for final_v9.
Based on final_v6 data, rebuilds train/val/test splits across all 6 data sources,
preserves negative samples (empty labels), and applies OpenCV environmental augmentation.
"""
import os
import sys
import json
import shutil
import random
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# === Config ===
SRC_DATA = os.path.join(ROOT, "data", "final_v6")
OUTPUT = os.path.join(ROOT, "data", "final_v9")
RANDOM_SEED = 42

# Augmentation config: what fraction of train positive samples get each augmentation
AUG_CONFIG = {
    "dark": 0.20,   # 20% of train positives get dark augmentation
    "fog": 0.20,    # 20% get fog
    "rain": 0.20,   # 20% get rain
}
# Total augmented: ~60% of train positives get one random enhancement
# The original images are always kept, augmented versions are added on top

CLASS_NAMES = {0: "military_vehicle"}
NC = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# OpenCV Environmental Augmentation Functions
# ============================================================

def add_dark(img, darkness=0.45):
    """Simulate low-light / nighttime conditions."""
    h, w = img.shape[:2]
    # Lower brightness
    result = cv2.convertScaleAbs(img, alpha=darkness, beta=-25)
    # Add slight blue-purple cast (moonlight)
    tint = np.full((h, w, 3), (45, 25, 70), dtype=np.uint8)
    result = cv2.addWeighted(result, 0.88, tint, 0.12, 0)
    # Add noise (low-light sensor noise)
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result


def add_fog(img, intensity=None):
    """Simulate fog / haze conditions."""
    h, w = img.shape[:2]
    if intensity is None:
        intensity = random.uniform(0.2, 0.4)
    fog_overlay = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    result = cv2.addWeighted(img, 1 - intensity, fog_overlay, intensity, 0)
    return result


def add_rain(img, intensity=0.25, drop_count=None):
    """Simulate rainy conditions with streak lines."""
    h, w = img.shape[:2]
    if drop_count is None:
        drop_count = random.randint(400, 800)

    rain_layer = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(drop_count):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(12, 30)
        angle = random.randint(75, 105)
        dx = int(length * math.cos(math.radians(angle)))
        dy = int(length * math.sin(math.radians(angle)))
        brightness = random.randint(160, 220)
        thickness = random.randint(1, 2)
        cv2.line(rain_layer, (x, y), (x + dx, y + dy),
                 (brightness, brightness, brightness), thickness)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    result = cv2.addWeighted(img, 1.0, rain_layer, intensity, 0)
    # Darken slightly (overcast)
    result = cv2.convertScaleAbs(result, alpha=0.85, beta=-15)
    return result


AUG_FUNCTIONS = {
    "dark": add_dark,
    "fog": add_fog,
    "rain": add_rain,
}


# ============================================================
# Dataset Builder
# ============================================================

def read_image(path):
    """Read image with Unicode path support."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    """Write image with Unicode path support."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    return False


def collect_pairs(base_dir):
    """
    Walk through base_dir and collect (img_path, label_path, has_boxes) tuples.
    Also categorise by source dataset via filename pattern.
    """
    pairs = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            img_path = os.path.join(img_dir, f)
            base_name = os.path.splitext(f)[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            boxes = []
            has_boxes = False
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls = int(float(parts[0]))
                            coords = [float(p) for p in parts[1:5]]
                            boxes.append((cls, coords))
                            has_boxes = True

            # Determine source dataset from filename pattern
            f_lower = f.lower()
            if ".rf." in f_lower:
                source = "camouflage_roboflow"
            elif "mhcd_" in f_lower:
                source = "mhcd2022"
            elif any(kw in f_lower for kw in ["0000"]) and ".rf." in f_lower:
                source = "mhcd2022"
            else:
                # For non-RF files, use folder name heuristic
                # These came from mixed/tank/tank2/armored datasets
                source = "other_military"

            pairs.append({
                "img_path": img_path,
                "lbl_path": lbl_path,
                "base_name": base_name,
                "boxes": boxes,
                "has_boxes": has_boxes,
                "source": source,
            })
    return pairs


def split_pairs(pairs, train_ratio=0.80, val_ratio=0.10):
    """
    Split pairs into train/val/test, stratified by source and has_boxes.
    Ensures each source is represented in val and test.
    """
    random.shuffle(pairs)

    # Group by source and has_boxes
    groups = defaultdict(list)
    for p in pairs:
        key = (p["source"], p["has_boxes"])
        groups[key].append(p)

    train, val, test = [], [], []

    for key, group in groups.items():
        n = len(group)
        random.shuffle(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_dataset(split_pairs, split_name, output_base, apply_aug=False):
    """Write images and labels for a split."""
    img_dir = os.path.join(output_base, split_name, "images")
    lbl_dir = os.path.join(output_base, split_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    stats = {"images": 0, "boxes": 0, "empty": 0, "augmented": 0}
    class_counts = defaultdict(int)

    for p in split_pairs:
        # Copy original image
        src_img = p["img_path"]
        base = p["base_name"]

        # Generate a unique name to avoid conflicts
        dst_name = f"{base}.jpg"
        dst_img = os.path.join(img_dir, dst_name)

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        stats["images"] += 1

        # Write label
        dst_lbl = os.path.join(lbl_dir, f"{base}.txt")
        if not p["has_boxes"]:
            # Empty label file
            with open(dst_lbl, "w") as f:
                pass
            stats["empty"] += 1
        else:
            with open(dst_lbl, "w") as f:
                for cls, coords in p["boxes"]:
                    f.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    class_counts[cls] += 1
                    stats["boxes"] += 1

            # Apply augmentation (train only)
            if apply_aug and p["has_boxes"]:
                # Each positive train image gets one random augmentation (if selected)
                # We apply each augmentation independently with its configured probability
                for aug_name, prob in AUG_CONFIG.items():
                    if random.random() < prob:
                        img = read_image(dst_img)
                        if img is None:
                            continue

                        aug_fn = AUG_FUNCTIONS[aug_name]
                        aug_img = aug_fn(img)

                        aug_name_full = f"{base}_aug_{aug_name}.jpg"
                        aug_img_path = os.path.join(img_dir, aug_name_full)
                        imwrite(aug_img_path, aug_img)

                        # Same labels for augmented image
                        aug_lbl_path = os.path.join(lbl_dir, f"{base}_aug_{aug_name}.txt")
                        with open(aug_lbl_path, "w") as f:
                            for cls, coords in p["boxes"]:
                                f.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

                        stats["augmented"] += 1
                        # Note: augmented images are added on top, original remains

    return dict(class_counts), stats


def generate_yaml(output_dir):
    """Generate data.yaml."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    config = {
        "path": output_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NC,
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def main():
    print("=" * 60)
    print("Building final_v9 dataset from final_v6 + OpenCV augmentation")
    print("=" * 60)

    # Step 1: Collect all pairs from final_v6
    print("\n[1/5] Collecting data from final_v6...")
    all_pairs = collect_pairs(SRC_DATA)
    positive = [p for p in all_pairs if p["has_boxes"]]
    negative = [p for p in all_pairs if not p["has_boxes"]]
    print(f"  Total: {len(all_pairs)} images")
    print(f"  Positive (has boxes): {len(positive)}")
    print(f"  Negative (empty labels): {len(negative)}")

    # Source distribution
    src_dist = defaultdict(lambda: {"positive": 0, "negative": 0})
    for p in all_pairs:
        key = "positive" if p["has_boxes"] else "negative"
        src_dist[p["source"]][key] += 1
    print("  Source distribution:")
    for src, counts in sorted(src_dist.items()):
        print(f"    {src}: {counts['positive']} positive, {counts['negative']} negative")

    # Step 2: Split
    print("\n[2/5] Splitting into train/val/test (80/10/10)...")
    train, val, test = split_pairs(all_pairs, train_ratio=0.80, val_ratio=0.10)
    print(f"  train: {len(train)} ({len([p for p in train if not p['has_boxes']])} negative)")
    print(f"  val:   {len(val)} ({len([p for p in val if not p['has_boxes']])} negative)")
    print(f"  test:  {len(test)} ({len([p for p in test if not p['has_boxes']])} negative)")

    # Step 3: Write dataset
    print("\n[3/5] Writing train set (with augmentation)...")
    os.makedirs(OUTPUT, exist_ok=True)

    train_class_dist, train_stats = write_dataset(train, "train", OUTPUT, apply_aug=True)
    print(f"  Original images: {train_stats['images']}")
    print(f"  Augmented images added: {train_stats['augmented']}")
    print(f"  Total train images: {train_stats['images'] + train_stats['augmented']}")
    print(f"  Boxes: {train_stats['boxes']}")
    print(f"  Negative (empty): {train_stats['empty']}")
    print(f"  Class distribution: {dict(train_class_dist)}")

    print("\n[4/5] Writing val set...")
    val_class_dist, val_stats = write_dataset(val, "val", OUTPUT, apply_aug=False)
    print(f"  Images: {val_stats['images']}")
    print(f"  Boxes: {val_stats['boxes']}")
    print(f"  Negative: {val_stats['empty']}")

    print("\n[5/5] Writing test set...")
    test_class_dist, test_stats = write_dataset(test, "test", OUTPUT, apply_aug=False)
    print(f"  Images: {test_stats['images']}")
    print(f"  Boxes: {test_stats['boxes']}")
    print(f"  Negative: {test_stats['empty']}")

    # Step 4: Generate data.yaml
    yaml_path = generate_yaml(OUTPUT)
    print(f"\ndata.yaml written to: {yaml_path}")

    # Final summary
    total_imgs = (train_stats['images'] + train_stats['augmented'] +
                  val_stats['images'] + test_stats['images'])
    total_boxes = train_stats['boxes'] + val_stats['boxes'] + test_stats['boxes']
    total_negative = train_stats['empty'] + val_stats['empty'] + test_stats['empty']

    print(f"\n{'=' * 60}")
    print(f"Dataset final_v9 ready at: {OUTPUT}")
    print(f"  Train: {train_stats['images']} + {train_stats['augmented']} aug = {train_stats['images'] + train_stats['augmented']}")
    print(f"  Val:   {val_stats['images']}")
    print(f"  Test:  {test_stats['images']}")
    print(f"  Total: {total_imgs}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Total negative samples: {total_negative} ({total_negative/total_imgs*100:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
