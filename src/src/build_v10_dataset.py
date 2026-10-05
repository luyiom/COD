"""
Build final_v10 dataset from 6 source datasets.
Strategy:
- v6 kept as baseline (includes all 6 source datasets + empty labels)
- Mixed dataset 1: NO augmentation (already has diverse environments + flips)
- Other 5 datasets: fog + rain augmentation only
- Preserve empty labels from v6
"""

import os
import sys
import json
import shutil
import random
import math
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_BASE = os.path.join(ROOT, "data", "final_v6")  # baseline: already has all data from 6 sources
OUTPUT = os.path.join(ROOT, "data", "final_v10")
RANDOM_SEED = 42

CLASS_NAMES = {0: "military_vehicle"}
NC = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Augmentation Config
# ============================================================
# Mixed dataset 1 stays untouched (no augmentation)
# Other 5 datasets get fog + rain
AUG_PROB_FOG = 0.30   # 30% get fog
AUG_PROB_RAIN = 0.30  # 30% get rain
# Each positive image from non-mixed sources gets ~60% chance of one or both augmentations

# ============================================================
# Source Identification
# ============================================================
# In final_v6, images were renamed to Roboflow .rf. format,
# EXCEPT MHCD2022 which kept 'mhcd_' prefix.
# We need to identify which .rf. files came from which source.
#
# Strategy: use the v6 label files as ground truth for "which images
# have empty labels", and for augmentation grouping we use a different approach:
#
# Since v6 flattened all sources into .rf. filenames, we can't perfectly
# separate by source. INSTEAD, we apply augmentation rules based on
# heuristics and accept minor overlap:
#
# Heuristic for "mixed dataset 1":
#   - Files where the original image contained empty labels (negative samples)
#     came from mixed dataset 1 (only mixed had 417 empty labels)
#   - But mixed also has 6698 positive images mixed in...
#
# Better approach: Rebuild from v6 as-is, but use the filename pattern
# to separate sources if possible. The v6 build script would have recorded
# which source each file came from.
#
# SIMPLEST APPROACH: Don't try to separate. Apply augmentation to a random
# subset of ALL positive images in v6 train. Mixed dataset 1 is ~55% of data,
# so augmenting all positive images means mixed1 gets augmented too.
#
# USER'S REQUEST: mixed1 (55% of data) should NOT get augmentation.
# Other 5 sources (45% of data) SHOULD get fog + rain.
#
# Since we can't separate the .rf. files, we need to re-build from
# the original 6 source datasets.


def read_image(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    return False


# ============================================================
# OpenCV Augmentation Functions
# ============================================================

def add_fog(img, intensity=None):
    h, w = img.shape[:2]
    if intensity is None:
        intensity = random.uniform(0.2, 0.4)
    fog_overlay = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    return cv2.addWeighted(img, 1 - intensity, fog_overlay, intensity, 0)


def add_rain(img, intensity=0.25, drop_count=None):
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
    result = cv2.convertScaleAbs(result, alpha=0.85, beta=-15)
    return result


# ============================================================
# Source Dataset Collectors
# ============================================================

def collect_yolo_dataset(data_dir):
    """Collect pairs from a YOLO-format directory (images + labels)."""
    pairs = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            img_path = os.path.join(root, f)
            base = os.path.splitext(f)[0]
            # Try to find label
            lbl_path = None
            # Same dir
            candidate = os.path.join(root, base + ".txt")
            if os.path.exists(candidate):
                lbl_path = candidate
            else:
                # labels dir
                lbl_dir = root.replace("images", "labels")
                candidate = os.path.join(lbl_dir, base + ".txt")
                if os.path.exists(candidate):
                    lbl_path = candidate

            boxes = []
            has_boxes = False
            if lbl_path and os.path.exists(lbl_path):
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
            elif lbl_path is None:
                # No label file at all — treat as empty
                pass

            pairs.append({
                "img_path": img_path,
                "boxes": boxes,
                "has_boxes": has_boxes,
                "base_name": base,
            })

    return pairs


def collect_mhcd_dataset(data_dir):
    """Collect pairs from MHCD2022 (JSON annotations)."""
    pairs = []
    for f in os.listdir(data_dir):
        if not f.endswith(".jpg"):
            continue
        base = os.path.splitext(f)[0]
        json_path = os.path.join(data_dir, base + ".json")
        if not os.path.exists(json_path):
            continue

        img_path = os.path.join(data_dir, f)
        img = read_image(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes = []
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label not in ("tank", "military vehicle"):
                continue
            points = shape.get("points", [])
            if len(points) != 2:
                continue
            (x1, y1), (x2, y2) = points
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            boxes.append((0, [xc, yc, bw, bh]))

        pairs.append({
            "img_path": img_path,
            "boxes": boxes,
            "has_boxes": len(boxes) > 0,
            "base_name": f"mhcd_{base}",
        })

    return pairs


# ============================================================
# Main Build Logic
# ============================================================

def main():
    print("=" * 60)
    print("Building final_v10")
    print("  - mixed1: NO augmentation")
    print("  - tank, tank2, camo, armored, MHCD: fog + rain")
    print("=" * 60)

    os.makedirs(OUTPUT, exist_ok=True)

    # Collect from 6 source datasets
    sources = {}
    source_pairs = {}

    print("\n[1/5] Collecting source datasets...")

    # Mixed dataset 1
    md = os.path.join(ROOT, "datasets", "raw", "混合数据集1")
    p = collect_yolo_dataset(md)
    source_pairs["mixed1"] = p
    print(f"  {md}: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # Tank
    md = os.path.join(ROOT, "datasets", "raw", "坦克数据集")
    p = collect_yolo_dataset(md)
    source_pairs["tank"] = p
    print(f"  {md}: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # Tank2
    md = os.path.join(ROOT, "datasets", "raw", "坦克数据集2")
    p = collect_yolo_dataset(md)
    source_pairs["tank2"] = p
    print(f"  {md}: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # Camouflage
    md = os.path.join(ROOT, "datasets", "raw", "伪装坦克数据集", "tank_dataset")
    p = collect_yolo_dataset(md)
    source_pairs["camouflage"] = p
    print(f"  camouflage: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # Armored
    md = os.path.join(ROOT, "datasets", "raw", "装甲车数据集")
    p = collect_yolo_dataset(md)
    source_pairs["armored"] = p
    print(f"  {md}: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # MHCD2022
    md = os.path.join(ROOT, "datasets", "raw", "挑过Military-Camouflage-MHCD2022", "数据集")
    p = collect_mhcd_dataset(md)
    source_pairs["mhcd"] = p
    print(f"  MHCD2022: {len(p)} images ({sum(1 for x in p if x['has_boxes'])} positive, {sum(1 for x in p if not x['has_boxes'])} negative)")

    # Summary
    total_all = sum(len(v) for v in source_pairs.values())
    total_pos = sum(sum(1 for x in v if x['has_boxes']) for v in source_pairs.values())
    total_neg = sum(sum(1 for x in v if not x['has_boxes']) for v in source_pairs.values())
    print(f"\n  Total: {total_all} ({total_pos} positive, {total_neg} negative)")

    # Step 2: Split each source into train/val/test (80/10/10)
    print("\n[2/5] Splitting 80/10/10 per source...")
    split_data = {"train": [], "val": [], "test": []}

    for src_name, pairs in source_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)

        for p in pairs[:n_train]:
            p["source"] = src_name
            split_data["train"].append(p)
        for p in pairs[n_train:n_train + n_val]:
            p["source"] = src_name
            split_data["val"].append(p)
        for p in pairs[n_train + n_val:]:
            p["source"] = src_name
            split_data["test"].append(p)

    random.shuffle(split_data["train"])
    random.shuffle(split_data["val"])
    random.shuffle(split_data["test"])

    for split_name, pairs in split_data.items():
        pos = sum(1 for x in pairs if x['has_boxes'])
        neg = sum(1 for x in pairs if not x['has_boxes'])
        print(f"  {split_name}: {len(pairs)} images ({pos} positive, {neg} negative)")

    # Step 3: Write dataset with selective augmentation
    print("\n[3/5] Writing with selective augmentation...")

    aug_stats = {"fog": 0, "rain": 0, "total_original": 0}

    for split_name, pairs in split_data.items():
        img_dir = os.path.join(OUTPUT, split_name, "images")
        lbl_dir = os.path.join(OUTPUT, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for p in pairs:
            src_name = p["source"]
            base = p["base_name"]

            # Ensure unique names
            dst_name = f"{src_name}_{base}.jpg"
            dst_img = os.path.join(img_dir, dst_name)
            dst_lbl = os.path.join(lbl_dir, f"{src_name}_{base}.txt")

            # Copy original image
            shutil.copy2(p["img_path"], dst_img)
            aug_stats["total_original"] += 1

            # Write label
            with open(dst_lbl, "w") as lf:
                for cls, coords in p["boxes"]:
                    lf.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

            # Apply augmentation (train only, positive only, non-mixed1 only)
            if split_name == "train" and p["has_boxes"] and src_name != "mixed1":
                img = read_image(dst_img)
                if img is None:
                    continue

                # Fog
                if random.random() < AUG_PROB_FOG:
                    aug_img = add_fog(img)
                    aug_name = f"{src_name}_{base}_aug_fog.jpg"
                    imwrite(os.path.join(img_dir, aug_name), aug_img)
                    aug_lbl_path = os.path.join(lbl_dir, f"{src_name}_{base}_aug_fog.txt")
                    with open(aug_lbl_path, "w") as lf:
                        for cls, coords in p["boxes"]:
                            lf.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    aug_stats["fog"] += 1

                # Rain
                if random.random() < AUG_PROB_RAIN:
                    aug_img = add_rain(img)
                    aug_name = f"{src_name}_{base}_aug_rain.jpg"
                    imwrite(os.path.join(img_dir, aug_name), aug_img)
                    aug_lbl_path = os.path.join(lbl_dir, f"{src_name}_{base}_aug_rain.txt")
                    with open(aug_lbl_path, "w") as lf:
                        for cls, coords in p["boxes"]:
                            lf.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    aug_stats["rain"] += 1

    print(f"  Original images: {aug_stats['total_original']}")
    print(f"  Fog augmented: {aug_stats['fog']}")
    print(f"  Rain augmented: {aug_stats['rain']}")
    print(f"  Total train images (with aug): {aug_stats['total_original'] + aug_stats['fog'] + aug_stats['rain']}")

    # Step 4: Generate data.yaml
    yaml_path = os.path.join(OUTPUT, "data.yaml")
    config = {
        "path": OUTPUT,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NC,
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Step 5: Print final stats
    print(f"\n[4/5] data.yaml: {yaml_path}")
    print(f"\n[5/5] Final stats:")

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT, split, "images")
        lbl_dir = os.path.join(OUTPUT, split, "labels")
        n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        n_pos = 0
        n_neg = 0
        n_boxes = 0
        cls_dist = defaultdict(int)
        for lbl in os.listdir(lbl_dir):
            if not lbl.endswith('.txt'):
                continue
            with open(os.path.join(lbl_dir, lbl)) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                n_pos += 1
            else:
                n_neg += 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_dist[int(float(parts[0]))] += 1
                    n_boxes += 1

        # Split augmented from original
        aug_count = len([f for f in os.listdir(img_dir) if '_aug_' in f.lower()])

        print(f"  {split}: {n_imgs} images ({aug_count} augmented, {n_pos} positive, {n_neg} negative), {n_boxes} boxes")

    print(f"\n{'='*60}")
    print(f"Dataset ready at: {OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
