"""
Build final dataset for military vehicle detection.
Sources:
  - data/final_v6 (baseline, has 5 of 6 sources already merged)
  - datasets/labelme/labelme_output/伪装坦克数据集目标检测 (new camouflage data, 121 imgs)
  - datasets/labelme/labelme_output/挑过Military-Camouflage-MHCD2022/数据集 (new MHCD data, 336 imgs)

Strategy:
  1. Final_v6 = baseline data (mixed1, tank, tank2, armored are clean).
  2. Remove old mhcd_* and camo_* files from final_v6.
  3. Convert new camouflage + MHCD images+labels → YOLO format.
  4. Split new images using final_v6's existing ratio (80/10/10).
  5. Apply fog + rain augmentation to train split only.
  6. Output → data/final_v10 (overwrite old v10 which used different datasets)
"""
import os
import sys
import json
import shutil
import random
import math
from collections import defaultdict

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6_DIR = os.path.join(ROOT, "data", "final_v6")
LABELME_DIR = os.path.join(ROOT, "datasets", "labelme", "labelme_output")
OUTPUT = os.path.join(ROOT, "data", "final_v10")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAME = "military_vehicle"
NC = 1

# ============================================================
# Augmentation Config (same as original v10)
# ============================================================
AUG_PROB_FOG = 0.30
AUG_PROB_RAIN = 0.30


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
# Step 1: Load new camouflage + MHCD data from labelme_output
# ============================================================
def load_labelme_json(json_path, img_path):
    """Convert labelme JSON to YOLO format (normalized bbox)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_w = data.get("imageWidth", 640)
    img_h = data.get("imageHeight", 640)
    boxes = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        # Map all military vehicle labels to class 0
        if label.lower() in ("vehicle", "tank", "truck", "military vehicle", "military_vehicle"):
            points = shape.get("points", [])
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                bw = abs(x2 - x1) / img_w
                bh = abs(y2 - y1) / img_h
                boxes.append((0, [xc, yc, bw, bh]))
    return boxes


def collect_labelme_images(dir_path, source_prefix):
    """Collect all images + labels from a labelme directory."""
    items = []
    jpg_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for jpg_f in jpg_files:
        img_path = os.path.join(dir_path, jpg_f)
        base_name = os.path.splitext(jpg_f)[0]
        json_f = base_name + ".json"
        json_path = os.path.join(dir_path, json_f)
        if not os.path.exists(json_path):
            continue
        boxes = load_labelme_json(json_path, img_path)
        items.append({
            "img_path": img_path,
            "boxes": boxes,
            "has_boxes": len(boxes) > 0,
            "base_name": base_name,
            "source": source_prefix,
        })
    return items


# ============================================================
# Step 2: Copy final_v6 baseline (minus old mhcd_* and camo_*)
# ============================================================
def copy_v6_filtered(out_dir, split_name):
    """Copy v6 images/labels, skipping mhcd_* and camo_* files."""
    v6_img_dir = os.path.join(V6_DIR, split_name, "images")
    v6_lbl_dir = os.path.join(V6_DIR, split_name, "labels")

    out_img_dir = os.path.join(out_dir, split_name, "images")
    out_lbl_dir = os.path.join(out_dir, split_name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    copied = 0
    skipped = 0

    for img_f in os.listdir(v6_img_dir):
        if not img_f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        # Skip old mhcd and camo files
        if img_f.lower().startswith('mhcd_') or img_f.lower().startswith('camo_'):
            # Also skip augmented versions of camo files
            skipped += 1
            continue

        # Copy image
        shutil.copy2(os.path.join(v6_img_dir, img_f), os.path.join(out_img_dir, img_f))

        # Copy corresponding label if exists
        base = os.path.splitext(img_f)[0]
        for ext in ['.txt']:
            lbl_src = os.path.join(v6_lbl_dir, base + ext)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(out_lbl_dir, base + ext))
                break
        copied += 1

    return copied, skipped


# ============================================================
# Main Build
# ============================================================
def main():
    print("=" * 60)
    print("Building final_v10 from:")
    print("  - final_v6 (cleaned: remove old mhcd/camo)")
    print("  - labelme_output/伪装坦克数据集目标检测 (121 new)")
    print("  - labelme_output/挑过MHCD (336 new)")
    print("=" * 60)

    if os.path.exists(OUTPUT):
        print(f"\nRemoving existing: {OUTPUT}")
        shutil.rmtree(OUTPUT)
    os.makedirs(OUTPUT, exist_ok=True)

    # ======== Collect new camouflage/MHCD data ========
    print("\n[1/5] Loading new camouflage + MHCD data...")
    camo_dir = os.path.join(LABELME_DIR, "伪装坦克数据集目标检测")
    mhcd_dir = os.path.join(LABELME_DIR, "挑过Military-Camouflage-MHCD2022", "数据集")

    camo_items = collect_labelme_images(camo_dir, source_prefix="camo")
    mhcd_items = collect_labelme_images(mhcd_dir, source_prefix="mhcd")
    new_items = camo_items + mhcd_items

    camo_pos = sum(1 for x in camo_items if x["has_boxes"])
    mhcd_pos = sum(1 for x in mhcd_items if x["has_boxes"])
    print(f"  伪装坦克: {len(camo_items)} images ({camo_pos} with boxes)")
    print(f"  MHCD: {len(mhcd_items)} images ({mhcd_pos} with boxes)")
    print(f"  Total new: {len(new_items)} images")

    # ======== Split new data 80/10/10 ========
    print("\n[2/5] Splitting new data 80/10/10...")
    random.shuffle(new_items)
    n = len(new_items)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    new_split = {
        "train": new_items[:n_train],
        "val": new_items[n_train:n_train + n_val],
        "test": new_items[n_train + n_val:],
    }
    for sn, items in new_split.items():
        pos = sum(1 for x in items if x["has_boxes"])
        print(f"  new_{sn}: {len(items)} images ({pos} positive)")

    # Save new splits separately for merging later
    all_new = {
        "train": [],
        "val": [],
        "test": [],
    }

    # ======== Step 3: Write output ========
    print("\n[3/5] Building final dataset...")
    aug_stats = {"fog": 0, "rain": 0, "source_new": 0, "source_v6": 0}

    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT, split_name, "images")
        lbl_dir = os.path.join(OUTPUT, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # 3a. Copy filtered v6 data
        v6_copied, v6_skipped = copy_v6_filtered(OUTPUT, split_name)
        aug_stats["source_v6"] += v6_copied
        print(f"  {split_name}: copied {v6_copied} v6 images (skipped {v6_skipped} old mhcd/camo)")

        # 3b. Write new camouflage/MHCD images (YOLO format)
        for item in new_split[split_name]:
            img = read_image(item["img_path"])
            if img is None:
                continue

            # Resize to 640x640 for consistency
            h, w = img.shape[:2]
            scale_factor_w = 1.0
            scale_factor_h = 1.0
            if w != 640 or h != 640:
                img_resized = cv2.resize(img, (640, 640))
                scale_factor_w = 640.0 / w
                scale_factor_h = 640.0 / h
            else:
                img_resized = img

            src = item["source"]
            base = item["base_name"]
            # Sanitize base name (remove leading dash)
            safe_base = base.lstrip("-").replace(" ", "_")
            dst_name = f"{src}_{safe_base}.jpg"
            dst_img = os.path.join(img_dir, dst_name)
            dst_lbl = os.path.join(lbl_dir, f"new_{src}_{safe_base}.txt")

            # Write image
            imwrite(dst_img, img_resized)

            # Write YOLO label with rescaled boxes
            with open(dst_lbl, "w") as lf:
                for cls_id, (xc, yc, bw, bh) in item["boxes"]:
                    if scale_factor_w != 1.0 or scale_factor_h != 1.0:
                        xc *= scale_factor_w
                        bw *= scale_factor_w
                        yc *= scale_factor_h
                        bh *= scale_factor_h
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            aug_stats["source_new"] += 1

        print(f"    + {len(new_split[split_name])} new images")
        print(f"    Total {split_name}: {v6_copied + len(new_split[split_name])} images")

    # ======== Step 4: Apply fog/rain augmentation (train only) ========
    print("\n[4/5] Applying fog + rain augmentation (train only)...")
    train_img_dir = os.path.join(OUTPUT, "train", "images")
    train_lbl_dir = os.path.join(OUTPUT, "train", "labels")
    train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_f in train_imgs:
        img_path = os.path.join(train_img_dir, img_f)
        base = os.path.splitext(img_f)[0]
        lbl_path = None
        for ext in ['.txt']:
            test_path = os.path.join(train_lbl_dir, base + ext)
            if os.path.exists(test_path):
                lbl_path = test_path
                break
        if lbl_path is None:
            continue

        # Only augment images that have boxes (positive samples)
        with open(lbl_path, "r") as f:
            labels = f.read().strip()
        if not labels:
            continue

        img = read_image(img_path)
        if img is None:
            continue

        # Fog
        if random.random() < AUG_PROB_FOG:
            aug_img = add_fog(img)
            aug_name = f"{base}_aug_fog.jpg"
            imwrite(os.path.join(train_img_dir, aug_name), aug_img)
            shutil.copy2(lbl_path, os.path.join(train_lbl_dir, f"{base}_aug_fog.txt"))
            aug_stats["fog"] += 1

        # Rain
        if random.random() < AUG_PROB_RAIN:
            aug_img = add_rain(img)
            aug_name = f"{base}_aug_rain.jpg"
            imwrite(os.path.join(train_img_dir, aug_name), aug_img)
            shutil.copy2(lbl_path, os.path.join(train_lbl_dir, f"{base}_aug_rain.txt"))
            aug_stats["rain"] += 1

    print(f"  Fog: {aug_stats['fog']} | Rain: {aug_stats['rain']}")

    # ======== Step 5: Generate data.yaml ========
    print("\n[5/5] Generating data.yaml...")
    yaml_path = os.path.join(OUTPUT, "data.yaml")
    config = {
        "path": OUTPUT.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NC,
        "names": [CLASS_NAME],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # ======== Final stats ========
    print(f"\n{'='*60}")
    print("FINAL STATS")
    print(f"{'='*60}")
    total_all = 0
    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT, split_name, "images")
        lbl_dir = os.path.join(OUTPUT, split_name, "labels")
        n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        n_aug = len([f for f in os.listdir(img_dir) if '_aug_' in f])

        # Count positive/negative/boxes
        n_pos = 0
        n_neg = 0
        n_boxes = 0
        for lbl_f in os.listdir(lbl_dir):
            if not lbl_f.endswith('.txt'):
                continue
            with open(os.path.join(lbl_dir, lbl_f)) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                n_pos += 1
            else:
                n_neg += 1
            n_boxes += len(lines)

        # Count new vs v6
        n_new = len([f for f in os.listdir(img_dir) if f.startswith(('camo_', 'mhcd_', 'new_camo_', 'new_mhcd_'))])
        total_all += n_imgs
        print(f"  {split_name}: {n_imgs} images ({n_aug} aug, {n_new} new), {n_pos} pos, {n_neg} neg, {n_boxes} boxes")

    print(f"\n  TOTAL: {total_all} images")
    print(f"  Source: v6 filtered + {aug_stats['source_new']} new (camo+MHCD) + {aug_stats['fog']+aug_stats['rain']} augmented")
    print(f"\n  Old mhcd/camo files removed: ~{v6_skipped} across all splits")
    print(f"  Output: {OUTPUT}")


if __name__ == "__main__":
    main()
