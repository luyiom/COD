"""
Convert MHCD2022 dataset JSON annotations to YOLO format,
merge with existing data, and generate new training dataset.
"""
import sys, os, json, shutil, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "datasets", "raw", "挑过Military-Camouflage-MHCD2022", "数据集")
EXISTING_DATA = os.path.join(ROOT, "data", "augmented")
OUTPUT_DIR = os.path.join(ROOT, "data", "final_v4")

# Map MHCD labels to our classes
LABEL_MAP = {"tank": 0, "military vehicle": 0}  # Both -> class 0 (military vehicle)
CLASS_NAMES = {0: "military_vehicle"}


def read_image(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def json_to_yolo(json_path, img_w, img_h):
    """Convert LabelMe JSON annotation to YOLO format boxes."""
    boxes = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if label not in LABEL_MAP:
            continue
        cls = LABEL_MAP[label]
        points = shape.get("points", [])
        if len(points) != 2:
            continue
        (x1, y1), (x2, y2) = points
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        boxes.append([cls, x_center, y_center, width, height])
    return boxes


def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def main():
    # Step 1: Convert MHCD dataset to YOLO format
    print("=" * 60)
    print("Converting MHCD2022 dataset...")

    mhcd_pairs = []
    kept = 0
    removed_blur = 0

    for f in os.listdir(SRC_DIR):
        if not f.endswith(".jpg"):
            continue
        base = os.path.splitext(f)[0]
        json_path = os.path.join(SRC_DIR, f"{base}.json")
        if not os.path.exists(json_path):
            continue

        img_path = os.path.join(SRC_DIR, f)
        img = read_image(img_path)
        if img is None:
            continue

        if is_blurry(img):
            removed_blur += 1
            continue

        h, w = img.shape[:2]
        boxes = json_to_yolo(json_path, w, h)
        if boxes:
            mhcd_pairs.append((img_path, boxes, base))
            kept += 1

    print(f"MHCD: {kept} kept, {removed_blur} blurry removed")

    # Step 2: Count class distribution
    class_counts = defaultdict(int)
    for _, boxes, _ in mhcd_pairs:
        for b in boxes:
            class_counts[b[0]] += 1
    print(f"MHCD class distribution: {dict(class_counts)}")

    # Step 3: Load existing augmented data
    print("\nLoading existing data...")
    existing_pairs = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(EXISTING_DATA, split, "images")
        lbl_dir = os.path.join(EXISTING_DATA, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, f)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
            boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append([int(float(parts[0]))] + [float(p) for p in parts[1:5]])
            existing_pairs[split].append((img_path, boxes, os.path.splitext(f)[0]))

    existing_counts = {}
    for split, pairs in existing_pairs.items():
        cc = defaultdict(int)
        for _, boxes, _ in pairs:
            for b in boxes:
                cc[b[0]] += 1
        existing_counts[split] = cc
        print(f"  {split}: {len(pairs)} images, classes={dict(cc)}")

    # Step 4: Split MHCD data (70/15/15)
    random.shuffle(mhcd_pairs)
    n = len(mhcd_pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    # Step 5: Write final dataset
    print(f"\nWriting final dataset to {OUTPUT_DIR}...")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Copy existing data
        for img_path, boxes, base in existing_pairs[split]:
            shutil.copy2(img_path, os.path.join(img_dir, os.path.basename(img_path)))
            with open(os.path.join(lbl_dir, base + ".txt"), "w") as lf:
                for b in boxes:
                    lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        # Add MHCD data
        if split == "train":
            mhcd_subset = mhcd_pairs[:n_train]
        elif split == "val":
            mhcd_subset = mhcd_pairs[n_train:n_train + n_val]
        else:
            mhcd_subset = mhcd_pairs[n_train + n_val:]

        for img_path, boxes, base in mhcd_subset:
            dst_img = os.path.join(img_dir, f"mhcd_{base}.jpg")
            shutil.copy2(img_path, dst_img)
            with open(os.path.join(lbl_dir, f"mhcd_{base}.txt"), "w") as lf:
                for b in boxes:
                    lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    # Step 6: Generate data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    config = {
        "path": OUTPUT_DIR,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["military_vehicle"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Step 7: Print final stats
    print("\n=== Final Dataset Stats ===")
    total_imgs = 0
    total_boxes = 0
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        cc = defaultdict(int)
        for lbl in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cc[int(float(parts[0]))] += 1
        n_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        n_boxes = sum(cc.values())
        total_imgs += n_imgs
        total_boxes += n_boxes
        print(f"  {split}: {n_imgs} images, {n_boxes} boxes | {dict(cc)}")
    print(f"\nTotal: {total_imgs} images, {total_boxes} boxes")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    random.seed(42)
    main()
