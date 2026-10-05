"""
Merge camouflage classes (1,2,3 -> 1) and apply data augmentation
to camouflage images to address class imbalance.
"""
import sys
import os
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FINAL = os.path.join(ROOT, "data", "final")
DATA_AUG = os.path.join(ROOT, "data", "augmented")
CLASS_NAMES = {0: "no_camouflage", 1: "camouflaged"}


def read_image(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def augment_image(img):
    """Apply a random augmentation to the image."""
    augs = []
    h, w = img.shape[:2]

    # Original (always include)
    augs.append(("original", img))

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augs.append(("flip", flipped))

    # Brightness +/-
    for factor in [0.7, 1.3]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augs.append((f"bright_{factor}", bright))

    # HSV shift (simulate different lighting)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for shift in [-20, 20]:
        hsv_shifted = hsv.copy()
        hsv_shifted[:, :, 0] = (hsv_shifted[:, :, 0] + shift) % 180
        hsv_shifted[:, :, 2] = np.clip(hsv_shifted[:, :, 2] * 0.8, 0, 255)
        shifted = cv2.cvtColor(hsv_shifted.astype(np.uint8), cv2.COLOR_HSV2BGR)
        augs.append((f"hsv_{shift}", shifted))

    # Gaussian blur (mild) - simulates slight motion/atmosphere
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    augs.append(("blur", blurred))

    # Rotation ±5°
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augs.append((f"rot_{angle}", rotated))

    # Noise
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    noisy = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    augs.append(("noise", noisy))

    return augs


def merge_and_augment():
    """Main processing: merge classes + augment camouflage images."""
    os.makedirs(DATA_AUG, exist_ok=True)

    for split in ["train", "val", "test"]:
        src_img_dir = os.path.join(DATA_FINAL, split, "images")
        src_lbl_dir = os.path.join(DATA_FINAL, split, "labels")
        dst_img_dir = os.path.join(DATA_AUG, split, "images")
        dst_lbl_dir = os.path.join(DATA_AUG, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir):
            continue

        camo_images = []  # (img_path, label_lines) for images with camouflage

        for f in os.listdir(src_img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            img_path = os.path.join(src_img_dir, f)
            base = os.path.splitext(f)[0]
            label_path = os.path.join(src_lbl_dir, base + ".txt")

            # Read original labels and merge classes
            merged_boxes = []
            has_camouflage = False
            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(float(parts[0]))
                        # Merge: 0 stays 0, 1/2/3 → 1
                        new_cls = 1 if cls >= 1 else 0
                        if new_cls == 1:
                            has_camouflage = True
                        merged_boxes.append([new_cls] + [float(p) for p in parts[1:5]])

            if has_camouflage:
                camo_images.append((img_path, merged_boxes))
            else:
                # Copy non-camouflage image directly
                shutil.copy2(img_path, os.path.join(dst_img_dir, f))
                if merged_boxes:
                    with open(os.path.join(dst_lbl_dir, base + ".txt"), "w") as lf:
                        for box in merged_boxes:
                            lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                else:
                    # Write empty label file
                    with open(os.path.join(dst_lbl_dir, base + ".txt"), "w") as lf:
                        pass

        # Augment camouflage images (train set only)
        if split == "train":
            total_new = 0
            for img_path, boxes in camo_images:
                img = read_image(img_path)
                if img is None:
                    continue
                augs = augment_image(img)
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                for aug_name, aug_img in augs:
                    new_name = f"{base_name}_aug_{aug_name}.jpg"
                    imwrite(os.path.join(dst_img_dir, new_name), aug_img)
                    # Copy same labels (boxes don't change for flip/color augs)
                    with open(os.path.join(dst_lbl_dir, f"{base_name}_aug_{aug_name}.txt"), "w") as lf:
                        for box in boxes:
                            # Adjust for horizontal flip
                            if aug_name == "flip":
                                box = box.copy()
                                box[1] = 1.0 - box[1]  # flip x_center
                            lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    total_new += 1
            print(f"{split}: {len(camo_images)} camo images -> {total_new} augmented samples")

        # Also copy original camouflage images
        for img_path, boxes in camo_images:
            f = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dst_img_dir, f))
            base = os.path.splitext(f)[0]
            with open(os.path.join(dst_lbl_dir, base + ".txt"), "w") as lf:
                for box in boxes:
                    lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

    # Generate data.yaml
    yaml_path = os.path.join(DATA_AUG, "data.yaml")
    config = {
        "path": DATA_AUG,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["no_camouflage", "camouflaged"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Print stats
    print("\n=== Final Stats ===")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATA_AUG, split, "images")
        lbl_dir = os.path.join(DATA_AUG, split, "labels")
        if os.path.exists(img_dir):
            class_counts = defaultdict(int)
            for lbl in os.listdir(lbl_dir):
                with open(os.path.join(lbl_dir, lbl)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_counts[int(float(parts[0]))] += 1
            print(f"{split}: {len(os.listdir(img_dir))} images | "
                  f"class0={class_counts[0]}, class1={class_counts[1]}")

    print(f"\ndata.yaml: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    merge_and_augment()
