"""
Data cleaning: remove blurry, corrupted, and undersized images.
Uses Laplacian variance to detect blur.
Handles Unicode paths on Windows via cv2.imdecode.
"""
import sys
import os
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from config import DATASETS, DATA_CLEANED, BLUR_THRESHOLD, ROOT


def read_image(path):
    """Read image with Unicode path support on Windows."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    """Check if image is blurry using Laplacian variance."""
    img = read_image(image_path)
    if img is None:
        return True, "corrupted"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, f"blurry (score={score:.1f})"


def is_too_small(image_path, min_w=50, min_h=50):
    """Check if image is too small."""
    img = read_image(image_path)
    if img is None:
        return True, "corrupted"
    h, w = img.shape[:2]
    return (w < min_w or h < min_h), f"too_small ({w}x{h})"


def find_label_path(img_path, root_dir):
    """Find corresponding YOLO label file for an image."""
    base = os.path.splitext(img_path)[0]
    # Try common label locations
    candidates = [
        base + ".txt",
        base.replace("images", "labels") + ".txt",
        img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt",
    ]
    for c in candidates:
        # Handle paths via regex replacement for YOLO dir structure
        for subdir in ["train", "val", "valid", "test"]:
            alt = c.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            if os.path.exists(alt):
                return alt
    # Raw replacement
    raw = img_path.replace("images", "labels")
    raw = os.path.splitext(raw)[0] + ".txt"
    if os.path.exists(raw):
        return raw
    return None


def process_dataset(name, src_dir, dst_dir):
    """Copy clean images and labels from src_dir to dst_dir."""
    removed = []
    kept = 0
    total = 0

    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            total += 1
            img_path = os.path.join(root, f)
            rel = os.path.relpath(img_path, src_dir)

            # Check image quality
            blurry, blur_reason = is_blurry(img_path)
            small, size_reason = is_too_small(img_path)

            if blurry or small:
                reason = blur_reason if blurry else size_reason
                removed.append({"file": rel, "reason": reason})
                continue

            # Copy image
            dst_img_rel = os.path.join(name, rel)
            dst_img = os.path.join(dst_dir, dst_img_rel)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(img_path, dst_img)

            # Copy label if exists
            label_path = find_label_path(img_path, src_dir)
            if label_path:
                label_rel = os.path.relpath(label_path, src_dir)
                dst_label = os.path.join(dst_dir, name, label_rel)
                os.makedirs(os.path.dirname(dst_label), exist_ok=True)
                shutil.copy2(label_path, dst_label)

            kept += 1

    return {"name": name, "total": total, "kept": kept, "removed": removed}


def main():
    print("=" * 60)
    print("Data Cleaning Report")
    print("=" * 60)

    all_reports = []
    total_removed = 0
    total_kept = 0

    for name, src_dir in DATASETS.items():
        if not os.path.exists(src_dir):
            print(f"  SKIP: {name} (not found: {src_dir})")
            continue

        print(f"\n[{name}] scanning...")
        report = process_dataset(name, src_dir, DATA_CLEANED)
        all_reports.append(report)
        total_removed += len(report["removed"])
        total_kept += report["kept"]

        print(f"  Total:    {report['total']}")
        print(f"  Kept:     {report['kept']}")
        print(f"  Removed:  {len(report['removed'])}")
        for r in report["removed"][:3]:
            print(f"    - {r['file']}: {r['reason']}")
        if len(report["removed"]) > 3:
            print(f"    ... and {len(report['removed'])-3} more")

    print(f"\n{'='*60}")
    print(f"Summary: {total_kept} kept, {total_removed} removed out of {total_kept+total_removed} total")
    print(f"Cleaned data saved to: {DATA_CLEANED}")

    report_path = os.path.join(ROOT, "data", "cleaning_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
