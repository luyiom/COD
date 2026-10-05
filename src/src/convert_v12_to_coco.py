"""
Convert final_v12 (YOLO format) to COCO JSON format for MMDetection.
"""
import os
import json
import cv2
from datetime import datetime

DATASET_DIR = r"D:\lingma项目\伪装坦克项目2\data\final_v12"
OUTPUT_DIR = r"D:\lingma项目\伪装坦克项目2\data\final_v12_coco"

CATEGORIES = [{"id": 1, "name": "military_vehicle", "supercategory": "vehicle"}]


def convert_split(split_name):
    img_dir = os.path.join(DATASET_DIR, split_name, "images")
    lbl_dir = os.path.join(DATASET_DIR, split_name, "labels")

    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    jpg_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for jpg_f in jpg_files:
        img_path = os.path.join(img_dir, jpg_f)
        base = os.path.splitext(jpg_f)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")

        # Read image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        img_id += 1

        images.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": jpg_f,
        })

        # Read labels
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    x1 = xc - bw / 2
                    y1 = yc - bh / 2

                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,  # military_vehicle
                        "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                    })

    coco = {
        "info": {
            "description": "final_v12 military vehicle detection",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, split_name, "annotations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"  {split_name}: {len(images)} images, {len(annotations)} boxes -> {out_path}")
    return len(images), len(annotations)


def main():
    print("Converting final_v12 to COCO format...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_img = 0
    total_ann = 0
    for split in ["train", "val", "test"]:
        n_img, n_ann = convert_split(split)
        total_img += n_img
        total_ann += n_ann
    print(f"\nDone! {total_img} images, {total_ann} boxes")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
