#!/usr/bin/env python3
"""将YOLO检测框格式转换为LabelMe JSON格式"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image

def det_to_labelme(img_path, label_path, output_dir, class_names):
    """
    将YOLO检测框转换为LabelMe JSON格式
    """
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as e:
        print(f"  无法读取图片 {img_path}: {e}")
        return None

    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])

                    # YOLO检测框转坐标
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

                    shape = {
                        "label": label,
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    labelme_data["shapes"].append(shape)
        except Exception as e:
            print(f"  无法读取标注 {label_path}: {e}")

    json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_name)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        return json_path
    except Exception as e:
        print(f"  无法保存JSON {json_path}: {e}")
        return None


def convert_dataset(dataset_dir, output_dir, class_names):
    """
    转换整个数据集
    """
    img_dir = Path(dataset_dir) / 'images'
    lbl_dir = Path(dataset_dir) / 'labels'

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"错误: 找不到 images 或 labels 目录")
        return

    os.makedirs(output_dir, exist_ok=True)

    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
    converted = 0
    copied = 0

    for img_f in img_files:
        lbl_f = lbl_dir / (img_f.stem + '.txt')
        if lbl_f.exists():
            result = det_to_labelme(str(img_f), str(lbl_f), output_dir, class_names)
            if result:
                converted += 1
                try:
                    shutil.copy2(str(img_f), os.path.join(output_dir, img_f.name))
                    copied += 1
                except Exception as e:
                    print(f"  无法复制图片 {img_f}: {e}")

    print(f"  转换 {converted} 个标注, 复制 {copied} 张图片 -> {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO检测框转LabelMe格式')
    parser.add_argument('dataset', help='数据集路径')
    parser.add_argument('-o', '--output', default='./datasets/labelme/labelme_output', help='输出目录')
    parser.add_argument('--names', nargs='+', default=['tank'], help='类别名列表')
    args = parser.parse_args()

    class_names = args.names

    print(f"数据集: {args.dataset}")
    print(f"类别: {class_names}")
    print()

    convert_dataset(args.dataset, args.output, class_names)

    print(f"\n转换完成！")
    print(f"LabelMe格式数据已保存到: {args.output}")
    print(f"\n使用LabelMe打开:")
    print(f"  1. 启动 LabelMe")
    print(f"  2. 点击 'Open Dir' 选择: {args.output}")


if __name__ == '__main__':
    main()
