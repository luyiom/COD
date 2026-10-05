#!/usr/bin/env python3
"""将YOLO格式标注转换为LabelMe JSON格式（同时复制图片文件）"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image

def yolo_to_labelme(img_path, label_path, output_dir, class_names):
    """
    将单个图片的YOLO标注转换为LabelMe JSON格式
    """
    try:
        # 使用PIL读取图片
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as e:
        print(f"  无法读取图片 {img_path}: {e}")
        return None

    # 创建LabelMe JSON结构
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,  # 不包含图片数据，只保存路径
        "imageHeight": h,
        "imageWidth": w
    }

    # 读取YOLO标注并转换
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

                    # YOLO格式转坐标
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    # 创建LabelMe形状
                    shape = {
                        "label": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    labelme_data["shapes"].append(shape)
        except Exception as e:
            print(f"  无法读取标注 {label_path}: {e}")

    # 保存JSON文件
    json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_name)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        return json_path
    except Exception as e:
        print(f"  无法保存JSON {json_path}: {e}")
        return None


def convert_dataset(dataset_dir, output_dir, class_names, splits=None):
    """
    转换整个数据集（同时复制图片文件）
    """
    if splits is None:
        splits = ['train', 'valid', 'test']

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        img_dir = Path(dataset_dir) / split / 'images'
        lbl_dir = Path(dataset_dir) / split / 'labels'
        out_split_dir = os.path.join(output_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        if not img_dir.exists():
            # 尝试非标准结构
            alt_img = Path(dataset_dir) / 'images'
            alt_lbl = Path(dataset_dir) / 'labels'
            if alt_img.exists():
                img_dir, lbl_dir = alt_img, alt_lbl
                out_split_dir = output_dir  # 直接输出到根目录
                os.makedirs(out_split_dir, exist_ok=True)

        if not img_dir.exists():
            print(f"  跳过 {split}: 图片目录不存在")
            continue

        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        converted = 0
        copied = 0

        for img_f in img_files:
            lbl_f = lbl_dir / (img_f.stem + '.txt')
            if lbl_f.exists():
                # 转换标注
                result = yolo_to_labelme(str(img_f), str(lbl_f), out_split_dir, class_names)
                if result:
                    converted += 1
                    # 复制图片文件
                    try:
                        shutil.copy2(str(img_f), os.path.join(out_split_dir, img_f.name))
                        copied += 1
                    except Exception as e:
                        print(f"  无法复制图片 {img_f}: {e}")

        print(f"  {split}: 转换 {converted} 个标注, 复制 {copied} 张图片 -> {out_split_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO格式转LabelMe格式（包含图片）')
    parser.add_argument('dataset', help='数据集路径')
    parser.add_argument('-o', '--output', default='./datasets/labelme/labelme_output', help='输出目录')
    parser.add_argument('--names', nargs='+', default=['object'], help='类别名列表')
    args = parser.parse_args()

    class_names = args.names

    print(f"数据集: {args.dataset}")
    print(f"类别: {class_names}")
    print()

    convert_dataset(args.dataset, args.output, class_names)

    print(f"\n转换完成！")
    print(f"LabelMe格式数据已保存到: {args.output}")
    print(f"\n使用LabelMe打开方式:")
    print(f"  1. 启动 LabelMe")
    print(f"  2. 点击 'Open Dir' 选择: {args.output}")
    print(f"  3. LabelMe会自动加载图片和JSON标注文件")


if __name__ == '__main__':
    main()
