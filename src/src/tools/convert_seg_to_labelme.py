#!/usr/bin/env python3
"""将YOLO分割格式转换为LabelMe JSON格式（支持多边形分割，修复class ID映射）"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image

def yolo_seg_to_labelme(img_path, label_path, output_dir, class_names, class_id_map=None):
    """
    将YOLO分割标注转换为LabelMe JSON格式
    YOLO分割格式: class x1 y1 x2 y2 x3 y3 ... (归一化坐标)
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
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    # 读取YOLO分割标注并转换
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 7:  # 至少 class + 3对坐标
                        continue

                    cls_id = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))

                    # YOLO分割格式转多边形坐标
                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * w)
                        y = int(coords[i+1] * h)
                        points.append([x, y])

                    # 获取类别名（支持class ID映射）
                    if class_id_map and cls_id in class_id_map:
                        label = class_id_map[cls_id]
                    elif 0 <= cls_id < len(class_names):
                        label = class_names[cls_id]
                    else:
                        label = str(cls_id)

                    # 创建LabelMe多边形形状
                    shape = {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
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


def convert_segmentation_dataset(dataset_dir, output_dir, class_names, class_id_map=None):
    """
    转换分割数据集（非标准结构）
    """
    # 尝试不同的目录结构
    img_dirs = [
        Path(dataset_dir) / 'images',
        Path(dataset_dir) / 'tank_dataset' / 'images',
        Path(dataset_dir) / 'train' / 'images',
    ]
    lbl_dirs = [
        Path(dataset_dir) / 'labels',
        Path(dataset_dir) / 'tank_dataset' / 'labels',
        Path(dataset_dir) / 'train' / 'labels',
    ]

    img_dir = None
    lbl_dir = None

    for img_d, lbl_d in zip(img_dirs, lbl_dirs):
        if img_d.exists() and lbl_d.exists():
            img_dir = img_d
            lbl_dir = lbl_d
            break

    if not img_dir:
        print(f"错误: 找不到图片目录")
        return

    os.makedirs(output_dir, exist_ok=True)

    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
    converted = 0
    copied = 0

    for img_f in img_files:
        lbl_f = lbl_dir / (img_f.stem + '.txt')
        if lbl_f.exists():
            # 转换标注
            result = yolo_seg_to_labelme(str(img_f), str(lbl_f), output_dir, class_names, class_id_map)
            if result:
                converted += 1
                # 复制图片文件
                try:
                    shutil.copy2(str(img_f), os.path.join(output_dir, img_f.name))
                    copied += 1
                except Exception as e:
                    print(f"  无法复制图片 {img_f}: {e}")

    print(f"  转换 {converted} 个分割标注, 复制 {copied} 张图片 -> {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO分割格式转LabelMe格式')
    parser.add_argument('dataset', help='数据集路径')
    parser.add_argument('-o', '--output', default='./datasets/labelme/labelme_output', help='输出目录')
    parser.add_argument('--names', nargs='+', default=['object'], help='类别名列表')
    parser.add_argument('--class-map', nargs='+', help='class ID映射，格式: old_id:new_name，如 "3:tank"')
    args = parser.parse_args()

    class_names = args.names

    # 解析class ID映射
    class_id_map = {}
    if args.class_map:
        for mapping in args.class_map:
            if ':' in mapping:
                old_id, new_name = mapping.split(':', 1)
                class_id_map[int(old_id)] = new_name

    print(f"数据集: {args.dataset}")
    print(f"类别: {class_names}")
    if class_id_map:
        print(f"Class ID映射: {class_id_map}")
    print(f"标注类型: 分割多边形")
    print()

    convert_segmentation_dataset(args.dataset, args.output, class_names, class_id_map)

    print(f"\n转换完成！")
    print(f"LabelMe格式数据已保存到: {args.output}")
    print(f"\n使用LabelMe打开方式:")
    print(f"  1. 启动 LabelMe")
    print(f"  2. 点击 'Open Dir' 选择: {args.output}")
    print(f"  3. LabelMe会自动加载图片和JSON标注文件")
    print(f"\n修改分割标注:")
    print(f"  - 点击 'Create Polygons' 按钮")
    print(f"  - 在图片上点击绘制多边形")
    print(f"  - 按 Enter 确认")


if __name__ == '__main__':
    main()
