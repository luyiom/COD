#!/usr/bin/env python3
"""可视化数据集标注 —— 把标注框/分割多边形画在图片上，保存到指定目录供查看。"""

import os
import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

def pil_to_cv2(img):
    """PIL Image 转 OpenCV 格式（用于计算）"""
    img = img.convert('RGB')
    import cv2
    import numpy as np
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_yolo_boxes(img_path, label_path, names):
    """画 YOLO 检测框"""
    try:
        img = Image.open(str(img_path))
    except Exception as e:
        print(f"  无法读取图片 {img_path}: {e}")
        return None

    draw = ImageDraw.Draw(img)
    w, h = img.size

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
                # 检测框格式: class x_center y_center width height (归一化)
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                label = names.get(cls_id, str(cls_id))
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                draw.text((x1, max(y1 - 15, 5)), label, fill='green')
    except Exception as e:
        print(f"  无法读取标注 {label_path}: {e}")
    return img

def draw_yolo_segments(img_path, label_path, names):
    """画 YOLO 分割多边形"""
    try:
        img = Image.open(str(img_path))
    except Exception as e:
        print(f"  无法读取图片 {img_path}: {e}")
        return None

    draw = ImageDraw.Draw(img)
    w, h = img.size

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
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append((x, y))
                label = names.get(cls_id, str(cls_id))
                if len(points) > 1:
                    draw.line(points + [points[0]], fill='yellow', width=2)
                if points:
                    draw.text(points[0], label, fill='yellow')
    except Exception as e:
        print(f"  无法读取标注 {label_path}: {e}")
    return img

def visualize_dataset(dataset_dir, output_dir, names, max_samples=50, has_seg=False, splits=None):
    """遍历数据集，画标注并保存"""
    if splits is None:
        splits = ['train', 'valid', 'test']

    os.makedirs(output_dir, exist_ok=True)
    draw_fn = draw_yolo_segments if has_seg else draw_yolo_boxes

    for split in splits:
        img_dir = Path(dataset_dir) / split / 'images'
        lbl_dir = Path(dataset_dir) / split / 'labels'
        if not img_dir.exists():
            # 尝试非标准结构（如伪装坦克数据集）
            alt_img = Path(dataset_dir) / 'images'
            alt_lbl = Path(dataset_dir) / 'labels'
            if alt_img.exists():
                img_dir, lbl_dir = alt_img, alt_lbl

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  跳过 {split}: 目录不存在")
            continue

        # 优先选有目标的图
        img_files = sorted(img_dir.glob('*'))
        # 过滤只保留有对应标注文件的图
        paired = []
        for img_f in img_files:
            lbl_f = lbl_dir / (img_f.stem + '.txt')
            if lbl_f.exists():
                paired.append((img_f, lbl_f))

        sample = random.sample(paired, min(max_samples, len(paired)))
        split_out = os.path.join(output_dir, split)
        os.makedirs(split_out, exist_ok=True)

        success = 0
        for img_f, lbl_f in sample:
            result = draw_fn(img_f, lbl_f, names)
            if result is not None:
                out_path = os.path.join(split_out, img_f.name)
                result.save(out_path)
                success += 1

        print(f"  {split}: 成功保存 {success}/{len(sample)} 张可视化图片 -> {split_out}")

def main():
    parser = argparse.ArgumentParser(description='可视化数据集标注')
    parser.add_argument('dataset', help='数据集路径')
    parser.add_argument('-o', '--output', default='./vis_output', help='输出目录')
    parser.add_argument('-k', '--num', type=int, default=30, help='每个split采样数量')
    parser.add_argument('--seg', action='store_true', help='分割标注模式')
    parser.add_argument('--names', nargs='+', default=['object'], help='类别名列表')
    args = parser.parse_args()

    names = {i: name for i, name in enumerate(args.names)}
    splits = ['train', 'valid', 'test']

    print(f"数据集: {args.dataset}")
    print(f"类别: {names}")
    print(f"标注类型: {'分割多边形' if args.seg else '检测框'}")
    print()

    visualize_dataset(args.dataset, args.output, names, args.num, args.seg, splits)
    print(f"\n完成！图片已保存到: {args.output}")
    print(f"可以直接用文件管理器打开该目录查看。")

if __name__ == '__main__':
    main()
