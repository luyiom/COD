#!/usr/bin/env python3
"""将YOLO分割标注转换为YOLO检测框格式（bounding box）"""

import os
from pathlib import Path

def polygon_to_bbox(points):
    """
    将多边形点转换为边界框 [x_center, y_center, width, height]
    points: [[x1, y1], [x2, y2], ...]
    返回: (xc, yc, bw, bh) 归一化坐标
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # 计算中心点和宽高
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return x_center, y_center, width, height


def convert_seg_to_det(dataset_dir, output_dir, class_names=None):
    """
    将分割标注转换为检测框标注
    """
    if class_names is None:
        class_names = ['tank']

    # 尝试不同的目录结构
    img_dirs = [
        Path(dataset_dir) / 'images',
        Path(dataset_dir) / 'tank_dataset' / 'images',
    ]
    lbl_dirs = [
        Path(dataset_dir) / 'labels',
        Path(dataset_dir) / 'tank_dataset' / 'labels',
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

    # 创建输出目录
    out_img_dir = Path(output_dir) / 'images'
    out_lbl_dir = Path(output_dir) / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片文件
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))

    converted = 0
    for img_f in img_files:
        lbl_f = lbl_dir / (img_f.stem + '.txt')

        if not lbl_f.exists():
            continue

        # 读取分割标注
        try:
            with open(lbl_f, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  无法读取标注 {lbl_f}: {e}")
            continue

        # 转换为检测框
        det_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:  # 至少 class + 3对坐标
                continue

            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            # 多边形点
            points = []
            for i in range(0, len(coords), 2):
                points.append([coords[i], coords[i+1]])

            # 转换为边界框
            xc, yc, bw, bh = polygon_to_bbox(points)

            # 确保class ID在有效范围内
            if cls_id >= len(class_names):
                cls_id = 0  # 默认使用第一个类别

            # 写入检测框格式: class x_center y_center width height
            det_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # 保存检测框标注
        out_lbl_path = out_lbl_dir / (img_f.stem + '.txt')
        try:
            with open(out_lbl_path, 'w', encoding='utf-8') as f:
                f.writelines(det_lines)

            # 复制图片
            import shutil
            shutil.copy2(str(img_f), str(out_img_dir / img_f.name))

            converted += 1
        except Exception as e:
            print(f"  无法保存 {out_lbl_path}: {e}")

    print(f"  转换 {converted} 个文件")
    print(f"  图片保存到: {out_img_dir}")
    print(f"  标注保存到: {out_lbl_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO分割格式转YOLO检测框格式')
    parser.add_argument('dataset', help='分割数据集路径')
    parser.add_argument('-o', '--output', default='./det_output', help='输出目录')
    parser.add_argument('--names', nargs='+', default=['tank'], help='类别名列表')
    args = parser.parse_args()

    print(f"源数据集: {args.dataset}")
    print(f"输出目录: {args.output}")
    print(f"类别: {args.names}")
    print()

    convert_seg_to_det(args.dataset, args.output, args.names)

    print(f"\n转换完成！")
    print(f"\n生成的检测框数据集结构:")
    print(f"  {args.output}/")
    print(f"  ├── images/")
    print(f"  └── labels/")
    print(f"\n使用LabelImg打开:")
    print(f"  1. 启动 LabelImg")
    print(f"  2. 点击 'Open Dir' 选择: {args.output}/images")
    print(f"  3. 点击 'Change Save Dir' 选择: {args.output}/labels")


if __name__ == '__main__':
    main()
