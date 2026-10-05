"""
Build final_v12 dataset from datasets/labelme/cleanshujuji/ (6 datasets).
1. Labelme JSON -> YOLO txt (single class: military_vehicle)
2. Split unsplit datasets (camo, MHCD)
3. Merge into train/val/test with prefix naming
"""
import os
import json
import random
import shutil
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "datasets", "labelme", "cleanshuji")
OUTPUT = os.path.join(ROOT, "data", "final_v12")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

VALID_LABELS = {'vehicle', 'tank', 'truck', 'military vehicle', 'military_vehicle'}


def convert_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_w = data.get('imageWidth', 640)
    img_h = data.get('imageHeight', 640)
    labels = []
    for s in data.get('shapes', []):
        lbl = s.get('label', '').strip().lower()
        if lbl not in VALID_LABELS:
            continue
        points = s.get('points', [])
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            bw = abs(x2 - x1) / img_w
            bh = abs(y2 - y1) / img_h
            labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return labels


def process_dir(src_dir, dst_split, prefix):
    """Copy images + convert labels for a directory."""
    jpgs = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    count = 0
    for jpg_f in jpgs:
        base = os.path.splitext(jpg_f)[0]
        json_f = base + '.json'
        json_path = os.path.join(src_dir, json_f)
        if not os.path.exists(json_path):
            continue
        labels = convert_json(json_path)
        dst_name = f"{prefix}_{jpg_f}"
        dst_base = f"{prefix}_{base}"
        shutil.copy2(
            os.path.join(src_dir, jpg_f),
            os.path.join(OUTPUT, dst_split, 'images', dst_name)
        )
        with open(os.path.join(OUTPUT, dst_split, 'labels', dst_base + '.txt'), 'w') as f:
            f.write('\n'.join(labels) + '\n')
        count += 1
    return count


def split_items(items, n_train, n_val, n_test):
    """Split shuffled items list into train/val/test."""
    return {
        'train': items[:n_train],
        'val': items[n_train:n_train + n_val],
        'test': items[n_train + n_val:],
    }


def main():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    for sp in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT, sp, 'labels'), exist_ok=True)

    total = {'train': 0, 'val': 0, 'test': 0}

    # ====== 1. 伪装坦克数据集目标检测 (flat, 121 imgs) ======
    print("1/6 伪装坦克数据集目标检测 (flat -> 80/10/10)")
    cam_dir = os.path.join(BASE, "伪装坦克数据集目标检测")
    items = []
    for f in os.listdir(cam_dir):
        if f.endswith('.jpg') and os.path.exists(os.path.join(cam_dir, os.path.splitext(f)[0] + '.json')):
            items.append(f)
    random.shuffle(items)
    n = len(items)
    splits = split_items(items, int(n * 0.80), int(n * 0.10), n - int(n * 0.80) - int(n * 0.10))
    for sp, files in splits.items():
        for jpg_f in files:
            base = os.path.splitext(jpg_f)[0]
            labels = convert_json(os.path.join(cam_dir, base + '.json'))
            dst_name = f"camo_{jpg_f}"
            shutil.copy2(os.path.join(cam_dir, jpg_f), os.path.join(OUTPUT, sp, 'images', dst_name))
            with open(os.path.join(OUTPUT, sp, 'labels', f"camo_{base}.txt"), 'w') as f:
                f.write('\n'.join(labels) + '\n')
            total[sp] += 1
        print(f"  {sp}: {len(files)}")

    # ====== 2. 坦克数据集 (train/valid/test) ======
    print("2/6 坦克数据集")
    for sdir, dsp in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        n = process_dir(os.path.join(BASE, '坦克数据集', sdir), dsp, 'tank')
        total[dsp] += n
        print(f"  {sdir} -> {dsp}: {n}")

    # ====== 3. 坦克数据集2 (train/valid/test) ======
    print("3/6 坦克数据集2")
    for sdir, dsp in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        n = process_dir(os.path.join(BASE, '坦克数据集2', sdir), dsp, 'tank2')
        total[dsp] += n
        print(f"  {sdir} -> {dsp}: {n}")

    # ====== 4. MHCD (use ImageSets for split) ======
    print("4/6 挑过MHCD")
    mhcd_img = os.path.join(BASE, '挑过Military-Camouflage-MHCD2022', '数据集')
    mhcd_ann = os.path.join(BASE, '挑过Military-Camouflage-MHCD2022', 'Annotations')
    imgsets = os.path.join(BASE, '挑过Military-Camouflage-MHCD2022', 'ImageSets', 'Main')

    # Read split IDs
    split_ids = {}
    for fn in ['train.txt', 'val.txt', 'test.txt']:
        fp = os.path.join(imgsets, fn)
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                split_ids[fn.replace('.txt', '')] = set(l.strip() for l in f.readlines())

    # Map image IDs to paths
    img_map = {}
    for f in os.listdir(mhcd_img):
        if f.endswith('.jpg'):
            img_map[os.path.splitext(f)[0]] = f

    used = set()
    for split_key, sp in [('val', 'val'), ('test', 'test'), ('train', 'train')]:
        ids = split_ids.get(split_key, set())
        for img_id in ids:
            if img_id in img_map and img_id not in used:
                used.add(img_id)
                jpg_f = img_map[img_id]
                # Find JSON - prefer dataset/ version
                json_p = os.path.join(mhcd_img, img_id + '.json')
                if not os.path.exists(json_p):
                    json_p = os.path.join(mhcd_ann, img_id + '.json')
                if os.path.exists(json_p):
                    labels = convert_json(json_p)
                    dst_name = f"mhcd_{jpg_f}"
                    shutil.copy2(os.path.join(mhcd_img, jpg_f), os.path.join(OUTPUT, sp, 'images', dst_name))
                    with open(os.path.join(OUTPUT, sp, 'labels', f"mhcd_{img_id}.txt"), 'w') as f:
                        f.write('\n'.join(labels) + '\n')
                    total[sp] += 1

    # Remaining MHCD -> 80/10/10
    remaining = [(img_id, img_map[img_id]) for img_id in img_map if img_id not in used]
    if remaining:
        random.shuffle(remaining)
        n = len(remaining)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        for sp, items in [('train', remaining[:n_train]), ('val', remaining[n_train:n_train + n_val]), ('test', remaining[n_train + n_val:])]:
            for img_id, jpg_f in items:
                json_p = os.path.join(mhcd_img, img_id + '.json')
                if not os.path.exists(json_p):
                    json_p = os.path.join(mhcd_ann, img_id + '.json')
                if os.path.exists(json_p):
                    labels = convert_json(json_p)
                    dst_name = f"mhcd_{jpg_f}"
                    shutil.copy2(os.path.join(mhcd_img, jpg_f), os.path.join(OUTPUT, sp, 'images', dst_name))
                    with open(os.path.join(OUTPUT, sp, 'labels', f"mhcd_{img_id}.txt"), 'w') as f:
                        f.write('\n'.join(labels) + '\n')
                    total[sp] += 1
    for sp in ['train', 'val', 'test']:
        n_mhcd = len([f for f in os.listdir(os.path.join(OUTPUT, sp, 'images')) if f.startswith('mhcd_')])
        print(f"  {sp}: {n_mhcd} mhcd images")

    # ====== 5. 混合数据集1 (train/valid/test) ======
    print("5/6 混合数据集1")
    for sdir, dsp in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        n = process_dir(os.path.join(BASE, '混合数据集1', sdir), dsp, 'mixed1')
        total[dsp] += n
        print(f"  {sdir} -> {dsp}: {n}")

    # ====== 6. 装甲车数据集 (train/valid, test empty) ======
    print("6/6 装甲车数据集")
    for sdir, dsp in [('train', 'train'), ('valid', 'val')]:
        n = process_dir(os.path.join(BASE, '装甲车数据集', sdir), dsp, 'armored')
        total[dsp] += n
        print(f"  {sdir} -> {dsp}: {n}")

    # Move 50% of armored val to test (since original test was empty)
    armored_val_imgs = [f for f in os.listdir(os.path.join(OUTPUT, 'val', 'images')) if f.startswith('armored_')]
    if armored_val_imgs:
        random.shuffle(armored_val_imgs)
        move_n = len(armored_val_imgs) // 2
        for img_f in armored_val_imgs[:move_n]:
            shutil.move(
                os.path.join(OUTPUT, 'val', 'images', img_f),
                os.path.join(OUTPUT, 'test', 'images', img_f)
            )
            lbl_f = os.path.splitext(img_f)[0] + '.txt'
            src_lbl = os.path.join(OUTPUT, 'val', 'labels', lbl_f)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, os.path.join(OUTPUT, 'test', 'labels', lbl_f))
        total['val'] -= move_n
        total['test'] += move_n
        print(f"  Moved {move_n} armored val -> test")

    # ====== Generate data.yaml ======
    config = {
        'path': OUTPUT,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['military_vehicle'],
    }
    with open(os.path.join(OUTPUT, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # ====== Final stats ======
    print(f"\n=== final_v12 ===")
    grand_total = 0
    grand_boxes = 0
    for sp in ['train', 'val', 'test']:
        img_dir = os.path.join(OUTPUT, sp, 'images')
        lbl_dir = os.path.join(OUTPUT, sp, 'labels')
        n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        n_pos = 0; n_neg = 0; n_boxes = 0
        for lf in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lf)) as f:
                lines = [l for l in f if l.strip()]
            if lines:
                n_pos += 1; n_boxes += len(lines)
            else:
                n_neg += 1
        grand_total += n_imgs; grand_boxes += n_boxes
        print(f"  {sp}: {n_imgs} imgs ({n_pos} pos, {n_neg} neg), {n_boxes} boxes")
    print(f"  Total: {grand_total} images, {grand_boxes} boxes")
    print(f"\nDone! {OUTPUT}")


if __name__ == '__main__':
    main()
