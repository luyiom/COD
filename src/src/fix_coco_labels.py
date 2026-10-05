"""Fix the degenerate bounding box in COCO train annotations."""
import os, json, shutil

coco_file = r"D:\lingma项目\伪装坦克项目2\data\final_v12_coco\train\annotations.json"
backup_file = coco_file + ".bak"
shutil.copy2(coco_file, backup_file)

with open(coco_file) as f:
    coco = json.load(f)

# Remove invalid boxes
bad_ann_ids = []
good_annotations = []
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        bad_ann_ids.append(ann["id"])
        print(f"Removed ann_id={ann['id']} image_id={ann['image_id']} bbox={ann['bbox']}")
    else:
        good_annotations.append(ann)

coco["annotations"] = good_annotations

with open(coco_file, "w") as f:
    json.dump(coco, f)

print(f"\nRemoved {len(bad_ann_ids)} bad boxes. Annotations: {len(good_annotations)}")
print(f"Backup saved to {backup_file}")
