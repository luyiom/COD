"""Train torchvision Faster R-CNN on final_v12 dataset."""
import os, sys, json, math, time, random
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

ROOT = r"D:\lingma项目\伪装坦克项目2"
COCO_DIR = os.path.join(ROOT, "data", "final_v12_coco")
WORK_DIR = os.path.join(ROOT, "runs", "v12_models", "faster_rcnn_v12")
os.makedirs(WORK_DIR, exist_ok=True)

CATEGORIES = {1: "military_vehicle"}
NUM_CLASSES = 2  # background + military_vehicle


class CocoDataset(Dataset):
    def __init__(self, ann_file, img_dir):
        with open(ann_file) as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.img_dir = img_dir
        # Build annotation index
        self.ann_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)

        anns = self.ann_by_image.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # military_vehicle

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if len(boxes) == 0:
            # torchvision requires at least one box; use dummy (ignored during training)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load dataset
    for split in ["train", "val"]:
        ann_file = os.path.join(COCO_DIR, split, "annotations.json")
        img_dir = os.path.join(ROOT, "data", "final_v12", split, "images")
        ds = CocoDataset(ann_file, img_dir)
        if split == "train":
            train_dataset = ds
            print(f"Train: {len(ds)} images")
        else:
            val_dataset = ds
            print(f"Val: {len(ds)} images")

    # DataLoader with persistent workers
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        multiprocessing_context="spawn",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Model
    print("Building Faster R-CNN...")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training
    num_epochs = 100
    print(f"Starting training: {num_epochs} epochs")
    best_map = 0
    results = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        lr_scheduler.step()

        # Validation every epoch
        model.eval()
        # Simple mAP estimation (use COCO evaluator)
        from torchvision.ops import box_iou

        all_preds = []
        all_gts = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                preds = model(images)

                for pred, target in zip(preds, targets):
                    # Keep top predictions
                    keep = pred["scores"] > 0.3
                    filtered = {
                        "boxes": pred["boxes"][keep].cpu(),
                        "scores": pred["scores"][keep].cpu(),
                        "labels": pred["labels"][keep].cpu(),
                    }
                    all_preds.append(filtered)
                    all_gts.append({k: v.cpu() for k, v in target.items()})

        # Calculate approximate mAP@50
        ap_sum = 0
        count = 0
        for pred, gt in zip(all_preds, all_gts):
            if len(gt["boxes"]) == 0:
                continue
            if len(pred["boxes"]) == 0:
                continue
            ious = box_iou(pred["boxes"], gt["boxes"])
            best_iou = ious.max(dim=1).values
            tp = (best_iou > 0.5).sum().item()
            fp = len(pred["boxes"]) - tp
            fn = len(gt["boxes"]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap_sum += precision
            count += 1

        approx_map50 = ap_sum / max(count, 1)

        train_loss_avg = total_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"ep{epoch:>3} | loss={train_loss_avg:.4f} | approx_mAP50={approx_map50:.4f} | {elapsed:.0f}s")

        results.append({
            "epoch": epoch, "loss": train_loss_avg, "approx_mAP50": approx_map50,
        })

        if approx_map50 > best_map:
            best_map = approx_map50
            torch.save(model.state_dict(), os.path.join(WORK_DIR, "best.pth"))
            print(f"  -> new best! mAP50={best_map:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(WORK_DIR, f"epoch{epoch}.pth"))

    # Save final
    torch.save(model.state_dict(), os.path.join(WORK_DIR, "last.pth"))

    # Save results CSV
    csv_path = os.path.join(WORK_DIR, "results.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,loss,approx_mAP50\n")
        for r in results:
            f.write(f"{r['epoch']},{r['loss']:.6f},{r['approx_mAP50']:.6f}\n")

    print(f"\nTraining complete! Best mAP50={best_map:.4f}")
    print(f"Results: {WORK_DIR}")


if __name__ == "__main__":
    main()
