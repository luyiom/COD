"""
Train YOLO model for military vehicle detection on final_v10 dataset.
Optimized for RTX 5070 12GB.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "final_v10", "data.yaml")
PROJECT = os.path.join(ROOT, "runs")
NAME = "camouflage_v10"
MODEL_NAME = os.path.join(ROOT, "models", "yolo26s.pt")


def main():
    print("=" * 60)
    print(f"Training: {MODEL_NAME} on final_v10")
    print(f"Data: {DATA_YAML}")
    print(f"Output: {PROJECT}/{NAME}")
    print(f"Workers: 8 | Batch: 16 | Epochs: 100 | multi_scale: 0 | Device: cuda:0")
    print("=" * 60)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        patience=30,
        batch=16,
        imgsz=640,
        device=0,
        workers=8,

        # ======== 优化器 ========
        lr0=0.008,
        lrf=0.001,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # ======== 在线增强 ========
        hsv_h=0.025,
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=10,
        translate=0.2,
        scale=0.7,
        shear=0,
        perspective=0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.5,

        # ======== 训练策略 ========
        close_mosaic=20,
        rect=False,
        multi_scale=0,              # 关闭多尺度, workers=8稳定, 与v8一致

        # ======== Loss权重 ========
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # ======== GPU优化 ========
        half=True,
        amp=True,
        cache=False,

        # ======== 保存 ========
        save=True,
        save_period=10,
        exist_ok=True,
        project=PROJECT,
        name=NAME,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
