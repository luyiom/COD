"""
Train YOLO model for military vehicle detection on final_v9 dataset.
Optimized for RTX 5070 12GB.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "final_v9", "data.yaml")
PROJECT = os.path.join(ROOT, "runs")
NAME = "camouflage_v9"
MODEL_NAME = os.path.join(ROOT, "models", "yolo26s.pt")  # 已迁移至 models/


def main():
    print("=" * 60)
    print(f"Training: {MODEL_NAME} on final_v9")
    print(f"Data: {DATA_YAML}")
    print(f"Output: {PROJECT}/{NAME}")
    print("=" * 60)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=200,
        patience=40,
        batch=16,
        imgsz=640,
        device=0,
        workers=4,

        # Optimizer
        lr0=0.008,
        lrf=0.001,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=10,
        translate=0.2,
        scale=0.7,
        shear=0,
        perspective=0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.5,

        # Training strategy
        close_mosaic=20,
        rect=False,
        multi_scale=0.5,

        # Loss weights
        box=8.0,
        cls=0.3,
        dfl=1.5,

        # GPU optimization
        half=True,
        amp=True,
        cache=False,

        # Saving
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
