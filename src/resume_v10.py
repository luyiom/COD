"""
Resume V10 training from last.pt (epoch 80 → 100).
Workers=2 to avoid Windows pagefile OOM with 32+ processes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAST_PT = os.path.join(ROOT, "runs", "camouflage_v10", "weights", "last.pt")
PROJECT = os.path.join(ROOT, "runs")
NAME = "camouflage_v10"


def main():
    print("=" * 60)
    print(f"Resuming from: {LAST_PT}")
    print(f"Workers: 2 | Epochs: 100 | Device: cuda:0")
    print("=" * 60)

    model = YOLO(LAST_PT)

    results = model.train(
        resume=True,
        data=os.path.join(ROOT, "data", "final_v10", "data.yaml"),
        epochs=100,
        patience=30,
        batch=16,
        imgsz=640,
        device=0,
        workers=2,

        lr0=0.008,
        lrf=0.001,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,           # skip warmup when resuming

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

        close_mosaic=20,
        rect=False,
        multi_scale=0,

        box=8.0,
        cls=0.3,
        dfl=1.5,

        half=True,
        amp=True,
        cache=False,

        save=True,
        save_period=5,
        exist_ok=True,
        project=PROJECT,
        name=NAME,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
