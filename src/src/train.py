"""
Train YOLOv8 model for camouflage military vehicle detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from ultralytics import YOLO

from config import DATA_FINAL, MODELS_DIR, YOLO_CONFIG, ROOT


def main():
    data_yaml = os.path.join(DATA_FINAL, "data.yaml")
    if not os.path.exists(data_yaml):
        # Try to generate it first
        print("data.yaml not found. Running data preparation...")
        from data_preparation import main as prep_main
        prep_main()

    data_yaml = os.path.join(DATA_FINAL, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found!")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("YOLOv8 Training for Camouflage Military Vehicle Detection")
    print("=" * 60)
    print(f"Data config: {data_yaml}")
    print(f"Device: {YOLO_CONFIG['device']}")
    print(f"Epochs: {YOLO_CONFIG['epochs']}")
    print(f"Batch size: {YOLO_CONFIG['batch']}")

    # Load model
    model = YOLO(YOLO_CONFIG["model"])

    # Train
    results = model.train(
        data=data_yaml,
        epochs=YOLO_CONFIG["epochs"],
        batch=YOLO_CONFIG["batch"],
        imgsz=YOLO_CONFIG["imgsz"],
        patience=YOLO_CONFIG["patience"],
        device=YOLO_CONFIG["device"],
        project=os.path.join(ROOT, "runs"),
        name="camouflage_detection",
        exist_ok=True,
    )

    # Export best model
    best_pt = os.path.join(ROOT, "runs", "camouflage_detection", "weights", "best.pt")
    if os.path.exists(best_pt):
        import shutil
        dst = os.path.join(MODELS_DIR, "best.pt")
        shutil.copy2(best_pt, dst)
        print(f"\nBest model saved to: {dst}")

    print("\nTraining complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
