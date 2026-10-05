import os

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source datasets
DATASETS = {
    "mixed": os.path.join(ROOT, "datasets", "raw", "混合数据集1"),
    "tank": os.path.join(ROOT, "datasets", "raw", "坦克数据集"),
    "tank2": os.path.join(ROOT, "datasets", "raw", "坦克数据集2"),
    "camouflage": os.path.join(ROOT, "datasets", "raw", "伪装坦克数据集", "tank_dataset"),
    "armored": os.path.join(ROOT, "datasets", "raw", "装甲车数据集"),
}

# Output directories
DATA_CLEANED = os.path.join(ROOT, "data", "cleaned")
DATA_MERGED = os.path.join(ROOT, "data", "merged")
DATA_FINAL = os.path.join(ROOT, "data", "final_v8")
MODELS_DIR = os.path.join(ROOT, "models")

# Class names
CLASS_NAMES = {
    0: "military_vehicle",
}

# Blur detection threshold (Laplacian variance)
BLUR_THRESHOLD = 100

# Train/val/test split ratios
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# YOLO training params
YOLO_CONFIG = {
    "model": os.path.join(MODELS_DIR, "yolov8n.pt"),
    "epochs": 100,
    "batch": 8,  # CPU training: smaller batch
    "imgsz": 640,
    "patience": 20,
    "device": "cpu",
}

# CLIP model for camouflage assessment
CLIP_MODEL = "openai/clip-vit-base-patch32"
