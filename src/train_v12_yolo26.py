"""
Train YOLO26s on final_v12 dataset.
Windows-optimized: runs train() in a subprocess to isolate CUDA,
allowing workers>0 without DataLoader deadlock.
"""
import os
import sys
import subprocess
import multiprocessing



ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "final_v12", "data.yaml")
MODEL_PATH = os.path.join(ROOT, "models", "yolo26s.pt")
PROJECT = os.path.join(ROOT, "runs", "v12_models")
NAME = "yolo26s_v12"


TRAIN_CODE = r"""
import os, sys
import torch
# 必须在子进程代码的第一行禁用 cuDNN
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False

sys.path.insert(0, r'{root}/src')
from ultralytics import YOLO

model = YOLO(r'{model_path}')
results = model.train(
    data=r'{data_yaml}',
    epochs=100,
    patience=30,
    batch=16, # =16
    imgsz=640,
    device=0,
    workers=4,

    lr0=0.008,
    lrf=0.001,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

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
    amp=True, # True
    cache=False,

    save=True,
    save_period=10,
    exist_ok=True,
    project=r'{project}',
    name=r'{name}',
)
print("DONE")
"""


def main():
    print("=" * 60)
    print("Training: YOLO26s on final_v12 (workers=4, subprocess)")
    print(f"Data: {DATA_YAML}")
    print(f"Output: {PROJECT}/{NAME}")
    print("=" * 60)

    code = TRAIN_CODE.format(
        root=ROOT,
        model_path=MODEL_PATH,
        data_yaml=DATA_YAML,
        project=PROJECT,
        name=NAME,
    )

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    p = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        cwd=ROOT,
    )
    p.wait()

    if p.returncode == 0:
        print(f"\nTraining complete!")
        print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")
    else:
        print(f"\nTraining failed with exit code {p.returncode}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
