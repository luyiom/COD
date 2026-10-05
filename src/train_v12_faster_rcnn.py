"""
Train Faster R-CNN on final_v12 (COCO format) using MMDetection.
"""
import os
import sys
import subprocess

ROOT = r"D:\lingma项目\伪装坦克项目2"
DATA_ROOT = os.path.join(ROOT, "data", "final_v12_coco")
WORK_DIR = os.path.join(ROOT, "runs", "v12_models", "faster_rcnn_v12")

# MMDetection training config as Python code
TRAIN_SCRIPT = f"""
import os
import sys
sys.path.insert(0, r'{ROOT}')

from mmengine.config import Config
from mmdet.apis import init_detector, train_detector

# Build config from scratch
from mmdet.models import FasterRCNN
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmdet.models.dense_heads import RPNHead
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules import (
    AnchorGenerator, BBoxAssigner, BBoxSampler, BBoxCoder,
)

# Use built-in config as base
from mmdet.configs.faster_rcnn.faster-rcnn_r50_fpn_1x_coco import (
    model as base_model,
    train_pipeline, test_pipeline,
    train_dataloader as base_train_loader,
    val_dataloader as base_val_loader,
    optim_wrapper, param_scheduler,
    default_hooks, train_cfg, val_cfg, test_cfg,
    load_from,
)

# Override for single class
model = base_model
model.roi_head.bbox_head.num_classes = 1

# Data
data_root = r'{DATA_ROOT}'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=dict(classes=('military_vehicle',)),
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        metainfo=dict(classes=('military_vehicle',)),
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'val/annotations.json'),
    metric='bbox',
)

test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# LR schedule
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[60, 80], gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False

# Build full config
cfg = Config(dict(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    val_evaluator=val_evaluator,
    test_evaluator=test_evaluator,
    optim_wrapper=optim_wrapper,
    param_scheduler=param_scheduler,
    train_cfg=train_cfg,
    val_cfg=val_cfg,
    test_cfg=test_cfg,
    default_hooks=default_hooks,
    load_from=load_from,
    resume=resume,
    work_dir=r'{WORK_DIR}',
    env_cfg=dict(cudnn_benchmark=True),
    vis_backends=[dict(type='LocalVisBackend')],
    visualizer=dict(type='DetLocalVisualizer'),
))
cfg.dump(os.path.join(r'{WORK_DIR}', 'config.py'))

print('Config built, starting training...')
print('Work dir: {WORK_DIR}')

train_detector(model, [cfg.train_dataloader], cfg, cfg.train_cfg, cfg.optim_wrapper, cfg.param_scheduler, cfg.val_cfg, [cfg.val_dataloader], [cfg.val_evaluator])
print('Training done!')
"""


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=" * 60)
    print("Training Faster R-CNN on final_v12")
    print(f"Work dir: {WORK_DIR}")
    print("=" * 60)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    p = subprocess.Popen([sys.executable, "-c", TRAIN_SCRIPT], env=env, cwd=ROOT)
    p.wait()
    if p.returncode == 0:
        print("Training complete!")
    else:
        print(f"Failed with exit code {p.returncode}")


if __name__ == "__main__":
    main()
