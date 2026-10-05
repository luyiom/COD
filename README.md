# 伪装坦克 / 军事车辆目标检测项目

基于 YOLO 的**伪装与军事车辆目标检测**系统。包含：原始数据采集、标注转换、数据集构建、模型训练，以及配套的检测 Web 服务（FastAPI 后端 + Vue 前端）和 AI 伪装等级评估（Qwen 多模态模型）。

> 代码由 AI 辅助编写，数据集由项目所有者人工采集与筛选。

---

## 目录结构

```
伪装坦克项目2/
├── README.md                 # 本文件：项目总览与说明
├── requirements.txt          # Python 依赖
├── start.bat                 # 一键启动检测服务（端口 8000）
├── .env                      # 环境变量（含 Qwen API Key，勿外泄）
├── datasets/                 # 所有数据集（核心资产）
│   ├── raw/                  # 原始采集数据集（已人工筛选）
│   │   ├── 混合数据集1/                      # 7113 张（主数据，Roboflow military-vehicle）
│   │   ├── 坦克数据集/                        # 3172 张（Roboflow tank-detection）
│   │   ├── 坦克数据集2/                       # 1279 张（Roboflow onyx）
│   │   ├── 装甲车数据集/                      #  978 张（Roboflow vehicles）
│   │   ├── 伪装坦克数据集/                    #  121 张（含 tank_dataset 子目录）
│   │   └── 挑过Military-Camouflage-MHCD2022/ #  336 张（筛选后的公开 MHCD2022，VOC/XML 格式）
│   └── labelme/              # 由原始数据生成的 LabelMe 标注中间产物
│       ├── labelme_output/   # 全量标注（json + jpg + xml）
│       └── cleanshujuji/     # 清洗后的子集（构建最终训练集的源）
├── data/                     # 最终训练集（YOLO 格式，由构建脚本生成）
│   ├── final_v4 ~ final_v12  # 各版本训练集（类别见各自 data.yaml）
│   ├── augmented/ cleaned/ merged/  # 数据增强 / 清洗 / 合并中间产物
├── src/                      # 全部 Python 代码
│   ├── config.py             # 路径与训练超参数配置（ROOT 自动推导项目根）
│   ├── build_v9/v10/v12_dataset.py , build_final_dataset.py  # 各版本数据集构建
│   ├── train_v9/v10/v11/v12_yolo26.py , train_v12_faster_rcnn.py , train_v12_torchvision.py  # 训练脚本
│   ├── data_*.py , data_augment.py  # 数据清洗 / 增强 / 合并 / 准备
│   ├── camouflage_agent.py , qwen_agent.py  # 伪装等级评估 Agent
│   ├── convert_v12_to_coco.py , fix_coco_labels.py  # COCO 格式转换
│   └── tools/                # 零散工具脚本
│       ├── convert_*_to_labelme.py  # YOLO ↔ LabelMe 格式转换
│       └── visualize_dataset.py     # 数据集可视化抽查
├── app/                      # FastAPI 检测后端（main.py + static + templates）
├── frontend/                 # Vue3 + Vite 前端
├── models/                   # 模型权重
│   ├── yolo26n.pt yolo26s.pt yolov8n.pt  # 预训练基座
│   └── best.pt best_v10.pt yolo26s_v12.pt camouflage_v2.pt last.pt  # 训练产出
└── runs/                     # 训练过程输出（检查点、曲线图、CSV）
```

---

## 数据集说明

### 原始数据集（`datasets/raw/`）
| 数据集 | 图片数 | 说明 |
|---|---|---|
| 混合数据集1 | 7113 | Roboflow 军事车辆检测，主力数据源 |
| 坦克数据集 | 3172 | Roboflow tank-detection-medgr |
| 坦克数据集2 | 1279 | Roboflow onyx-rrnib |
| 装甲车数据集 | 978 | Roboflow vehicles-xhdpx |
| 伪装坦克数据集 | 121 | 伪装坦克（内部含 `tank_dataset` 子目录） |
| 挑过Military-Camouflage-MHCD2022 | 336 | 公开 MHCD2022 数据集中人工筛选的部分（VOC/XML 格式） |

### 最终训练集（`data/`）
- **`final_v12`**（当前主版本，单类 `military_vehicle`）：由 `src/build_v12_dataset.py` 从 `datasets/labelme/cleanshuji` 构建。
- `final_v11` / `final_v9`：单类军事车辆。
- `final`：4 类伪装分级 —— `no_camouflage / light_camouflage / medium_camouflage / heavy_camouflage`。
- `augmented`：含雾/雨增强的二分类集（`no_camouflage / camouflaged`）。

---

## 环境与运行

1. 安装依赖：`pip install -r requirements.txt`
2. 启动检测服务：双击 `start.bat`，或执行
   `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
   浏览器打开 http://localhost:8000
3. 前端（可选）：`cd frontend && npm install && npm run dev`

---

## 训练流程

1. （如需重建数据集）运行 `src/build_v12_dataset.py` 等构建脚本，生成 `data/final_vXX`。
2. 训练：`python src/train_v12_yolo26.py`（或其它版本训练脚本）。

---

## 整理记录（本次）

- 原 `z数据集备份` 经全量比对，是 `datasets/labelme/labelme_output` 的冗余副本（仅 3 个旧版标注 JSON 不同），已删除以节省约 **806 MB**。
- 根目录零散脚本 `convert_*.py`、`visualize_dataset.py` 已移入 `src/tools/`；根目录 3 个预训练权重已移入 `models/`。
- 已统一修正所有脚本中对数据集 / 模型路径的硬编码引用（统一基于项目根 `ROOT` 推导）。
- 修复 `src/train_v12_faster_rcnn.py` 中一处 f-string 语法错误（原 AI 生成笔误，会导致该脚本无法运行）。
- `data/*.yaml` 中的绝对路径已规范为相对路径 `.`，便于在其他机器上训练。

---

## 注意事项

- **`.env` 含 Qwen API Key，对外分享时请勿一并打包或泄露。**
- `datasets/raw/` 是你的原始资产，请勿随意删除。
- 脚本中的 `ROOT` 由文件位置自动推导项目根目录，正常情况下无需手动修改。
- `runs/` 为训练产物（含检查点），体积较大，按需保留或清理。
