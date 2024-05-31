# DATA620004-MidTerm-Task2

## Introduction

本项目为复旦大学研究生课程DATA620004——神经网络和深度学习期中作业Task2的代码仓库

## Requirements

```bash
pip install -r requirements.txt
```

## How to Run

### 数据要求

请把VOC2007和VOC2012下载到 `./vocdata`路径下，然后把VOC2007的内容放在 `./vocdata/VOC2007`下面，把VOC2012的内容放在 `./vocdata/VOC2012`下面。

如果你的数据集与本项目不适配，你可以通过如下路径做进一步设置：

`.configs/`

### 模型训练

你可以通过下面两个命令分别训练Faster R-CNN和YOLOv3

```bash
# 训练Faster R-CNN
bash train_faster.sh
# 训练YOLOv3
bash train_yolov3.sh
```

如果你需要进一步设置训练参数，你可以修改这两个文件：

1. 在 `./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco_lzf.py`内调整Faster R-CNN的模型和优化器参数，在 `.configs/_base_/datasets/voc0712_lzf.py`内调整batch size
2. 在 `./configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py`内调整YOLOv3的模型和优化器参数，在 `.configs/_base_/datasets/voc0712_lzf.py`内调整batch size

### 模型部署

您可以通过下面的命令，使用Faster R-CNN和YOLOv3两个模型进行目标检测的预测：

```bash
bash prediction.sh
```

您需要在这个文件中修改图像路径和输出路径。

### 模型测评

您可以通过下面两个命令分别对Faster R-CNN和YOLOv3进行测评：

```bash
# 评测Faster R-CNN
bash test_faster.sh
# 评测YOLOv3
bash test_yolov3.sh
```

请您将路径修改为权重的存储路径。
