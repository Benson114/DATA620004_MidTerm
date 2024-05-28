# DATA620004-MidTerm-Task1

## Introduction

本项目为复旦大学研究生课程DATA620004——神经网络和深度学习期中作业Task1的代码仓库

## Requirements

```bash
pip install -r requirements.txt
```

## How to Run

### 数据下载

在代码仓库下创建目录`data/`，从[CUB-200-2011]( https://data.caltech.edu/records/65de6-vp158)下载CUB-200-2011数据集，放到`data/`目录下，随后在命令行运行`TarData.py`

```bash
python TarData.py
```

### 模型训练

在`config.py`内调整配置参数，在三个训练脚本内分别调整训练超参数，运行即可

```bash
# 从全模型参数初始化开始训练
python train_from_init.py
# 从预训练权重开始训练，预训练权重和新全连接层分阶段训练
python train_stage.py
# 从预训练权重开始训练，预训练权重和新全连接层同时训练
python train_joint.py
```

