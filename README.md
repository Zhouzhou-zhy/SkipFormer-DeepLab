<div align="center">

# SkipFormer-DeepLab

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**多模型图像语义分割研究框架** | Multi-model semantic segmentation research framework

基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 扩展，集成了 SkipFormer-DeepLab、SwinUnet、Transunet、Samba-UNet 和 ViT-UNet 等前沿架构。

</div>

---

## 支持的模型 | Supported Models

| 目录                  | 模型            | 构造方式                                                                                |
| --------------------- | --------------- | --------------------------------------------------------------------------------------- |
| `UNet/`               | UNet            | `UNet(n_channels, n_classes, bilinear=False)`                                           |
| `vit_unet/`           | ViT-UNet        | `Vit_Unet(n_channels, n_classes, bilinear=False, vit_depth=6, vit_heads=8, base_cn=32)` |
| `Swin_Unet/`          | SwinUnet        | `SwinUnet(img_size=256, n_classes=7, n_channels=3, zero_head=False, vis=False)`         |
| `Transunet/`          | TransUnet       | `get_transNet(n_classes)`                                                               |
| `UNet_Samba/`         | Samba-UNet      | `Samba_Unet(n_channels, n_classes, bilinear=False, base_cn=32)`                         |
| `SkipFormer_DeepLab/` | DeepLabV3+ 变体 | 工厂函数（见下表）                                                                      |

### SkipFormer_DeepLab 变体

| 工厂函数                                      | 骨干网络                |
| --------------------------------------------- | ----------------------- |
| `deeplabv3_resnet50(num_classes=8, ...)`      | **实际使用 ResNet-101** |
| `deeplabv3plus_mobilevit(num_classes=8, ...)` | MobileViT               |
| `deeplabv3plus_mvit_unet(num_classes=8)`      | ViT-UNet                |
| `deeplabv3_xception(num_classes=21, ...)`     | Xception                |

---

## 环境要求 | Requirements

- **PyTorch** (推荐容器：`nvcr.io/nvidia/pytorch:22.11-py3`)
- **CUDA**（Samba_Unet 需要 GPU）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 数据集准备 | Dataset

本项目主要面向 **LoveDA** 遥感图像语义分割任务。

在 `train.py:28-30` 中修改数据路径：

```python
dir_img = Path("/path/to/train/images")
dir_mask = Path("/path/to/train/labels")
dir_checkpoint = Path("/path/to/checkpoints")
```

数据集通过 `utils/data_loading.py` 中的 `BasicDataset` 加载，该模块会自动扫描所有掩码文件以确定类别值。

---

## 训练 | Training

```bash
# 训练 U-Net（默认）
python train.py --epochs 50 --batch-size 11 --classes 8 --amp

# 使用其他模型：修改 train.py 第 315-319 行的注释
```

可配置参数：

| 参数           | 说明             | 默认值 |
| -------------- | ---------------- | ------ |
| `--epochs`     | 训练轮数         | 50     |
| `--batch-size` | 批次大小         | 11     |
| `--classes`    | 类别数           | 8      |
| `--amp`        | 混合精度训练     | False  |
| `--bilinear`   | 双线性上采样     | False  |
| `--load`       | 从检查点恢复训练 | None   |
| `--scale`      | 图像缩放比例     | 0.5    |

### 训练策略

- **优化器**: AdamW (`lr=1e-5`, `weight_decay=0.05`)
- **学习率调度器**: OneCycleLR (`max_lr=1e-3`, `pct_start=0.3`)
- **损失函数**: CrossEntropyLoss + Dice 损失
- **评估指标**: mIoU、mF1、OA（整体精度）

---

## 预测 | Prediction

```bash
# 单图预测（类别数为 3）
python predict.py -m checkpoints/model.pth -i ./input_images --classes 3

# 多类别预测可视化
python multipredict.py -m checkpoints/model.pth -i ./input_images --classes 3
```

---

## 评估指标 | Evaluation

训练过程中使用 `evaluate.py:compute_miou()` 进行验证，计算以下指标：

- **mIoU** (mean Intersection over Union)：所有类别的平均交并比
- **mF1** (mean F1 Score)：所有类别的平均 F1 分数
- **OA** (Overall Accuracy)：整体像素分类精度
- **类 IoU**：每个类别的单独 IoU 值

---

## 项目结构 | Project Structure

```
SkipFormer-DeepLab/
├── UNet/                    # 标准 U-Net
├── vit_unet/                # ViT-UNet（编码器最深一层融合 ViT）
├── Swin_Unet/               # Swin Transformer U-Net
├── Transunet/               # TransUnet
├── UNet_Samba/              # Samba 状态空间模型 U-Net（需 CUDA）
├── SkipFormer_DeepLab/      # DeepLabV3+ 多骨干网络
│   └── backbone/            #   resnet, xception, mobilevit, mvit_unet
├── utils/                   # 数据加载、Dice 损失、工具函数
├── train.py                 # 训练脚本（主入口）
├── predict.py               # 单图像推理
├── multipredict.py           # 多类别批量推理与可视化
├── evaluate.py              # mIoU、mF1、OA 评估
├── Dockerfile               # Docker 构建文件
└── requirements.txt         # 依赖列表
```

---

## Docker

```bash
docker build -t skipformer-deeplab .
docker run --gpus all -v /path/to/data:/data skipformer-deeplab
```

基于 `nvcr.io/nvidia/pytorch:22.11-py3`，已预装 PyTorch 及 CUDA 工具链。

---

## 致谢 | Acknowledgments

本项目基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 构建，并集成了以下工作的实现：

- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Mamba](https://github.com/state-spaces/mamba)
- [Deeplab](https://github.com/VainF/DeepLabV3Plus-Pytorch)

## 许可证 | License

[GNU General Public License v3.0](LICENSE)
