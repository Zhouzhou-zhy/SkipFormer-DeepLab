from samba import Samba
import torch.nn as nn
from unet_parts import *


class Samba_Unet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        base_cn=32,
    ):
        super(Samba_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, base_cn)
        self.samba_backbone = Samba(
            in_chans=n_channels,
            embed_dims=[base_cn * 2, base_cn * 4, base_cn * 8, base_cn * 16],
            depths=[3, 4, 6, 3],
        )
        factor = 2 if bilinear else 1
        self.up1 = Up(base_cn * 16, base_cn * 8 // factor, bilinear)
        self.up2 = Up(base_cn * 8, base_cn * 4 // factor, bilinear)
        self.up3 = Up(base_cn * 4, base_cn * 2 // factor, bilinear)
        self.up4 = Up(base_cn * 2, base_cn, bilinear)
        self.outc = OutConv(base_cn, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        features = self.samba_backbone(x)
        x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == "__main__":
    # 1. 实例化 Backbone
    # 默认参数会生成 4 个阶段的特征，通道数分别为 64, 128, 320, 448
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告：Mamba 不支持 CPU 运行，请确保你在 GPU 环境下。")
    backbone = Samba_Unet(n_channels=3, n_classes=2, base_cn=32)
    backbone.to(device)
    # 将模型放入评估模式（如果在推理）或训练模式
    backbone.eval()

    # 2. 构造一个假输入图像 (Batch_Size=2, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 3. 提取特征
    with torch.no_grad():
        features = backbone(dummy_input.cuda())
        print("输出的分割图形状:", features.shape)
