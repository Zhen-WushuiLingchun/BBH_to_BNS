"""
model_resnet_v2.py — 适配 baseline 接口的 1D ResNet
====================================================

与 model_resnet.py (之前版本) 的关键区别:
  1. num_classes=2 (不是 1), 配合 CrossEntropyLoss + softmax[:, 1]
     这样 evaluate_gpu 函数可以直接复用
  2. in_channels=2 (H1+L1 双探测器作为两个通道)
     baseline 的 MyNet 用 Conv2d(1, ..., kernel=(1, K)) 处理 (1, 2, N),
     等价于 Conv1d(2, ..., kernel=K) 处理 (2, N)

输入:  (batch, 2, 8192)   — 2 探测器, 1s × 8192Hz (白化后裁切)
输出:  (batch, 2)          — 2 类 logits
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Basic residual block"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet1D(nn.Module):
    """
    1D ResNet-18 — baseline 接口兼容版

    Parameters
    ----------
    in_channels : int
        输入通道数. H1+L1 → 2.
    num_classes : int
        输出类数. 二分类用 2, 配合 CrossEntropyLoss.
    """

    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()

        # stem
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # residual layers
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, in_channels, length) — 注意不需要额外的维度
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # (batch, 512, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        return x


# ============================================================
# 轻量版 (训练更快, 适合调试)
# ============================================================
class ResNet1D_Small(nn.Module):
    """3 层 ResNet, 参数量约为 ResNet-18 的 1/4"""

    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResBlock1D(32, 64, stride=1), ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    # 模拟 baseline 的输入维度
    # baseline: (batch, 1, 2, 16384) 的 2D tensor
    # 我们: (batch, 2, 8192) 的 1D tensor (裁切到 1s = 8192 点)
    batch_size = 4

    model = ResNet1D(in_channels=2, num_classes=2)
    x = torch.randn(batch_size, 2, 8192)
    y = model(x)
    print(f"ResNet1D: input {x.shape} -> output {y.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    model_s = ResNet1D_Small(in_channels=2, num_classes=2)
    y2 = model_s(x)
    print(f"\nResNet1D_Small: input {x.shape} -> output {y2.shape}")
    print(f"  参数量: {sum(p.numel() for p in model_s.parameters()):,}")

    # 验证 softmax[:, 1] 和 baseline evaluate_gpu 兼容
    softmax = nn.Softmax(dim=-1)
    probs = softmax(y)
    print(f"\nsoftmax output: {probs.shape}")
    print(f"  class 1 probs: {probs[:, 1].detach().numpy()}")
