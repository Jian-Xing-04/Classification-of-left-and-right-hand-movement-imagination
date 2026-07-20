import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs

    输入形状: (batch_size, 1, n_channels, n_timepoints)
    对应加载器输出: (n_samples, 14, 256) → 需 unsqueeze(1) 变为 (batch, 1, 14, 256)
    """

    def __init__(self, n_channels=config.EEGNET_N_CHANNELS,
                 n_timepoints=config.EEGNET_N_TIMEPOINTS,
                 n_classes=config.EEGNET_N_CLASSES,
                 F1=config.EEGNET_F1, D=config.EEGNET_D, F2=config.EEGNET_F2,
                 kernel1=config.EEGNET_KERNEL1, dropout=config.EEGNET_DROPOUT):
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # Block 1: 时序卷积
        self.conv1 = nn.Conv2d(1, F1, (1, kernel1), padding=(0, kernel1 // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: 深度可分离卷积 (空间滤波 + 逐点卷积)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.dropout2 = nn.Dropout(dropout)

        # Block 3: 可分离卷积
        self.conv3_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False,
                                     padding=(0, 8))
        self.conv3_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.dropout3 = nn.Dropout(dropout)

        # 分类头
        self.flatten_size = F2 * (n_timepoints // 4 // 8)
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        # x: (batch, 1, C, T)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3_depth(x)
        x = self.conv3_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = EEGNet()
    dummy = torch.randn(4, 1, config.EEGNET_N_CHANNELS, config.EEGNET_N_TIMEPOINTS)
    out = model(dummy)
    print(f"输入: {dummy.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
