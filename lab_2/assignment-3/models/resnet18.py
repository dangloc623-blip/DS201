import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_1x1conv: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, use_1x1conv=False),
            ResidualBlock(64, 64, use_1x1conv=False)
        )

        self.pool_between_1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, use_1x1conv=True),
            ResidualBlock(128, 128, use_1x1conv=False)
        )

        self.pool_between_2_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, use_1x1conv=True),
            ResidualBlock(256, 256, use_1x1conv=False)
        )

        self.pool_between_3_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, use_1x1conv=True),
            ResidualBlock(512, 512, use_1x1conv=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.stage1(x)
        x = self.pool_between_1_2(x)

        x = self.stage2(x)
        x = self.pool_between_2_3(x)

        x = self.stage3(x)
        x = self.pool_between_3_4(x)

        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits
