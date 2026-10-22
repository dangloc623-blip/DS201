import torch
from torch import nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_3a = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.inception_3b = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_4a = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.inception_4b = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.inception_4c = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.inception_4d = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.inception_4e = InceptionBlock(528, 256, (160, 320), (32, 128), 128)
        self.pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_5a = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.inception_5b = InceptionBlock(832, 384, (192, 384), (48, 128), 128)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.pool_2(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool_3(x)

        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pool_4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
