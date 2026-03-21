from __future__ import annotations

import torch
from torch import nn


class StudentBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv(images)
        pooled = self.pool(features)
        batch_size, hidden_dim, height, width = pooled.shape
        return pooled.view(batch_size, hidden_dim, height * width).transpose(1, 2)
