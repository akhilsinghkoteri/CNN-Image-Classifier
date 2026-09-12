"""
model.py
--------
Defines a custom Convolutional Neural Network for CIFAR-10 classification.

Architecture:
    [Conv-BN-ReLU-Conv-BN-ReLU-Pool] x 3  ->  Flatten -> FC -> Dropout -> FC (10 classes)

This is intentionally a "from scratch" CNN (no pretrained weights) so the
project demonstrates understanding of core building blocks: convolutions,
batch normalization, pooling, dropout, and fully connected layers.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers + batch norm + ReLU, followed by max pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    Custom CNN for 32x32x3 CIFAR-10 images -> 10 class logits.

    Input:  (batch, 3, 32, 32)
    Output: (batch, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 32x32 -> 16x16
            ConvBlock(32, 64),   # 16x16 -> 8x8
            ConvBlock(64, 128),  # 8x8   -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model
    model = CIFAR10CNN()
    dummy_input = torch.randn(8, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: torch.Size([8, 10])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
