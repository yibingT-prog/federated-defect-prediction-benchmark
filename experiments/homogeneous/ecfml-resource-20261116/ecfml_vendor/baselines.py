from __future__ import annotations

import torch
from torch import nn

from .model import make_activation


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = make_activation(activation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(out + identity)


class ResNet18Tabular(nn.Module):
    """ResNet-18 style 1D adaptation for 61-dimensional tabular metrics."""

    def __init__(self, input_dim: int = 61, num_classes: int = 2, activation: str = "relu") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            make_activation(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, 2, activation=activation)
        self.layer2 = self._make_layer(32, 64, 2, stride=2, activation=activation)
        self.layer3 = self._make_layer(64, 128, 2, stride=2, activation=activation)
        self.layer4 = self._make_layer(128, 256, 2, stride=2, activation=activation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        activation: str = "relu",
    ) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride, activation=activation)]
        layers.extend(BasicBlock1D(out_channels, out_channels, activation=activation) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.fc(self.pool(x).squeeze(-1))


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            make_activation(activation),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet1DTabular(nn.Module):
    """Compact U-Net style 1D adaptation for tabular vectors."""

    def __init__(self, input_dim: int = 61, num_classes: int = 2, activation: str = "relu") -> None:
        super().__init__()
        self.enc1 = ConvBlock1D(1, 32, activation)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(32, 64, activation)
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock1D(64, 128, activation)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(128, 64, activation)
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(64, 32, activation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = crop_or_pad(dec2, enc2.shape[-1])
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = crop_or_pad(dec1, enc1.shape[-1])
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.fc(self.pool(dec1).squeeze(-1))


def crop_or_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def build_baseline(name: str, input_dim: int = 61, num_classes: int = 2, activation: str = "relu") -> nn.Module:
    name = name.lower()
    if name in {"resnet", "resnet18", "resnet-18"}:
        return ResNet18Tabular(input_dim=input_dim, num_classes=num_classes, activation=activation)
    if name in {"unet", "u-net", "unet1d"}:
        return UNet1DTabular(input_dim=input_dim, num_classes=num_classes, activation=activation)
    raise ValueError("baseline name must be resnet18 or unet.")
