"""A compact U-Net for binary talc segmentation.

Small on purpose: the talc dataset only has a few dozen annotated panoramas, so a
lightweight encoder/decoder (base width 32) with batch-norm generalises better
than a heavy backbone would on this many images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-18 encoder.

    A pretrained encoder is the decisive choice on this tiny (~40 image) talc
    dataset: its low-level texture filters transfer directly and stop the decoder
    from collapsing to a "predict one big blob" solution, which a from-scratch
    U-Net does here. Input side length must be a multiple of 32.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)

        self.input_block = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 64, /2
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1  # 64,  /4
        self.layer2 = encoder.layer2  # 128, /8
        self.layer3 = encoder.layer3  # 256, /16
        self.layer4 = encoder.layer4  # 512, /32

        self.up4 = _UpBlock(512, 256, 256)  # /16
        self.up3 = _UpBlock(256, 128, 128)  # /8
        self.up2 = _UpBlock(128, 64, 64)    # /4
        self.up1 = _UpBlock(64, 64, 64)     # /2
        self.up0 = _UpBlock(64, 0, 32)      # /1
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.input_block(x)   # 64,  /2
        x1 = self.layer1(self.pool(x0))  # 64,  /4
        x2 = self.layer2(x1)       # 128, /8
        x3 = self.layer3(x2)       # 256, /16
        x4 = self.layer4(x3)       # 512, /32

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1)
        return self.head(d0)
