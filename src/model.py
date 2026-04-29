from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV3UNet(nn.Module):
    """MobileNetV3-small encoder with a light UNet decoder for binary masks."""

    def __init__(
        self,
        pretrained: bool = False,
        decoder_channels: Tuple[int, int, int, int] = (128, 64, 32, 16),
        dropout: float = 0.0,
    ):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights).features

        # For 384x384 input:
        # x0: 192x192, 16 channels
        # x1:  96x96, 16 channels
        # x2:  48x48, 24 channels
        # x3:  24x24, 48 channels
        # x4:  12x12, 576 channels
        self.enc0 = backbone[0]
        self.enc1 = backbone[1]
        self.enc2 = nn.Sequential(backbone[2], backbone[3])
        self.enc3 = nn.Sequential(backbone[4], backbone[5], backbone[6], backbone[7], backbone[8])
        self.enc4 = nn.Sequential(backbone[9], backbone[10], backbone[11], backbone[12])

        c3, c2, c1, c0 = decoder_channels
        self.dec3 = DecoderBlock(576, 48, c3)
        self.dec2 = DecoderBlock(c3, 24, c2)
        self.dec1 = DecoderBlock(c2, 16, c1)
        self.dec0 = DecoderBlock(c1, 16, c0)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

    def freeze_encoder(self) -> None:
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)

        logits = self.head(self.dropout(d0))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = MobileNetV3UNet(pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
