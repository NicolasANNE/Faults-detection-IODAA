"""Residual U-Net for single-channel fault segmentation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        group_norm_groups: int,
        dropout: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(group_norm_groups, out_channels)
        self.activation1 = activation()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(group_norm_groups, out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation2 = activation()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.activation1(self.norm1(self.conv1(inputs)))
        outputs = self.dropout(outputs)
        outputs = self.norm2(self.conv2(outputs))
        return self.activation2(outputs + residual)


class Unet_base(nn.Module):
    """Four-stage residual U-Net with GroupNorm and bilinear upsampling."""

    def __init__(
        self,
        in_channels: int = 19,
        out_channels: int = 1,
        base_channels: int = 32,
        group_norm_groups: int = 8,
        dropout: float = 0.2,
        prevalence: float = 0.0118,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        activations: dict[str, type[nn.Module]] = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        activation_layer = activations[activation.lower()]
        channels = (base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels)

        self.encoder1 = _ResidualConvBlock(
            in_channels,
            channels[0],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.encoder2 = _ResidualConvBlock(
            channels[0],
            channels[1],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.encoder3 = _ResidualConvBlock(
            channels[1],
            channels[2],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.bottleneck = _ResidualConvBlock(
            channels[2],
            channels[3],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up3 = nn.Conv2d(channels[3], channels[2], kernel_size=3, padding=1)
        self.decoder3 = _ResidualConvBlock(
            2 * channels[2],
            channels[2],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.up2 = nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1)
        self.decoder2 = _ResidualConvBlock(
            2 * channels[1],
            channels[1],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.up1 = nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1)
        self.decoder1 = _ResidualConvBlock(
            2 * channels[0],
            channels[0],
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            activation=activation_layer,
        )
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        initial_bias = math.log(prevalence / (1 - prevalence))
        nn.init.constant_(self.output.bias, initial_bias)

    @staticmethod
    def _decode(
        outputs: torch.Tensor,
        skip: torch.Tensor,
        projection: nn.Module,
        block: nn.Module,
    ) -> torch.Tensor:
        outputs = F.interpolate(
            outputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs = projection(outputs)
        return block(torch.cat((outputs, skip), dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        outputs = self.bottleneck(self.pool(encoder3))
        outputs = self._decode(outputs, encoder3, self.up3, self.decoder3)
        outputs = self._decode(outputs, encoder2, self.up2, self.decoder2)
        outputs = self._decode(outputs, encoder1, self.up1, self.decoder1)
        return self.output(outputs)
