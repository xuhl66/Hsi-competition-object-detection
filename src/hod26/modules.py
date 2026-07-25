from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SelfSensingRegrouping(nn.Module):
    """
    Differentiable version of HYPER-DET SSRM.

    Three image-conditioned spectral groups produce the RGB-compatible branch.
    The initialization follows separated visible wavelengths while training can
    move each group to the most discriminative bands.
    """

    def __init__(self, bands: int = 16, groups: int = 3, temperature: float = 0.7):
        super().__init__()
        hidden = max(4, bands // 4)
        self.bands = bands
        self.groups = groups
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(bands, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, groups * bands),
        )
        self.group_logits = nn.Parameter(torch.full((groups, bands), -4.0))
        initial_bands = [13, 8, 5] if groups == 3 and bands == 16 else [
            round(index * (bands - 1) / max(groups - 1, 1))
            for index in range(groups)
        ]
        with torch.no_grad():
            for group, center in enumerate(initial_bands):
                positions = torch.arange(bands)
                self.group_logits[group].copy_(
                    -0.5 * ((positions - center).float() / 1.5) ** 2
                )
        self.last_attention: torch.Tensor | None = None

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        descriptor = cube.mean(dim=(2, 3))
        dynamic = self.mlp(descriptor).view(-1, self.groups, self.bands)
        logits = dynamic + self.group_logits.unsqueeze(0)
        attention = torch.softmax(logits / self.temperature, dim=-1)
        self.last_attention = attention
        return torch.einsum("bgc,bhwc->bghw", attention, cube.permute(0, 2, 3, 1))

    def diversity_loss(self) -> torch.Tensor:
        if self.last_attention is None:
            return self.group_logits.sum() * 0.0
        attention = self.last_attention
        gram = attention @ attention.transpose(1, 2)
        diagonal = torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        off_diagonal = gram - diagonal
        return off_diagonal.square().mean()


class SpectralSpatialStage(nn.Module):
    """Parallel spatial and spectral depthwise 3-D convolution stage."""

    def __init__(self, in_channels: int, out_channels: int, spectral_stride: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(spectral_stride, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.spatial = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=out_channels,
            bias=False,
        )
        self.spectral = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=out_channels,
            bias=False,
        )
        self.mix = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.down(inputs)
        features = self.mix(self.spatial(inputs) + self.spectral(inputs))
        return self.activation(inputs + features)


class Projection2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.project(inputs.mean(dim=2))


class HyperspectralPyramid(nn.Module):
    """Five-stage HYPER-DET HSI branch returning P2/P3/P4/P5."""

    def __init__(
        self,
        output_channels: Sequence[int] = (256, 512, 512, 512),
    ):
        super().__init__()
        channels = (8, 16, 32, 64, 128)
        self.stage1 = SpectralSpatialStage(1, channels[0], spectral_stride=2)
        self.stage2 = SpectralSpatialStage(channels[0], channels[1], spectral_stride=2)
        self.stage3 = SpectralSpatialStage(channels[1], channels[2], spectral_stride=2)
        self.stage4 = SpectralSpatialStage(channels[2], channels[3], spectral_stride=2)
        self.stage5 = SpectralSpatialStage(channels[3], channels[4], spectral_stride=1)
        self.projections = nn.ModuleList(
            [
                Projection2d(channels[1], output_channels[0]),
                Projection2d(channels[2], output_channels[1]),
                Projection2d(channels[3], output_channels[2]),
                Projection2d(channels[4], output_channels[3]),
            ]
        )

    def forward(self, cube: torch.Tensor) -> list[torch.Tensor]:
        features = cube.unsqueeze(1)
        features = self.stage1(features)
        stage2 = self.stage2(features)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        return [
            projection(feature)
            for projection, feature in zip(
                self.projections, (stage2, stage3, stage4, stage5), strict=True
            )
        ]


class PyramidSelect(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        return pyramid[self.index]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(inputs, 1)
        maximum = F.adaptive_max_pool2d(inputs, 1)
        return torch.sigmoid(self.network(average + maximum))


class HomogeneousUniqueFusion(nn.Module):
    """HYPER-DET HUFF: homogeneous weighting plus unique residual extraction."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.rgb_unique = ChannelAttention(channels, reduction)
        self.hsi_unique = ChannelAttention(channels, reduction)
        self.branch_logits = nn.Parameter(torch.zeros(2))
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        rgb, hsi = inputs
        if rgb.shape != hsi.shape:
            raise ValueError(f"HUFF shape mismatch: {rgb.shape} vs {hsi.shape}")
        joint = rgb + hsi
        gate = torch.sigmoid(self.local_gate(joint) + self.global_gate(joint))
        homogeneous = rgb * gate + hsi * (1.0 - gate)
        unique = rgb * self.rgb_unique(rgb) - hsi * self.hsi_unique(hsi)
        weights = torch.softmax(self.branch_logits, dim=0)
        fused = weights[0] * homogeneous + weights[1] * unique
        return self.output(fused + 0.5 * joint)


class ObjectAwareRefinement(nn.Module):
    """OSSDet-style object activation branch retained as train-time supervision."""

    def __init__(self, channels: int, gain: float = 0.75):
        super().__init__()
        hidden = max(32, channels // 4)
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        initial = math.log(math.expm1(max(gain, 1e-3)))
        self.raw_gain = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.last_logits: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.mask_head(features)
        self.last_logits = logits
        gain = F.softplus(self.raw_gain)
        return features * (1.0 + gain * logits.sigmoid())
