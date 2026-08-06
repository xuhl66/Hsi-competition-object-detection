"""High-capacity HSI Co-DINO with OSSDet-style object-aware refinement."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmdet.models.backbones.vit import ViT
from mmdet.models.builder import BACKBONES, DETECTORS, NECKS
from mmdet.models.necks.sfp import SFP
from mmdet.models.necks.sfp import LayerNorm as SFPChannelLayerNorm
from projects.models.co_detr import CoDETR
from torch import nn
from torch.nn import functional as F


class Float32ChannelLayerNorm(nn.Module):
    """AMP-safe per-pixel channel normalization for NCHW feature maps."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first",
    ) -> None:
        super().__init__()
        channels = int(normalized_shape)
        if channels < 1:
            raise ValueError("normalized_shape must be positive")
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"Unsupported data format: {data_format}")
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)
        self.data_format = data_format
        self.normalized_shape = (channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        # The upstream SFP implementation computes channel variance directly
        # in FP16 under autocast. Near-constant spectral features can then
        # produce a finite forward value but a non-finite SubBackward0. Keep
        # the identical affine normalization in FP32 and cast only its output.
        with torch.cuda.amp.autocast(enabled=False):
            value = x.float()
            if self.data_format == "channels_first":
                value = value.permute(0, 2, 3, 1)
            value = F.layer_norm(
                value,
                self.normalized_shape,
                self.weight.float(),
                self.bias.float(),
                self.eps,
            )
            if self.data_format == "channels_first":
                value = value.permute(0, 3, 1, 2).contiguous()
        return value.to(dtype=output_dtype)


def _replace_sfp_layer_norms(module: nn.Module) -> None:
    """Replace upstream FP16-unsafe SFP norms without changing state keys."""

    for name, child in tuple(module.named_children()):
        if isinstance(child, SFPChannelLayerNorm):
            replacement = Float32ChannelLayerNorm(
                child.normalized_shape[0],
                eps=child.eps,
                data_format=child.data_format,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_sfp_layer_norms(child)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            Float32ChannelLayerNorm(out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class SpectralSpatialStage(nn.Module):
    """Spatial downsampling plus local spectral/detail refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.depthwise = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=3,
            groups=out_channels,
        )
        self.mix = ConvNormAct(
            out_channels, out_channels, kernel_size=1, activate=False
        )
        hidden = max(out_channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        residual = self.mix(self.depthwise(x))
        return x + residual * self.channel_gate(residual)


@BACKBONES.register_module()
class HOD26HSIViT(ViT):
    """Co-DINO ViT-L with a native 16-band, multi-resolution second path.

    The pretrained ViT still consumes a three-channel broadband proxy. Its
    linear projection starts as deterministic band averages, while a nonlinear
    residual begins at exactly zero. Native spectra and first differences are
    retained separately for P2--P5 fusion in :class:`HOD26SpectralSFP`.
    """

    def __init__(
        self,
        *args,
        input_bands: int = 16,
        spectral_channels: Sequence[int] = (64, 96, 128, 192),
        **kwargs,
    ) -> None:
        if input_bands != 16:
            raise ValueError("HOD26HSIViT requires the official 16 bands")
        if len(spectral_channels) != 4:
            raise ValueError("spectral_channels must contain P2--P5 widths")
        kwargs["in_chans"] = 3
        super().__init__(*args, **kwargs)
        self.input_bands = int(input_bands)
        self.spectral_channels = tuple(int(x) for x in spectral_channels)

        self.proxy_linear = nn.Conv2d(input_bands, 3, 1, bias=False)
        self.proxy_residual = nn.Sequential(
            nn.Conv2d(input_bands * 2 - 1, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=True,
        )

        channels = self.spectral_channels
        self.spectral_stem = nn.Sequential(
            ConvNormAct(input_bands * 2 - 1, channels[0], stride=2),
            SpectralSpatialStage(channels[0], channels[0], stride=2),
        )
        self.spectral_stages = nn.ModuleList(
            [
                SpectralSpatialStage(channels[0], channels[1], stride=2),
                SpectralSpatialStage(channels[1], channels[2], stride=2),
                SpectralSpatialStage(channels[2], channels[3], stride=2),
            ]
        )
        self._init_hsi_path()

    def _init_hsi_path(self) -> None:
        with torch.no_grad():
            self.proxy_linear.weight.zero_()
            # HOD26 bands are ordered by the 4x4 mosaic position. These broad,
            # overlapping groups deliberately preserve all spectral regions.
            groups = ((10, 16), (5, 11), (0, 6))
            for output, (start, stop) in enumerate(groups):
                self.proxy_linear.weight[output, start:stop, 0, 0] = 1.0 / (
                    stop - start
                )
            final = self.proxy_residual[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def spectral_evidence(x: torch.Tensor) -> torch.Tensor:
        derivative = x[:, 1:] - x[:, :-1]
        return torch.cat((x, derivative), dim=1)

    def broadband_proxy(self, x: torch.Tensor) -> torch.Tensor:
        evidence = self.spectral_evidence(x)
        proxy = self.proxy_linear(x) + self.proxy_residual(evidence)
        return (proxy - self.imagenet_mean) / self.imagenet_std

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.input_bands:
            raise ValueError(
                f"Expected BCHW with {self.input_bands} bands, got {tuple(x.shape)}"
            )
        evidence = self.spectral_evidence(x)
        spectral = self.spectral_stem(evidence)
        spectral_outputs = [spectral]
        for stage in self.spectral_stages:
            spectral = stage(spectral)
            spectral_outputs.append(spectral)

        proxy = self.broadband_proxy(x)
        vit_output = super().forward(proxy)[0]
        return [vit_output, *spectral_outputs]


class SpectralFusionBlock(nn.Module):
    """Zero-start gated detail/context fusion for one feature level."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.detail = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                groups=out_channels,
            ),
            ConvNormAct(out_channels, out_channels, kernel_size=1, activate=False),
        )
        hidden = max(out_channels // 8, 16)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gate = nn.Parameter(torch.zeros(out_channels))

    def forward(self, base: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        spectral = self.project(spectral)
        if spectral.shape[-2:] != base.shape[-2:]:
            spectral = F.interpolate(
                spectral,
                size=base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        detail = self.detail(spectral)
        channel = self.context(torch.cat((base, spectral), dim=1))
        spatial = self.spatial(
            torch.cat(
                (
                    spectral.mean(dim=1, keepdim=True),
                    spectral.amax(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        delta = detail * channel * spatial
        gate = torch.tanh(self.gate).view(1, -1, 1, 1)
        return base + gate * delta


class ObjectActivationMask(nn.Module):
    """Predict the OSSDet low-level object activation mask."""

    def __init__(self, channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            Float32ChannelLayerNorm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ObjectAwareFeatureGate(nn.Module):
    """Identity-start foreground/background modulation driven by a mask."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.output = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.output.weight)

    def forward(
        self, features: torch.Tensor, object_probability: torch.Tensor
    ) -> torch.Tensor:
        if object_probability.shape[-2:] != features.shape[-2:]:
            object_probability = F.interpolate(
                object_probability,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        foreground = features * object_probability
        background = features * (1.0 - object_probability)
        return features + self.output(torch.cat((foreground, background), dim=1))


class CrossSpectralObjectRefinement(nn.Module):
    """CAFR-style bottom-up object and cross-spectral feature propagation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.align_low = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )
        self.high_query = nn.Linear(channels, channels, bias=False)
        self.low_key = nn.Linear(channels, channels, bias=False)
        self.spatial_gate = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            Float32ChannelLayerNorm(channels),
            nn.GELU(),
        )
        self.output = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        high: torch.Tensor,
        refined_low: torch.Tensor,
        object_probability: torch.Tensor,
    ) -> torch.Tensor:
        low = self.align_low(refined_low)
        if low.shape[-2:] != high.shape[-2:]:
            low = F.interpolate(
                low,
                size=high.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        object_probability = F.interpolate(
            object_probability,
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        high_vector = high.mean(dim=(2, 3))
        low_vector = low.mean(dim=(2, 3))
        query = self.high_query(high_vector)
        key = self.low_key(low_vector)
        affinity = torch.softmax(
            query.unsqueeze(2) * key.unsqueeze(1) / math.sqrt(self.channels),
            dim=-1,
        )
        high_weight = (
            torch.sigmoid(torch.bmm(affinity, low_vector.unsqueeze(-1)).squeeze(-1))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        low_weight = (
            torch.sigmoid(
                torch.bmm(affinity.transpose(1, 2), high_vector.unsqueeze(-1)).squeeze(
                    -1
                )
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        spatial = torch.sigmoid(self.spatial_gate(low)) * object_probability
        delta = self.fuse(torch.cat((high * high_weight, low * low_weight), dim=1))
        return high + self.output(delta * spatial)


def build_gaussian_object_targets(
    gt_bboxes: Sequence[torch.Tensor],
    *,
    batch_input_shape: Sequence[int],
    feature_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize box-centred, truncated 2-D Gaussian activation targets."""

    input_height, input_width = (int(value) for value in batch_input_shape)
    feature_height, feature_width = (int(value) for value in feature_shape)
    if min(input_height, input_width, feature_height, feature_width) < 1:
        raise ValueError("Input and feature shapes must be positive")
    targets = torch.zeros(
        (len(gt_bboxes), 1, feature_height, feature_width),
        device=device,
        dtype=dtype,
    )
    scale_x = feature_width / input_width
    scale_y = feature_height / input_height
    with torch.no_grad():
        for batch_index, boxes in enumerate(gt_bboxes):
            for box in boxes.detach().to(device=device, dtype=dtype):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) * 0.5 * scale_x
                center_y = (y1 + y2) * 0.5 * scale_y
                sigma_x = torch.clamp((x2 - x1) * scale_x / 4.0, min=0.75)
                sigma_y = torch.clamp((y2 - y1) * scale_y / 4.0, min=0.75)
                radius_x = int(torch.ceil(3.0 * sigma_x).item())
                radius_y = int(torch.ceil(3.0 * sigma_y).item())
                left = max(0, int(torch.floor(center_x).item()) - radius_x)
                right = min(
                    feature_width,
                    int(torch.floor(center_x).item()) + radius_x + 1,
                )
                top = max(0, int(torch.floor(center_y).item()) - radius_y)
                bottom = min(
                    feature_height,
                    int(torch.floor(center_y).item()) + radius_y + 1,
                )
                if right <= left or bottom <= top:
                    continue
                xs = torch.arange(left, right, device=device, dtype=dtype)
                ys = torch.arange(top, bottom, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                gaussian = torch.exp(
                    -0.5
                    * (
                        ((xx - center_x) / sigma_x).square()
                        + ((yy - center_y) / sigma_y).square()
                    )
                )
                destination = targets[batch_index, 0, top:bottom, left:right]
                targets[batch_index, 0, top:bottom, left:right] = torch.maximum(
                    destination, gaussian
                )
    return targets


@NECKS.register_module()
class HOD26SpectralSFP(SFP):
    """Spectral SFP plus object mask and CAFR-style P2--P5 propagation."""

    def __init__(
        self,
        *args,
        spectral_channels: Sequence[int] = (64, 96, 128, 192),
        object_mask_channels: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        _replace_sfp_layer_norms(self)
        if len(spectral_channels) != 4:
            raise ValueError("spectral_channels must contain P2--P5 widths")
        self.spectral_fusions = nn.ModuleList(
            [
                SpectralFusionBlock(int(channels), self.out_channels)
                for channels in spectral_channels
            ]
        )
        self.object_activation = ObjectActivationMask(
            self.out_channels, hidden_channels=object_mask_channels
        )
        self.object_gate = ObjectAwareFeatureGate(self.out_channels)
        self.object_refinements = nn.ModuleList(
            [CrossSpectralObjectRefinement(self.out_channels) for _ in range(3)]
        )
        self.last_object_logits: torch.Tensor | None = None

    def init_weights(self) -> None:
        super().init_weights()
        # SFP's recursive Xavier initializer also visits the new modules.
        # Restore exact identity for every residual path after it completes.
        self.object_gate.reset_identity()
        for refinement in self.object_refinements:
            refinement.reset_identity()

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 5:
            raise ValueError("HOD26SpectralSFP expects ViT plus native P2--P5 features")
        base = list(super().forward([inputs[0]]))
        if len(base) != 5:
            raise RuntimeError("V4 requires the official five-level P2 SFP")
        for level, fusion in enumerate(self.spectral_fusions):
            base[level] = fusion(base[level], inputs[level + 1])

        object_logits = self.object_activation(base[0])
        self.last_object_logits = object_logits
        object_probability = object_logits.sigmoid()
        base[0] = self.object_gate(base[0], object_probability)
        for level, refinement in enumerate(self.object_refinements, start=1):
            base[level] = refinement(
                base[level],
                base[level - 1],
                object_probability,
            )
        return tuple(base)

    def object_activation_loss(
        self,
        gt_bboxes: Sequence[torch.Tensor],
        *,
        batch_input_shape: Sequence[int],
        background_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the paper's intersection/difference activation objective."""

        logits = self.last_object_logits
        if logits is None:
            raise RuntimeError("Object activation loss requested before neck forward")
        prediction = logits.float().sigmoid()
        target = build_gaussian_object_targets(
            gt_bboxes,
            batch_input_shape=batch_input_shape,
            feature_shape=prediction.shape[-2:],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        support = target > 0
        reduce_dims = (1, 2, 3)
        target_mass = target.sum(dim=reduce_dims)
        prediction_mass = prediction.sum(dim=reduce_dims).clamp_min(1e-6)
        valid = target_mass > 0

        intersection = (prediction * target).sum(dim=reduce_dims)
        intersection_loss = torch.where(
            valid,
            1.0 - intersection / target_mass.clamp_min(1e-6),
            torch.zeros_like(target_mass),
        )
        difference_loss = (prediction * (~support).to(dtype=prediction.dtype)).sum(
            dim=reduce_dims
        ) / prediction_mass
        loss = (intersection_loss + float(background_weight) * difference_loss).mean()
        diagnostics = {
            "object_activation_intersection": intersection_loss.detach().mean(),
            "object_activation_difference": difference_loss.detach().mean(),
            "object_mask_foreground_response": (
                (prediction * support).sum()
                / support.sum().clamp_min(1).to(dtype=prediction.dtype)
            ).detach(),
            "object_mask_background_response": (
                (prediction * (~support)).sum()
                / (~support).sum().clamp_min(1).to(dtype=prediction.dtype)
            ).detach(),
        }
        return loss, diagnostics


@DETECTORS.register_module()
class HOD26ObjectAwareCoDETR(CoDETR):
    """Co-DINO with box-supervised OSSDet activation as a joint train loss."""

    def __init__(
        self,
        *args,
        object_activation_loss_weight: float = 0.6,
        object_activation_background_weight: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.object_activation_loss_weight = float(object_activation_loss_weight)
        self.object_activation_background_weight = float(
            object_activation_background_weight
        )
        if self.object_activation_loss_weight <= 0:
            raise ValueError("object_activation_loss_weight must be positive")
        if self.object_activation_background_weight < 0:
            raise ValueError("object_activation_background_weight must be non-negative")
        if not isinstance(self.neck, HOD26SpectralSFP):
            raise TypeError("HOD26ObjectAwareCoDETR requires HOD26SpectralSFP")

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs,
    ):
        losses = super().forward_train(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            proposals=proposals,
            **kwargs,
        )
        activation_loss, diagnostics = self.neck.object_activation_loss(
            gt_bboxes,
            batch_input_shape=img.shape[-2:],
            background_weight=self.object_activation_background_weight,
        )
        losses["loss_object_activation"] = (
            self.object_activation_loss_weight * activation_loss
        )
        losses.update(diagnostics)
        return losses
