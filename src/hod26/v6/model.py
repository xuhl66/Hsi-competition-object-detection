"""Independent CoSpec-DINO V6 model for 16-band HOD26 detection."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmcv.ops import ModulatedDeformConv2d
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_overlaps, reduce_mean
from mmdet.models.backbones.vit import ViT
from mmdet.models.builder import BACKBONES, DETECTORS, HEADS, NECKS
from mmdet.models.necks.sfp import SFP
from mmdet.models.necks.sfp import LayerNorm as SFPChannelLayerNorm
from projects.models.co_atss_head import CoATSSHead
from projects.models.co_detr import CoDETR
from projects.models.co_dino_head import CoDINOHead
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .constants import FOLD0_CLASS_COUNTS, WAVELENGTHS_NM


class Float32ChannelLayerNorm(nn.Module):
    """AMP-safe per-pixel LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(channels)))
        self.bias = nn.Parameter(torch.zeros(int(channels)))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        with torch.cuda.amp.autocast(enabled=False):
            output = F.layer_norm(
                value.float().permute(0, 2, 3, 1),
                (value.shape[1],),
                self.weight.float(),
                self.bias.float(),
                self.eps,
            ).permute(0, 3, 1, 2).contiguous()
        return output.to(dtype=dtype)


class Float32GroupNorm(nn.GroupNorm):
    """Batch-size-one-safe GN with FP32 statistics and dtype restoration."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        groups = min(32, int(channels))
        if int(channels) % groups:
            raise ValueError(f"V6 channels={channels} are not divisible by {groups} GN groups")
        super().__init__(groups, int(channels), eps=eps, affine=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        with torch.cuda.amp.autocast(enabled=False):
            output = F.group_norm(
                value.float(), self.num_groups, self.weight.float(),
                self.bias.float(), self.eps,
            )
        return output.to(dtype=dtype)

def _replace_sfp_norms(module: nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, SFPChannelLayerNorm):
            replacement = Float32ChannelLayerNorm(child.normalized_shape[0], child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_sfp_norms(child)


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
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False,
            ),
            Float32GroupNorm(out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class SelfExcitedSpectralBlock(nn.Module):
    """Local spectral-spatial residual with bounded self-excitation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            ConvNormAct(channels, channels, groups=channels),
            ConvNormAct(channels, channels, kernel_size=1, activate=False),
        )
        hidden = max(16, channels // 4)
        self.excitation = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.local(value)
        spatial = self.excitation(
            torch.cat((residual.mean(1, keepdim=True), residual.amax(1, keepdim=True)), 1)
        )
        # The amplification is bounded in [1, 2].  Unlike the old scalar gate,
        # it cannot suppress the entire HSI path to zero.
        return value + residual * (1.0 + spatial) * self.channel(residual)


class SpectralDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int = 2) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(
            *(SelfExcitedSpectralBlock(out_channels) for _ in range(blocks))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(value))


@BACKBONES.register_module()
class HOD26V6WavelengthViT(ViT):
    """Public ViT-L plus raw/shape/dI-dlambda wavelength-aware HSI stream."""

    def __init__(
        self,
        *args,
        input_bands: int = 16,
        spectral_channels: Sequence[int] = (96, 128, 192),
        wavelength_basis_size: int = 8,
        use_spectral_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        if input_bands != 16 or len(WAVELENGTHS_NM) != input_bands:
            raise ValueError("V6 is locked to the official 16 VIS3 bands")
        if len(spectral_channels) != 3:
            raise ValueError("V6 spectral_channels must describe P2/P3/P4")
        if wavelength_basis_size != 8:
            raise ValueError("The audited V6 wavelength basis has eight channels")
        kwargs["in_chans"] = 3
        super().__init__(*args, **kwargs)
        self.input_bands = int(input_bands)
        self.spectral_channels = tuple(int(v) for v in spectral_channels)
        self.use_spectral_checkpoint = bool(use_spectral_checkpoint)

        wavelengths = torch.tensor(WAVELENGTHS_NM, dtype=torch.float32)
        axis = (wavelengths - wavelengths.mean()) / (wavelengths.max() - wavelengths.min())
        basis = []
        for frequency in range(1, 5):
            basis.extend((torch.sin(math.pi * frequency * axis), torch.cos(math.pi * frequency * axis)))
        self.register_buffer("wavelengths_nm", wavelengths, persistent=True)
        self.register_buffer("wavelength_basis", torch.stack(basis, dim=1), persistent=True)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=True,
        )

        self.proxy_linear = nn.Conv2d(16, 3, 1, bias=False)
        self.proxy_residual = nn.Sequential(
            nn.Conv2d(16, 24, 1), nn.GELU(),
            nn.Conv2d(24, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )
        evidence_channels = 16 + 16 + 15 + wavelength_basis_size
        c2, c3, c4 = self.spectral_channels
        self.spectral_stem = nn.Sequential(
            ConvNormAct(evidence_channels, c2, stride=2),
            SelfExcitedSpectralBlock(c2),
            SelfExcitedSpectralBlock(c2),
        )
        self.spectral_stages = nn.ModuleList(
            (SpectralDownStage(c2, c3), SpectralDownStage(c3, c4))
        )
        self._reset_v6_entry()

    def _reset_v6_entry(self) -> None:
        with torch.no_grad():
            self.proxy_linear.weight.zero_()
            # Wavelength-ordered long/middle/short broad responses provide a
            # stable public-ViT input before the learned residual adapts.
            for output, (start, stop) in enumerate(((10, 16), (5, 11), (0, 6))):
                self.proxy_linear.weight[output, start:stop, 0, 0] = 1.0 / (stop - start)
            nn.init.zeros_(self.proxy_residual[-1].weight)
            nn.init.zeros_(self.proxy_residual[-1].bias)

    def init_weights(self) -> None:
        super().init_weights()
        self._reset_v6_entry()

    def spectral_evidence(self, half_cube: torch.Tensor) -> torch.Tensor:
        rms = half_cube.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
        shape = half_cube / rms
        delta_nm = self.wavelengths_nm[1:] - self.wavelengths_nm[:-1]
        mean_delta = delta_nm.mean()
        derivative = (half_cube[:, 1:] - half_cube[:, :-1]) * (
            mean_delta / delta_nm
        ).view(1, -1, 1, 1)
        basis = torch.einsum(
            "bchw,ck->bkhw", shape, self.wavelength_basis.to(dtype=shape.dtype)
        ) / math.sqrt(self.input_bands)
        return torch.cat((half_cube, shape, derivative, basis), dim=1)

    def broadband_proxy(self, cube: torch.Tensor) -> torch.Tensor:
        proxy = self.proxy_linear(cube) + self.proxy_residual(cube)
        return (proxy - self.imagenet_mean) / self.imagenet_std

    def forward(self, cube: torch.Tensor) -> list[torch.Tensor]:
        if cube.ndim != 4 or cube.shape[1] != self.input_bands:
            raise ValueError(f"Expected Bx16xHxW, got {tuple(cube.shape)}")
        # The self-excited spectral Jacobian overflows in FP16 even when the
        # scalar loss is finite. Keep this compact branch in FP32; ViT-L,
        # fusion and detector heads remain under the outer AMP context.
        with torch.cuda.amp.autocast(enabled=False):
            half = F.interpolate(
                cube.float(), scale_factor=0.5, mode="bilinear", align_corners=False
            )
            evidence = self.spectral_evidence(half)
            if self.training and self.use_spectral_checkpoint:
                spectral = checkpoint(self.spectral_stem, evidence, use_reentrant=False)
            else:
                spectral = self.spectral_stem(evidence)
            spectral_outputs = [spectral]
            for stage in self.spectral_stages:
                if self.training and self.use_spectral_checkpoint:
                    spectral = checkpoint(stage, spectral, use_reentrant=False)
                else:
                    spectral = stage(spectral)
                spectral_outputs.append(spectral)
        visual = super().forward(self.broadband_proxy(cube))[0]
        return [visual, *spectral_outputs]


class BidirectionalDeformableFusion(nn.Module):
    """Memory-bounded two-way DCN sampling between visual and HSI streams."""

    def __init__(self, spectral_channels: int, visual_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.visual_query = ConvNormAct(visual_channels, hidden, kernel_size=1)
        self.spectral_query = ConvNormAct(spectral_channels, hidden, kernel_size=1)
        self.visual_source = ConvNormAct(visual_channels, hidden, kernel_size=1)
        self.spectral_source = ConvNormAct(spectral_channels, hidden, kernel_size=1)
        self.visual_offsets = nn.Conv2d(hidden, 27, 3, padding=1)
        self.spectral_offsets = nn.Conv2d(hidden, 27, 3, padding=1)
        self.sample_spectral = ModulatedDeformConv2d(hidden, hidden, 3, padding=1, bias=False)
        self.sample_visual = ModulatedDeformConv2d(hidden, hidden, 3, padding=1, bias=False)
        self.visual_output = nn.Conv2d(hidden * 2, visual_channels, 1, bias=False)
        self.spectral_output = nn.Conv2d(hidden * 2, spectral_channels, 1, bias=False)
        self.reset_identity()

    def reset_identity(self) -> None:
        for predictor in (self.visual_offsets, self.spectral_offsets):
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)
        # The deterministic coefficient is exactly zero in the initialization
        # checkpoint.  Tiny non-zero projections let every DCN coordinate get
        # gradients on the first prospective update without perturbing the
        # public detector by more than O(1e-8).
        nn.init.normal_(self.visual_output.weight, std=1e-4)
        nn.init.normal_(self.spectral_output.weight, std=1e-4)

    @staticmethod
    def _offset_mask(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offset, mask = value[:, :18], value[:, 18:]
        return offset, mask.sigmoid()

    def forward(
        self, visual: torch.Tensor, spectral: torch.Tensor, coefficient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spectral.shape[-2:] != visual.shape[-2:]:
            spectral = F.interpolate(spectral, visual.shape[-2:], mode="bilinear", align_corners=False)
        visual_query = self.visual_query(visual)
        spectral_query = self.spectral_query(spectral)
        v_offset, v_mask = self._offset_mask(self.visual_offsets(visual_query))
        s_offset, s_mask = self._offset_mask(self.spectral_offsets(spectral_query))
        sampled_visual = self.sample_visual(self.visual_source(visual), s_offset, s_mask)
        spectral_delta = self.spectral_output(torch.cat((spectral_query, sampled_visual), 1))
        coefficient = coefficient.to(device=visual.device, dtype=visual.dtype)
        refined_spectral = spectral + coefficient * spectral_delta
        # The visual update samples the reverse-updated spectral stream, so
        # both directions are causally connected to the detector loss.
        sampled_spectral = self.sample_spectral(
            self.spectral_source(refined_spectral), v_offset, v_mask
        )
        visual_delta = self.visual_output(torch.cat((visual_query, sampled_spectral), 1))
        return visual + coefficient * visual_delta, refined_spectral


@NECKS.register_module()
class HOD26V6CrossSpectralSFP(SFP):
    """P2/P3/P4 forced bidirectional fusion; no learnable global bypass gate."""

    def __init__(
        self,
        *args,
        spectral_channels: Sequence[int] = (96, 128, 192),
        fusion_open_updates: int = 6000,
        fusion_hidden: int = 64,
        use_fusion_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        _replace_sfp_norms(self)
        if len(spectral_channels) != 3 or fusion_open_updates < 1:
            raise ValueError("Invalid V6 fusion contract")
        self.fusion_open_updates = int(fusion_open_updates)
        self.use_fusion_checkpoint = bool(use_fusion_checkpoint)
        self.fusions = nn.ModuleList(
            BidirectionalDeformableFusion(int(ch), self.out_channels, fusion_hidden)
            for ch in spectral_channels
        )
        self.spectral_proposal_laterals = nn.ModuleList(
            ConvNormAct(int(channels), self.out_channels, kernel_size=1)
            for channels in spectral_channels
        )
        self.spectral_proposal_down = nn.ModuleList(
            ConvNormAct(self.out_channels, self.out_channels, stride=2)
            for _ in range(3)
        )
        self.register_buffer("fusion_progress", torch.tensor(0.0), persistent=True)
        self.last_spectral_features: tuple[torch.Tensor, ...] | None = None
        self.last_spectral_proposal_features: tuple[torch.Tensor, ...] | None = None

    def init_weights(self) -> None:
        super().init_weights()
        for fusion in self.fusions:
            fusion.reset_identity()

    def set_successful_updates(self, successful_updates: int) -> None:
        progress = min(1.0, max(0.0, int(successful_updates) / self.fusion_open_updates))
        self.fusion_progress.fill_(progress)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 4:
            raise ValueError("V6 SFP expects ViT plus P2/P3/P4 spectral tensors")
        visual = list(super().forward([inputs[0]]))
        if len(visual) != 5:
            raise RuntimeError("V6 requires the five-level P2-enabled SFP")
        spectral_outputs = []
        for level, fusion in enumerate(self.fusions):
            if self.training and self.use_fusion_checkpoint:
                visual[level], spectral = checkpoint(
                    fusion,
                    visual[level],
                    inputs[level + 1],
                    self.fusion_progress,
                    use_reentrant=False,
                )
            else:
                visual[level], spectral = fusion(
                    visual[level], inputs[level + 1], self.fusion_progress
                )
            spectral_outputs.append(spectral)
        self.last_spectral_features = tuple(spectral_outputs)
        if self.training:
            proposal = [
                lateral(value)
                for lateral, value in zip(
                    self.spectral_proposal_laterals, spectral_outputs, strict=True
                )
            ]
            value = proposal[-1]
            for downsample in self.spectral_proposal_down:
                value = downsample(value)
                proposal.append(value)
            self.last_spectral_proposal_features = tuple(proposal)
        else:
            # The spectral proposal route is training-only.  Inference remains
            # one shared Co-DINO decoder and does not pay for this pyramid.
            self.last_spectral_proposal_features = None
        return tuple(visual)


def effective_number_class_weights(
    counts: Sequence[int] = FOLD0_CLASS_COUNTS,
    beta: float = 0.999,
    cap: float = 1.5,
) -> torch.Tensor:
    if len(counts) != 18 or not 0 < beta < 1 or cap < 1:
        raise ValueError("Invalid V6 effective-number class-weight contract")
    values = torch.tensor(counts, dtype=torch.float64)
    effective = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), values)) / (1.0 - beta)
    weights = effective.rsqrt()
    weights = weights / weights.mean()
    weights = weights.clamp(max=float(cap))
    weights = weights / weights.mean()
    return weights.float()


def _rank_weights(
    quality: torch.Tensor,
    bbox_targets: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_images: int,
    queries_per_image: int,
    num_classes: int,
    tau: float,
) -> torch.Tensor:
    """Rank identical one-to-many GT targets independently in each image."""

    weights = quality.new_ones(quality.shape)
    offset = 0
    for image in range(num_images):
        start, stop = image * queries_per_image, (image + 1) * queries_per_image
        positive = ((labels[start:stop] >= 0) & (labels[start:stop] < num_classes)).nonzero().flatten()
        count = len(positive)
        if not count:
            continue
        targets = bbox_targets[start:stop][positive]
        _, inverse = torch.unique(targets, dim=0, return_inverse=True)
        local_quality = quality[offset : offset + count]
        local_weights = local_quality.new_ones(count)
        for group in range(int(inverse.max().item()) + 1):
            members = (inverse == group).nonzero().flatten()
            order = torch.argsort(local_quality[members], descending=True)
            ranks = torch.arange(len(members), device=quality.device, dtype=quality.dtype)
            local_weights[members[order]] = torch.exp(-ranks / float(tau))
        weights[offset : offset + count] = local_weights
        offset += count
    if offset != quality.numel():
        raise RuntimeError("V6 one-to-many rank accounting is inconsistent")
    return weights


@HEADS.register_module()
class HOD26V6AlignCoDINOHead(CoDINOHead):
    """Co-DINO with Align-DETR IA-BCE quality targets and aux ranking."""

    def __init__(
        self,
        *args,
        align_alpha: float = 0.25,
        align_gamma: float = 2.0,
        align_rank_tau: float = 1.5,
        class_weight_beta: float = 0.999,
        class_weight_cap: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= align_alpha <= 1 or align_gamma <= 0 or align_rank_tau <= 0:
            raise ValueError("Invalid Align-DETR loss hyperparameters")
        self.align_alpha = float(align_alpha)
        self.align_gamma = float(align_gamma)
        self.align_rank_tau = float(align_rank_tau)
        self.register_buffer(
            "hod26_class_weights",
            effective_number_class_weights(beta=class_weight_beta, cap=class_weight_cap),
            persistent=True,
        )

    def _aligned_losses(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
        img_metas,
        *,
        rank_one_to_many: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_images, queries_per_image = cls_scores.shape[:2]
        flat_logits = cls_scores.reshape(-1, self.cls_out_channels)
        loss_logits = flat_logits.float()
        flat_boxes = bbox_preds.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1).to(dtype=loss_logits.dtype)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        positive = ((labels >= 0) & (labels < self.num_classes)).nonzero().flatten()

        probability = loss_logits.sigmoid()
        positive_weights = torch.zeros_like(probability)
        negative_weights = probability.pow(self.align_gamma)
        rank = probability.new_ones(len(positive))
        if len(positive):
            positive_labels = labels[positive].long()
            predicted = bbox_cxcywh_to_xyxy(flat_boxes[positive])
            target = bbox_cxcywh_to_xyxy(bbox_targets[positive])
            iou = bbox_overlaps(predicted.detach(), target, is_aligned=True).clamp(0, 1)
            confidence = probability[positive, positive_labels]
            quality = (
                confidence.pow(self.align_alpha)
                * iou.pow(1.0 - self.align_alpha)
            ).clamp_min(0.01).detach()
            if rank_one_to_many:
                rank = _rank_weights(
                    quality,
                    bbox_targets,
                    labels,
                    num_images=num_images,
                    queries_per_image=queries_per_image,
                    num_classes=self.num_classes,
                    tau=self.align_rank_tau,
                )
            aligned_target = (quality * rank).to(dtype=probability.dtype)
            class_scale = self.hod26_class_weights[positive_labels].to(probability.dtype)
            positive_weights[positive, positive_labels] = aligned_target * class_scale
            negative_weights[positive, positive_labels] = (1.0 - aligned_target) * class_scale
        classification = (
            F.softplus(-loss_logits) * positive_weights
            + F.softplus(loss_logits) * negative_weights
        ) * label_weights[:, None]
        normalizer = reduce_mean(probability.new_tensor([max(len(positive), 1)])).clamp_min(1)
        loss_cls = classification.sum() / normalizer

        factors = []
        for meta, prediction in zip(img_metas, bbox_preds):
            image_h, image_w, _ = meta["img_shape"]
            factors.append(
                prediction.new_tensor((image_w, image_h, image_w, image_h))
                .view(1, 4).repeat(prediction.shape[0], 1)
            )
        factors = torch.cat(factors, 0)
        regression_weights = bbox_weights
        if rank_one_to_many and len(positive):
            regression_weights = regression_weights.clone()
            regression_weights[positive] *= rank[:, None]
        boxes_xyxy = bbox_cxcywh_to_xyxy(flat_boxes) * factors
        targets_xyxy = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        box_normalizer = reduce_mean(flat_boxes.new_tensor([max(len(positive), 1)])).clamp_min(1).item()
        loss_iou = self.loss_iou(
            boxes_xyxy, targets_xyxy, regression_weights, avg_factor=box_normalizer
        )
        loss_bbox = self.loss_bbox(
            flat_boxes, bbox_targets, regression_weights, avg_factor=box_normalizer
        )
        return loss_cls, loss_bbox, loss_iou

    def loss_single(
        self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list,
        img_metas, gt_bboxes_ignore_list=None,
    ):
        targets = self.get_targets(
            [value for value in cls_scores], [value for value in bbox_preds],
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list,
        )
        labels, label_weights, bbox_targets, bbox_weights = (
            torch.cat(values, 0) for values in targets[:4]
        )
        return self._aligned_losses(
            cls_scores, bbox_preds, labels, label_weights, bbox_targets,
            bbox_weights, img_metas, rank_one_to_many=False,
        )

    def loss_single_aux(
        self, cls_scores, bbox_preds, labels, label_weights, bbox_targets,
        bbox_weights, img_metas, gt_bboxes_ignore_list=None,
    ):
        try:
            losses = self._aligned_losses(
                cls_scores, bbox_preds, labels, label_weights, bbox_targets,
                bbox_weights, img_metas, rank_one_to_many=True,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError("V6 auxiliary Align loss target contract failed") from error
        return tuple(value * self.lambda_1 for value in losses)

    def loss_dn_single(
        self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
        img_metas, dn_meta,
    ):
        targets = self.get_dn_target(
            [value for value in dn_bbox_preds], gt_bboxes_list, gt_labels_list,
            img_metas, dn_meta,
        )
        labels, label_weights, bbox_targets, bbox_weights = (
            torch.cat(values, 0) for values in targets[:4]
        )
        return self._aligned_losses(
            dn_cls_scores, dn_bbox_preds, labels, label_weights, bbox_targets,
            bbox_weights, img_metas, rank_one_to_many=False,
        )


@HEADS.register_module()
class HOD26V6SpectralProposalHead(CoATSSHead):
    """Class-balanced, small-level-emphasized one-to-many proposal route."""

    def __init__(
        self,
        *args,
        class_weight_beta: float = 0.999,
        class_weight_cap: float = 1.5,
        level_loss_weights: Sequence[float] = (1.50, 1.25, 1.00, 0.75, 0.50, 0.25),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "hod26_class_weights",
            effective_number_class_weights(beta=class_weight_beta, cap=class_weight_cap),
            persistent=True,
        )
        self.level_loss_weights = tuple(float(value) for value in level_loss_weights)
        if any(value <= 0 for value in self.level_loss_weights):
            raise ValueError("V6 proposal level weights must be positive")

    def loss_single(
        self, anchors, cls_score, bbox_pred, centerness, labels,
        label_weights, bbox_targets, img_metas, num_total_samples,
    ):
        adjusted = label_weights.clone()
        positive = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=True)
        if len(positive[0]):
            adjusted[positive] *= self.hod26_class_weights[labels[positive]].to(adjusted.dtype)
        return super().loss_single(
            anchors, cls_score, bbox_pred, centerness, labels, adjusted,
            bbox_targets, img_metas, num_total_samples,
        )

    def loss(self, *args, **kwargs):
        result = super().loss(*args, **kwargs)
        if result is None:
            return None
        for key in ("loss_cls", "loss_bbox", "loss_centerness"):
            values = result[key]
            if len(values) != len(self.level_loss_weights):
                raise RuntimeError("V6 proposal feature-level count changed")
            result[key] = [
                value * weight for value, weight in zip(values, self.level_loss_weights, strict=True)
            ]
        return result


@DETECTORS.register_module()
class HOD26V6CoSpecDINO(CoDETR):
    """Single-output V6 detector with training-only collaborative routes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.backbone, HOD26V6WavelengthViT):
            raise TypeError("V6 requires the wavelength-aware ViT-L")
        if not isinstance(self.neck, HOD26V6CrossSpectralSFP):
            raise TypeError("V6 requires forced cross-spectral SFP")
        if not isinstance(self.query_head, HOD26V6AlignCoDINOHead):
            raise TypeError("V6 requires the aligned Co-DINO head")
        if len(self.bbox_head) != 1 or not isinstance(
            self.bbox_head[0], HOD26V6SpectralProposalHead
        ):
            raise TypeError("V6 requires exactly one spectral proposal route")

    def set_v6_successful_updates(self, successful_updates: int) -> None:
        self.neck.set_successful_updates(successful_updates)

    @staticmethod
    def _suffix_losses(losses, index: int, weight: float = 1.0):
        output = {}
        for name, value in losses.items():
            key = f"{name}{index}"
            if isinstance(value, (list, tuple)):
                output[key] = [item * weight for item in value]
            else:
                output[key] = value * weight
        return output

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
        """Run Co-DETR while supervising its ATSS route from HSI features.

        The public query and RoI routes retain the upstream feature flow.  Only
        the training-only one-to-many ATSS route consumes the direct P2--P7
        spectral pyramid; its positive coordinates are then injected into the
        shared query decoder exactly as in Co-DETR.
        """

        if hasattr(self, "mask_head"):
            raise RuntimeError("The audited V6 detector does not support a mask head")
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        if not self.with_attn_mask:
            for img_meta in img_metas:
                input_h, input_w = img_meta["batch_input_shape"]
                img_meta["img_shape"] = [input_h, input_w, 16]

        features = self.extract_feat(img, img_metas)
        spectral_proposals = self.neck.last_spectral_proposal_features
        if spectral_proposals is None or len(spectral_proposals) != 6:
            raise RuntimeError("V6 direct spectral P2--P7 proposal pyramid is missing")

        losses = {}
        detr_results = self.query_head.forward_train(
            features, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        query_losses, encoded_features = detr_results[:2]
        losses.update(query_losses)

        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get(
                "rpn_proposal", self.test_cfg[self.head_idx].rpn
            )
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                encoded_features,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        positive_coords = []
        for index, roi_head in enumerate(self.roi_head):
            roi_losses = roi_head.forward_train(
                encoded_features,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                gt_masks,
                **kwargs,
            )
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop("pos_coords"))
            else:
                roi_losses.pop("pos_coords", None)
            losses.update(self._suffix_losses(roi_losses, index))

        for index, bbox_head in enumerate(self.bbox_head):
            bbox_losses = bbox_head.forward_train(
                spectral_proposals,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
            if self.with_pos_coord:
                positive_coords.append(bbox_losses.pop("pos_coords"))
            else:
                bbox_losses.pop("pos_coords", None)
            losses.update(
                self._suffix_losses(bbox_losses, index + len(self.roi_head))
            )

        if self.with_pos_coord:
            for index, coordinates in enumerate(positive_coords):
                auxiliary = self.query_head.forward_train_aux(
                    encoded_features,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_ignore,
                    coordinates,
                    index,
                )
                losses.update(self._suffix_losses(auxiliary, index))
        return losses
