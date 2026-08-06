"""V5 localization-first Co-DINO extensions for HOD26.

The detector keeps the converged V4 Co-DINO/ViT-L graph and adds three
targeted changes:

* exact linear collapse of the redundant adjacent-band evidence;
* scale-aware, explicitly negative salience supervision;
* D-FINE FDR, LQE and GO-LSD on the six primary Co-DINO decoder layers.

At transition zero the query boxes/scores and object-aware feature path are
the V4 functions.  A resume-safe hook opens the new paths during the first
4,000 optimizer updates.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_overlaps, reduce_mean
from mmdet.models.builder import BACKBONES, DETECTORS, HEADS, NECKS
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.models.co_detr import CoDETR
from projects.models.co_dino_head import CoDINOHead
from projects.models.transformer import DinoTransformerDecoder
from torch import nn
from torch.nn import functional as F

from hod26.v4.model import (
    HOD26HSIViT,
    HOD26SpectralSFP,
    ObjectActivationMask,
    build_gaussian_object_targets,
)


def collapse_adjacent_difference_weight(weight: torch.Tensor) -> torch.Tensor:
    """Collapse ``[raw16, x[i+1]-x[i]]`` convolution weights to raw16.

    The transformation is algebraically exact for any kernel size and bias.
    """

    if weight.ndim != 4 or weight.shape[1] != 31:
        raise ValueError(f"Expected OIHW convolution with 31 inputs, got {weight.shape}")
    raw = weight[:, :16]
    difference = weight[:, 16:]
    collapsed = raw.clone()
    collapsed[:, 0] -= difference[:, 0]
    collapsed[:, 1:15] += difference[:, :14] - difference[:, 1:]
    collapsed[:, 15] += difference[:, 14]
    return collapsed


def _weighting_function(
    reg_max: int,
    up: torch.Tensor,
    reg_scale: torch.Tensor,
) -> torch.Tensor:
    """D-FINE's non-uniform localization bins (official formulation)."""

    upper1 = up.abs() * reg_scale.abs()
    upper2 = upper1 * 2
    step = (upper1 + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [(step**i) - 1 for i in range(1, reg_max // 2)]
    return torch.cat([-upper2, *left, torch.zeros_like(up), *right, upper2])


def _integral(corners: torch.Tensor, project: torch.Tensor, reg_max: int) -> torch.Tensor:
    probability = F.softmax(corners.reshape(-1, reg_max + 1), dim=-1)
    values = F.linear(probability, project.reshape(1, -1))
    return values.reshape(*corners.shape[:-1], 4)


def _distance_to_box(
    points: torch.Tensor,
    distance: torch.Tensor,
    reg_scale: torch.Tensor,
) -> torch.Tensor:
    scale = reg_scale.abs()
    x1 = points[..., 0] - (0.5 * scale + distance[..., 0]) * (
        points[..., 2] / scale
    )
    y1 = points[..., 1] - (0.5 * scale + distance[..., 1]) * (
        points[..., 3] / scale
    )
    x2 = points[..., 0] + (0.5 * scale + distance[..., 2]) * (
        points[..., 2] / scale
    )
    y2 = points[..., 1] + (0.5 * scale + distance[..., 3]) * (
        points[..., 3] / scale
    )
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1
    )


def _translate_distance(
    values: torch.Tensor,
    project: torch.Tensor,
    reg_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = values.reshape(-1)
    diffs = project.unsqueeze(0) - flat.unsqueeze(1)
    indices = (diffs <= 0).sum(dim=1) - 1
    indices = indices.clamp(0, reg_max - 1)
    left = project[indices]
    right = project[indices + 1]
    right_weight = ((flat - left) / (right - left).clamp_min(1e-12)).clamp(0, 1)
    left_weight = 1.0 - right_weight
    return indices, right_weight, left_weight


def _box_to_distribution(
    points: torch.Tensor,
    target_xyxy: torch.Tensor,
    project: torch.Tensor,
    reg_scale: torch.Tensor,
    reg_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = reg_scale.abs()
    unit_w = points[:, 2] / scale + 1e-16
    unit_h = points[:, 3] / scale + 1e-16
    values = torch.stack(
        (
            (points[:, 0] - target_xyxy[:, 0]) / unit_w - 0.5 * scale,
            (points[:, 1] - target_xyxy[:, 1]) / unit_h - 0.5 * scale,
            (target_xyxy[:, 2] - points[:, 0]) / unit_w - 0.5 * scale,
            (target_xyxy[:, 3] - points[:, 1]) / unit_h - 0.5 * scale,
        ),
        dim=-1,
    )
    return _translate_distance(values, project, reg_max)


def global_optimal_bbox_targets(
    layer_targets: Sequence[tuple],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build D-FINE GO-LSD's matching union from Co-DINO target tensors.

    Every query that is positive in any decoder layer remains positive. If
    different layers assign that query to different objects, the assignment
    occurring most often wins, matching D-FINE's global-optimal tie policy.
    """

    if not layer_targets:
        raise ValueError("GO-LSD requires at least one decoder target set")
    image_count = len(layer_targets[0][2])
    if any(len(targets[2]) != image_count for targets in layer_targets):
        raise ValueError("Inconsistent batch size across decoder targets")
    union_targets: list[torch.Tensor] = []
    union_weights: list[torch.Tensor] = []
    for image_index in range(image_count):
        candidates = torch.stack(
            [targets[2][image_index] for targets in layer_targets]
        )
        valid = torch.stack(
            [targets[3][image_index][:, 0] > 0 for targets in layer_targets]
        )
        # Count how often every candidate assignment occurs for each query.
        # The tensorized form avoids a per-query CUDA synchronization.
        same_assignment = (
            candidates[:, None] == candidates[None, :]
        ).all(dim=-1)
        counts = (
            same_assignment & valid[:, None] & valid[None, :]
        ).sum(dim=1)
        counts = counts.masked_fill(~valid, -1)
        selected_layer = counts.argmax(dim=0)
        query_index = torch.arange(
            candidates.shape[1], device=candidates.device
        )
        target = candidates[selected_layer, query_index]
        positive = valid.any(dim=0)
        target = target * positive[:, None].to(target.dtype)
        weight = torch.zeros_like(layer_targets[0][3][image_index])
        weight[positive] = 1
        union_targets.append(target)
        union_weights.append(weight)
    return union_targets, union_weights


class FDRMLP(nn.Module):
    """D-FINE-compatible three-layer localization MLP."""

    def __init__(self, hidden_dim: int, reg_max: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 4 * (reg_max + 1)),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            value = F.relu(layer(value))
        return self.layers[-1](value)


class LocationQualityEstimator(nn.Module):
    """Official D-FINE top-k distribution statistic and zero-safe correction."""

    def __init__(self, reg_max: int, k: int = 4) -> None:
        super().__init__()
        self.reg_max = int(reg_max)
        self.k = int(k)
        self.reg_conf = FDRMLP.__new__(FDRMLP)
        nn.Module.__init__(self.reg_conf)
        self.reg_conf.layers = nn.ModuleList(
            (nn.Linear(4 * (k + 1), 64), nn.Linear(64, 1))
        )

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        batch, queries, _ = corners.shape
        probability = F.softmax(
            corners.reshape(batch, queries, 4, self.reg_max + 1), dim=-1
        )
        topk = probability.topk(self.k, dim=-1).values
        statistic = torch.cat((topk, topk.mean(dim=-1, keepdim=True)), dim=-1)
        value = statistic.reshape(batch, queries, -1)
        value = F.relu(self.reg_conf.layers[0](value))
        return self.reg_conf.layers[1](value)


@BACKBONES.register_module()
class HOD26V5HSIViT(HOD26HSIViT):
    """V4 ViT-L HSI path using only the 16 independent sensor bands."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old_proxy = self.proxy_residual[0]
        self.legacy_proxy_input = old_proxy
        self.proxy_residual[0] = nn.Conv2d(
            16,
            old_proxy.out_channels,
            old_proxy.kernel_size,
            stride=old_proxy.stride,
            padding=old_proxy.padding,
            bias=old_proxy.bias is not None,
        )
        old_stem = self.spectral_stem[0][0]
        self.legacy_spectral_input = old_stem
        self.spectral_stem[0][0] = nn.Conv2d(
            16,
            old_stem.out_channels,
            old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            groups=old_stem.groups,
            bias=old_stem.bias is not None,
        )
        self.register_buffer("v5_transition", torch.tensor(0.0), persistent=True)
        for parameter in self.legacy_proxy_input.parameters():
            parameter.requires_grad_(False)
        for parameter in self.legacy_spectral_input.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def spectral_evidence(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def legacy_spectral_evidence(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, x[:, 1:] - x[:, :-1]), dim=1)

    def set_v5_progress(self, update: int, transition_updates: int = 4000) -> None:
        ratio = min(max(int(update), 0) / max(int(transition_updates), 1), 1.0)
        self.v5_transition.fill_(ratio)

    def _blend_input_convolution(
        self,
        raw_input: torch.Tensor,
        legacy_input: torch.Tensor,
        raw_convolution: nn.Module,
        legacy_convolution: nn.Module,
    ) -> torch.Tensor:
        transition = float(self.v5_transition.item())
        if transition <= 0:
            return legacy_convolution(legacy_input)
        if transition >= 1:
            return raw_convolution(raw_input)
        legacy = legacy_convolution(legacy_input)
        raw = raw_convolution(raw_input)
        return legacy + self.v5_transition.to(legacy.dtype) * (raw - legacy)

    def broadband_proxy(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._blend_input_convolution(
            x,
            self.legacy_spectral_evidence(x),
            self.proxy_residual[0],
            self.legacy_proxy_input,
        )
        for index in range(1, len(self.proxy_residual)):
            hidden = self.proxy_residual[index](hidden)
        proxy = self.proxy_linear(x) + hidden
        return (proxy - self.imagenet_mean) / self.imagenet_std

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.input_bands:
            raise ValueError(
                f"Expected BCHW with {self.input_bands} bands, got {tuple(x.shape)}"
            )
        legacy_evidence = self.legacy_spectral_evidence(x)
        spectral = self._blend_input_convolution(
            x,
            legacy_evidence,
            self.spectral_stem[0][0],
            self.legacy_spectral_input,
        )
        for index in range(1, len(self.spectral_stem[0])):
            spectral = self.spectral_stem[0][index](spectral)
        spectral = self.spectral_stem[1](spectral)
        spectral_outputs = [spectral]
        for stage in self.spectral_stages:
            spectral = stage(spectral)
            spectral_outputs.append(spectral)
        proxy = self.broadband_proxy(x)
        vit_output = super(HOD26HSIViT, self).forward(proxy)[0]
        return [vit_output, *spectral_outputs]


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HOD26V5DinoTransformerDecoder(DinoTransformerDecoder):
    """Co-DINO decoder with D-FINE refinement fed into successive references."""

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        valid_ratios=None,
        reg_branches=None,
        **kwargs,
    ):
        fdr_branches = getattr(self, "_v5_fdr_branches_ref", None)
        if fdr_branches is None:
            return super().forward(
                query,
                *args,
                reference_points=reference_points,
                valid_ratios=valid_ratios,
                reg_branches=reg_branches,
                **kwargs,
            )
        transition = getattr(self, "v5_transition")
        reg_max = int(getattr(self, "v5_reg_max"))
        reg_scale = getattr(self, "v5_reg_scale")
        project = _weighting_function(reg_max, getattr(self, "v5_up"), reg_scale)

        output = query
        previous_output = 0
        cumulative_corners = 0
        base_reference = None
        intermediate = []
        intermediate_references = [reference_points]
        all_corners = []
        all_bases = []
        all_boxes = []

        for layer_index, layer in enumerate(self.layers):
            reference_input = reference_points[:, :, None] * torch.cat(
                [valid_ratios, valid_ratios], -1
            )[:, None]
            sine = self.gen_sineembed_for_position(
                reference_input[:, :, 0, :], self.embed_dims // 2
            )
            query_pos = self.ref_point_head(sine).permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_input,
                **kwargs,
            )
            output_bqc = output.permute(1, 0, 2)
            coarse = (
                reg_branches[layer_index](output_bqc)
                + inverse_sigmoid(reference_points, eps=1e-3)
            ).sigmoid()
            if base_reference is None:
                base_reference = coarse.detach()
            # Co-DINO query activations have a wider dynamic range than the
            # native D-FINE decoder. Keep distribution logits/integration in
            # FP32 so DN queries cannot overflow under AMP.
            with torch.cuda.amp.autocast(enabled=False):
                fdr_input = output_bqc.float()
                if isinstance(previous_output, torch.Tensor):
                    fdr_input = fdr_input + previous_output
                corners = fdr_branches[layer_index](fdr_input)
                if isinstance(cumulative_corners, torch.Tensor):
                    corners = corners + cumulative_corners
                refined = _distance_to_box(
                    base_reference.float(),
                    _integral(corners, project.float(), reg_max),
                    reg_scale.float(),
                )
                blended = coarse.float() + transition.float() * (
                    refined - coarse.float()
                )
            blended = torch.cat(
                (
                    blended[..., :2].clamp(1e-5, 1 - 1e-5),
                    blended[..., 2:].clamp(1e-5, 1.5),
                ),
                dim=-1,
            ).to(dtype=coarse.dtype)

            all_corners.append(corners)
            all_bases.append(base_reference)
            all_boxes.append(blended)
            cumulative_corners = corners
            previous_output = output_bqc.detach().float()
            reference_points = blended.detach()
            output = output_bqc.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_references.append(blended)

        self._v5_last_corners = torch.stack(all_corners)
        self._v5_last_base_references = torch.stack(all_bases)
        self._v5_last_boxes = torch.stack(all_boxes)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_references)
        return output, reference_points


@HEADS.register_module()
class HOD26V5CoDINOHead(CoDINOHead):
    """Primary Co-DINO query head with pretrained D-FINE localization heads."""

    def __init__(
        self,
        *args,
        reg_max: int = 32,
        reg_scale: float = 8.0,
        transition_updates: int = 4000,
        fgl_weight: float = 0.15,
        go_lsd_weight: float = 1.5,
        go_lsd_temperature: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        layers = int(self.transformer.decoder.num_layers)
        self.reg_max = int(reg_max)
        self.transition_updates = int(transition_updates)
        self.fgl_weight = float(fgl_weight)
        self.go_lsd_weight = float(go_lsd_weight)
        self.go_lsd_temperature = float(go_lsd_temperature)
        self.fdr_branches = nn.ModuleList(
            FDRMLP(self.embed_dims, self.reg_max) for _ in range(layers)
        )
        self.lqe_layers = nn.ModuleList(
            LocationQualityEstimator(self.reg_max) for _ in range(layers)
        )
        self.register_buffer("v5_transition", torch.tensor(0.0), persistent=True)
        self.register_buffer("fdr_up", torch.tensor([0.5]), persistent=True)
        self.register_buffer(
            "fdr_reg_scale", torch.tensor([float(reg_scale)]), persistent=True
        )
        self._v5_main_cache = None
        self._v5_collect_targets = False
        self._v5_collect_dn_targets = False
        self._v5_targets = []
        self._v5_dn_targets = []

    def set_v5_progress(self, update: int) -> None:
        ratio = min(max(int(update), 0) / max(self.transition_updates, 1), 1.0)
        self.v5_transition.fill_(ratio)

    def forward(
        self,
        mlvl_feats,
        img_metas,
        dn_label_query=None,
        dn_bbox_query=None,
        attn_mask=None,
    ):
        if float(self.v5_transition.item()) <= 0:
            # Make update zero a literal V4 forward, not merely an algebraic
            # approximation of it. This keeps the public-board-winning parent
            # as the exact starting function before the new path opens.
            object.__setattr__(
                self.transformer.decoder, "_v5_fdr_branches_ref", None
            )
            self._v5_main_cache = None
            return super().forward(
                mlvl_feats,
                img_metas,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
            )
        batch_size = mlvl_feats[0].size(0)
        input_h, input_w = img_metas[0]["batch_input_shape"]
        img_masks = mlvl_feats[0].new_ones((batch_size, input_h, input_w))
        for image_index in range(batch_size):
            image_h, image_w, _ = img_metas[image_index]["img_shape"]
            img_masks[image_index, :image_h, :image_w] = 0
        masks = []
        positions = []
        for feature in mlvl_feats:
            mask = F.interpolate(img_masks[None], size=feature.shape[-2:]).bool()[0]
            masks.append(mask)
            positions.append(self.positional_encoding(mask))

        decoder = self.transformer.decoder
        # Bypass nn.Module.__setattr__: this is a non-owning runtime reference.
        # Registering it on the decoder would duplicate all FDR state keys.
        object.__setattr__(decoder, "_v5_fdr_branches_ref", self.fdr_branches)
        decoder.v5_transition = self.v5_transition
        decoder.v5_reg_max = self.reg_max
        decoder.v5_reg_scale = self.fdr_reg_scale
        decoder.v5_up = self.fdr_up
        hs, references, topk_score, topk_anchor, encoder = self.transformer(
            mlvl_feats,
            masks,
            None,
            positions,
            dn_label_query,
            dn_bbox_query,
            attn_mask,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )
        encoded_features = []
        start = 0
        for feature in mlvl_feats:
            batch, channels, height, width = feature.shape
            end = start + height * width
            value = encoder[start:end].permute(1, 2, 0).contiguous()
            encoded_features.append(value.reshape(batch, channels, height, width))
            start = end
        encoded_features.append(self.downsample(encoded_features[-1]))

        hs = hs.permute(0, 2, 1, 3)
        if dn_label_query is not None and dn_label_query.size(1) == 0:
            hs[0] += self.label_embedding.weight[0, 0] * 0.0
        corners = decoder._v5_last_corners
        boxes = decoder._v5_last_boxes
        base_references = decoder._v5_last_base_references
        classes = []
        for layer_index in range(hs.shape[0]):
            score = self.cls_branches[layer_index](hs[layer_index])
            correction = self.lqe_layers[layer_index](corners[layer_index])
            score = score + self.v5_transition.to(score.dtype) * correction.to(
                score.dtype
            )
            classes.append(score)
        classes = torch.stack(classes)
        self._v5_main_cache = {
            "corners": corners,
            "base_references": base_references,
            "boxes": boxes,
            "classes": classes,
        }
        return classes, boxes, topk_score, topk_anchor, encoded_features

    def get_targets(self, *args, **kwargs):
        result = super().get_targets(*args, **kwargs)
        if self._v5_collect_targets:
            self._v5_targets.append(result)
        return result

    def get_dn_target(self, *args, **kwargs):
        result = super().get_dn_target(*args, **kwargs)
        if self._v5_collect_dn_targets:
            self._v5_dn_targets.append(result)
        return result

    def _fgl_loss(
        self,
        corners: torch.Tensor,
        references: torch.Tensor,
        boxes: torch.Tensor,
        targets,
    ) -> torch.Tensor:
        bbox_targets = torch.cat(targets[2], dim=0)
        bbox_weights = torch.cat(targets[3], dim=0)
        positive = bbox_weights[:, 0] > 0
        if not positive.any():
            return corners.sum() * 0
        pred = corners.reshape(-1, 4 * (self.reg_max + 1))[positive]
        ref = references.reshape(-1, 4)[positive].detach()
        target = bbox_targets[positive]
        target_xyxy = bbox_cxcywh_to_xyxy(target)
        project = _weighting_function(
            self.reg_max, self.fdr_up, self.fdr_reg_scale
        ).to(dtype=pred.dtype)
        index, right_weight, left_weight = _box_to_distribution(
            ref, target_xyxy, project, self.fdr_reg_scale, self.reg_max
        )
        logits = pred.reshape(-1, self.reg_max + 1)
        left = F.cross_entropy(logits, index, reduction="none")
        right = F.cross_entropy(logits, index + 1, reduction="none")
        predicted_boxes = boxes.reshape(-1, 4)[positive]
        iou = bbox_overlaps(
            bbox_cxcywh_to_xyxy(predicted_boxes.detach()),
            target_xyxy,
            is_aligned=True,
        ).clamp_min(0)
        weights = iou[:, None].expand(-1, 4).reshape(-1)
        normalizer = reduce_mean(pred.new_tensor([positive.sum()])).clamp_min(1)
        return (
            (left * left_weight + right * right_weight) * weights
        ).sum() / normalizer

    def _go_lsd_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        teacher_classes: torch.Tensor,
        student_boxes: torch.Tensor,
        targets,
    ) -> torch.Tensor:
        temperature = self.go_lsd_temperature
        student_flat = student.reshape(-1, self.reg_max + 1)
        teacher_flat = teacher.detach().reshape(-1, self.reg_max + 1)
        divergence = F.kl_div(
            F.log_softmax(student_flat / temperature, dim=-1),
            F.softmax(teacher_flat / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (temperature**2)
        confidence = teacher_classes.detach().sigmoid().amax(dim=-1).reshape(-1)
        positive = torch.cat(targets[3], dim=0)[:, 0] > 0
        if positive.any():
            target_boxes = torch.cat(targets[2], dim=0)[positive]
            positive_iou = bbox_overlaps(
                bbox_cxcywh_to_xyxy(
                    student_boxes.reshape(-1, 4)[positive].detach()
                ),
                bbox_cxcywh_to_xyxy(target_boxes),
                is_aligned=True,
            ).clamp_min(0)
            confidence = confidence.clone()
            confidence[positive] = positive_iou.to(confidence.dtype)
        confidence = confidence[:, None].expand(-1, 4).reshape(-1)
        mask = positive[:, None].expand(-1, 4).reshape(-1)
        weighted = divergence * confidence
        pos_count = mask.sum().float()
        neg_count = (~mask).sum().float()
        pos_loss = weighted[mask].mean() if mask.any() else weighted.sum() * 0
        neg_loss = weighted[~mask].mean() if (~mask).any() else weighted.sum() * 0
        pos_scale = pos_count.sqrt()
        neg_scale = neg_count.sqrt()
        return (pos_loss * pos_scale + neg_loss * neg_scale) / (
            pos_scale + neg_scale
        ).clamp_min(1)

    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_topk_scores,
        enc_topk_anchors,
        enc_outputs,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        dn_meta=None,
        gt_bboxes_ignore=None,
    ):
        cache = self._v5_main_cache
        self._v5_targets = []
        self._v5_dn_targets = []
        self._v5_collect_targets = True
        self._v5_collect_dn_targets = True
        try:
            losses = super().loss(
                all_cls_scores,
                all_bbox_preds,
                enc_topk_scores,
                enc_topk_anchors,
                enc_outputs,
                gt_bboxes_list,
                gt_labels_list,
                img_metas,
                dn_meta,
                gt_bboxes_ignore,
            )
        finally:
            self._v5_collect_targets = False
            self._v5_collect_dn_targets = False
        if cache is None:
            # DDP has find_unused_parameters=False. Keep the dormant FDR/LQE
            # weights in the update-zero graph without changing the loss.
            dormant = all_cls_scores.sum() * 0
            for parameter in self.fdr_branches.parameters():
                dormant = dormant + parameter.sum() * 0
            for parameter in self.lqe_layers.parameters():
                dormant = dormant + parameter.sum() * 0
            losses["loss_fgl"] = dormant
            return losses

        layers = cache["corners"].shape[0]
        if len(self._v5_targets) < layers:
            raise RuntimeError("Incomplete GO-LSD target capture")
        decoder_targets = self._v5_targets[-layers:]
        union_bbox_targets, union_bbox_weights = global_optimal_bbox_targets(
            self._v5_targets
        )
        localization_targets = (
            decoder_targets[-1][0],
            decoder_targets[-1][1],
            union_bbox_targets,
            union_bbox_weights,
            0,
            0,
        )
        pad = int(dn_meta["pad_size"]) if dn_meta is not None else 0
        transition = self.v5_transition.clamp(0, 1)
        for layer_index in range(layers):
            corners = cache["corners"][layer_index, :, pad:]
            references = cache["base_references"][layer_index, :, pad:]
            boxes = cache["boxes"][layer_index, :, pad:]
            fgl = self._fgl_loss(
                corners, references, boxes, localization_targets
            )
            prefix = "" if layer_index == layers - 1 else f"d{layer_index}."
            losses[f"{prefix}loss_fgl"] = transition * self.fgl_weight * fgl
            if layer_index < layers - 1:
                go = self._go_lsd_loss(
                    corners,
                    cache["corners"][-1, :, pad:],
                    cache["classes"][-1, :, pad:],
                    boxes,
                    localization_targets,
                )
                losses[f"{prefix}loss_go_lsd"] = (
                    transition * self.go_lsd_weight * go
                )

        if pad and len(self._v5_dn_targets) >= layers:
            dn_targets = self._v5_dn_targets[-layers:]
            for layer_index in range(layers):
                corners = cache["corners"][layer_index, :, :pad]
                references = cache["base_references"][layer_index, :, :pad]
                boxes = cache["boxes"][layer_index, :, :pad]
                fgl = self._fgl_loss(
                    corners, references, boxes, dn_targets[layer_index]
                )
                prefix = "dn_" if layer_index == layers - 1 else f"d{layer_index}.dn_"
                losses[f"{prefix}loss_fgl"] = transition * self.fgl_weight * fgl
                if layer_index < layers - 1:
                    go = self._go_lsd_loss(
                        corners,
                        cache["corners"][-1, :, :pad],
                        cache["classes"][-1, :, :pad],
                        boxes,
                        dn_targets[layer_index],
                    )
                    losses[f"{prefix}loss_go_lsd"] = (
                        transition * self.go_lsd_weight * go
                    )
        return losses


def _resize_probability(
    probability: torch.Tensor, size: Sequence[int]
) -> torch.Tensor:
    if tuple(probability.shape[-2:]) == tuple(size):
        return probability
    return F.interpolate(probability, size=size, mode="bilinear", align_corners=False)


@NECKS.register_module()
class HOD26V5SalienceSFP(HOD26SpectralSFP):
    """V4 object refinement driven by multi-level scale-aware salience."""

    def __init__(
        self,
        *args,
        transition_updates: int = 4000,
        salience_size_boundaries: Sequence[int] = (64, 128, 256),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transition_updates = int(transition_updates)
        self.salience_size_boundaries = tuple(int(x) for x in salience_size_boundaries)
        self.salience_high = nn.ModuleList(
            ObjectActivationMask(self.out_channels) for _ in range(3)
        )
        self.register_buffer("v5_transition", torch.tensor(0.0), persistent=True)
        self.last_salience_logits: list[torch.Tensor] | None = None

    def set_v5_progress(self, update: int) -> None:
        ratio = min(max(int(update), 0) / max(self.transition_updates, 1), 1.0)
        self.v5_transition.fill_(ratio)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 5:
            raise ValueError("HOD26V5SalienceSFP expects ViT plus P2--P5 spectra")
        base = list(super(HOD26SpectralSFP, self).forward([inputs[0]]))
        for level, fusion in enumerate(self.spectral_fusions):
            base[level] = fusion(base[level], inputs[level + 1])
        logits = [self.object_activation(base[0])]
        logits.extend(head(base[level]) for level, head in enumerate(self.salience_high, 1))
        self.last_object_logits = logits[0]
        self.last_salience_logits = logits
        p2 = logits[0].sigmoid()
        multilevel = torch.stack(
            [_resize_probability(value.sigmoid(), p2.shape[-2:]) for value in logits]
        ).mean(dim=0)
        transition = self.v5_transition.to(p2.dtype)
        effective = p2 + transition * 0.5 * (multilevel - p2)
        base[0] = self.object_gate(base[0], effective)
        for level, refinement in enumerate(self.object_refinements, start=1):
            base[level] = refinement(base[level], base[level - 1], effective)
        return tuple(base)

    @staticmethod
    def _focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = logits.float().sigmoid()
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), reduction="none"
        )
        positive_weight = target.float()
        negative_weight = (1.0 - target.float()) * probability.square()
        positive = (loss * positive_weight).sum() / positive_weight.sum().clamp_min(1)
        negative = (loss * negative_weight).sum() / (
            (1.0 - target.float()).sum().clamp_min(1)
        )
        return positive + negative

    def salience_loss(
        self,
        gt_bboxes: Sequence[torch.Tensor],
        *,
        batch_input_shape: Sequence[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.last_salience_logits
        if logits is None:
            raise RuntimeError("Salience loss requested before neck forward")
        per_level_boxes: list[list[torch.Tensor]] = [
            [] for _ in range(len(logits))
        ]
        boundaries = self.salience_size_boundaries
        for boxes in gt_bboxes:
            level_indices = []
            if len(boxes):
                sizes = (
                    (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
                    * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
                ).sqrt()
                level_indices = [
                    torch.bucketize(size, size.new_tensor(boundaries)).item()
                    for size in sizes
                ]
            for level in range(len(logits)):
                selected = [
                    box
                    for box, assigned in zip(boxes, level_indices)
                    if int(assigned) == level
                ]
                per_level_boxes[level].append(
                    torch.stack(selected) if selected else boxes.new_zeros((0, 4))
                )

        losses = []
        foreground = []
        background = []
        for level, level_logits in enumerate(logits):
            target = build_gaussian_object_targets(
                per_level_boxes[level],
                batch_input_shape=batch_input_shape,
                feature_shape=level_logits.shape[-2:],
                device=level_logits.device,
                dtype=torch.float32,
            )
            losses.append(self._focal(level_logits, target))
            probability = level_logits.float().sigmoid()
            support = target > 0
            foreground.append(
                (probability * support).sum()
                / support.sum().clamp_min(1).to(probability.dtype)
            )
            background.append(
                (probability * ~support).sum()
                / (~support).sum().clamp_min(1).to(probability.dtype)
            )
        transition_weight = 0.1 + 0.9 * self.v5_transition.clamp(0, 1)
        loss = transition_weight * torch.stack(losses).mean()
        diagnostics = {
            "salience_foreground_response": torch.stack(foreground).mean().detach(),
            "salience_background_response": torch.stack(background).mean().detach(),
            "salience_transition": self.v5_transition.detach().clone(),
        }
        return loss, diagnostics


@DETECTORS.register_module()
class HOD26V5CoDETR(CoDETR):
    """Single-model V5 detector with joint scale-aware salience training."""

    def __init__(
        self,
        *args,
        salience_loss_weight: float = 0.6,
        object_activation_loss_weight=None,
        object_activation_background_weight=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.salience_loss_weight = float(salience_loss_weight)
        if not isinstance(self.neck, HOD26V5SalienceSFP):
            raise TypeError("HOD26V5CoDETR requires HOD26V5SalienceSFP")
        if not isinstance(self.query_head, HOD26V5CoDINOHead):
            raise TypeError("HOD26V5CoDETR requires HOD26V5CoDINOHead")

    def set_v5_progress(self, update: int) -> None:
        self.backbone.set_v5_progress(update)
        self.neck.set_v5_progress(update)
        self.query_head.set_v5_progress(update)

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
        salience, diagnostics = self.neck.salience_loss(
            gt_bboxes, batch_input_shape=img.shape[-2:]
        )
        losses["loss_salience"] = self.salience_loss_weight * salience
        losses.update(diagnostics)
        if float(self.backbone.v5_transition.item()) <= 0:
            # The raw-16 convolutions are intentionally dormant on the exact
            # V4 anchor update. Keep them visible to DDP without perturbing
            # either the forward function or the scalar objective.
            dormant = img.sum() * 0
            for parameter in self.backbone.proxy_residual[0].parameters():
                dormant = dormant + parameter.sum() * 0
            for parameter in self.backbone.spectral_stem[0][0].parameters():
                dormant = dormant + parameter.sum() * 0
            losses["loss_spectral_reparameterization_anchor"] = dormant
        return losses
