"""Clean V5R localization lineage.

V5R preserves V5's high-potential spectral and salience changes but replaces
the faulty localization integration.  The repair follows the official D-FINE
structure: an independently supervised pre-bbox head defines the fixed FDR
anchor, while Co-DETR auxiliary heads always use the native Co-DINO decoder.
"""

from __future__ import annotations

import copy

import torch
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.models.builder import DETECTORS, HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.models.co_dino_head import CoDINOHead
from torch.nn import functional as F

from hod26.v5.model import (
    HOD26V5CoDETR,
    HOD26V5CoDINOHead,
    HOD26V5DinoTransformerDecoder,
    _distance_to_box,
    _integral,
    _weighting_function,
    global_optimal_bbox_targets,
)


def _bounded_box(box: torch.Tensor) -> torch.Tensor:
    """Keep normalized centers valid while allowing large boundary objects."""

    return torch.cat(
        (
            box[..., :2].clamp(1e-5, 1 - 1e-5),
            box[..., 2:].clamp(1e-5, 1.5),
        ),
        dim=-1,
    )


class HOD26V5RDecoderMixin:
    """Marker used by tests and fail-closed runtime assertions."""


@HEADS.register_module()
class HOD26V5RCoDINOHead(HOD26V5CoDINOHead):
    """D-FINE localization with a supervised, independent initial anchor."""

    def __init__(self, *args, pre_loss_weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Official D-FINE uses a distinct traditional 4-D pre-bbox head.  A
        # V4 layer-0 clone gives an exact domain-trained anchor at bootstrap,
        # but it receives fresh Adam moments because its gradient coordinate
        # is now independent.
        self.pre_bbox_head = copy.deepcopy(self.reg_branches[0])
        self.pre_loss_weight = float(pre_loss_weight)
        if self.pre_loss_weight <= 0:
            raise ValueError("pre_loss_weight must be positive")

    def forward(
        self,
        mlvl_feats,
        img_metas,
        dn_label_query=None,
        dn_bbox_query=None,
        attn_mask=None,
    ):
        transition = float(self.v5_transition.item())
        decoder = self.transformer.decoder
        if transition <= 0:
            # The bootstrap update remains the literal public-board-winning
            # V4 function.  New parameters stay in the zero-valued loss graph.
            object.__setattr__(decoder, "_v5_fdr_branches_ref", None)
            object.__setattr__(decoder, "_v5r_pre_bbox_head_ref", None)
            self._v5_main_cache = None
            return CoDINOHead.forward(
                self,
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

        # These are non-owning runtime references.  Registering either module
        # on the decoder would duplicate checkpoint keys.
        object.__setattr__(decoder, "_v5_fdr_branches_ref", self.fdr_branches)
        object.__setattr__(decoder, "_v5r_pre_bbox_head_ref", self.pre_bbox_head)
        object.__setattr__(decoder, "v5_transition", self.v5_transition)
        object.__setattr__(decoder, "v5_reg_max", self.reg_max)
        object.__setattr__(decoder, "v5_reg_scale", self.fdr_reg_scale)
        object.__setattr__(decoder, "v5_up", self.fdr_up)
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
        native_boxes = decoder._v5r_last_native_boxes
        pre_boxes = decoder._v5r_last_pre_boxes
        classes = []
        native_classes = []
        for layer_index in range(hs.shape[0]):
            native_score = self.cls_branches[layer_index](hs[layer_index])
            correction = self.lqe_layers[layer_index](corners[layer_index])
            score = native_score + self.v5_transition.to(
                native_score.dtype
            ) * correction.to(native_score.dtype)
            native_classes.append(native_score)
            classes.append(score)
        classes = torch.stack(classes)
        native_classes = torch.stack(native_classes)
        self._v5_main_cache = {
            "corners": corners,
            "base_references": base_references,
            "boxes": boxes,
            "classes": classes,
            "native_boxes": native_boxes,
            "pre_boxes": pre_boxes,
            "pre_classes": native_classes[0],
        }
        return classes, boxes, topk_score, topk_anchor, encoded_features

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Run Co-DETR auxiliary supervision in native Co-DINO coordinates."""

        decoder = self.transformer.decoder
        old_fdr = getattr(decoder, "_v5_fdr_branches_ref", None)
        old_pre = getattr(decoder, "_v5r_pre_bbox_head_ref", None)
        object.__setattr__(decoder, "_v5_fdr_branches_ref", None)
        object.__setattr__(decoder, "_v5r_pre_bbox_head_ref", None)
        try:
            return CoDINOHead.forward_aux(
                self, mlvl_feats, img_metas, aux_targets, head_idx
            )
        finally:
            object.__setattr__(decoder, "_v5_fdr_branches_ref", old_fdr)
            object.__setattr__(decoder, "_v5r_pre_bbox_head_ref", old_pre)

    @staticmethod
    def _matched_iou(boxes: torch.Tensor, targets) -> torch.Tensor:
        bbox_targets = torch.cat(targets[2], dim=0)
        bbox_weights = torch.cat(targets[3], dim=0)
        positive = bbox_weights[:, 0] > 0
        flat_boxes = boxes.reshape(-1, 4)
        if not positive.any():
            return flat_boxes.sum().detach() * 0
        return (
            bbox_overlaps(
                bbox_cxcywh_to_xyxy(flat_boxes[positive].detach()),
                bbox_cxcywh_to_xyxy(bbox_targets[positive]),
                is_aligned=True,
            )
            .clamp_min(0)
            .mean()
            .detach()
        )

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
        decoder_targets = []
        decoder_dn_targets = []
        pre_targets = None
        pre_dn_targets = None
        try:
            # Bypass V5's old FDR loss wrapper but retain Co-DINO's exact
            # encoder, decoder and DN objectives.
            losses = CoDINOHead.loss(
                self,
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
            if cache is not None:
                layers = cache["corners"].shape[0]
                if len(self._v5_targets) < layers:
                    raise RuntimeError("Incomplete decoder target capture")
                decoder_targets = list(self._v5_targets[-layers:])
                decoder_dn_targets = list(self._v5_dn_targets[-layers:])
                pad = int(dn_meta["pad_size"]) if dn_meta is not None else 0

                pre_classes = cache["pre_classes"][:, pad:]
                pre_boxes = cache["pre_boxes"][:, pad:]
                pre_cls, pre_bbox, pre_iou = self.loss_single(
                    pre_classes,
                    pre_boxes,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore,
                )
                pre_targets = self._v5_targets[-1]
                pre_scale = self.v5_transition.clamp(0, 1) * self.pre_loss_weight
                losses["loss_pre_cls"] = pre_scale * pre_cls
                losses["loss_pre_bbox"] = pre_scale * pre_bbox
                losses["loss_pre_iou"] = pre_scale * pre_iou

                if pad:
                    dn_pre_classes = cache["pre_classes"][:, :pad]
                    dn_pre_boxes = cache["pre_boxes"][:, :pad]
                    per_image_dn_meta = [dn_meta for _ in img_metas]
                    dn_pre_cls, dn_pre_bbox, dn_pre_iou = self.loss_dn_single(
                        dn_pre_classes,
                        dn_pre_boxes,
                        gt_bboxes_list,
                        gt_labels_list,
                        img_metas,
                        per_image_dn_meta,
                    )
                    pre_dn_targets = self._v5_dn_targets[-1]
                    losses["dn_loss_pre_cls"] = pre_scale * dn_pre_cls
                    losses["dn_loss_pre_bbox"] = pre_scale * dn_pre_bbox
                    losses["dn_loss_pre_iou"] = pre_scale * dn_pre_iou
        finally:
            self._v5_collect_targets = False
            self._v5_collect_dn_targets = False

        if cache is None:
            dormant = all_cls_scores.sum() * 0
            for parameter in self.fdr_branches.parameters():
                dormant = dormant + parameter.sum() * 0
            for parameter in self.lqe_layers.parameters():
                dormant = dormant + parameter.sum() * 0
            for parameter in self.pre_bbox_head.parameters():
                dormant = dormant + parameter.sum() * 0
            losses["loss_fdr_bootstrap_anchor"] = dormant
            return losses

        layers = cache["corners"].shape[0]
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
            fgl = self._fgl_loss(corners, references, boxes, localization_targets)
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
                losses[f"{prefix}loss_go_lsd"] = transition * self.go_lsd_weight * go

        if pad and len(decoder_dn_targets) == layers:
            for layer_index in range(layers):
                corners = cache["corners"][layer_index, :, :pad]
                references = cache["base_references"][layer_index, :, :pad]
                boxes = cache["boxes"][layer_index, :, :pad]
                fgl = self._fgl_loss(
                    corners, references, boxes, decoder_dn_targets[layer_index]
                )
                prefix = "dn_" if layer_index == layers - 1 else f"d{layer_index}.dn_"
                losses[f"{prefix}loss_fgl"] = transition * self.fgl_weight * fgl
                if layer_index < layers - 1:
                    go = self._go_lsd_loss(
                        corners,
                        cache["corners"][-1, :, :pad],
                        cache["classes"][-1, :, :pad],
                        boxes,
                        decoder_dn_targets[layer_index],
                    )
                    losses[f"{prefix}loss_go_lsd"] = (
                        transition * self.go_lsd_weight * go
                    )

        # Always-on diagnostics make the exact V5 failure mode observable:
        # main/DN localization can no longer collapse invisibly behind stable
        # encoder and auxiliary-head losses.
        final_boxes = cache["boxes"][-1, :, pad:]
        final_native = cache["native_boxes"][-1, :, pad:]
        losses["fdr_main_matched_iou"] = self._matched_iou(
            final_boxes, decoder_targets[-1]
        )
        losses["fdr_pre_matched_iou"] = self._matched_iou(
            cache["pre_boxes"][:, pad:], pre_targets
        )
        losses["fdr_refinement_l1"] = (
            (final_boxes.detach() - final_native.detach()).abs().mean()
        )
        losses["fdr_corner_abs_mean"] = (
            cache["corners"][-1, :, pad:].detach().abs().mean()
        )
        losses["fdr_box_saturation_fraction"] = (
            (
                (final_boxes[..., :2] <= 1.01e-5)
                | (final_boxes[..., :2] >= 1 - 1.01e-5)
                | (final_boxes[..., 2:] <= 1.01e-5)
                | (final_boxes[..., 2:] >= 1.499)
            )
            .float()
            .mean()
            .detach()
        )
        if pad and pre_dn_targets is not None:
            losses["fdr_dn_matched_iou"] = self._matched_iou(
                cache["boxes"][-1, :, :pad], decoder_dn_targets[-1]
            )
            losses["fdr_dn_pre_matched_iou"] = self._matched_iou(
                cache["pre_boxes"][:, :pad], pre_dn_targets
            )
        return losses


from mmcv.cnn.bricks.transformer import (  # noqa: E402
    TRANSFORMER_LAYER_SEQUENCE,
)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HOD26V5RDinoTransformerDecoder(
    HOD26V5RDecoderMixin, HOD26V5DinoTransformerDecoder
):
    """Continuous Co-DINO -> supervised-pre-head FDR decoder."""

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
        pre_bbox_head = getattr(self, "_v5r_pre_bbox_head_ref", None)
        if fdr_branches is None and pre_bbox_head is None:
            return super().forward(
                query,
                *args,
                reference_points=reference_points,
                valid_ratios=valid_ratios,
                reg_branches=reg_branches,
                **kwargs,
            )
        if fdr_branches is None or pre_bbox_head is None:
            raise RuntimeError("Partial V5R localization runtime installation")
        if reg_branches is None:
            raise RuntimeError("V5R requires native Co-DINO regression branches")

        transition = getattr(self, "v5_transition")
        reg_max = int(getattr(self, "v5_reg_max"))
        reg_scale = getattr(self, "v5_reg_scale")
        project = _weighting_function(reg_max, getattr(self, "v5_up"), reg_scale)

        output = query
        previous_output = 0
        cumulative_corners = 0
        base_reference = None
        pre_boxes = None
        intermediate = []
        intermediate_references = [reference_points]
        all_corners = []
        all_bases = []
        all_boxes = []
        all_native_boxes = []

        for layer_index, layer in enumerate(self.layers):
            if reference_points.shape[-1] != 4:
                raise RuntimeError("V5R requires 4-D reference boxes")
            reference_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            )
            sine = self.gen_sineembed_for_position(
                reference_input[:, :, 0, :], self.embed_dims // 2
            )
            query_pos = self.ref_point_head(sine).clamp(-10, 10).permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_input,
                **kwargs,
            )
            output_bqc = output.permute(1, 0, 2)
            normalized_bqc = self.norm(output_bqc)
            inverse_reference = inverse_sigmoid(reference_points, eps=1e-3)

            # Co-DINO uses the unnormalized branch output to advance decoder
            # references, but its supervised prediction is recomputed from the
            # normalized intermediate state.  Keeping both removes V5's
            # discontinuous update-0 -> update-1 coordinate jump.
            native_next = (
                reg_branches[layer_index](output_bqc) + inverse_reference
            ).sigmoid()
            native_box = (
                reg_branches[layer_index](normalized_bqc) + inverse_reference
            ).sigmoid()
            if layer_index == 0:
                pre_boxes = (
                    pre_bbox_head(normalized_bqc) + inverse_reference
                ).sigmoid()
                base_reference = pre_boxes.detach()

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
                predicted_box = native_box.float() + transition.float() * (
                    refined - native_box.float()
                )
                next_reference = native_next.float() + transition.float() * (
                    refined - native_next.float()
                )
            predicted_box = _bounded_box(predicted_box).to(native_box.dtype)
            next_reference = _bounded_box(next_reference).to(native_next.dtype)

            all_corners.append(corners)
            all_bases.append(base_reference)
            all_boxes.append(predicted_box)
            all_native_boxes.append(native_box)
            cumulative_corners = corners
            previous_output = output_bqc.detach().float()
            reference_points = next_reference.detach()
            output = output_bqc.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(normalized_bqc.permute(1, 0, 2))
                intermediate_references.append(predicted_box)

        if pre_boxes is None:
            raise RuntimeError("V5R decoder produced no pre-bbox anchor")
        self._v5_last_corners = torch.stack(all_corners)
        self._v5_last_base_references = torch.stack(all_bases)
        self._v5_last_boxes = torch.stack(all_boxes)
        self._v5r_last_native_boxes = torch.stack(all_native_boxes)
        self._v5r_last_pre_boxes = pre_boxes
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_references)
        return output, reference_points


@DETECTORS.register_module()
class HOD26V5RCoDETR(HOD26V5CoDETR):
    """Clean single-model detector with a synchronized transition coordinate."""

    def __init__(self, *args, transition_updates: int = 12000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.query_head, HOD26V5RCoDINOHead):
            raise TypeError("HOD26V5RCoDETR requires HOD26V5RCoDINOHead")
        self.transition_updates = int(transition_updates)
        if self.transition_updates <= 0:
            raise ValueError("transition_updates must be positive")

    def set_v5_progress(self, update: int) -> None:
        self.backbone.set_v5_progress(update, self.transition_updates)
        self.neck.set_v5_progress(update)
        self.query_head.set_v5_progress(update)
