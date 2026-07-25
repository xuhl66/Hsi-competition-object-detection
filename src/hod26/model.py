from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import C2PSA, C3k2, Concat, Conv, Detect, SPPF
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights

from .modules import (
    HomogeneousUniqueFusion,
    HyperspectralPyramid,
    ObjectAwareRefinement,
    PyramidSelect,
    SelfSensingRegrouping,
)


YOLO11L_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
)


def ensure_rgb_pretraining(path: str | Path) -> dict[str, Any]:
    """Download the declared public COCO checkpoint and record provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading declared external pretraining: {YOLO11L_URL}")
        request = urllib.request.Request(
            YOLO11L_URL, headers={"User-Agent": "hod26/0.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
        partial.replace(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "file": str(path.resolve()),
        "url": YOLO11L_URL,
        "sha256": sha256,
        "source_dataset": "MS COCO",
        "purpose": "RGB branch initialization only",
        "declared_external_pretraining": True,
    }
    provenance_path = path.with_suffix(path.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return provenance


def _attach_graph_metadata(module: nn.Module, index: int, source: int | list[int]) -> None:
    module.i = index
    module.f = source
    module.type = module.__class__.__name__
    module.np = sum(parameter.numel() for parameter in module.parameters())


class HyperDetectionLoss:
    def __init__(
        self,
        model: "HyperDetModel",
        object_weight: float,
        diversity_weight: float,
    ):
        self.base = v8DetectionLoss(model)
        self.model = model
        self.object_weight = object_weight
        self.diversity_weight = diversity_weight

    @staticmethod
    def _gaussian_target(
        batch: dict[str, torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        boxes = batch["bboxes"].to(device)
        batch_indices = batch["batch_idx"].to(device).long()
        for index in range(boxes.shape[0]):
            cx, cy, box_width, box_height = boxes[index]
            center_x = cx * width
            center_y = cy * height
            sigma_x = torch.clamp(box_width * width / 4.0, min=0.75)
            sigma_y = torch.clamp(box_height * height / 4.0, min=0.75)
            radius_x = int(torch.ceil(3.0 * sigma_x).item())
            radius_y = int(torch.ceil(3.0 * sigma_y).item())
            left = max(0, int(center_x.item()) - radius_x)
            right = min(width, int(center_x.item()) + radius_x + 1)
            top = max(0, int(center_y.item()) - radius_y)
            bottom = min(height, int(center_y.item()) + radius_y + 1)
            if right <= left or bottom <= top:
                continue
            xs = torch.arange(left, right, device=device, dtype=dtype)
            ys = torch.arange(top, bottom, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            gaussian = torch.exp(
                -0.5
                * (
                    ((xx - center_x) / sigma_x) ** 2
                    + ((yy - center_y) / sigma_y) ** 2
                )
            )
            destination = target[batch_indices[index], 0, top:bottom, left:right]
            target[batch_indices[index], 0, top:bottom, left:right] = torch.maximum(
                destination, gaussian
            )
        return target

    def _object_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.model.object_refinement.last_logits
        if logits is None:
            return next(self.model.parameters()).sum() * 0.0
        target = self._gaussian_target(
            batch,
            batch_size=logits.shape[0],
            height=logits.shape[2],
            width=logits.shape[3],
            device=logits.device,
            dtype=logits.dtype,
        )
        probability = logits.sigmoid()
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = 0.75 * target + 0.25 * (1.0 - target)
        focal = (alpha * (1.0 - p_t).square() * cross_entropy).mean()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return focal + dice

    def __call__(self, predictions, batch):
        total, items = self.base(predictions, batch)
        object_loss = self._object_loss(batch)
        diversity_loss = self.model.ssrm.diversity_loss()
        auxiliary = (
            self.object_weight * object_loss
            + self.diversity_weight * diversity_loss
        )
        batch_size = int(batch["img"].shape[0])
        total = total + auxiliary * batch_size
        items = items.clone()
        items[1] = items[1] + auxiliary.detach()
        self.model.last_aux_losses = {
            "object": float(object_loss.detach()),
            "ssrm_diversity": float(diversity_loss.detach()),
        }
        return total, items


class HyperDetModel(DetectionModel):
    """
    Single HYPER-DET/OSSDet detector with a YOLO11 localization head.

    The graph retains separate RGB-compatible and 3-D hyperspectral branches,
    fuses them at four resolutions, and emits one set of predictions.
    """

    def __init__(
        self,
        num_classes: int = 18,
        input_bands: int = 16,
        ssrm_temperature: float = 0.7,
        object_refine_gain: float = 0.75,
        object_aux_weight: float = 0.6,
        ssrm_diversity_weight: float = 0.02,
        pretrained_rgb: str | Path | None = None,
        names: Sequence[str] | None = None,
    ):
        nn.Module.__init__(self)
        self.yaml = {"nc": num_classes, "ch": input_bands, "scale": "l"}
        self.names = {
            index: name
            for index, name in enumerate(
                names if names is not None else [str(i) for i in range(num_classes)]
            )
        }
        self.nc = num_classes
        self.inplace = True
        self.end2end = False
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.object_aux_weight = object_aux_weight
        self.ssrm_diversity_weight = ssrm_diversity_weight
        self.last_aux_losses: dict[str, float] = {}

        layers: list[nn.Module] = []
        save: set[int] = set()

        def add(module: nn.Module, source: int | list[int] = -1) -> int:
            index = len(layers)
            _attach_graph_metadata(module, index, source)
            layers.append(module)
            references = [source] if isinstance(source, int) else source
            save.update(reference for reference in references if reference != -1)
            return index

        # Raw input and RGB-compatible HYPER-DET branch.
        add(nn.Identity())  # 0, preserved raw 16-band input
        self.ssrm = SelfSensingRegrouping(
            input_bands, groups=3, temperature=ssrm_temperature
        )
        add(self.ssrm, 0)  # 1
        add(Conv(3, 64, 3, 2))  # 2, P1
        add(Conv(64, 128, 3, 2))  # 3
        add(C3k2(128, 256, n=2, c3k=True, e=0.25))  # 4, RGB P2
        add(Conv(256, 256, 3, 2))  # 5
        add(C3k2(256, 512, n=2, c3k=True, e=0.25))  # 6, RGB P3
        add(Conv(512, 512, 3, 2))  # 7
        add(C3k2(512, 512, n=2, c3k=True))  # 8, RGB P4
        add(Conv(512, 512, 3, 2))  # 9
        add(C3k2(512, 512, n=2, c3k=True))  # 10
        add(SPPF(512, 512, 5))  # 11
        add(C2PSA(512, 512, n=2))  # 12, RGB P5

        # Five-stage 3-D HSI branch and four HUFF fusion points.
        self.hsi_pyramid = HyperspectralPyramid((256, 512, 512, 512))
        add(self.hsi_pyramid, 0)  # 13
        add(PyramidSelect(0), 13)  # 14
        add(HomogeneousUniqueFusion(256), [4, 14])  # 15, fused P2
        self.object_refinement = ObjectAwareRefinement(
            256, gain=object_refine_gain
        )
        add(self.object_refinement)  # 16, refined P2
        add(PyramidSelect(1), 13)  # 17
        add(HomogeneousUniqueFusion(512), [6, 17])  # 18, fused P3
        add(PyramidSelect(2), 13)  # 19
        add(HomogeneousUniqueFusion(512), [8, 19])  # 20, fused P4
        add(PyramidSelect(3), 13)  # 21
        add(HomogeneousUniqueFusion(512), [12, 21])  # 22, fused P5

        # P2-P5 FPN/PAN and one decoupled DFL detection head.
        add(nn.Upsample(scale_factor=2, mode="nearest"), 22)  # 23
        add(Concat(1), [-1, 20])  # 24
        add(C3k2(1024, 512, n=2, c3k=True, shortcut=False))  # 25
        add(nn.Upsample(scale_factor=2, mode="nearest"))  # 26
        add(Concat(1), [-1, 18])  # 27
        add(C3k2(1024, 256, n=2, c3k=True, shortcut=False))  # 28
        add(nn.Upsample(scale_factor=2, mode="nearest"))  # 29
        add(Concat(1), [-1, 16])  # 30
        add(C3k2(512, 128, n=2, c3k=True, shortcut=False))  # 31, P2
        add(Conv(128, 128, 3, 2))  # 32
        add(Concat(1), [-1, 28])  # 33
        add(C3k2(384, 256, n=2, c3k=True, shortcut=False))  # 34, P3
        add(Conv(256, 256, 3, 2))  # 35
        add(Concat(1), [-1, 25])  # 36
        add(C3k2(768, 512, n=2, c3k=True, shortcut=False))  # 37, P4
        add(Conv(512, 512, 3, 2))  # 38
        add(Concat(1), [-1, 22])  # 39
        add(C3k2(1024, 512, n=2, c3k=True, shortcut=False))  # 40, P5

        Detect.legacy = False
        detect = Detect(num_classes, ch=(128, 256, 512, 512))
        add(detect, [31, 34, 37, 40])  # 41

        self.model = nn.Sequential(*layers)
        self.save = sorted(save)
        initialize_weights(self)
        detect.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
        self.stride = detect.stride
        detect.inplace = True
        detect.bias_init()

        self.pretraining_provenance: dict[str, Any] | None = None
        self.transfer_report: dict[str, Any] = {}
        if pretrained_rgb:
            self.pretraining_provenance = ensure_rgb_pretraining(pretrained_rgb)
            self.transfer_report = self.load_rgb_pretraining(pretrained_rgb)

    def init_criterion(self):
        return HyperDetectionLoss(
            self,
            object_weight=self.object_aux_weight,
            diversity_weight=self.ssrm_diversity_weight,
        )

    @staticmethod
    def _copy_exact(source: nn.Module, destination: nn.Module) -> tuple[int, int]:
        source_state = source.state_dict()
        destination_state = destination.state_dict()
        copied = {
            name: tensor
            for name, tensor in source_state.items()
            if name in destination_state and destination_state[name].shape == tensor.shape
        }
        destination.load_state_dict(copied, strict=False)
        return len(copied), len(destination_state)

    def load_rgb_pretraining(self, path: str | Path) -> dict[str, Any]:
        """Transfer matching RGB backbone/head tensors from public YOLO11l."""
        source_model = YOLO(str(path)).model
        source_layers = source_model.model
        mappings = [
            *[(source, source + 2) for source in range(11)],
            (13, 25),
            (16, 28),
            (17, 35),
            (19, 37),
            (20, 38),
            (22, 40),
        ]
        copied, available = 0, 0
        details = []
        for source_index, destination_index in mappings:
            matched, total = self._copy_exact(
                source_layers[source_index], self.model[destination_index]
            )
            copied += matched
            available += total
            details.append(
                {
                    "source_layer": source_index,
                    "destination_layer": destination_index,
                    "tensors": matched,
                    "destination_tensors": total,
                }
            )

        source_detect = source_layers[-1]
        destination_detect = self.model[-1]
        detect_pairs = [(0, 1), (1, 2), (2, 3)]
        for source_scale, destination_scale in detect_pairs:
            for branch_name in ("cv2", "cv3"):
                matched, total = self._copy_exact(
                    getattr(source_detect, branch_name)[source_scale],
                    getattr(destination_detect, branch_name)[destination_scale],
                )
                copied += matched
                available += total
                details.append(
                    {
                        "source_detect": f"{branch_name}.{source_scale}",
                        "destination_detect": f"{branch_name}.{destination_scale}",
                        "tensors": matched,
                        "destination_tensors": total,
                    }
                )
        return {
            "copied_tensors": copied,
            "candidate_destination_tensors": available,
            "details": details,
        }

    def parameter_groups(
        self,
        learning_rate: float,
        rgb_multiplier: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        rgb_prefixes = tuple(f"model.{index}." for index in range(2, 13))
        rgb_parameters: list[nn.Parameter] = []
        new_decay: list[nn.Parameter] = []
        new_no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(rgb_prefixes):
                rgb_parameters.append(parameter)
            elif parameter.ndim <= 1 or name.endswith(".bias"):
                new_no_decay.append(parameter)
            else:
                new_decay.append(parameter)
        return [
            {
                "params": rgb_parameters,
                "lr": learning_rate * rgb_multiplier,
                "weight_decay": weight_decay,
                "group_name": "pretrained_rgb",
                "lr_multiplier": rgb_multiplier,
            },
            {
                "params": new_decay,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": "new_decay",
                "lr_multiplier": 1.0,
            },
            {
                "params": new_no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": "new_no_decay",
                "lr_multiplier": 1.0,
            },
        ]

    def summary(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "parameters": parameters,
            "trainable_parameters": trainable,
            "strides": self.stride.tolist(),
            "classes": self.nc,
            "pretraining": self.pretraining_provenance,
            "transfer": self.transfer_report,
        }


def build_model(config: dict[str, Any], classes: Sequence[str]) -> HyperDetModel:
    model_cfg = config["model"]
    pretrained = model_cfg.get("pretrained_rgb")
    return HyperDetModel(
        num_classes=int(model_cfg["num_classes"]),
        input_bands=int(model_cfg["input_bands"]),
        ssrm_temperature=float(model_cfg["ssrm_temperature"]),
        object_refine_gain=float(model_cfg["object_refine_gain"]),
        object_aux_weight=float(model_cfg["object_aux_weight"]),
        ssrm_diversity_weight=float(model_cfg["ssrm_diversity_weight"]),
        pretrained_rgb=pretrained,
        names=classes,
    )


def unwrap_model(model: nn.Module) -> HyperDetModel:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def update_ema(
    ema: nn.Module,
    model: nn.Module,
    decay: float,
    updates: int,
    tau: float = 2000.0,
) -> float:
    # A ramp prevents the EMA from retaining initialization for many epochs.
    # This is the same functional form used by Ultralytics ModelEMA.
    effective_decay = decay * (1.0 - math.exp(-updates / tau))
    source = unwrap_model(model).state_dict()
    for name, value in ema.state_dict().items():
        source_value = source[name].detach()
        if value.dtype.is_floating_point:
            value.mul_(effective_decay).add_(
                source_value, alpha=1.0 - effective_decay
            )
        else:
            value.copy_(source_value)
    return effective_decay
