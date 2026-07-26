from __future__ import annotations

import copy
import json
import math
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as TVF

from hod26.v2.compat import install_deim_compatibility

install_deim_compatibility()

from engine.backbone.hgnetv2 import HGNetv2
from engine.core import register
from engine.data._misc import (
    BoundingBoxes,
    Image,
    SanitizeBoundingBoxes,
    convert_to_tv_tensor,
)
from engine.data.dataloader import BaseCollateFunction
from engine.data.dataset._dataset import DetDataset
from engine.misc import MetricLogger, SmoothedValue, dist_utils
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from engine.solver.det_engine import evaluate
from engine.solver.det_solver import DetSolver

from hod26.data import decode_x2cube


HOD26_CLASSES = (
    "apple",
    "apple_plastic",
    "badminton",
    "banana",
    "banana_plastic",
    "car",
    "car_toy",
    "charger_head",
    "e-bike",
    "egg",
    "egg_plastic",
    "egg_wood",
    "orange",
    "orange_plastic",
    "people",
    "rubik",
    "stone_block",
    "table_tennis",
)

# torchvision / COCO contiguous label order.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Reuse semantically useful classifier priors; None intentionally keeps the
# target head's fresh initialization.
HOD26_TO_COCO: dict[str, tuple[str, ...] | None] = {
    "apple": ("apple",),
    "apple_plastic": ("apple",),
    "badminton": ("tennis racket",),
    "banana": ("banana",),
    "banana_plastic": ("banana",),
    "car": ("car",),
    "car_toy": ("car",),
    "charger_head": ("cell phone", "remote", "mouse"),
    "e-bike": ("bicycle", "motorcycle"),
    "egg": None,
    "egg_plastic": None,
    "egg_wood": None,
    "orange": ("orange",),
    "orange_plastic": ("orange",),
    "people": ("person",),
    "rubik": None,
    "stone_block": None,
    "table_tennis": ("sports ball",),
}


def _load_cube_tensor(
    path: Path,
    low: np.ndarray,
    high: np.ndarray,
) -> Image:
    mosaic = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mosaic is None:
        raise FileNotFoundError(path)
    if mosaic.dtype != np.uint16 or mosaic.ndim != 2:
        raise ValueError(
            f"Expected an official uint16 mosaic at {path}, "
            f"got dtype={mosaic.dtype}, shape={mosaic.shape}"
        )
    cube = decode_x2cube(mosaic).astype(np.float32)
    cube = np.clip((cube - low) / (high - low), 0.0, 1.0)
    tensor = torch.from_numpy(cube).permute(2, 0, 1).contiguous()
    return Image(tensor)


@register()
class HSICocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    """COCO annotations with exact on-the-fly X2Cube decoding."""

    __inject__ = ["transforms"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        stats_file: str,
        transforms: Any,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
    ) -> None:
        if return_masks:
            raise ValueError("HOD26 is a bounding-box-only task")
        if remap_mscoco_category:
            raise ValueError("HOD26 category ids are already contiguous 0..17")
        super().__init__(img_folder, ann_file)
        stats = json.loads(Path(stats_file).read_text(encoding="utf-8"))
        self.low = np.asarray(stats["low"], dtype=np.float32).reshape(1, 1, 16)
        self.high = np.asarray(stats["high"], dtype=np.float32).reshape(1, 1, 16)
        self.high = np.maximum(self.high, self.low + 1.0)
        self._transforms = transforms
        self.img_folder = Path(img_folder)
        self.ann_file = str(ann_file)
        self.return_masks = False
        self.remap_mscoco_category = False

    @property
    def transforms(self) -> Any:
        return self._transforms

    @transforms.setter
    def transforms(self, value: Any) -> None:
        self._transforms = value

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target

    def load_item(self, index: int):
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image = _load_cube_tensor(
            self.img_folder / image_info["file_name"],
            self.low,
            self.high,
        )
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if (width, height) != (
            int(image_info["width"]),
            int(image_info["height"]),
        ):
            raise ValueError(
                f"Decoded shape mismatch for {image_info['file_name']}: "
                f"{width}x{height} vs annotation "
                f"{image_info['width']}x{image_info['height']}"
            )
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        boxes = torch.as_tensor(
            [item["bbox"] for item in annotations], dtype=torch.float32
        ).reshape(-1, 4)
        if boxes.numel():
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
        labels = torch.as_tensor(
            [item["category_id"] for item in annotations],
            dtype=torch.int64,
        )
        area = torch.as_tensor(
            [item.get("area", item["bbox"][2] * item["bbox"][3])
             for item in annotations],
            dtype=torch.float32,
        )
        iscrowd = torch.as_tensor(
            [item.get("iscrowd", 0) for item in annotations],
            dtype=torch.int64,
        )
        if boxes.numel():
            keep = (boxes[:, 2] > boxes[:, 0]) & (
                boxes[:, 3] > boxes[:, 1]
            )
            boxes, labels = boxes[keep], labels[keep]
            area, iscrowd = area[keep], iscrowd[keep]
        target = {
            "boxes": convert_to_tv_tensor(
                boxes,
                key="boxes",
                box_format="XYXY",
                spatial_size=(height, width),
            ),
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([width, height]),
            "idx": torch.tensor([index]),
        }
        return image, target

    @property
    def categories(self):
        return self.coco.dataset["categories"]

    @property
    def category2name(self):
        return {item["id"]: item["name"] for item in self.categories}

    @property
    def category2label(self):
        return {item["id"]: item["id"] for item in self.categories}

    @property
    def label2category(self):
        return {item["id"]: item["id"] for item in self.categories}


@register()
class HSISpectralAugment(nn.Module):
    """Physically conservative, band-correlated hyperspectral augmentation."""

    def __init__(
        self,
        p: float = 0.8,
        gain: float = 0.12,
        slope: float = 0.06,
        noise_std: float = 0.008,
        band_dropout: float = 0.06,
    ) -> None:
        super().__init__()
        self.p = p
        self.gain = gain
        self.slope = slope
        self.noise_std = noise_std
        self.band_dropout = band_dropout

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        if random.random() >= self.p:
            return image, target, dataset
        tensor = image.as_subclass(torch.Tensor)
        channels = tensor.shape[0]
        axis = torch.linspace(
            -1.0, 1.0, channels, dtype=tensor.dtype, device=tensor.device
        ).view(channels, 1, 1)
        global_gain = 1.0 + random.uniform(-self.gain, self.gain)
        spectral_slope = random.uniform(-self.slope, self.slope)
        tensor = tensor * (global_gain + spectral_slope * axis)
        if self.noise_std > 0:
            sigma = random.uniform(0.0, self.noise_std)
            tensor = tensor + torch.randn_like(tensor) * sigma
        if self.band_dropout > 0 and random.random() < self.band_dropout:
            index = random.randrange(channels)
            replacement = tensor.mean(dim=0, keepdim=True)
            tensor[index : index + 1] = replacement
        return Image(tensor.clamp_(0.0, 1.0)), target, dataset


@register()
class HSIRandomVerticalFlip(T.RandomVerticalFlip):
    pass


def _clone_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in target.items()
    }


@register()
class HSIMosaic(nn.Module):
    """Four-image rectangular mosaic that preserves all 16 bands."""

    def __init__(
        self,
        output_size: Sequence[int] = (224, 448),
        rotation_range: float = 7.0,
        translation_range: Sequence[float] = (0.08, 0.08),
        scaling_range: Sequence[float] = (0.65, 1.35),
        probability: float = 0.65,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        self.resize = T.Resize(self.output_size, antialias=True)
        self.affine = T.RandomAffine(
            degrees=rotation_range,
            translate=tuple(float(value) for value in translation_range),
            scale=tuple(float(value) for value in scaling_range),
            fill=fill_value,
        )
        self.probability = probability

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs
        if random.random() >= self.probability:
            return image, target, dataset

        samples = [(image, _clone_target(target))]
        for index in random.choices(range(len(dataset)), k=3):
            item_image, item_target = dataset.load_item(index)
            samples.append((item_image, _clone_target(item_target)))
        resized = [self.resize(item_image, item_target) for item_image, item_target in samples]
        tile_height, tile_width = self.output_size
        canvas_height, canvas_width = tile_height * 2, tile_width * 2
        canvas = torch.zeros(
            (
                int(image.shape[0]),
                canvas_height,
                canvas_width,
            ),
            dtype=image.dtype,
        )
        offsets = ((0, 0), (tile_width, 0), (0, tile_height), (tile_width, tile_height))
        boxes, labels, area, iscrowd = [], [], [], []
        for (item_image, item_target), (offset_x, offset_y) in zip(
            resized, offsets, strict=True
        ):
            canvas[
                :,
                offset_y : offset_y + tile_height,
                offset_x : offset_x + tile_width,
            ] = item_image.as_subclass(torch.Tensor)
            item_boxes = item_target["boxes"].as_subclass(torch.Tensor).clone()
            item_boxes += torch.tensor(
                [offset_x, offset_y, offset_x, offset_y],
                dtype=item_boxes.dtype,
            )
            boxes.append(item_boxes)
            labels.append(item_target["labels"])
            area.append(item_target["area"])
            iscrowd.append(item_target["iscrowd"])

        merged_target = {
            "boxes": convert_to_tv_tensor(
                torch.cat(boxes),
                key="boxes",
                box_format="XYXY",
                spatial_size=(canvas_height, canvas_width),
            ),
            "labels": torch.cat(labels),
            "area": torch.cat(area),
            "iscrowd": torch.cat(iscrowd),
            "image_id": target["image_id"],
            "orig_size": torch.tensor([canvas_width, canvas_height]),
            "idx": target["idx"],
        }
        mosaic, merged_target = self.affine(Image(canvas), merged_target)
        return mosaic, merged_target, dataset


@register()
class HSIBatchCollateFunction(BaseCollateFunction):
    """Aspect-preserving multi-scale collate for the native ~2:1 sensor."""

    def __init__(
        self,
        stop_epoch: int = 40,
        ema_restart_decay: float = 0.9999,
        widths: Sequence[int] = (768, 832, 896, 960),
        mixup_prob: float = 0.25,
        mixup_epochs: Sequence[int] = (4, 32),
        base_size: int | None = None,
        base_size_repeat: int | None = None,
    ) -> None:
        super().__init__()
        self.stop_epoch = int(stop_epoch)
        self.ema_restart_decay = float(ema_restart_decay)
        self.widths = tuple(int(value) for value in widths)
        if any(value % 32 for value in self.widths):
            raise ValueError("Every multi-scale width must be divisible by 32")
        self.mixup_prob = float(mixup_prob)
        self.mixup_epochs = tuple(int(value) for value in mixup_epochs)

    def __call__(self, items):
        images = torch.stack(
            [item[0].as_subclass(torch.Tensor) for item in items]
        )
        targets = [item[1] for item in items]
        if (
            self.mixup_prob > 0
            and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[1]
            and random.random() < self.mixup_prob
        ):
            beta = random.uniform(0.45, 0.55)
            rolled = images.roll(1, dims=0)
            images = images * beta + rolled * (1.0 - beta)
            shifted = targets[-1:] + targets[:-1]
            mixed = []
            for left, right in zip(targets, shifted, strict=True):
                target = _clone_target(left)
                for key in ("boxes", "labels", "area", "iscrowd"):
                    target[key] = torch.cat((left[key], right[key]))
                target["mixup"] = torch.tensor(
                    [beta] * len(left["labels"])
                    + [1.0 - beta] * len(right["labels"]),
                    dtype=torch.float32,
                )
                mixed.append(target)
            targets = mixed
        if self.epoch < self.stop_epoch and self.widths:
            width = random.choice(self.widths)
            images = F.interpolate(
                images,
                size=(width // 2, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return images, targets


class _ConvGNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, stride: int) -> None:
        groups = math.gcd(c2, 16)
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride, 1, bias=False),
            nn.GroupNorm(groups, c2),
            nn.SiLU(inplace=True),
        )


@register()
class HSIHGNetv2(HGNetv2):
    """
    Exact pretrained HGNetv2-B5 plus a learnable 16->3 proxy and a
    zero-initialized multi-scale spectral residual path.
    """

    def __init__(
        self,
        name: str,
        input_bands: int = 16,
        spectral_channels: Sequence[int] = (64, 96, 128, 192),
        use_lab: bool = False,
        return_idx: Sequence[int] = (1, 2, 3),
        freeze_stem_only: bool = False,
        freeze_at: int = -1,
        freeze_norm: bool = False,
        pretrained: bool = False,
        local_model_dir: str = "storage/pretrained/v2/hgnetv2/",
    ) -> None:
        super().__init__(
            name=name,
            use_lab=use_lab,
            return_idx=list(return_idx),
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )
        if input_bands != 16:
            raise ValueError("The official sensor has exactly 16 bands")
        channels = tuple(int(value) for value in spectral_channels)
        if len(channels) != 4:
            raise ValueError("spectral_channels must contain four levels")

        self.proxy_projection = nn.Conv2d(input_bands, 3, 1, bias=True)
        self.proxy_refine = nn.Sequential(
            nn.Conv2d(input_bands, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=True),
        )
        with torch.no_grad():
            self.proxy_projection.weight.zero_()
            self.proxy_projection.bias.zero_()
            # Broad spectral groups yield a stable pseudo-RGB image before
            # fine-tuning while every band receives gradient from step one.
            groups = (range(10, 16), range(5, 11), range(0, 6))
            for output, indices in enumerate(groups):
                indices = tuple(indices)
                self.proxy_projection.weight[output, indices, 0, 0] = (
                    1.0 / len(indices)
                )
            nn.init.zeros_(self.proxy_refine[-1].weight)
            nn.init.zeros_(self.proxy_refine[-1].bias)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=True,
        )

        self.spectral_stem = nn.Sequential(
            _ConvGNAct(input_bands, channels[0], 2),
            _ConvGNAct(channels[0], channels[0], 2),
        )
        self.spectral_levels = nn.ModuleList(
            [
                _ConvGNAct(channels[0], channels[1], 2),
                _ConvGNAct(channels[1], channels[2], 2),
                _ConvGNAct(channels[2], channels[3], 2),
            ]
        )
        rgb_channels = [self._out_channels[index] for index in self.return_idx]
        self.spectral_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(spec_c, rgb_c, 1, bias=False),
                    nn.GroupNorm(math.gcd(rgb_c, 32), rgb_c),
                )
                for spec_c, rgb_c in zip(
                    channels[1:], rgb_channels, strict=True
                )
            ]
        )
        self.spectral_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(rgb_c)) for rgb_c in rgb_channels]
        )

    def forward(self, x: torch.Tensor):
        proxy = self.proxy_projection(x) + self.proxy_refine(x)
        proxy = (proxy - self.image_mean) / self.image_std

        rgb = self.stem(proxy)
        rgb_outputs = []
        for index, stage in enumerate(self.stages):
            rgb = stage(rgb)
            if index in self.return_idx:
                rgb_outputs.append(rgb)

        spectral = self.spectral_stem(x)
        spectral_outputs = []
        for level in self.spectral_levels:
            spectral = level(spectral)
            spectral_outputs.append(spectral)

        fused = []
        for rgb_feature, spectral_feature, projection, gate in zip(
            rgb_outputs,
            spectral_outputs,
            self.spectral_projections,
            self.spectral_gates,
            strict=True,
        ):
            residual = projection(spectral_feature)
            gain = torch.tanh(gate).view(1, -1, 1, 1)
            fused.append(rgb_feature + gain * residual)
        return fused


def _is_classification_parameter(name: str) -> bool:
    return (
        name == "decoder.denoising_class_embed.weight"
        or name.startswith("decoder.enc_score_head.")
        or name.startswith("decoder.dec_score_head.")
    )


def _semantic_head_tensor(
    current: torch.Tensor,
    pretrained: torch.Tensor,
    parameter_name: str,
) -> torch.Tensor | None:
    if pretrained.ndim != current.ndim or pretrained.shape[1:] != current.shape[1:]:
        return None
    if pretrained.shape[0] not in (80, 81) or current.shape[0] not in (18, 19):
        return None
    result = current.clone()
    coco_index = {name: index for index, name in enumerate(COCO_CLASSES)}
    for target_index, target_name in enumerate(HOD26_CLASSES):
        sources = HOD26_TO_COCO[target_name]
        if not sources:
            continue
        indices = [coco_index[source] for source in sources]
        result[target_index].copy_(pretrained[indices].mean(dim=0))
    if pretrained.shape[0] == 81 and current.shape[0] == 19:
        result[-1].copy_(pretrained[-1])
    return result


class HSIDetSolver(DetSolver):
    """DEIM solver with safe and explicit 80->18 semantic head transfer."""

    def load_tuning_state(self, path: str):
        state = (
            torch.hub.load_state_dict_from_url(path, map_location="cpu")
            if path.startswith("http")
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        pretrained = (
            state["ema"]["module"]
            if "ema" in state
            else state.get("model", state)
        )
        model = dist_utils.de_parallel(self.model)
        current = model.state_dict()
        matched: dict[str, torch.Tensor] = {}
        semantic = []
        mismatched = []
        for name, tensor in current.items():
            source = pretrained.get(name)
            if source is None:
                continue
            if source.shape == tensor.shape:
                matched[name] = source
                continue
            if _is_classification_parameter(name):
                mapped = _semantic_head_tensor(tensor, source, name)
                if mapped is not None:
                    matched[name] = mapped
                    semantic.append(name)
                    continue
            mismatched.append(name)
        model.load_state_dict(matched, strict=False)
        trainable_new = sorted(set(current) - set(matched))
        print(
            json.dumps(
                {
                    "v2_pretrained_transfer": {
                        "checkpoint": path,
                        "exact_tensors": len(matched) - len(semantic),
                        "semantic_head_tensors": len(semantic),
                        "new_or_unmatched_tensors": len(trainable_new),
                        "shape_mismatches": mismatched,
                        "semantic_parameters": semantic,
                    }
                },
                indent=2,
            )
        )

    def _train_one_epoch_accumulated(
        self,
        epoch: int,
        scheduler: FlatCosineLRScheduler | None,
        accumulation_steps: int,
    ) -> dict[str, float]:
        self.model.train()
        self.criterion.train()
        logger = MetricLogger(delimiter="  ")
        logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        self.optimizer.zero_grad(set_to_none=True)
        max_batches = int(self.cfg.yaml_cfg.get("max_train_batches", 0))
        epoch_batches = (
            min(len(self.train_dataloader), max_batches)
            if max_batches > 0
            else len(self.train_dataloader)
        )
        updates_per_epoch = math.ceil(epoch_batches / accumulation_steps)
        update_index = 0

        for batch_index, (samples, targets) in enumerate(
            logger.log_every(
                self.train_dataloader,
                self.cfg.print_freq,
                header,
            )
        ):
            samples = samples.to(self.device, non_blocking=True)
            targets = [
                {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in target.items()
                }
                for target in targets
            ]
            synchronize = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == epoch_batches
            )
            context = (
                self.model.no_sync()
                if not synchronize and hasattr(self.model, "no_sync")
                else nullcontext()
            )
            with context:
                if self.scaler is not None:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                    ):
                        outputs = self.model(samples, targets=targets)
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=False,
                    ):
                        loss_dict = self.criterion(
                            outputs,
                            targets,
                            epoch=epoch,
                            step=batch_index,
                            global_step=epoch * len(self.train_dataloader)
                            + batch_index,
                            epoch_step=len(self.train_dataloader),
                        )
                    loss = sum(loss_dict.values())
                    self.scaler.scale(loss / accumulation_steps).backward()
                else:
                    outputs = self.model(samples, targets=targets)
                    loss_dict = self.criterion(
                        outputs,
                        targets,
                        epoch=epoch,
                        step=batch_index,
                        global_step=epoch * len(self.train_dataloader)
                        + batch_index,
                        epoch_step=len(self.train_dataloader),
                    )
                    loss = sum(loss_dict.values())
                    (loss / accumulation_steps).backward()

            if synchronize:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.clip_max_norm,
                    )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
                if scheduler is not None:
                    global_update = epoch * updates_per_epoch + update_index
                    self.optimizer = scheduler.step(
                        global_update,
                        self.optimizer,
                    )
                update_index += 1

            reduced = dist_utils.reduce_dict(loss_dict)
            reduced_loss = sum(reduced.values())
            if not math.isfinite(float(reduced_loss)):
                raise FloatingPointError(
                    f"Non-finite V2 loss at epoch={epoch}, "
                    f"batch={batch_index}: {reduced}"
                )
            logger.update(loss=reduced_loss, **reduced)
            logger.update(lr=self.optimizer.param_groups[0]["lr"])
            if batch_index + 1 >= epoch_batches:
                break

        logger.synchronize_between_processes()
        print("Averaged stats:", logger)
        return {
            name: meter.global_avg for name, meter in logger.meters.items()
        }

    def fit(self):
        """Competition-oriented fit loop with true global-batch accumulation."""
        self.train()
        accumulation = int(
            self.cfg.yaml_cfg.get("gradient_accumulation_steps", 1)
        )
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        max_batches = int(self.cfg.yaml_cfg.get("max_train_batches", 0))
        epoch_batches = (
            min(len(self.train_dataloader), max_batches)
            if max_batches > 0
            else len(self.train_dataloader)
        )
        updates_per_epoch = math.ceil(epoch_batches / accumulation)
        scheduler = None
        if self.cfg.lrsheduler is not None:
            scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.cfg.lr_gamma,
                updates_per_epoch,
                total_epochs=self.cfg.epoches,
                warmup_iter=self.cfg.warmup_iter,
                flat_epochs=self.cfg.flat_epoch,
                no_aug_epochs=self.cfg.no_aug_epoch,
            )

        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        effective_batch = (
            int(self.cfg.yaml_cfg["train_dataloader"]["total_batch_size"])
            * accumulation
        )
        print(
            json.dumps(
                {
                    "v2_training": {
                        "trainable_parameters": trainable,
                        "gradient_accumulation_steps": accumulation,
                        "effective_global_batch": effective_batch,
                        "updates_per_epoch": updates_per_epoch,
                    }
                },
                indent=2,
            )
        )

        best_ap = -1.0
        start_epoch = self.last_epoch + 1
        start_time = time.time()
        eval_every = int(self.cfg.yaml_cfg.get("eval_every", 1))
        for epoch in range(start_epoch, self.cfg.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = self._train_one_epoch_accumulated(
                epoch,
                scheduler,
                accumulation,
            )
            if scheduler is None:
                if (
                    self.lr_warmup_scheduler is None
                    or self.lr_warmup_scheduler.finished()
                ):
                    self.lr_scheduler.step()
            self.last_epoch = epoch

            should_evaluate = (
                (epoch + 1) % eval_every == 0
                or epoch + 1 == self.cfg.epoches
            )
            test_stats: dict[str, Any] = {}
            coco_evaluator = None
            if should_evaluate:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                )
                current_ap = float(test_stats["coco_eval_bbox"][0])
                if current_ap > best_ap:
                    best_ap = current_ap
                    if self.output_dir:
                        dist_utils.save_on_master(
                            self.state_dict(),
                            self.output_dir / "best.pth",
                        )

            if self.output_dir:
                dist_utils.save_on_master(
                    self.state_dict(),
                    self.output_dir / "last.pth",
                )
                if (epoch + 1) % int(self.cfg.checkpoint_freq) == 0:
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / f"checkpoint{epoch + 1:04}.pth",
                    )
            log_stats = {
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
                "epoch": epoch,
                "best_ap_50_95": best_ap,
                "n_parameters": trainable,
            }
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    eval_dir = self.output_dir / "eval"
                    eval_dir.mkdir(exist_ok=True)
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        eval_dir / "latest.pth",
                    )
        print(
            "Training time",
            str(timedelta(seconds=int(time.time() - start_time))),
        )
