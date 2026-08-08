from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "storage/upstream/Co-DETR"
if importlib.util.find_spec("mmcv") is None:
    pytest.skip("V6 tests require the pinned Co-DINO environment", allow_module_level=True)
sys.path.insert(0, str(UPSTREAM))

from mmcv import Config  # noqa: E402

from hod26.v6.constants import (  # noqa: E402
    FOLD0_CLASS_COUNTS,
    OFFICIAL_CHECKPOINT_SHA256,
    WAVELENGTHS_NM,
)
from hod26.v6.data import (  # noqa: E402
    HOD26V6FixedSensorShift,
    apply_fixed_sensor_shift,
    four_scene_mosaic,
)
from hod26.v6.infer import _gaussian_soft_nms  # noqa: E402
from hod26.v6.model import (  # noqa: E402
    BidirectionalDeformableFusion,
    HOD26V6AlignCoDINOHead,
    HOD26V6WavelengthViT,
    ConvNormAct,
    Float32GroupNorm,
    HOD26V6CoSpecDINO,
    _rank_weights,
    effective_number_class_weights,
)
from hod26.v6.hooks import HOD26V6EMAHook  # noqa: E402
from hod26.v6.optim import (  # noqa: E402
    _is_new_v6_parameter,
    piecewise_cosine_ratio,
    stable_clip_grad_norm_,
)


def test_wavelength_axis_is_physical_monotone_and_nonuniform() -> None:
    wavelengths = np.asarray(WAVELENGTHS_NM, np.float32)
    spacing = np.diff(wavelengths)
    assert len(wavelengths) == 16
    assert np.all(spacing > 0)
    assert float(spacing.max() - spacing.min()) > 1.0


def test_effective_number_weights_are_bounded_and_favor_rare_classes() -> None:
    weights = effective_number_class_weights()
    assert tuple(weights.shape) == (18,)
    assert torch.isfinite(weights).all()
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    stone = 16
    badminton = 2
    assert FOLD0_CLASS_COUNTS[stone] < FOLD0_CLASS_COUNTS[badminton]
    assert float(weights[stone]) > float(weights[badminton])
    assert float(weights.max()) < 1.6


def test_align_rank_weights_are_per_gt_quality_ordered() -> None:
    quality = torch.tensor([0.9, 0.4, 0.8, 0.2])
    boxes = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.1, 0.1, 0.2, 0.2],
            [0.7, 0.7, 0.1, 0.1],
            [0.7, 0.7, 0.1, 0.1],
        ]
    )
    labels = torch.tensor([5, 5, 8, 8])
    weights = _rank_weights(
        quality,
        boxes,
        labels,
        num_images=1,
        queries_per_image=4,
        num_classes=18,
        tau=1.5,
    )
    expected_tail = math.exp(-1 / 1.5)
    torch.testing.assert_close(
        weights, torch.tensor([1.0, expected_tail, 1.0, expected_tail])
    )


def test_align_loss_keeps_soft_targets_in_fp32_under_amp() -> None:
    source = inspect.getsource(HOD26V6AlignCoDINOHead._aligned_losses)
    assert "loss_logits = flat_logits.float()" in source
    assert "probability = loss_logits.sigmoid()" in source
    assert "F.softplus(-loss_logits)" in source


def test_v6_ema_persists_successful_step_time_constant() -> None:
    hook = HOD26V6EMAHook(momentum=0.0001, total_iter=123)
    assert hook.total_iter == 123
    with pytest.raises(ValueError):
        HOD26V6EMAHook(momentum=0.0001, total_iter=0)


def test_new_spectral_and_fusion_convolutions_use_spatial_group_norm() -> None:
    block = ConvNormAct(16, 96)
    assert isinstance(block[1], Float32GroupNorm)
    assert block[1].num_groups == 32
    value = torch.zeros(1, 16, 8, 12, requires_grad=True)
    output = block(value)
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()

def test_spectral_encoder_is_an_amp_safe_fp32_island() -> None:
    source = inspect.getsource(HOD26V6WavelengthViT.forward)
    assert "with torch.cuda.amp.autocast(enabled=False):" in source
    assert "cube.float()" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv2 smoke needs CUDA")
def test_bidirectional_fusion_identity_and_every_path_gradient() -> None:
    torch.manual_seed(2608)
    block = BidirectionalDeformableFusion(6, 8, hidden=8).cuda().train()
    visual = torch.randn(1, 8, 9, 13, device="cuda", requires_grad=True)
    spectral = torch.randn(1, 6, 9, 13, device="cuda", requires_grad=True)
    with torch.no_grad():
        zero_visual, zero_spectral = block(
            visual.detach(), spectral.detach(), torch.tensor(0.0, device="cuda")
        )
    torch.testing.assert_close(zero_visual, visual.detach(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(zero_spectral, spectral.detach(), rtol=0.0, atol=0.0)
    output_visual, output_spectral = block(
        visual, spectral, torch.tensor(1.0, device="cuda")
    )
    (output_visual.square().mean() + output_spectral.square().mean()).backward()
    missing = [
        name
        for name, parameter in block.named_parameters()
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
    ]
    assert not missing
    assert all(
        "gate" not in name and "coefficient" not in name and "progress" not in name
        for name, _ in block.named_parameters()
    )


def test_native_hsi_mosaic_preserves_sixteen_bands_and_four_scenes() -> None:
    samples = []
    for index in range(4):
        image = np.full((8, 12, 16), index / 4, np.float32)
        boxes = np.asarray([[1, 1, 6, 5]], np.float32)
        labels = np.asarray([index], np.int64)
        samples.append((image, boxes, labels))
    np.random.seed(17)
    image, boxes, labels = four_scene_mosaic(samples)
    assert image.shape == (16, 24, 16)
    assert boxes.shape == (4, 4)
    assert sorted(labels.tolist()) == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", HOD26V6FixedSensorShift.MODES)
def test_fixed_sensor_shifts_are_deterministic(mode: str) -> None:
    rng = np.random.RandomState(8)
    cube = rng.rand(9, 13, 16).astype(np.float32)
    first = apply_fixed_sensor_shift(cube, mode, image_id=42)
    second = apply_fixed_sensor_shift(cube, mode, image_id=42)
    np.testing.assert_array_equal(first, second)
    assert first.shape == cube.shape
    assert first.dtype == np.float32
    assert 0 <= float(first.min()) <= float(first.max()) <= 1


def test_gaussian_soft_nms_protocol_is_frozen() -> None:
    detections = np.asarray([[1, 2, 10, 12, 0.9]], np.float32)
    with mock.patch("mmcv.ops.soft_nms") as soft_nms:
        soft_nms.return_value = (detections.copy(), np.asarray([0]))
        result = _gaussian_soft_nms([detections])
    assert result.shape == (1, 5)
    assert soft_nms.call_args.kwargs == {
        "iou_threshold": 0.55,
        "sigma": 0.5,
        "min_score": 0.0001,
        "method": "gaussian",
        "offset": 0,
    }


def test_update_schedule_and_optimizer_families_are_explicit() -> None:
    points = [(0, 0.02), (1500, 1.0), (6000, 1.0), (96000, 0.004)]
    assert piecewise_cosine_ratio(0, points) == 0.02
    assert piecewise_cosine_ratio(1500, points) == 1.0
    assert piecewise_cosine_ratio(999999, points) == 0.004
    assert _is_new_v6_parameter("backbone.spectral_stem.0.0.weight")
    assert _is_new_v6_parameter("neck.fusions.0.visual_output.weight")
    assert _is_new_v6_parameter("neck.spectral_proposal_laterals.0.0.weight")


def test_stable_gradient_clip_does_not_zero_fp32_overflowing_global_norm() -> None:
    first = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))
    second = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))
    first.grad = torch.full_like(first, 1e20)
    second.grad = torch.full_like(second, 1e20)
    norm = stable_clip_grad_norm_((first, second), max_norm=0.1, norm_type=2)
    assert norm is not None and torch.isfinite(norm)
    assert float(norm) == pytest.approx(4e20, rel=1e-6)
    clipped = torch.linalg.vector_norm(
        torch.cat((first.grad.double(), second.grad.double())), ord=2
    )
    assert float(clipped) == pytest.approx(0.1, rel=1e-5)
    assert bool((first.grad != 0).all())

def test_formal_fold0_and_full_data_contracts() -> None:
    fold = Config.fromfile(str(REPO_ROOT / "configs/v6/co_spec_dino_vitl_fold0.py"))
    assert fold.model.type == "HOD26V6CoSpecDINO"
    assert fold.model.query_head.type == "HOD26V6AlignCoDINOHead"
    assert fold.model.bbox_head[0].type == "HOD26V6SpectralProposalHead"
    assert fold.model.backbone.input_bands == 16
    assert fold.model.backbone.use_spectral_checkpoint is True
    assert fold.model.neck.use_fusion_checkpoint is True
    assert fold.runner.max_epochs * 1200 == 96000
    assert tuple(fold.data.train.pipeline[2].stage_updates) == (6000, 56400, 80400)
    assert "repeat_class_factors" not in fold.data.train
    assert fold.optimizer.lr == pytest.approx(1.25e-5)
    assert fold.optimizer.paramwise_cfg.visual_lr_mult == 1.0
    assert fold.optimizer.paramwise_cfg.new_lr_mult == 8.0
    assert fold.optimizer.paramwise_cfg.class_lr_mult == 4.0
    assert fold.fp16 is None
    assert fold.optimizer_config.type == "HOD26V6StableFp16OptimizerHook"
    assert fold.optimizer_config.loss_scale.init_scale == 1.0
    assert fold.checkpoint_config.interval == 2
    assert fold.evaluation.interval == 2

    full = Config.fromfile(str(REPO_ROOT / "configs/v6/co_spec_dino_vitl_full.py"))
    assert full.data.train.ann_file.endswith("/full.json")
    assert full.data.train.pipeline[0].stats_file.endswith("/full_band_stats.json")
    assert full.data.train.pipeline[2].mosaic_probability == 0.0
    assert tuple(full.data.train.pipeline[2].widths.polish) == (1280,)
    assert full.evaluation.interval == 999999


def test_direct_spectral_proposal_is_not_a_renamed_visual_route() -> None:
    source = inspect.getsource(HOD26V6CoSpecDINO.forward_train)
    assert "last_spectral_proposal_features" in source
    assert "bbox_head.forward_train" in source
    assert "spectral_proposals" in source
    assert "forward_train_aux" in source


def test_v6_is_independent_and_uses_only_one_public_checkpoint() -> None:
    assert len(OFFICIAL_CHECKPOINT_SHA256) == 64
    for path in (REPO_ROOT / "src/hod26/v6").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "hod26.v4" not in source
        assert "hod26.v5" not in source
        assert "hod26.v5r" not in source


def test_all_launches_detach_from_ssh_and_inference_selects_stats() -> None:
    for name in ("launch_v6_detached.sh", "launch_v6_full_detached.sh"):
        source = (REPO_ROOT / "tools" / name).read_text(encoding="utf-8")
        assert "nohup setsid env" in source
        assert "</dev/null" in source
        assert "tty=" in source.lower() or "TTY=" in source
    infer_source = (REPO_ROOT / "src/hod26/v6/infer.py").read_text(encoding="utf-8")
    infer_launcher = (REPO_ROOT / "tools/infer_v6_champion.sh").read_text(encoding="utf-8")
    full_train = (REPO_ROOT / "tools/train_v6_full.sh").read_text(encoding="utf-8")
    full_launch = (REPO_ROOT / "tools/launch_v6_full_detached.sh").read_text(encoding="utf-8")
    assert "hod26.v6.load_gate resume" in infer_launcher
    assert "V6_FULL_AUDIT_ONLY" in full_train
    assert "co_spec_dino_vitl_(fold0|full).py" in full_launch
    assert 'stats_path: str | Path = "auto"' in infer_source
    assert "hod26_v6_full_data_stage" in infer_source
