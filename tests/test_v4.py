from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "storage/upstream/Co-DETR"

if importlib.util.find_spec("mmcv") is None:
    raise unittest.SkipTest(
        "V4 tests require the pinned Co-DINO/MMCV training environment"
    )

sys.path.insert(0, str(UPSTREAM))
from mmcv import Config  # noqa: E402

from hod26.v4.checkpoint import _semantic_rows  # noqa: E402
from hod26.v4.constants import COCO_CLASSES, HOD26_CLASSES  # noqa: E402
from hod26.v4.data import (  # noqa: E402
    HOD26CocoDataset,
    _four_image_mosaic,
    normalize_cube,
)
from hod26.v4.infer import _invert_boxes, _soft_nms_class  # noqa: E402
from hod26.v4.model import (  # noqa: E402
    CrossSpectralObjectRefinement,
    Float32ChannelLayerNorm,
    ObjectAwareFeatureGate,
    SpectralFusionBlock,
    build_gaussian_object_targets,
)
from hod26.v4.optimize import (  # noqa: E402
    ViewSpec,
    _hard_nms,
    _weighted_box_fusion,
)
from hod26.v4.optim import piecewise_cosine_ratio  # noqa: E402
from hod26.v4.soup import average_checkpoints  # noqa: E402


class V4Tests(unittest.TestCase):
    def test_soft_nms_method_is_explicit(self) -> None:
        detections = np.asarray([[1.0, 2.0, 10.0, 12.0, 0.9]], dtype=np.float32)
        with mock.patch("mmcv.ops.soft_nms") as soft_nms:
            soft_nms.return_value = (detections.copy(), np.asarray([0]))
            result = _soft_nms_class(
                [detections],
                iou_threshold=0.55,
                sigma=0.5,
                min_score=0.0001,
                method="gaussian",
            )
        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(soft_nms.call_args.kwargs["method"], "gaussian")

        with self.assertRaisesRegex(ValueError, "Unsupported Soft-NMS"):
            _soft_nms_class(
                [detections],
                iou_threshold=0.55,
                sigma=0.5,
                min_score=0.0001,
                method="invalid",
            )

    def test_channel_layer_norm_keeps_half_backward_finite(self) -> None:
        layer = Float32ChannelLayerNorm(8)
        values = torch.full((2, 8, 5, 7), 1024.0, dtype=torch.float16)
        values.requires_grad_(True)
        weights = torch.linspace(
            -1.0,
            1.0,
            values.numel(),
            dtype=torch.float32,
        ).reshape(values.shape)
        output = layer(values)
        loss = (output.float() * weights).sum()
        loss.backward()
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_zero_start_spectral_fusion_is_exact_identity(self) -> None:
        torch.manual_seed(7)
        block = SpectralFusionBlock(6, 8).eval()
        base = torch.randn(2, 8, 9, 13)
        spectral = torch.randn(2, 6, 5, 7)
        with torch.no_grad():
            output = block(base, spectral)
        torch.testing.assert_close(output, base, rtol=0.0, atol=0.0)
        self.assertTrue(torch.count_nonzero(block.gate).item() == 0)

    def test_object_refinement_paths_start_as_exact_identity(self) -> None:
        torch.manual_seed(11)
        features = torch.randn(2, 8, 16, 24)
        mask = torch.rand(2, 1, 16, 24)
        gate = ObjectAwareFeatureGate(8).eval()
        with torch.no_grad():
            gated = gate(features, mask)
        torch.testing.assert_close(gated, features, rtol=0.0, atol=0.0)

        high = torch.randn(2, 8, 8, 12)
        refinement = CrossSpectralObjectRefinement(8).eval()
        with torch.no_grad():
            refined = refinement(high, features, mask)
        torch.testing.assert_close(refined, high, rtol=0.0, atol=0.0)

    def test_gaussian_object_targets_follow_boxes(self) -> None:
        targets = build_gaussian_object_targets(
            [torch.tensor([[32.0, 16.0, 96.0, 80.0]])],
            batch_input_shape=(128, 256),
            feature_shape=(32, 64),
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(targets.shape), (1, 1, 32, 64))
        peak = np.unravel_index(int(targets.argmax()), targets.shape)
        self.assertEqual(peak[2:], (12, 16))
        self.assertGreater(float(targets[0, 0, 12, 16]), 0.95)
        self.assertEqual(float(targets[0, 0, 0, 0]), 0.0)

    def test_native_mosaic_preserves_bands_boxes_and_labels(self) -> None:
        samples = []
        for index in range(4):
            image = np.full((6, 10, 16), index / 4, dtype=np.float32)
            boxes = np.asarray([[1.0, 1.0, 5.0, 4.0]], dtype=np.float32)
            labels = np.asarray([index], dtype=np.int64)
            samples.append((image, boxes, labels))
        np.random.seed(17)
        image, boxes, labels = _four_image_mosaic(samples)
        self.assertEqual(image.shape, (12, 20, 16))
        self.assertEqual(boxes.shape, (4, 4))
        self.assertEqual(sorted(labels.tolist()), [0, 1, 2, 3])
        self.assertGreaterEqual(float(boxes.min()), 0.0)
        self.assertLessEqual(float(boxes[:, 0::2].max()), 20.0)
        self.assertLessEqual(float(boxes[:, 1::2].max()), 12.0)

    def test_normalization_and_flip_box_inverse(self) -> None:
        cube = np.arange(2 * 3 * 16, dtype=np.float32).reshape(2, 3, 16)
        low = np.zeros(16, dtype=np.float32)
        high = np.full(16, 100.0, dtype=np.float32)
        normalized = normalize_cube(cube, low, high)
        self.assertEqual(normalized.shape, (2, 3, 16))
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

        width, height = 502, 248
        original = np.asarray([[31.0, 27.0, 141.0, 119.0]], dtype=np.float32)
        flipped = original.copy()
        flipped[:, 0] = width - original[:, 2]
        flipped[:, 2] = width - original[:, 0]
        sx, sy = 2.0, 2.0
        network_boxes = flipped * np.asarray((sx, sy, sx, sy), np.float32)
        restored = _invert_boxes(
            network_boxes,
            sx=sx,
            sy=sy,
            original_width=width,
            original_height=height,
            flip="horizontal",
        )
        np.testing.assert_allclose(restored, original, rtol=0.0, atol=1e-6)

    def test_absolute_update_schedule_and_tail_clamp(self) -> None:
        points = [(0, 0.02), (1500, 1.0), (12000, 1.0), (101232, 0.003)]
        self.assertEqual(piecewise_cosine_ratio(0, points), 0.02)
        self.assertEqual(piecewise_cosine_ratio(1500, points), 1.0)
        self.assertEqual(piecewise_cosine_ratio(12000, points), 1.0)
        self.assertEqual(piecewise_cosine_ratio(999999, points), 0.003)
        self.assertGreater(piecewise_cosine_ratio(500, points), 0.02)

    def test_semantic_class_initialization(self) -> None:
        source = torch.arange(80, dtype=torch.float32).view(80, 1)
        target = torch.zeros(18, 1)
        remapped = _semantic_rows(source, target)
        coco_index = {name: index for index, name in enumerate(COCO_CLASSES)}
        self.assertEqual(
            remapped[HOD26_CLASSES.index("apple")].item(),
            float(coco_index["apple"]),
        )
        expected_ebike = (coco_index["bicycle"] + coco_index["motorcycle"]) / 2
        self.assertEqual(remapped[HOD26_CLASSES.index("e-bike")].item(), expected_ebike)

    def test_single_checkpoint_soup_uses_primary_ema_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.pth", root / "b.pth"]
            for path, value in zip(paths, (1.0, 3.0)):
                torch.save(
                    {
                        "meta": {"config": "same", "epoch": int(value)},
                        "state_dict": {
                            "weight": torch.tensor([value]),
                            "counter": torch.tensor(4, dtype=torch.int64),
                            "ema_weight": torch.tensor([99.0]),
                        },
                    },
                    path,
                )
            output = root / "soup.pth"
            report = average_checkpoints(paths, output, weights=(1.0, 3.0))
            payload = torch.load(output, map_location="cpu")
            self.assertEqual(payload["state_dict"]["weight"].item(), 2.5)
            self.assertNotIn("ema_weight", payload["state_dict"])
            self.assertTrue(report["single_output_checkpoint"])
            self.assertFalse(report["prediction_ensemble"])

    def test_tta_view_parser_and_box_merge_primitives(self) -> None:
        self.assertEqual(ViewSpec.parse("1280:horizontal").key, "w1280_horizontal")
        with self.assertRaises(ValueError):
            ViewSpec.parse("1279:none")

        detections = np.asarray(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [1.0, 1.0, 11.0, 11.0, 0.8],
                [20.0, 20.0, 30.0, 30.0, 0.7],
            ],
            dtype=np.float32,
        )
        kept = _hard_nms(detections, 0.5)
        self.assertEqual(len(kept), 2)
        np.testing.assert_allclose(kept[:, 4], [0.9, 0.7], atol=1e-6)

        fused = _weighted_box_fusion(
            [
                detections[:1],
                np.asarray([[2.0, 0.0, 12.0, 10.0, 0.8]], dtype=np.float32),
                np.zeros((0, 5), dtype=np.float32),
            ],
            iou_threshold=0.5,
            score_mode="avg_all",
        )
        self.assertEqual(fused.shape, (1, 5))
        self.assertAlmostEqual(float(fused[0, 4]), (0.9 + 0.8) / 3, places=6)
        self.assertGreater(float(fused[0, 0]), 0.0)
        self.assertLess(float(fused[0, 0]), 2.0)

    def test_formal_config_invariants_and_bounded_sampling(self) -> None:
        config = Config.fromfile(
            str(REPO_ROOT / "configs/v4/co_dino_vitl_hsi_fold0.py")
        )
        self.assertEqual(config.model.type, "HOD26ObjectAwareCoDETR")
        self.assertEqual(config.model.object_activation_loss_weight, 0.6)
        self.assertEqual(config.model.object_activation_background_weight, 0.1)
        self.assertEqual(config.model.query_head.num_classes, 18)
        self.assertEqual(config.model.query_head.num_query, 1500)
        self.assertEqual(config.runner.max_epochs, 72)
        self.assertEqual(config.custom_hooks[1].priority, "HIGHEST")
        self.assertEqual(config.checkpoint_config.interval, 2)
        self.assertEqual(config.evaluation.interval, 2)
        self.assertEqual(tuple(config.evaluation.proposal_nums), (1, 10, 100))
        self.assertEqual(
            config.data.train.repeat_class_factors,
            {5: 2, 8: 2, 14: 2, 16: 2},
        )
        augment = config.data.train.pipeline[2]
        self.assertEqual(augment.mosaic_probability, 0.25)
        self.assertEqual(augment.mixup_probability, 0.0)

        dataset = HOD26CocoDataset(
            ann_file=str(REPO_ROOT / "storage/v2/dataset/fold0/train.json"),
            img_prefix=str(REPO_ROOT / "data/raw/data_train/data_train/VIS"),
            pipeline=[],
            classes=None,
            filter_empty_gt=False,
            repeat_class_factors={5: 2, 8: 2, 14: 2, 16: 2},
        )
        self.assertEqual(dataset.base_image_count, 2400)
        self.assertEqual(len(dataset), 2812)
        self.assertEqual(dataset.repeat_histogram, {1: 1988, 2: 412})
        context = dataset.prepare_train_img(0)
        self.assertIs(context["hod26_dataset"], dataset)
        self.assertEqual(context["hod26_sample_idx"], 0)
        np.random.seed(29)
        extra = dataset.sample_mosaic_indices(0, count=3)
        source_ids = [dataset.img_ids[0], *(dataset.img_ids[index] for index in extra)]
        self.assertEqual(len(set(source_ids)), 4)

    def test_fold_statistics_exclude_validation(self) -> None:
        payload = json.loads(
            (REPO_ROOT / "storage/v4/dataset/fold0_band_stats.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["scope"], "fold0_train_only")
        self.assertEqual(payload["excluded_validation_fold"], 0)
        self.assertEqual(payload["samples"], 2400)


if __name__ == "__main__":
    unittest.main()
