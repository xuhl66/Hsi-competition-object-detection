from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "storage/upstream/Co-DETR"

if importlib.util.find_spec("mmcv") is None:
    raise unittest.SkipTest(
        "V4 sliced tests require the pinned Co-DINO/MMCV environment"
    )

sys.path.insert(0, str(UPSTREAM))

from hod26.v4.sliced import (  # noqa: E402
    ALL_SLICES,
    FusionConfig,
    _box_is_slice_safe,
    _calibrated_score,
    _cluster_tile_candidates,
    _deployment_config,
    _greedy_one_to_one,
)


class V4SlicedTests(unittest.TestCase):
    def test_four_slices_cover_every_pixel_with_expected_overlap(self) -> None:
        width, height = 502, 248
        coverage = np.zeros((height, width), dtype=np.uint8)
        bounds = [spec.bounds(width, height) for spec in ALL_SLICES]
        for x0, y0, x1, y1 in bounds:
            coverage[y0:y1, x0:x1] += 1
            self.assertEqual((x1 - x0, y1 - y0), (301, 149))
        self.assertGreaterEqual(int(coverage.min()), 1)
        self.assertEqual(int(coverage.max()), 4)
        self.assertEqual(bounds[0], (0, 0, 301, 149))
        self.assertEqual(bounds[-1], (201, 99, 502, 248))

    def test_artificial_crop_edges_are_rejected_but_image_edges_are_valid(self) -> None:
        crop = np.asarray([201.0, 99.0, 502.0, 248.0], dtype=np.float32)
        internal_edge = np.asarray([201.0, 120.0, 240.0, 160.0])
        real_edge = np.asarray([460.0, 210.0, 502.0, 248.0])
        interior = np.asarray([250.0, 130.0, 300.0, 180.0])
        self.assertFalse(
            _box_is_slice_safe(internal_edge, crop, 502, 248, 0.015)
        )
        self.assertTrue(_box_is_slice_safe(real_edge, crop, 502, 248, 0.015))
        self.assertTrue(_box_is_slice_safe(interior, crop, 502, 248, 0.015))

    def test_greedy_matching_is_one_to_one(self) -> None:
        base = np.asarray(
            [[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]],
            dtype=np.float32,
        )
        candidates = np.asarray(
            [[0, 0, 10, 10, 0.9], [30, 30, 40, 40, 0.8]],
            dtype=np.float32,
        )
        matches, used = _greedy_one_to_one(
            base,
            candidates,
            eligible_base=np.ones(3, dtype=bool),
            candidate_mask=np.ones(2, dtype=bool),
            iou_threshold=0.5,
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(used, {0, 1})
        self.assertEqual(matches[0][:2], (0, 0))
        self.assertEqual(matches[1][:2], (2, 1))

    def test_clusters_never_use_two_detections_from_one_slice(self) -> None:
        candidates = [
            (0, np.asarray([0, 0, 10, 10, 0.9], dtype=np.float32)),
            (0, np.asarray([1, 0, 11, 10, 0.8], dtype=np.float32)),
            (1, np.asarray([0, 1, 10, 11, 0.7], dtype=np.float32)),
        ]
        clusters = _cluster_tile_candidates(candidates, iou_threshold=0.5)
        for cluster in clusters:
            tile_ids = [tile_id for tile_id, _ in cluster]
            self.assertEqual(len(tile_ids), len(set(tile_ids)))
        self.assertEqual(sorted(len(cluster) for cluster in clusters), [1, 2])

    def test_consistency_calibration_is_bounded_and_round_trips(self) -> None:
        config = FusionConfig(
            name="test",
            support_slope=0.3,
            iou_slope=1.0,
        )
        high = _calibrated_score(
            0.8,
            support_ratio=1.0,
            mean_iou=0.9,
            config=config,
        )
        low = _calibrated_score(
            0.8,
            support_ratio=0.0,
            mean_iou=0.4,
            config=config,
        )
        self.assertGreater(high, 0.8)
        self.assertLess(low, 0.8)
        self.assertLessEqual(high, 1.0)
        self.assertEqual(FusionConfig.from_dict(asdict(config)), config)

    def test_negligible_optional_addition_is_not_deployed(self) -> None:
        core = FusionConfig(name="core")
        addition = FusionConfig(name="addition", tile_enabled=True)
        report = {
            "selected": {
                "tune": {
                    "stage": "additions",
                    "config": asdict(addition),
                    "map_50_95": 0.700001,
                    "map_75": 0.8,
                    "map_small": 0.6,
                    "weak4_map_50_95": 0.5,
                }
            },
            "results": [
                {
                    "stage": "calibration",
                    "config": asdict(core),
                    "map_50_95": 0.700000,
                    "map_75": 0.8,
                    "map_small": 0.6,
                    "weak4_map_50_95": 0.5,
                }
            ],
        }
        deployed, decision = _deployment_config(report)
        self.assertEqual(deployed.name, "core")
        self.assertTrue(decision["tile_addition_rejected_as_noise"])


if __name__ == "__main__":
    unittest.main()
