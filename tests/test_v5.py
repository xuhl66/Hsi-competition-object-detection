from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "storage/upstream/Co-DETR"
if importlib.util.find_spec("mmcv") is None:
    raise unittest.SkipTest("V5 tests require the pinned Co-DINO environment")
sys.path.insert(0, str(UPSTREAM))

from mmcv import Config  # noqa: E402

from hod26.v5.data import spectral_augment_order_agnostic  # noqa: E402
from hod26.v5.hooks import HOD26V5InheritedEMAHook  # noqa: E402
from hod26.v5.load_gate import (  # noqa: E402
    RuntimeSchema,
    StateLoadGateError,
    validate_checkpoint_payload,
)
from hod26.v5.model import (  # noqa: E402
    FDRMLP,
    LocationQualityEstimator,
    collapse_adjacent_difference_weight,
    global_optimal_bbox_targets,
)
from hod26.v5.stateful import transplant_optimizer_state  # noqa: E402


class V5Tests(unittest.TestCase):
    @staticmethod
    def _gate_fixture(*, resume: bool = False):
        inherited = "backbone.pos_embed"
        fresh = "query_head.fdr_branches.0.layers.0.weight"
        fields = {
            "group_name": "mixed_decay",
            "initial_lr": 1e-5,
            "lr_scale": 1.0,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        }
        schema = RuntimeSchema(
            model={
                inherited: ((2,), "torch.float32"),
                fresh: ((2, 2), "torch.float32"),
            },
            trainable_shapes={inherited: (2,), fresh: (2, 2)},
            optimizer_groups=[
                {
                    "params": [0, 1],
                    "param_names": [inherited, fresh],
                    **fields,
                }
            ],
        )
        contract = {
            "schema_version": 1,
            "hod26_version": "v5",
            "lineage": {
                "parent_checkpoint_sha256": "parent-sha",
                "parent_epoch": 3,
            },
            "model": {"state_tensors": 2, "ema_pairs": 2},
            "optimizer": {
                "parent_attempted_iterations": 10,
                "parent_successful_steps": 8,
                "parent_amp_skipped_steps": 2,
                "parent_optimizer_states": 1,
                "target_trainable_tensors": 2,
                "bootstrap_stateful_states": 1,
                "fresh_states": 1,
                "semantic_salience_clone_states": 0,
                "fresh_prefixes": [
                    "query_head.fdr_branches.",
                    "query_head.lqe_layers.",
                ],
            },
            "fp16_scaler": {
                "parent_was_serialized": True,
                "parent_scale": 8.0,
                "parent_growth_tracker": 17,
                "bootstrap_inherited": False,
                "bootstrap_policy": "fresh_recalibration_for_changed_v5_graph",
                "resume_required": True,
            },
            "ema": {
                "inherited_attempted_age": 10,
                "current_lineage_update_order": "pre_optimizer_step_v4_compatible",
                "current_lineage_smooths_control_buffers": True,
            },
        }

        def adam(shape, step):
            return {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.ones(shape),
                "exp_avg_sq": torch.ones(shape),
            }

        state_dict = {
            inherited: torch.ones(2),
            fresh: torch.ones(2, 2),
            "ema_backbone_pos_embed": torch.ones(2),
            "ema_query_head_fdr_branches_0_layers_0_weight": torch.ones(2, 2),
        }
        optimizer_state = {0: adam((2,), 8)}
        meta = {
            "hod26_version": "v5",
            "optimizer_state_transplant": True,
            "optimizer_parent_iter": 10,
            "optimizer_parent_epoch": 3,
            "optimizer_parent_sha256": "parent-sha",
            "parent_optimizer_states": 1,
            "ema_inherited_updates": 10,
            "fresh_optimizer_names": [fresh],
            "epoch": 0,
            "iter": 0,
        }
        if resume:
            optimizer_state[0] = adam((2,), 11)
            optimizer_state[1] = adam((2, 2), 3)
            meta.update(
                epoch=2,
                iter=4,
                fp16={
                    "loss_scaler": {
                        "scale": 2.0,
                        "growth_factor": 2.0,
                        "backoff_factor": 0.5,
                        "growth_interval": 2000,
                        "_growth_tracker": 3,
                    }
                },
            )
        payload = {
            "meta": meta,
            "state_dict": state_dict,
            "optimizer": {
                "state": optimizer_state,
                "param_groups": [
                    {
                        "params": [0, 1],
                        "param_names": [inherited, fresh],
                        **fields,
                    }
                ],
            },
        }
        return payload, schema, contract

    def test_exact_spectral_reparameterization(self) -> None:
        torch.manual_seed(17)
        raw = torch.randn(2, 16, 13, 19, dtype=torch.float64)
        weight = torch.randn(7, 31, 3, 3, dtype=torch.float64)
        evidence = torch.cat((raw, raw[:, 1:] - raw[:, :-1]), dim=1)
        expected = F.conv2d(evidence, weight, padding=1)
        actual = F.conv2d(raw, collapse_adjacent_difference_weight(weight), padding=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fdr_and_lqe_shapes(self) -> None:
        features = torch.randn(2, 11, 256)
        corners = FDRMLP(256, 32)(features)
        self.assertEqual(tuple(corners.shape), (2, 11, 132))
        correction = LocationQualityEstimator(32)(corners)
        self.assertEqual(tuple(correction.shape), (2, 11, 1))
        self.assertTrue(torch.isfinite(correction).all())

    def test_band_augmentation_has_no_order_slope(self) -> None:
        image = np.ones((8, 12, 16), dtype=np.float32) * 0.5
        np.random.seed(23)
        output = spectral_augment_order_agnostic(
            image, gain=0.1, band_jitter=0.04, noise=0.0, dropout=0.0
        )
        factors = output.mean(axis=(0, 1))
        self.assertEqual(output.shape, image.shape)
        # Independent, zero-mean band jitter: no forced monotonic wavelength slope.
        self.assertFalse(np.all(np.diff(factors) >= 0))
        self.assertFalse(np.all(np.diff(factors) <= 0))

    def test_global_optimal_localization_target_union(self) -> None:
        def target(positive: dict[int, tuple[float, float, float, float]]):
            boxes = torch.zeros(4, 4)
            weights = torch.zeros(4, 4)
            for query, box in positive.items():
                boxes[query] = torch.tensor(box)
                weights[query] = 1
            return ([torch.zeros(4)], [torch.ones(4)], [boxes], [weights], 0, 0)

        box_a = (0.2, 0.3, 0.1, 0.1)
        box_b = (0.7, 0.6, 0.2, 0.2)
        layers = [
            target({0: box_a, 2: box_b}),
            target({0: box_a, 1: box_b}),
            target({0: box_b, 1: box_b}),
        ]
        boxes, weights = global_optimal_bbox_targets(layers)
        self.assertTrue(torch.equal(weights[0][:, 0], torch.tensor([1, 1, 1, 0])))
        torch.testing.assert_close(boxes[0][0], torch.tensor(box_a))
        torch.testing.assert_close(boxes[0][1], torch.tensor(box_b))
        torch.testing.assert_close(boxes[0][2], torch.tensor(box_b))

    def test_optimizer_state_transplant_preserves_parent_direction(self) -> None:
        def adam(shape, value):
            return {
                "step": torch.tensor(42168.0),
                "exp_avg": torch.full(shape, value),
                "exp_avg_sq": torch.full(shape, value + 1),
            }

        parent = {
            "state": {
                0: adam((3,), 1.0),
                1: adam((2, 31, 1, 1), 2.0),
                2: adam((4,), 3.0),
            },
            "param_groups": [
                {
                    "params": [0, 1, 2],
                    "param_names": [
                        "backbone.pos_embed",
                        "backbone.proxy_residual.0.weight",
                        "neck.object_activation.layers.1.weight",
                    ],
                }
            ],
        }
        target = {
            "state": {},
            "param_groups": [
                {
                    "params": [0, 1, 2, 3],
                    "param_names": [
                        "backbone.pos_embed",
                        "backbone.proxy_residual.0.weight",
                        "neck.salience_high.0.layers.1.weight",
                        "query_head.fdr_branches.0.layers.0.weight",
                    ],
                    "lr": 1e-5,
                }
            ],
        }
        shapes = {
            "backbone.pos_embed": (3,),
            "backbone.proxy_residual.0.weight": (2, 16, 1, 1),
            "neck.salience_high.0.layers.1.weight": (4,),
            "query_head.fdr_branches.0.layers.0.weight": (2, 2),
        }
        transplanted, report = transplant_optimizer_state(parent, target, shapes)
        self.assertEqual(report["direct_states"], 1)
        self.assertEqual(report["raw16_gradient_slice_states"], 1)
        self.assertEqual(report["salience_cloned_states"], 1)
        self.assertEqual(report["fresh_optimizer_states"], 1)
        self.assertEqual(len(transplanted["state"]), 3)
        torch.testing.assert_close(
            transplanted["state"][1]["exp_avg"],
            parent["state"][1]["exp_avg"][:, :16],
        )
        torch.testing.assert_close(
            transplanted["state"][2]["exp_avg"],
            parent["state"][2]["exp_avg"],
        )
        self.assertEqual(transplanted["param_groups"][0]["lr"], 1e-5)

    def test_inherited_ema_does_not_restart_warmup(self) -> None:
        hook = HOD26V5InheritedEMAHook(
            inherited_updates=42180,
            momentum=0.0001,
            total_iter=2000,
        )
        self.assertAlmostEqual(hook.momentum_fun(0), 0.0001, places=8)

    def test_state_load_gate_accepts_declared_bootstrap_scope(self) -> None:
        payload, schema, contract = self._gate_fixture()
        result = validate_checkpoint_payload(
            payload,
            schema,
            contract,
            mode="bootstrap",
            updates_per_epoch=None,
        )
        self.assertEqual(result.report["optimizer_states"], 1)
        self.assertEqual(result.report["fresh_lineage_states"], 1)
        self.assertIsNone(result.scaler)

    def test_state_load_gate_counts_successful_and_skipped_steps(self) -> None:
        payload, schema, contract = self._gate_fixture(resume=True)
        result = validate_checkpoint_payload(
            payload,
            schema,
            contract,
            mode="resume",
            updates_per_epoch=2,
        )
        self.assertEqual(result.report["successful_local_optimizer_steps"], 3)
        self.assertEqual(result.report["skipped_local_optimizer_steps"], 1)
        self.assertEqual(result.scaler["scale"], 2.0)

    def test_state_load_gate_rejects_missing_resume_scaler(self) -> None:
        payload, schema, contract = self._gate_fixture(resume=True)
        del payload["meta"]["fp16"]
        with self.assertRaisesRegex(StateLoadGateError, "requires an FP16 scaler"):
            validate_checkpoint_payload(
                payload,
                schema,
                contract,
                mode="resume",
                updates_per_epoch=2,
            )

    def test_state_load_gate_rejects_silent_optimizer_reordering(self) -> None:
        payload, schema, contract = self._gate_fixture()
        payload["optimizer"]["param_groups"][0]["param_names"].reverse()
        with self.assertRaisesRegex(StateLoadGateError, "parameter order mismatch"):
            validate_checkpoint_payload(
                payload,
                schema,
                contract,
                mode="bootstrap",
                updates_per_epoch=None,
            )

    def test_formal_contract_and_initialization_provenance(self) -> None:
        config = Config.fromfile(
            str(REPO_ROOT / "configs/v5/co_dino_vitl_fdr_salience_fold0.py")
        )
        self.assertEqual(config.model.type, "HOD26V5CoDETR")
        self.assertEqual(config.model.query_head.reg_max, 32)
        self.assertEqual(
            config.model.query_head.transformer.decoder.type,
            "HOD26V5DinoTransformerDecoder",
        )
        self.assertEqual(config.runner.max_epochs * 1406, 118104)
        self.assertEqual(config.checkpoint_config.interval, 2)
        self.assertEqual(config.evaluation.interval, 2)
        self.assertEqual(config.data.train.pipeline[2].mosaic_probability, 0.18)
        self.assertEqual(config.custom_hooks[1].type, "HOD26V5InheritedEMAHook")
        self.assertEqual(config.custom_hooks[1].inherited_updates, 42180)
        report = json.loads(
            (
                REPO_ROOT / "storage/pretrained/v5/"
                "v4_e30_dfine_to_v5_init.pth.provenance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["dfine_fdr_lqe_tensors"], 60)
        self.assertTrue(report["transition_zero"])
        self.assertEqual(len(report["collapsed_tensors"]), 2)
        self.assertEqual(report["frozen_v4_numeric_anchor_tensors"], 3)


if __name__ == "__main__":
    unittest.main()
