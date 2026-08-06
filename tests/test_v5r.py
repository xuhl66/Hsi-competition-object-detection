from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "storage/upstream/Co-DETR"
if importlib.util.find_spec("mmcv") is None:
    raise unittest.SkipTest("V5R tests require the pinned Co-DINO environment")
sys.path.insert(0, str(UPSTREAM))

from mmcv import Config  # noqa: E402
from projects.models.co_dino_head import CoDINOHead  # noqa: E402

from hod26.v5r.hooks import HOD26V5RInheritedEMAHook  # noqa: E402
from hod26.v5r.model import HOD26V5RCoDINOHead, _bounded_box  # noqa: E402
from hod26.v5r.optim import (  # noqa: E402
    _is_fast_hsi_parameter,
    _is_localization_parameter,
)
from hod26.v5r.stateful import transplant_optimizer_state  # noqa: E402


class V5RTests(unittest.TestCase):
    def test_localization_is_not_in_eight_x_hsi_family(self) -> None:
        for name in (
            "query_head.fdr_branches.0.layers.0.weight",
            "query_head.lqe_layers.0.reg_conf.layers.0.weight",
            "query_head.pre_bbox_head.0.weight",
        ):
            self.assertTrue(_is_localization_parameter(name))
            self.assertFalse(_is_fast_hsi_parameter(name))
        self.assertTrue(_is_fast_hsi_parameter("neck.salience_high.0.layers.0.weight"))

    def test_auxiliary_forward_temporarily_disables_fdr(self) -> None:
        fdr = object()
        pre = object()
        decoder = SimpleNamespace(
            _v5_fdr_branches_ref=fdr,
            _v5r_pre_bbox_head_ref=pre,
        )
        fake = SimpleNamespace(transformer=SimpleNamespace(decoder=decoder))
        test_case = self

        def native_forward(_head, *args, **kwargs):
            test_case.assertIsNone(decoder._v5_fdr_branches_ref)
            test_case.assertIsNone(decoder._v5r_pre_bbox_head_ref)
            return "native"

        with patch.object(CoDINOHead, "forward_aux", new=native_forward):
            result = HOD26V5RCoDINOHead.forward_aux(fake, None, None, None, 0)
        self.assertEqual(result, "native")
        self.assertIs(decoder._v5_fdr_branches_ref, fdr)
        self.assertIs(decoder._v5r_pre_bbox_head_ref, pre)

    def test_box_bound_is_finite_and_shape_preserving(self) -> None:
        value = torch.tensor([[-1.0, 2.0, -0.5, 4.0], [0.2, 0.3, 0.4, 0.5]])
        bounded = _bounded_box(value)
        self.assertEqual(bounded.shape, value.shape)
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue((bounded[:, :2] > 0).all())
        self.assertTrue((bounded[:, :2] < 1).all())
        self.assertTrue((bounded[:, 2:] > 0).all())
        self.assertTrue((bounded[:, 2:] <= 1.5).all())

    def test_optimizer_transplant_keeps_semantic_clones_fresh(self) -> None:
        def adam(shape):
            return {
                "step": torch.tensor(42168.0),
                "exp_avg": torch.ones(shape),
                "exp_avg_sq": torch.ones(shape),
            }

        parent = {
            "state": {
                0: adam((2,)),
                1: adam((3,)),
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "param_names": [
                        "backbone.pos_embed",
                        "neck.object_activation.layers.1.weight",
                    ],
                }
            ],
        }
        target = {
            "state": {},
            "param_groups": [
                {
                    "params": [0, 1, 2, 3, 4],
                    "param_names": [
                        "backbone.pos_embed",
                        "neck.object_activation.layers.1.weight",
                        "neck.salience_high.0.layers.1.weight",
                        "query_head.pre_bbox_head.0.weight",
                        "query_head.fdr_branches.0.layers.0.weight",
                    ],
                }
            ],
        }
        shapes = {
            "backbone.pos_embed": (2,),
            "neck.object_activation.layers.1.weight": (3,),
            "neck.salience_high.0.layers.1.weight": (3,),
            "query_head.pre_bbox_head.0.weight": (4, 4),
            "query_head.fdr_branches.0.layers.0.weight": (4, 4),
        }
        transplanted, report = transplant_optimizer_state(parent, target, shapes)
        self.assertEqual(len(transplanted["state"]), 2)
        self.assertEqual(report["fresh_salience_states"], 1)
        self.assertEqual(report["fresh_pre_bbox_states"], 1)
        self.assertEqual(report["fresh_fdr_states"], 1)
        self.assertEqual(report["salience_cloned_states"], 0)

    def test_post_step_ema_skips_overflow_and_copies_control(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        control = torch.tensor(0.5)
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        optimizer.state[parameter] = {
            "step": torch.tensor(8.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        hook = HOD26V5RInheritedEMAHook(
            inherited_updates=10,
            parent_successful_steps=8,
            momentum=0.1,
            total_iter=2,
        )
        hook.model_parameters = {"weight": parameter, "v5_transition": control}
        hook.param_ema_buffer = {
            "weight": "ema_weight",
            "v5_transition": "ema_v5_transition",
        }
        hook.model_buffers = {
            "ema_weight": torch.tensor([1.0]),
            "ema_v5_transition": torch.tensor(0.0),
        }
        hook._last_optimizer_step = 8
        runner = SimpleNamespace(
            optimizer=optimizer,
            iter=0,
            meta={},
            rank=0,
            log_buffer=SimpleNamespace(update=lambda *args, **kwargs: None),
        )
        hook.update_after_optimizer(runner)
        torch.testing.assert_close(
            hook.model_buffers["ema_weight"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            hook.model_buffers["ema_v5_transition"], torch.tensor(0.5)
        )
        optimizer.state[parameter]["step"] = torch.tensor(9.0)
        hook.update_after_optimizer(runner)
        self.assertGreater(hook.model_buffers["ema_weight"].item(), 1.0)

    def test_formal_config_and_clean_provenance(self) -> None:
        config = Config.fromfile(
            str(REPO_ROOT / "configs/v5r/co_dino_vitl_fdr_prehead_fold0.py")
        )
        self.assertEqual(config.model.type, "HOD26V5RCoDETR")
        self.assertEqual(config.model.transition_updates, 12000)
        self.assertEqual(config.model.query_head.pre_loss_weight, 1.0)
        self.assertEqual(
            config.model.query_head.transformer.decoder.type,
            "HOD26V5RDinoTransformerDecoder",
        )
        self.assertEqual(config.optimizer.paramwise_cfg.localization_lr_mult, 1.0)
        self.assertEqual(config.runner.max_epochs * 1406, 134976)
        self.assertEqual(config.data.train.pipeline[2].stage_epochs, (10, 36, 72))
        report = json.loads(
            (
                REPO_ROOT / "storage/pretrained/v5r/"
                "v4_e30_stateful_to_v5r_init.pth.provenance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["parent_optimizer_states"], 1015)
        self.assertEqual(report["transplanted_optimizer_states"], 1015)
        self.assertEqual(report["fresh_optimizer_states"], 81)
        self.assertEqual(report["salience_cloned_states"], 0)
        self.assertEqual(
            report["prohibited_faulty_v5_e18_sha256"],
            "14b8b7a9987cfc18ac32d6139b41d7f31fe664686b38f7e951c6a992c3751ff5",
        )


if __name__ == "__main__":
    unittest.main()
