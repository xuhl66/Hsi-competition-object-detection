"""V5 optimizer registration and parameter-family extension."""

from __future__ import annotations

from collections import defaultdict

from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS

from hod26.v4.optim import (
    HOD26LayerDecayOptimizerConstructor,
    _is_class_parameter,
    _no_decay,
    _vit_layer_id,
)


def _is_v5_parameter(name: str) -> bool:
    return name.startswith(
        (
            "backbone.proxy_",
            "backbone.spectral_",
            "neck.spectral_fusions",
            "neck.object_activation",
            "neck.salience_high",
            "neck.object_gate",
            "neck.object_refinements",
            "query_head.fdr_branches",
            "query_head.lqe_layers",
        )
    )


@OPTIMIZER_BUILDERS.register_module()
class HOD26V5LayerDecayOptimizerConstructor(HOD26LayerDecayOptimizerConstructor):
    def add_params(self, params, module, **kwargs) -> None:
        depth = int(self.paramwise_cfg.get("num_layers", 24))
        decay_rate = float(self.paramwise_cfg.get("layer_decay_rate", 0.8))
        new_lr_mult = float(self.paramwise_cfg.get("hsi_lr_mult", 8.0))
        class_lr_mult = float(self.paramwise_cfg.get("class_lr_mult", 4.0))
        grouped = defaultdict(lambda: {"params": [], "param_names": []})
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if _is_v5_parameter(name):
                scale, family = new_lr_mult, "v5"
            elif _is_class_parameter(name):
                scale, family = class_lr_mult, "class"
            elif name.startswith("backbone."):
                layer = _vit_layer_id(name, depth)
                scale, family = decay_rate ** (depth + 1 - layer), f"vit_{layer:02d}"
            else:
                scale, family = 1.0, "detector"
            decay = not _no_decay(name, parameter)
            group = grouped[(family, scale, decay)]
            group["params"].append(parameter)
            group["param_names"].append(name)
        for (family, scale, decay), group in grouped.items():
            group.update(
                group_name=f"{family}_{'decay' if decay else 'no_decay'}",
                lr=self.base_lr * scale,
                initial_lr=self.base_lr * scale,
                lr_scale=scale,
                weight_decay=self.base_wd if decay else 0.0,
            )
            params.append(group)
