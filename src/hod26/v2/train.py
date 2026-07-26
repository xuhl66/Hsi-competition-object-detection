from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _activate_upstream(path: str | Path) -> Path:
    upstream = Path(path).resolve()
    if not (upstream / "engine" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Official DEIM checkout not found at {upstream}. "
            "Run tools/bootstrap_v2.sh first."
        )
    sys.path.insert(0, str(upstream))
    return upstream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train HOD26 V2: 16-band DEIM-D-FINE-X"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--upstream",
        default="storage/upstream/DEIM",
    )
    parser.add_argument("-r", "--resume")
    parser.add_argument("-t", "--tuning")
    parser.add_argument("-d", "--device")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--summary-dir")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _activate_upstream(args.upstream)
    from hod26.v2.compat import install_deim_compatibility

    install_deim_compatibility()
    from engine.core import YAMLConfig, yaml_utils
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from hod26.v2.extensions import HSIDetSolver

    TASKS["hsi_detection"] = HSIDetSolver
    dist_utils.setup_distributed(
        args.print_rank,
        args.print_method,
        seed=args.seed,
    )
    if args.tuning and args.resume:
        raise ValueError("Use tuning or resume, never both")
    updates = yaml_utils.parse_cli(args.update)
    updates.update(
        {
            key: value
            for key, value in vars(args).items()
            if key not in {"update", "upstream"} and value is not None
        }
    )
    cfg = YAMLConfig(args.config, **updates)
    if args.resume or args.tuning:
        cfg.yaml_cfg.setdefault("HSIHGNetv2", {})["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
