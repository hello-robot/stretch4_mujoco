#!/usr/bin/env python
"""
`launch_scripts/run_eval.py` for the RBY1 configs, with the released policy bugs patched.

MolmoBot's own eval entry point cannot run any of the `MolmoBotRBY1*EvalConfig`
classes as shipped: the RBY1 policy chain reads three attributes that only the
Franka config and policy provide, so it raises before the first simulation step.
`rby1_interactive.py:patch_released_rby1_policy` documents and fixes all three;
this script applies it and then calls the same `run_evaluation` that
`run_eval.py` does, with the same arguments.

Use it exactly like the README's command:

    third_party/MolmoBot/MolmoBot/.venv/bin/python \\
        examples/machine_learning/molmobot/run_eval_rby1.py \\
        --checkpoint_path third_party/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-RBY1Multitask \\
        --benchmark_path <benchmark_dir> \\
        --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig \\
        --task_horizon 400

This scores each episode against its own recorded instruction and prints a
success rate. To type your own instructions instead, use `rby1_interactive.py`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rby1_interactive import patch_released_rby1_policy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--benchmark_path", type=str, required=True)
    p.add_argument("--eval_config_cls", type=str,
                   default="olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig")
    p.add_argument("--task_horizon", type=int, default=400)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mjthor-online-eval")
    p.add_argument("--use_filament", action="store_true")
    p.add_argument("--environment_light_intensity", type=float, default=None)
    args = p.parse_args(argv)

    patch_released_rby1_policy()

    from molmo_spaces.evaluation.eval_main import run_evaluation

    eval_config_cls = args.eval_config_cls
    if ":" in eval_config_cls:
        module_path, class_name = eval_config_cls.split(":")
        eval_config_cls = getattr(importlib.import_module(module_path), class_name)

    results = run_evaluation(
        eval_config_cls=eval_config_cls,
        benchmark_dir=Path(args.benchmark_path),
        checkpoint_path=Path(args.checkpoint_path),
        task_horizon_steps=args.task_horizon,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_filament=args.use_filament,
        environment_light_intensity=args.environment_light_intensity,
    )

    print(f"Success rate: {results.success_rate:.1%}")
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
