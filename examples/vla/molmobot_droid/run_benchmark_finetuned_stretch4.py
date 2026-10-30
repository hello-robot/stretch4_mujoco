"""
Run a MolmoSpaces benchmark with a fine-tuned MolmoBot-DROID (`run_finetune.py`) driving Stretch 4
in simulation: as `run_benchmark_stretch4.py`, except that when the policy's state and actions
are Stretch's joints (its run's --action_space stretch4) they drive it directly, with no
retargeting (`finetune/stretch4_env.py`); a policy of the Franka's joints (--action_space franka)
is retargeted, as `run_benchmark_stretch4.py` retargets MolmoBot-DROID's. Stretch's footprint goes
where the episode puts the Franka's base, and it starts where the Franka's start pose puts its
tool, as when the training episodes were recorded.

The checkpoint is the newest under `finetune.checkpoints.MODELS_DIR` unless --checkpoint names
one (or a run, for its newest); with --interactive (-i), they are listed to pick from. The
policy sees the cameras it was trained on, and Stretch has the gripper it was trained with
(the run's `spec.json`). Stretch starts where the training episodes did: the Franka start pose
(--custom_franka_start_pose) and grasp offset they used, unless given here.

Writes to <out>/<run name>/ what `run_benchmark_stretch4.py` writes; the run name starts with
`stretch4-finetuned` and names the checkpoint. Its results are a Stretch 4 run's, so
`compare_benchmarks.py --all` compares it with the retargeted ones.

Usage:
    python -m examples.vla.molmobot_droid.run_benchmark_finetuned_stretch4 --max-episodes 200 --no-rerun
    python -m examples.vla.molmobot_droid.run_benchmark_finetuned_stretch4 -i --no-wait-for-arrival
"""

from __future__ import annotations

import click

from examples.vla.molmobot_droid.finetune.checkpoints import (
    apply_to_params,
    apply_training_start,
    finetuned_checkpoint_options,
    resolve_checkpoint,
)
from examples.vla.molmobot_droid.finetune.stretch4_env import FinetunedStretch4SimEnv
from examples.vla.molmobot_droid.franka_retarget.follow_franka import follow_flags
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    Stretch4SimEnv,
    custom_start_option,
    params_from_kwargs,
    retarget_options,
)
from examples.vla.molmobot_droid.molmospaces import benchmark as bench
from examples.vla.molmobot_droid.run_benchmark_franka import benchmark_options
from examples.vla.molmobot_droid.run_benchmark_stretch4 import run_benchmark


@click.command()
@benchmark_options(checkpoint=False)
@finetuned_checkpoint_options
@retarget_options(gripper_option=False, camera_options=False)
@custom_start_option
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@click.option("--viewer", is_flag=True, help="Show stretch4_mujoco's viewer (default: headless).")
def main(benchmark, episodes, max_episodes, out, run_to_horizon, resume, list_only, checkpoint, interactive,
         custom_franka_start_pose, rerun, viewer, **kwargs):
    if list_only:
        click.echo("\n".join(bench.list_benchmarks()))
        return
    params = params_from_kwargs(kwargs)
    finetuned = resolve_checkpoint(checkpoint, interactive)
    apply_to_params(finetuned, params)
    custom_franka_start_pose = apply_training_start(finetuned, params, custom_franka_start_pose)
    # The cameras and the gripper are the checkpoint's, which its name says.
    flags = {"finetuned_checkpoint": finetuned.name, **follow_flags(params, custom_franka_start_pose)}
    flags.pop("use_parallel_gripper")
    flags.pop("rectify", None)
    run_benchmark(params, flags, "stretch4-finetuned", benchmark, episodes, max_episodes, out, finetuned.path,
                  run_to_horizon, resume, custom_franka_start_pose=custom_franka_start_pose, rerun=rerun,
                  viewer=viewer,
                  env_class=FinetunedStretch4SimEnv if finetuned.action_space == "stretch4" else Stretch4SimEnv,
                  title=f"Fine-tuned MolmoBot-DROID Stretch 4 benchmark: {finetuned.name}")


if __name__ == "__main__":
    main()
