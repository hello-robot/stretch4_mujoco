"""
A fine-tuned MolmoBot-DROID (`run_finetune.py`) driving Stretch 4 in simulation, in any
MolmoSpaces house: as `run_stretch4_sim.py`, except that when the policy's state and actions
are Stretch's joints (its run's --action_space stretch4) they drive it directly, with no
retargeting (`finetune/stretch4_env.py`). A policy of the Franka's joints (--action_space franka)
is retargeted, as `run_stretch4_sim.py` retargets MolmoBot-DROID's.

The checkpoint is the newest under `finetune.checkpoints.MODELS_DIR` unless --checkpoint names
one (or a run, for its newest); with --interactive (-i), they are listed to pick from. The
policy sees the cameras it was trained on, and Stretch has the gripper it was trained with
(the run's `spec.json`). Stretch starts where the training episodes did: the Franka start pose
(--custom_franka_start_pose) and grasp offset they used, unless given here.

Type an instruction and press Enter (see `franka_retarget/console.py`'s KEYS_HELP); `home` or
`reset` goes back to the start pose. The start pose commands (`jog`, `set`, `height`, `offset`)
move the virtual Franka Stretch starts from, as in `run_stretch4_sim.py`.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_finetuned_sim
    python -m examples.vla.molmobot_droid.run_stretch4_finetuned_sim -i --scene-id procthor-10k/val/0 --object-type apple
    python -m examples.vla.molmobot_droid.run_stretch4_finetuned_sim \\
        --checkpoint outputs/molmobot_droid/finetuning/models/<run>/step6000
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
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    Stretch4SimEnv,
    custom_start_option,
    params_from_kwargs,
    retarget_options,
)
from examples.vla.molmobot_droid.run_franka import scene_options
from examples.vla.molmobot_droid.run_stretch4_sim import run_sim


@click.command()
@scene_options(checkpoint=False)
@finetuned_checkpoint_options
@retarget_options(gripper_option=False, camera_options=False)
@custom_start_option
def main(scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, rerun, checkpoint, interactive,
         custom_franka_start_pose, **kwargs):
    params = params_from_kwargs(kwargs)
    finetuned = resolve_checkpoint(checkpoint, interactive)
    apply_to_params(finetuned, params)
    custom_franka_start_pose = apply_training_start(finetuned, params, custom_franka_start_pose)
    env_class = FinetunedStretch4SimEnv if finetuned.action_space == "stretch4" else Stretch4SimEnv
    run_sim(params, scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, finetuned.path, rerun,
            custom_franka_start_pose=custom_franka_start_pose, env_class=env_class,
            title=f"Fine-tuned MolmoBot-DROID Stretch 4: {finetuned.name}")


if __name__ == "__main__":
    main()
