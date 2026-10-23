"""
A fine-tuned MolmoBot-DROID (`run_finetune.py`) driving a real Stretch 4 directly, continuously:
as `run_stretch4_real.py`, except that the policy's state and actions are Stretch's joints, with
no retargeting (`finetune/stretch4_env.py`). It asks before it first moves the robot to the
start pose.

The checkpoint is the newest under `finetune.checkpoints.MODELS_DIR` unless --checkpoint names
one (or a run, for its newest); with --interactive (-i), they are listed to pick from. The
policy sees the cameras it was trained on, so run the robot's publisher with both head
fisheyes and the checkpoint's wrist camera side (the runner says which if they differ):

    python send_gripper_and_head_images_with_joint_states.py -r --head_camera left,right --wrist_camera left --wrist_resolution 400

The robot has to have the gripper the checkpoint was trained with. Stretch starts where the
training episodes did: the Franka start pose (--custom_franka_start_pose, or the one for
--object-height) and grasp offset they used, unless given here. The start pose commands (`jog`,
`set`, `height`, `offset`) move the virtual Franka Stretch starts from, as in
`run_stretch4_real.py`; in Rerun, the ghost Franka is the one with its tool where Stretch's is.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_finetuned_real --robot_ip 10.0.0.12 --slow
    python -m examples.vla.molmobot_droid.run_stretch4_finetuned_real --robot_ip 10.0.0.12 -i
"""

from __future__ import annotations

import time

import click
import numpy as np

from examples.vla.molmobot_droid.checkpoint import POLICY_DT
from examples.vla.molmobot_droid.droid import Observation
from examples.vla.molmobot_droid.finetune.checkpoints import (
    apply_to_params,
    apply_training_start,
    finetuned_checkpoint_options,
    resolve_checkpoint,
)
from examples.vla.molmobot_droid.finetune.stretch4_env import StretchJointControl
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    STEP_ARRIVAL_TIMEOUT,
    StretchTargets,
    custom_start_option,
    params_from_kwargs,
    prepare_exo,
    retarget_options,
    wrist_view,
)
from examples.vla.molmobot_droid.run_stretch4_real import RealStretch4Env, run_real


class FinetunedRealStretch4Env(StretchJointControl, RealStretch4Env):
    """`RealStretch4Env` driven by a fine-tuned policy (`StretchJointControl`)."""

    def world_from_footprint(self) -> np.ndarray:
        return self._state({"closest_joint_state": self.live_joint_state()})[0]

    def observe(self) -> Observation:
        # As `RealStretch4Env.observe()`: a message taken after the robot stopped, with the joints as of now.
        message = self.receive_settled() if self.params.wait_for_arrival else self.receiver.receive()
        message = {**message, "closest_joint_state": self.live_joint_state(), "observed_at": time.time()}
        self.last_message = message
        footprint, joints = self._state(message)
        # Only for Rerun's ghost (`LiveView`): no Franka is being driven.
        self.last_state8, _ = self.retargeter.stretch_to_franka(footprint, joints)
        side = self.update_exo_side()
        head, wrist = message["head_rgbs"][side], message["wrist_rgb"]
        return Observation(
            exo_rgb=prepare_exo(head, self.params, side, self.head_rectifiers.get(side)),
            wrist_rgb=wrist_view(wrist),
            state8=self.stretch_state8(footprint, joints),
            extra_cameras={"head_raw": head, "gripper_raw": wrist},
        )

    def step(self, action8: np.ndarray) -> StretchTargets:
        targets = self.stretch_targets(action8, self.world_from_footprint())
        self.last_miss = None
        start = time.monotonic()
        self.send(targets)
        if self.params.wait_for_arrival:
            self.wait_for_arrival(STEP_ARRIVAL_TIMEOUT)
        remaining = POLICY_DT - (time.monotonic() - start)
        if remaining > 0 and not self.interrupted():
            time.sleep(remaining)
        return targets


@click.command()
@click.option("--robot_ip", "--robot-ip", "robot_ip", required=True, help="Stretch 4's IP address.")
@click.option("--object-height", type=float, default=0.80, show_default=True,
              help="Height of the target object above the floor (m); sets the virtual Franka's height "
              "(not used with --custom_franka_start_pose).")
@click.option("--max-steps", type=int, default=0, show_default=True,
              help="Stop an instruction after this many steps (15/s). 0: run until told otherwise.")
@finetuned_checkpoint_options
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@retarget_options(gripper_option=False, camera_options=False)
@custom_start_option
def main(robot_ip, object_height, max_steps, checkpoint, interactive, rerun, custom_franka_start_pose, **kwargs):
    params = params_from_kwargs(kwargs)
    finetuned = resolve_checkpoint(checkpoint, interactive)
    apply_to_params(finetuned, params)
    custom_franka_start_pose = apply_training_start(finetuned, params, custom_franka_start_pose)
    run_real(params, robot_ip, object_height, max_steps, 0, finetuned.path, rerun, custom_franka_start_pose,
             env_class=FinetunedRealStretch4Env, tool_name=finetuned.tool_name,
             title=f"Fine-tuned MolmoBot-DROID Stretch 4 (real): {finetuned.name}")


if __name__ == "__main__":
    main()
