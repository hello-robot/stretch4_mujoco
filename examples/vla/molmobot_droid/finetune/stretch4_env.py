"""
Stretch 4 driven directly by a fine-tuned MolmoBot-DROID (`run_finetune.py`), whose state and
actions are Stretch's joints (`stretch4_joints`) rather than the Franka's: what the
`*_finetuned_*` runners step.

`StretchJointControl` turns a retargeting env (`Stretch4SimEnv` here, `RealStretch4Env` in
`run_stretch4_finetuned_real.py`) into one: `observe()` gives Stretch's state, its base heading
from the start pose, and its cameras as the checkpoint was trained on them; `step()` sends an
action's joints to Stretch as they are, with no retargeting. Only the start pose still goes
through the virtual Franka, since the training episodes began from the Franka's
(`move_to_franka_pose()`), and so do the start pose editor's commands (`franka_controls`).
"""

from __future__ import annotations

import math

import numpy as np

from examples.vla.molmobot_droid.checkpoint import GRIPPER_CLOSED
from examples.vla.molmobot_droid.droid import Observation
from examples.vla.molmobot_droid.finetune import stretch4_joints
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    Stretch4SimEnv,
    StretchJoints,
    StretchTargets,
    _wrap,
    planar_from_pose,
    planar_transform,
    prepare_exo,
    wrist_view,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import STRETCH_ROOT_BODY


def footprint_yaw(world_from_footprint: np.ndarray) -> float:
    return math.atan2(world_from_footprint[1, 0], world_from_footprint[0, 0])


class FrankaControls:
    """
    A `StretchJointControl` env as the start pose editor drives it: the virtual Franka's state
    and actions, retargeted, as the base env takes them. Everything else is the env's.
    """

    def __init__(self, env: StretchJointControl):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def observe(self) -> Observation:
        return self._env.franka_observe()

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        return self._env.franka_step(action8)


class StretchJointControl:
    """
    Mixed in before a retargeting env: its `observe()` and `step()` become Stretch's joints (see
    the module docstring), and its Franka ones are `franka_observe()` and `franka_step()`.

    The state's base heading is from where Stretch last arrived at a start pose
    (`move_to_franka_pose()`), as in the training episodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_yaw = footprint_yaw(self.world_from_footprint())
        """Stretch's heading at the start pose: the state's base_yaw is from there."""
        self.franka_controls = FrankaControls(self)
        """What the start pose editor drives."""

    def franka_observe(self) -> Observation:
        return super().observe()

    def franka_step(self, action8: np.ndarray) -> StretchTargets | None:
        return super().step(action8)

    def move_to_franka_pose(self, *args, **kwargs):
        """The base env's, then the base heading from there on is the state's 0."""
        result = super().move_to_franka_pose(*args, **kwargs)
        self.start_yaw = footprint_yaw(self.world_from_footprint())
        return result

    def stretch_state8(self, world_from_footprint: np.ndarray, joints: StretchJoints) -> np.ndarray:
        return stretch4_joints.state8(_wrap(footprint_yaw(world_from_footprint) - self.start_yaw), joints)

    def stretch_targets(self, action8: np.ndarray, world_from_footprint: np.ndarray) -> StretchTargets:
        """
        `action8`'s joints as Stretch's targets from where it is: the base turned by what is left
        to its heading. Where they put the tool becomes the retargeter's `last_target_tool_world`,
        which Rerun and `tool_error_m()` compare the tool with.
        """
        heading = footprint_yaw(world_from_footprint) - self.start_yaw
        lift, arm, wrist_yaw, wrist_pitch, wrist_roll = (float(v) for v in action8[1:6])
        targets = StretchTargets(
            _wrap(float(action8[0]) - heading), lift, arm, wrist_yaw, wrist_pitch, wrist_roll,
            gripper_closed=bool(action8[7] >= GRIPPER_CLOSED / 2),
        )
        turned = world_from_footprint @ planar_transform(0.0, 0.0, targets.base_rotate_by)
        joints = StretchJoints(lift, arm, wrist_yaw, wrist_pitch, wrist_roll, gripper_open_fraction=1.0)
        self.retargeter.last_target_tool_world = self.retargeter.stretch_tool_world(turned, joints)
        return targets


class FinetunedStretch4SimEnv(StretchJointControl, Stretch4SimEnv):
    """`Stretch4SimEnv` driven by a fine-tuned policy (`StretchJointControl`). Without the ghost
    Franka: the policy is not driving one."""

    @property
    def last_miss(self) -> None:
        """No action is retargeted, so none is missed."""
        return None

    def observe(self) -> Observation:
        cameras = self.fresh_camera_data()
        status = self.sim.pull_status()
        poses = self.sim.pull_body_poses()
        footprint = planar_from_pose(*poses[STRETCH_ROOT_BODY], self.stretch_scene.scene.floor_z)
        joints = self.joints(status)
        self.observed_state = (status.time, footprint, joints)
        self.mirror.update(status, poses)
        self.update_exo_side(joints)
        head, gripper = self.camera_images(cameras)
        return Observation(
            exo_rgb=prepare_exo(head, self.params, self.exo_camera),
            wrist_rgb=wrist_view(gripper),
            state8=self.stretch_state8(footprint, joints),
            extra_cameras={"gripper_raw": gripper, "head_raw": head},
        )

    def step(self, action8: np.ndarray) -> StretchTargets:
        targets = self.stretch_targets(action8, self.world_from_footprint())
        start = self.sim.pull_status().time
        self.send(targets)
        self.wait(start)
        return targets

    def stats(self) -> dict[str, float | int]:
        return {
            "stale_observations": self.stale_observations,
            "furniture_removed": len(self.stretch_scene.removed_bodies),
        }
