"""
Stretch 4's joints as the 8 values of MolmoBot-DROID's state and actions, for fine-tuning it to
drive Stretch 4 directly. They are the joints retargeting moves (`FrankaStretchRetargeter`), in
the Franka's slots:

    0 base_yaw     rad, the base's heading from where the episode started (where the Franka's
                   joint 1 turns the whole arm about the vertical)
    1 lift         m
    2 arm          m, the telescoping arm's extension
    3 wrist_yaw    rad
    4 wrist_pitch  rad
    5 wrist_roll   rad
    6 unused       always 0: the Franka has 7 arm joints, retargeting moves 6 of Stretch's
    7 gripper      state: the Robotiq driver angle the opening maps to (`robotiq_driver_angle()`,
                   as retargeting reports it); action: 0 open, 255 closed (the Franka's command)

Keeping 8 values, the gripper last, keeps the pretrained action head's shape, so its weights
load. All are absolute, as the Franka's joint actions are.

The h5 files (`finetune.dataset`) store them per move group (`MOVE_GROUPS`), as molmospaces
stores the Franka's ({"arm": [7 joints], "gripper": [1]}).
"""

from __future__ import annotations

import numpy as np

from examples.vla.molmobot_droid.checkpoint import GRIPPER_CLOSED, GRIPPER_OPEN, robotiq_driver_angle

JOINTS = ("base_yaw", "lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "unused", "gripper")

MOVE_GROUPS: dict[str, int] = {"base": 1, "lift": 1, "arm": 1, "wrist": 3, "unused": 1, "gripper": 1}
"""How many of the 8 values each move group holds, in order: the h5's `obs/agent/qpos` and
`actions/joint_pos` keys, and MolmoBot's action spec for them (`finetune.train`)."""


def state8(base_yaw: float, joints) -> np.ndarray:
    """The state for Stretch at `joints` (a `StretchJoints`), its base `base_yaw` (rad) from where
    the episode started."""
    return np.array([
        base_yaw, joints.lift, joints.arm, joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll, 0.0,
        robotiq_driver_angle(joints.gripper_open_fraction),
    ])


def action8(base_yaw: float, lift: float, arm: float, wrist_yaw: float, wrist_pitch: float, wrist_roll: float,
            gripper_closed: bool) -> np.ndarray:
    """The action commanding Stretch's joints to these positions (m, rad) and its gripper shut or open."""
    gripper = GRIPPER_CLOSED if gripper_closed else GRIPPER_OPEN
    return np.array([base_yaw, lift, arm, wrist_yaw, wrist_pitch, wrist_roll, 0.0, gripper])


def to_move_groups(values8) -> dict[str, list[float]]:
    """`values8` split into MOVE_GROUPS, as the h5 files store them."""
    values = [float(v) for v in values8]
    groups, start = {}, 0
    for name, size in MOVE_GROUPS.items():
        groups[name] = values[start : start + size]
        start += size
    return groups
