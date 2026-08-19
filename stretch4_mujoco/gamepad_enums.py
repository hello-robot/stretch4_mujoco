"""
Port of the real robot's `stretch4_body/core/gamepad_enums.py`.

The enums and their cycle orders are kept identical to the robot so muscle
memory carries over. Two robot-only concepts are stubbed:

* `play_sound_file()` prints instead of playing the robot's wav files.
* `GuardedContactSensitivity.apply()` is a no-op: MuJoCo has no guarded-contact
  layer to configure. The profile is still cycled and announced so the button
  behaves the same way.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

from stretch4_mujoco.enums.actuators import Actuators

if TYPE_CHECKING:
    from stretch4_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


class GuardedContactSensitivity(Enum):
    """
    Name of the sensitivity mode as defined in robot_params_SE4.
    Cycled with RT + B, exactly as on the robot. Has no effect in simulation.
    """

    HIGH_SENSITIVITY_NAV = 1
    HIGH_SENSITIVITY_MANIPULATION = 2  # Low Strength
    MEDIUM = 3  # Medium Strength (default in Stretch Body)
    STRONG_MANIPULATION = 4  # High Strength

    def _get_cycleable_options(self):
        return [
            GuardedContactSensitivity.HIGH_SENSITIVITY_MANIPULATION,
            GuardedContactSensitivity.MEDIUM,
            GuardedContactSensitivity.STRONG_MANIPULATION,
        ]

    def cycle(self, is_forward: bool):
        index_offset = 1 if is_forward else -1
        members = self._get_cycleable_options()
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    def play_sound_file(self):
        print(f"Contact sensitivity: {self.name}")

    def get_name(self):
        """Get the name mapping that works with Stretch Body"""
        if self == GuardedContactSensitivity.MEDIUM:
            return "default"
        return self.name.lower()

    def apply(self, sim: "StretchMujocoSimulator"):
        """No-op in simulation - there is no guarded contact layer to configure."""


class MotionProfile(Enum):
    """
    Name of the motion profile as defined in robot_params_SE4. (e.g. default, slow, fast, max)
    """

    SLOW = 1  # The ordering here is important for get_one_lower_speed()
    MEDIUM = 2  # This is called 'default' in Stretch Body
    FAST = 3
    MAX = 4

    def get_name(self):
        """Get the name mapping that works with Stretch Body"""
        if self == MotionProfile.MEDIUM:
            return "default"
        return self.name.lower()

    def get_one_lower_speed(self):
        if self is MotionProfile.SLOW:
            return self
        return self.cycle(is_forward=False, use_cyclable_options=False)

    def _get_cycleable_options(self):
        return [MotionProfile.SLOW, MotionProfile.MEDIUM, MotionProfile.FAST]

    def cycle(self, is_forward: bool, use_cyclable_options: bool = True):
        index_offset = 1 if is_forward else -1

        members = self._get_cycleable_options() if use_cyclable_options else list(type(self))
        index = members.index(self)
        return members[(index + index_offset) % len(members)]

    def play_sound_file(self):
        print(f"Motion profile: {self.name}")


class GripperHandedness(Enum):
    LEFT = 0
    RIGHT = 1

    def play_sound_file(self):
        print(f"{self.name}-handed mode")

    def move_to(self, sim: "StretchMujocoSimulator"):
        """Moves the gripper to achieve this handedness"""
        print(f"Moving wrist to {self}")

        yaw_to: float
        pitch_to: float
        roll_to: float
        if self is GripperHandedness.RIGHT:
            yaw_to = 0.0
            pitch_to = 0.0
            roll_to = 0.0
        elif self is GripperHandedness.LEFT:
            yaw_to = math.pi
            pitch_to = math.pi
            # The URDF/MJCF wrist_roll axis is mirrored relative to the robot's servo
            # convention (URDF range [-4.276, 1.135] vs servo range [-1.135, 4.276]),
            # so the robot's +pi is -pi here. See CommandWristRoll in gamepad_joints.py.
            roll_to = -math.pi
        else:
            raise NotImplementedError(f"No move_to defined for {self}")

        sim.end_of_arm.wrist_yaw.move_to(yaw_to)
        sim.end_of_arm.wrist_pitch.move_to(pitch_to)
        sim.end_of_arm.wrist_roll.move_to(roll_to)

        sim.wait_until_at_setpoint(Actuators.wrist_yaw, timeout=3.0)
        sim.wait_until_at_setpoint(Actuators.wrist_pitch, timeout=3.0)
        sim.wait_until_at_setpoint(Actuators.wrist_roll, timeout=3.0)
