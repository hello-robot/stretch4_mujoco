"""
Port of the real robot's `stretch4_body/core/gamepad_joints.py`.

The gamepad_joints library provides the abstract motion command classes for each
robot joint that can be used in a control loop to make a motion through gamepad
input elements (button presses, analog stick motions).

A gamepad joint command class provides the same four attributes as on the robot:

command_stick_to_motion()
    Supply a float value between -1.0 to 1.0 from a control loop.
    The value supplied and its sign determines the speed of joint motion and direction.

command_button_to_motion()
    Supply a direction integer, either +1 or -1, for the joint to move in that direction.

stop_motion()
    Use this method whenever a joint needs to be still with no motion in a control loop.

precision_mode
    Set this to a 0.0-1.0 value (the left trigger) to scale motions down.

Differences from the robot-side file:

* `RobotParams()` is not available in sim, so the SE4 motion profiles are baked
  into `ROBOT_PARAMS` below, copied verbatim from `robot_params_SE4.py`.
* Feetech joints are position-stepped rather than velocity-profiled by a servo,
  so `CommandFeetechJoint._move()` clamps its step to `velocity * dt`. That
  reproduces "move at most `velocity` over one control period", which is what
  the robot's move_by(delta, velocity) achieves with its trapezoidal profile.
* The gripper is commanded in aperture radians instead of the robot's percent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stretch4_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

# Copied from stretch4_body/robot/robot_params_SE4.py so the sim uses the same
# velocities and accelerations as the robot.
ROBOT_PARAMS = {
    "omnibase": {
        "motion": {
            # Base motion profiles. w_r: rotation (rad/s), xy_m: translation (m/s)
            "slow": {"accel_w_r": 1.0, "vel_w_r": 1.0, "accel_xy_m": 0.1, "vel_xy_m": 0.1},
            "default": {"accel_w_r": 2.0, "vel_w_r": 2.0, "accel_xy_m": 0.25, "vel_xy_m": 0.3},
            "fast": {"accel_w_r": 3.0, "vel_w_r": 3.0, "accel_xy_m": 0.4, "vel_xy_m": 0.4},
            "max": {"accel_w_r": 4.0, "vel_w_r": 4.0, "accel_xy_m": 0.5, "vel_xy_m": 0.6},
        }
    },
    "lift": {
        "motion": {
            "slow": {"accel_m": 0.2, "vel_m": 0.15},
            "default": {"accel_m": 0.3, "vel_m": 0.3},
            "fast": {"accel_m": 0.5, "vel_m": 0.4},
            "max": {"accel_m": 1.0, "vel_m": 0.5},
        }
    },
    "arm": {
        "motion": {
            "slow": {"accel_m": 0.1, "vel_m": 0.1},
            "default": {"accel_m": 0.4, "vel_m": 0.4},
            "fast": {"accel_m": 0.6, "vel_m": 0.6},
            "max": {"accel_m": 0.7, "vel_m": 0.7},
        }
    },
    "wrist_yaw": {
        "motion": {
            "slow": {"accel": 4.0, "vel": 4.0},
            "default": {"accel": 7.0, "vel": 7.0},
            "fast": {"accel": 9.0, "vel": 9.0},
            "max": {"accel": 12.0, "vel": 12.0},
        }
    },
    "wrist_pitch": {
        "motion": {
            "slow": {"accel": 4.0, "vel": 4.0},
            "default": {"accel": 7.0, "vel": 7.0},
            "fast": {"accel": 9.0, "vel": 9.0},
            "max": {"accel": 12.0, "vel": 12.0},
        }
    },
    "wrist_roll": {
        "motion": {
            "slow": {"accel": 4.0, "vel": 4.0},
            "default": {"accel": 7.0, "vel": 7.0},
            "fast": {"accel": 9.0, "vel": 9.0},
            "max": {"accel": 12.0, "vel": 12.0},
        }
    },
    "stretch_gripper": {
        "motion": {
            "slow": {"accel": 4.0, "vel": 1.0},
            "default": {"accel": 6.0, "vel": 6.0},
            "fast": {"accel": 6.0, "vel": 6.0},
            "max": {"accel": 6.0, "vel": 6.0},
        }
    },
}

# The robot's teleop control period. Used to convert velocity caps into per-step
# position deltas for the position-controlled sim joints.
DEFAULT_STEP_SLEEP = 1 / 15

# The URDF (and therefore MJCF) wrist_roll axis is mirrored relative to the robot's
# servo convention: the URDF limits are [-4.276, 1.135] where the servo's are
# [-1.135, 4.276]. Applying this sign to roll commands keeps LB/RB rolling the
# gripper the same way it does on the robot.
WRIST_ROLL_SIM_SIGN = -1.0


def map_to_range(value, new_min, new_max):
    # Ensure value is between 0 and 1
    value = max(0, min(1, value))
    return (value - 0) * (new_max - new_min) / (1 - 0) + new_min


def deg_to_rad(x):
    return math.pi * x / 180.0


class CommandBase:
    def __init__(self, motion_profile: str = "default", motion_profile_angular: str = "slow"):
        self.motion_profile = motion_profile
        self.motion_profile_angular = motion_profile_angular
        self.params = ROBOT_PARAMS["omnibase"]
        self.dead_zone = 0.0001

        self.accel_xy_max = self.params["motion"]["max"]["accel_xy_m"]
        self.accel_w_max = self.params["motion"]["max"]["accel_w_r"]

        self.precision_mode = 0.0

    def _get_motion_params(self, is_rotating: bool):
        motion_profile = self.motion_profile
        if is_rotating:
            motion_profile = self.motion_profile_angular

        vel_xy = self.params["motion"][motion_profile]["vel_xy_m"]
        accel_xy = self.params["motion"][motion_profile]["accel_xy_m"]
        vel_w = self.params["motion"][motion_profile]["vel_w_r"]
        accel_w = self.params["motion"][motion_profile]["accel_w_r"]

        return vel_xy, accel_xy, vel_w, accel_w

    def _move(self, x, y, w, accel_xy, accel_w, robot: "StretchMujocoSimulator"):
        scale = 1.0 - 0.75 * self.precision_mode
        robot.base.set_velocity(scale * x, scale * y, scale * w, accel_xy, accel_w)

    def command_stick_to_motion(self, x, y, w, robot: "StretchMujocoSimulator"):
        """Convert a stick axis value to robot base's driving motion.

        Args:
            x (float): Range [-1.0,+1.0], control linear x speed
            y (float): Range [-1.0,+1.0], control linear y speed
            w (float): Range [-1.0,+1.0], control angular speed
        """
        vel_xy, accel_xy, vel_w, accel_w = self._get_motion_params(is_rotating=abs(w) >= 0.1)

        v_x = vel_xy * (0 if abs(x) < self.dead_zone else x)
        v_y = vel_xy * (0 if abs(y) < self.dead_zone else y)
        v_w = vel_w * (0 if abs(w) < self.dead_zone else w)

        self._move(v_x, v_y, v_w, accel_xy, accel_w, robot)

    def stop_motion(self, robot: "StretchMujocoSimulator"):
        """Stop the joint motion. To be used whenever the controller is idle/no-inputs
        to stop unnecessary robot motion."""
        robot.base.set_velocity(0, 0, 0, self.accel_xy_max, self.accel_w_max)


class CommandLift:
    def __init__(self, motion_profile: str = "default"):
        self.motion_profile = motion_profile
        self.params = ROBOT_PARAMS["lift"]
        self.dead_zone = 0.0001
        self.max_linear_vel = self.params["motion"][self.motion_profile]["vel_m"]
        self.precision_mode = 0.0
        self.acc = self.params["motion"][self.motion_profile]["accel_m"]

    def _move(self, v_m, robot: "StretchMujocoSimulator"):
        scale = 1.0 - 0.75 * self.precision_mode
        v_m = v_m * scale
        robot.lift.set_velocity(v_m, a_m=self.acc)

    def command_stick_to_motion(self, x, robot: "StretchMujocoSimulator"):
        """Convert a stick axis value to robot lift motion.

        Args:
            x (float): Range [-1.0,+1.0], control lift speed
        """
        if abs(x) < self.dead_zone:
            x = 0
        v_m = map_to_range(abs(x), 0, self.max_linear_vel)
        v_m *= -1 if x < 0 else 1

        self._move(v_m, robot)

    def command_button_to_motion(self, direction, robot: "StretchMujocoSimulator"):
        """Make lift move based on a button state.

        Args:
            direction (int): Direction integer -1 or +1
        """
        v_m = self.max_linear_vel * direction
        self._move(v_m, robot)

    def stop_motion(self, robot: "StretchMujocoSimulator"):
        robot.lift.set_velocity(0, a_m=self.params["motion"]["max"]["accel_m"])


class CommandArm:
    def __init__(self, motion_profile: str = "default"):
        self.motion_profile = motion_profile
        self.params = ROBOT_PARAMS["arm"]
        self.dead_zone = 0.0001
        self.max_linear_vel = self.params["motion"][self.motion_profile]["vel_m"] * 0.75
        self.precision_mode = 0.0
        self.acc = self.params["motion"][self.motion_profile]["accel_m"]

    def _move(self, v_m, robot: "StretchMujocoSimulator"):
        scale = 1.0 - 0.75 * self.precision_mode
        v_m = v_m * scale
        robot.arm.set_velocity(v_m, a_m=self.acc)

    def command_stick_to_motion(self, x, robot: "StretchMujocoSimulator"):
        """Convert a stick axis value to robot arm motion.

        Args:
            x (float): Range [-1.0,+1.0], control arm speed
        """
        if abs(x) < self.dead_zone:
            x = 0

        v_m = map_to_range(abs(x), 0, self.max_linear_vel)
        v_m *= -1 if x < 0 else 1

        self._move(v_m, robot)

    def command_button_to_motion(self, direction, robot: "StretchMujocoSimulator"):
        """Make arm move based on a button state.

        Args:
            direction (int): Direction integer -1 or +1
        """
        v_m = self.max_linear_vel * direction
        self._move(v_m, robot)

    def stop_motion(self, robot: "StretchMujocoSimulator"):
        robot.arm.set_velocity(0, a_m=self.params["motion"]["max"]["accel_m"])


class CommandFeetechJoint:
    """Abstract motion command class for Feetech joints."""

    def __init__(self, name, dx_deg, vel_type, acc_type, dt: float = DEFAULT_STEP_SLEEP):
        self.params = ROBOT_PARAMS[name]
        self.name = name
        self.dead_zone = 0.001
        self.dx_deg = dx_deg
        self.max_vel = self.params["motion"][vel_type]["vel"]
        self.acc = self.params["motion"][acc_type]["accel"]
        self.precision_mode = 0.0
        self.dt = dt

    def _get_subsystem(self, robot: "StretchMujocoSimulator"):
        return getattr(robot.end_of_arm, self.name)

    def _move(self, dx_deg, robot: "StretchMujocoSimulator", velocity: float | None = None):
        scale = 1.0 - (0.95 * self.precision_mode)
        dx_deg = dx_deg * scale

        capped_velocity = min(self.max_vel, velocity) if velocity is not None else self.max_vel

        # The sim's move_by() applies the delta straight to the position actuator, so
        # the velocity cap has to be applied here as a per-step limit on the delta.
        dx_rad = deg_to_rad(dx_deg)
        max_step_rad = capped_velocity * self.dt
        dx_rad = max(-max_step_rad, min(max_step_rad, dx_rad))

        if dx_rad == 0.0:
            # A zero-delta move_by is a no-op on the robot, but not in sim: move_by
            # targets `current measured position + delta`, so on a gravity-loaded joint
            # (wrist_pitch) it re-targets the sagged position and ratchets the joint down
            # a little every call. Holding the existing setpoint is what "no motion" means
            # for a position actuator, so send nothing.
            return

        self._get_subsystem(robot).move_by(dx_rad, capped_velocity, self.acc)

    def command_button_to_motion(self, direction, robot: "StretchMujocoSimulator"):
        """Make servo move based on a button state.

        Args:
            direction (int): Direction integer -1 or +1
        """
        self._move(self.dx_deg * direction, robot)

    def command_stick_to_motion(self, x, robot: "StretchMujocoSimulator"):
        """Convert a stick axis value to a servo motion.

        Args:
            x (float): Range [-1.0,+1.0]
        """
        if abs(x) < self.dead_zone:
            x = 0

        self._move(self.dx_deg * x, robot)

    def stop_motion(self, robot: "StretchMujocoSimulator"):
        """Stop the joint motion. To be used whenever the controller is idle/no-inputs
        to stop unnecessary robot motion.

        The robot sends move_by(name, 0) here. In sim the position actuator already
        holds its setpoint, and re-sending a zero move_by would drift the joint (see
        `_move`), so this is a no-op.
        """


class CommandWristYaw(CommandFeetechJoint):
    """Wrist Yaw motion command class."""

    def __init__(
        self,
        name="wrist_yaw",
        dx_deg=15.0,
        motion_profile: str = "default",
        dt: float = DEFAULT_STEP_SLEEP,
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile, dt=dt)


class CommandWristPitch(CommandFeetechJoint):
    """Wrist Pitch motion command class."""

    def __init__(
        self,
        name="wrist_pitch",
        dx_deg=15.0,
        motion_profile: str = "default",
        dt: float = DEFAULT_STEP_SLEEP,
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile, dt=dt)


class CommandWristRoll(CommandFeetechJoint):
    """Wrist Roll motion command class.

    Commands are in URDF/MJCF sign convention, which is mirrored relative to the
    robot's servo convention (see WRIST_ROLL_SIM_SIGN). Callers that port a
    robot-side sign - `_map_joint_space` - flip it; the IK mapping already works
    in URDF space and does not.
    """

    def __init__(
        self,
        name="wrist_roll",
        dx_deg=15.0,
        motion_profile: str = "default",
        dt: float = DEFAULT_STEP_SLEEP,
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile, dt=dt)


class CommandStretchGripperPosition:
    """Gripper motion command class. Only simple open and close methods are provided,
    and it is expected to be controlled on a button state.

    The robot commands the stretch_gripper in percent; the sim's gripper actuator takes
    an aperture angle in radians (roughly 0.0 closed to 1.0 fully open), so
    `gripper_step_rad` replaces the robot's `gripper_rotate_pct`.
    """

    def __init__(self, motion_profile: str = "max", gripper_step_rad: float = 0.07):
        self.name = "stretch_gripper"
        self.params = ROBOT_PARAMS[self.name]
        self.gripper_step_rad = gripper_step_rad
        self.gripper_accel = self.params["motion"][motion_profile]["accel"]
        self.gripper_vel = self.params["motion"][motion_profile]["vel"]
        self.precision_mode = 0.0
        self.stop_reqd = False

    def _get_subsystem(self, robot: "StretchMujocoSimulator"):
        # stretch_gripper and parallel_gripper are the same actuator in sim.
        return robot.end_of_arm.stretch_gripper

    def _move(self, dx_rad, robot: "StretchMujocoSimulator"):
        scale = 1.0 - 0.75 * self.precision_mode
        dx_rad = dx_rad * scale
        self._get_subsystem(robot).move_by(dx_rad, self.gripper_vel, self.gripper_accel)
        self.stop_reqd = True

    def open_gripper(self, robot: "StretchMujocoSimulator"):
        self._move(self.gripper_step_rad, robot)

    def close_gripper(self, robot: "StretchMujocoSimulator"):
        self._move(-self.gripper_step_rad, robot)

    def stop_gripper(self, robot: "StretchMujocoSimulator"):
        """The robot quick-stops the servo here. In sim the gripper actuator holds its
        setpoint on its own, and a zero move_by would drift it against a grasped object,
        so this only clears the flag."""
        if self.stop_reqd:
            self.stop_reqd = False
