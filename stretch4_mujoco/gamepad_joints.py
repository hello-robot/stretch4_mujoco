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
* The robot's per-tick `move_by(dx_deg)` jogs become `set_velocity()` here, at
  the speed that step implies over the robot's control period. The sim joints
  run trapezoidal profiles of their own, so a stream of position deltas would
  fight them -- see `CommandFeetechJoint`.
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
    """Abstract motion command class for Feetech joints.

    Jogs are issued as a velocity, not as a position delta per tick. Re-issuing
    `move_by(dx)` every tick -- which is what this used to do, back when the sim
    had no velocity profiles -- chains each delta off the *commanded* position, so
    the goal runs away from the joint at the full jog rate while the profile can
    only close on it at `sqrt(2 * accel * error)`. The two balance at a permanent
    lag of `rate**2 / (2 * accel)`, which for the wrists' 3.93 rad/s and
    7 rad/s**2 is 1.1 rad: the wrist trails the stick by 63 degrees and, because
    that travel is already committed to the goal, keeps going for another 1.1 rad
    after the stick is released.
    """

    def __init__(self, name, dx_deg, vel_type, acc_type):
        self.params = ROBOT_PARAMS[name]
        self.name = name
        self.dead_zone = 0.001
        self.dx_deg = dx_deg
        self.max_vel = self.params["motion"][vel_type]["vel"]
        self.acc = self.params["motion"][acc_type]["accel"]
        self.precision_mode = 0.0

    def _get_subsystem(self, robot: "StretchMujocoSimulator"):
        return getattr(robot.end_of_arm, self.name)

    def _move(self, dx_deg, robot: "StretchMujocoSimulator", velocity: float | None = None):
        scale = 1.0 - (0.95 * self.precision_mode)

        # `dx_deg` is a per-tick step sized for the robot's control period. Held
        # down it means a continuous jog, so turn it into the speed it implies
        # rather than re-issuing it as a position delta every tick -- see the
        # class docstring for why that distinction matters so much here.
        v_rad = deg_to_rad(dx_deg) * scale / DEFAULT_STEP_SLEEP

        cap = min(self.max_vel, velocity) if velocity is not None else self.max_vel
        v_rad = max(-cap, min(cap, v_rad))

        self._get_subsystem(robot).set_velocity(v_rad, self.acc)

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

        Braking at the `max` acceleration rather than the jog's own, which is what
        the robot does for every other joint: letting go of the stick should stop
        the wrist, not coast it to a halt.
        """
        self._get_subsystem(robot).set_velocity(
            0.0, self.params["motion"]["max"]["accel"]
        )


class CommandWristYaw(CommandFeetechJoint):
    """Wrist Yaw motion command class."""

    def __init__(
        self,
        name="wrist_yaw",
        dx_deg=15.0,
        motion_profile: str = "default",
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile)


class CommandWristPitch(CommandFeetechJoint):
    """Wrist Pitch motion command class."""

    def __init__(
        self,
        name="wrist_pitch",
        dx_deg=15.0,
        motion_profile: str = "default",
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile)


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
    ):
        super().__init__(name, dx_deg, motion_profile, motion_profile)


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
        # Same per-tick-step -> jog-speed conversion as CommandFeetechJoint, and
        # for the same reason: chained move_by deltas leave the fingers trailing
        # the button and still closing after it is let go.
        v_rad = dx_rad * scale / DEFAULT_STEP_SLEEP
        v_rad = max(-self.gripper_vel, min(self.gripper_vel, v_rad))
        self._get_subsystem(robot).set_velocity(v_rad, self.gripper_accel)
        self.stop_reqd = True

    def open_gripper(self, robot: "StretchMujocoSimulator"):
        self._move(self.gripper_step_rad, robot)

    def close_gripper(self, robot: "StretchMujocoSimulator"):
        self._move(-self.gripper_step_rad, robot)

    def stop_gripper(self, robot: "StretchMujocoSimulator"):
        """The robot quick-stops the servo here.

        A zero jog holds the commanded aperture where it is, so a grasp keeps its
        squeeze: the profile stops advancing the setpoint but does not give any of
        it back, which is what keeps position error -- and grip force -- built up.
        """
        if self.stop_reqd:
            self._get_subsystem(robot).set_velocity(0.0, self.gripper_accel)
            self.stop_reqd = False
