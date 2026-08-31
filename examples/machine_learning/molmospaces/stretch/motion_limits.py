"""Hardware joint velocity/acceleration limits for the MolmoSpaces Stretch.

MolmoSpaces drives Stretch through position controllers whose output is written
straight into `mj_data.ctrl` (`Robot.compute_control()`), and MuJoCo puts no
velocity or acceleration limit on a position actuator -- it tracks whatever
setpoint it is given as hard as its gains and `forcerange` allow. So a policy
that commands a pose 40cm away gets it in a couple of control intervals, at
speeds the real lift or wrist could not reach: measured against the standalone
simulator, an unshaped lift move peaks around 2.3 m/s against the hardware's
0.3 m/s, and a wrist yaw move around 11.7 rad/s against 7.0.

`RateLimitedPositionController` wraps each of the robot's controllers and caps
their output's speed and acceleration at the numbers in `stretch4_mujoco.config`,
which are mirrored from stretch_body's `robot_params_SE4.py`. Because
`compute_control()` runs once per *control* step (`ctrl_dt_ms`, 2ms by default)
rather than once per policy step, the shaping happens at 500Hz and a single
policy action is spread over the ~33 control steps inside its 66ms tick.

The shaper is `SetpointRateLimiter` rather than the `TrapezoidalProfile` the
standalone simulator uses, and the difference matters: a controller here reissues
an absolute target every control step, so a profile that plans to a stop at its
current goal brakes into each of those samples instead of travelling through
them. See `SetpointRateLimiter`'s docstring for what that does to the gripper.

Policies see this as latency, not as a wall: they are closed loop against the
observed joint state, so a waypoint simply takes more steps to converge, and
`SimpleIKPolicyConfig.max_steps_per_waypoint` already budgets for slow
convergence. What it does mean is that a policy which assumed a commanded pose
was reached within one tick will now trail its command.
"""

import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.robots.robot_views.abstract import MoveGroup
from stretch4_mujoco import config
from stretch4_mujoco.trapezoidal_profile import SetpointRateLimiter

# The MJCF actuator behind each element of a move group's command vector, in the
# order `Stretch4RobotView` builds the group. The base is handled separately: its
# three DOFs are the virtual holonomic joints added by `add_robot_to_scene()`,
# which have no hardware counterpart to look up and take the omnibase limits.
#
# The gripper is deliberately absent, and is the one joint left unshaped here.
# Its limits do exist in `stretch4_mujoco.config` and the standalone simulator
# applies them; what makes them the wrong thing to apply *to a policy's finger
# command* is how such a command is built. The simple_ik expert commands
# `measured +- grasp_close_step_rad` (0.03 rad) every step, which is a force
# controller wearing a position command's clothes: at kp=500 that fixed offset is
# what holds grip force near 15Nm however rigid the object turns out to be.
# Shaping the setpoint attenuates the offset -- the setpoint spends the interval
# ramping towards `measured - 0.03` instead of sitting at it -- and with it the
# force, which measured out as the fingers closing 3.6x slower and the 'close'
# waypoint timing out with the object ungrasped.
#
# Nothing is lost by leaving it out. That same 0.03 rad per 66ms policy tick is
# 0.45 rad/s, against the 0.43 rad/s the hardware's servo manages across its
# range: the expert's grasp controller is already a rate limit at the hardware's
# own value, and there is nothing above it to cap.
MOVE_GROUP_ACTUATORS: dict[str, tuple[str, ...]] = {
    "lift": ("lift",),
    "arm": ("arm",),
    "wrist": ("wrist_yaw", "wrist_pitch", "wrist_roll"),
}


def move_group_limits(
    move_group_id: str, profile: str = config.DEFAULT_MOTION_PROFILE
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-DOF `(max_vel, max_accel, angular)` for one move group.

    `angular` marks DOFs that wrap, so a target the far side of +-pi is chased
    the short way round. Only the base yaw qualifies: it rides an unlimited hinge
    and `HoloJointsRobotBaseGroup.ctrl` already unwraps commands onto the
    revolution the base is currently on. The wrist joints are range-limited
    hinges whose travel exceeds pi (yaw spans -65..245 degrees), so wrapping them
    would fold a legitimate target back on itself.

    Returns:
        The three arrays, or `None` if this group has no recorded limits and
        should be driven unshaped.
    """
    if move_group_id == "base":
        (vel_xy, accel_xy), (vel_w, accel_w) = config.get_base_motion_limits(profile)
        return (
            np.array([vel_xy, vel_xy, vel_w]),
            np.array([accel_xy, accel_xy, accel_w]),
            np.array([False, False, True]),
        )

    actuators = MOVE_GROUP_ACTUATORS.get(move_group_id)
    if actuators is None:
        return None

    limits = [config.get_actuator_motion_limits(name, profile) for name in actuators]
    if any(limit is None for limit in limits):
        return None

    return (
        np.array([limit[0] for limit in limits]),
        np.array([limit[1] for limit in limits]),
        np.zeros(len(limits), dtype=bool),
    )


class RateLimitedPositionController(AbstractPositionController):
    """Wraps a position controller so its setpoint respects the joint's limits.

    Everything the robot asks of a controller -- `set_target`, `target`,
    `target_pos`, `stationary`, `set_to_stationary` -- is forwarded to the
    wrapped controller unchanged, so the action space, the noise model and the
    unnoised-command bookkeeping in `Robot._apply_action_noise_and_save_unnoised_cmd_jp`
    all see exactly what they saw before. Only `compute_ctrl_inputs()` differs.

    The ramp tracks the *commanded* setpoint rather than the measured joint
    position, which is what keeps grasping working: once the fingers stall on an
    object the setpoint carries on towards the commanded target, so position
    error -- and with it grip force -- builds as it did before.

    Three points resynchronise the ramp to reality rather than driving to it:
    `reset()`, any step on which the wrapped controller is `stationary`, and the
    first shaped step after a stationary spell. All three read the move group's
    measured `joint_pos`.

    The last of those is what makes an episode's base placement work. MolmoSpaces
    repositions a robot by writing `qpos` directly -- `robot_view.base.pose = ...`
    -- which leaves `ctrl` pointing at wherever the robot was before, and it does
    so *after* `Robot.reset()`. An unshaped controller survives that because the
    first thing it writes is an absolute target computed from the new pose. A
    ramp seeded at the old pose does not: it would command the base back through
    the origin at kp=25000 before crawling to the target at 0.3 m/s, which shows
    up as the robot lurching across the house on the first step of every episode
    and then timing out its waypoints metres short of the object.
    """

    def __init__(
        self,
        controller: AbstractPositionController,
        ctrl_dt: float,
        max_vel: np.ndarray,
        max_accel: np.ndarray,
        angular: np.ndarray | None = None,
    ) -> None:
        # Before `super().__init__`, which assigns `self.robot_move_group` and so
        # lands in the forwarding setter below.
        self._controller = controller
        super().__init__(controller.robot_move_group)
        self._ctrl_dt = ctrl_dt
        self._limiter = SetpointRateLimiter(max_vel, max_accel, angular)
        self._was_stationary = True
        self.reset()

    @property
    def robot_move_group(self) -> MoveGroup:
        return self._controller.robot_move_group

    @robot_move_group.setter
    def robot_move_group(self, move_group: MoveGroup) -> None:
        # `Controller.__init__` assigns this; forward it so the wrapper and the
        # wrapped controller cannot end up pointing at different groups.
        self._controller.robot_move_group = move_group

    @property
    def target(self):
        return self._controller.target

    @property
    def target_pos(self) -> np.ndarray:
        return self._controller.target_pos

    @property
    def stationary(self) -> bool:
        return self._controller.stationary

    def set_target(self, target) -> None:
        self._controller.set_target(target)

    def set_to_stationary(self) -> None:
        self._controller.set_to_stationary()

    def compute_ctrl_inputs(self) -> np.ndarray:
        target = np.asarray(self._controller.compute_ctrl_inputs(), dtype=float)
        if self._controller.stationary:
            self._limiter.reset(target)
            self._was_stationary = True
            return target
        if self._was_stationary:
            self._seed_from_robot()
            self._was_stationary = False
        return self._limiter.step(target, self._ctrl_dt)

    def reset(self) -> None:
        self._controller.reset()
        self._was_stationary = True
        self._seed_from_robot()

    def _seed_from_robot(self) -> None:
        """Start the ramp from where the joints actually are.

        `joint_pos` rather than `noop_ctrl`, because a `GripperGroup`'s noop is
        "hold fully open" or "hold fully closed" rather than its measured state.
        """
        self._limiter.reset(np.asarray(self.robot_move_group.joint_pos, dtype=float))


def rate_limited(
    controller: AbstractPositionController,
    move_group_id: str,
    ctrl_dt: float,
    profile: str = config.DEFAULT_MOTION_PROFILE,
) -> AbstractPositionController:
    """`controller`, wrapped if `move_group_id` has recorded limits."""
    import os
    if os.environ.get("STRETCH_DISABLE_RATE_LIMITS") == "1":
        return controller
    limits = move_group_limits(move_group_id, profile)
    if limits is None:
        return controller
    max_vel, max_accel, angular = limits
    return RateLimitedPositionController(controller, ctrl_dt, max_vel, max_accel, angular)
