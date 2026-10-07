import math
import time
from typing import TYPE_CHECKING

import click
import mujoco
import numpy as np

import stretch4_mujoco.utils as utils
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.safe_motions.safe_motion import SafeMotion

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer

WHEEL_ACTUATORS = (
    Actuators.left_wheel_vel,
    Actuators.right_wheel_vel,
    Actuators.back_wheel_vel,
)

SAFETY_ACTUATORS = (Actuators.lift, Actuators.arm)
"""The joints put into safety mode, matching `SafeMotionOvertiltAvoid` on the robot.

These are the two that move mass far from the base, so stopping them is what
keeps a lean from becoming a fall.
"""


class SafeMotionOvertiltAvoid(SafeMotion):
    """Stop the robot from tipping over once the base leans too far.

    The sim counterpart of
    `stretch4_body.behavior.safe_motions.safe_motion_overtilt_avoid`, and it
    takes the same two actions at the same 6 degree threshold:

    * the omni base goes to freewheel, so the wheels stop driving the robot
      further over and it is free to roll back down,
    * the lift and arm stop where they are, so the centre of mass stops moving
      out over the tipping edge.

    Motion is released once the base comes back under
    `gravity_tilt_release_deg`.
    """

    def __init__(self, mujoco_server: "MujocoServer"):
        super().__init__(name="safe_motion_overtilt_avoid", mujoco_server=mujoco_server)
        self.status = {"in_overtilt": False, "gravity_tilt": 0.0}
        self.ts_last_alert = 0.0

        self._wheel_actuator_ids = [
            actuator_id
            for actuator_id in (
                mujoco.mj_name2id(
                    mujoco_server.mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, wheel.name
                )
                for wheel in WHEEL_ACTUATORS
            )
            if actuator_id != -1
        ]
        # The wheels' own force limits, restored when the robot comes back level.
        self._wheel_forcerange: np.ndarray | None = None
        self._wheel_forcelimited: np.ndarray | None = None
        # Where each safetied joint came to rest, see `_enable_safety`.
        self._held_positions: dict[str, float] = {}

    @property
    def threshold_deg(self) -> float:
        return self.params["gravity_tilt_thresh_deg"]["default"]

    @property
    def release_deg(self) -> float:
        return self.params.get("gravity_tilt_release_deg", self.threshold_deg)

    def step(self) -> bool:
        was_in_overtilt = self.status["in_overtilt"]

        tilt_rad = utils.site_gravity_tilt(
            self.mujoco_server.mjmodel, self.mujoco_server.mjdata
        )
        self.status["gravity_tilt"] = tilt_rad

        # Hysteresis: a robot parked right on the threshold would otherwise
        # chatter in and out of the override at the control rate.
        limit_deg = self.release_deg if was_in_overtilt else self.threshold_deg
        in_overtilt = math.degrees(tilt_rad) > limit_deg
        self.status["in_overtilt"] = in_overtilt

        if in_overtilt:
            is_due = time.time() - self.ts_last_alert > self.params["alert_period"]
            if not was_in_overtilt or is_due:
                click.secho(
                    f"SafeMotionOvertiltAvoid triggered: "
                    f"gravity tilt {math.degrees(tilt_rad):.1f} deg",
                    fg="yellow",
                )
                self.ts_last_alert = time.time()
            self._enable_freewheel_base()
            for actuator in SAFETY_ACTUATORS:
                self._enable_safety(actuator.name)
        elif was_in_overtilt:
            self._disable_freewheel_base()
            self._held_positions.clear()
            click.secho(
                f"SafeMotionOvertiltAvoid released: "
                f"gravity tilt {math.degrees(tilt_rad):.1f} deg",
                fg="green",
            )

        return in_overtilt

    def _enable_freewheel_base(self) -> None:
        """Cut torque to the wheels, the sim's `omnibase.enable_freewheel_mode()`.

        Clamping the actuators' force to zero is the closest thing to a stepper
        in `MODE_FREEWHEEL`: the motors stop pushing, while joint damping and
        friction still act, so the base coasts rather than snapping to a halt.
        """
        if not self._wheel_actuator_ids:
            return

        mjmodel = self.mujoco_server.mjmodel
        if self._wheel_forcerange is None:
            self._wheel_forcerange = mjmodel.actuator_forcerange[
                self._wheel_actuator_ids
            ].copy()
            self._wheel_forcelimited = mjmodel.actuator_forcelimited[
                self._wheel_actuator_ids
            ].copy()

        mjmodel.actuator_forcerange[self._wheel_actuator_ids] = 0.0
        mjmodel.actuator_forcelimited[self._wheel_actuator_ids] = 1

        # Drop whatever the base was driving towards, so nothing resumes the
        # moment torque comes back.
        base_controller = self.mujoco_server.base_controller
        base_controller._clear_command(is_stop_motion=False)
        for profile in (
            base_controller.left_wheel_profile,
            base_controller.right_wheel_profile,
            base_controller.back_wheel_profile,
        ):
            profile.set_target_velocity(0.0)
        for actuator_id in self._wheel_actuator_ids:
            self.mujoco_server.mjdata.ctrl[actuator_id] = 0.0

    def _disable_freewheel_base(self) -> None:
        """Give the wheels their torque back, stopped rather than mid-command."""
        if self._wheel_forcerange is None:
            return

        mjmodel = self.mujoco_server.mjmodel
        mjmodel.actuator_forcerange[self._wheel_actuator_ids] = self._wheel_forcerange
        mjmodel.actuator_forcelimited[self._wheel_actuator_ids] = self._wheel_forcelimited
        self._wheel_forcerange = None
        self._wheel_forcelimited = None

        # Re-seed the wheel profiles off the wheels as they actually sit now;
        # they coasted while the base was freewheeling.
        self.mujoco_server.base_controller.profiles_initialized = False

    def _enable_safety(self, actuator_name: str) -> None:
        """Bring one joint to a controlled stop and hold it there.

        The sim's `enable_safety()`: the stepper's `MODE_SAFETY` decelerates the
        joint and holds position. That is a zero-velocity goal on the joint's
        motion profile, re-run here so it overrides the setpoint
        `push_command()` wrote earlier this cycle.

        Once the joint is down it is pinned to where it stopped rather than left
        on a zero-velocity goal. A caller that keeps re-commanding its original
        goal re-plans the profile every cycle, and each re-plan gets one tick of
        acceleration in before this override takes it back -- enough to walk the
        joint millimetres across the seconds a tilt lasts. On the robot there is
        nothing to walk: the stepper is latched in safety mode until something
        takes it out.
        """
        profile = self.mujoco_server.joint_profiles.get(actuator_name)
        if profile is None:
            return

        held = self._held_positions.get(actuator_name)
        if held is None:
            profile.set_target_velocity(0.0)
            held = profile.update(self.dt)
            if not profile.is_moving():
                self._held_positions[actuator_name] = held
        else:
            profile.set_position(held)

        self.mujoco_server.mjdata.actuator(actuator_name).ctrl = held
