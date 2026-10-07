import math
import time
from typing import TYPE_CHECKING

import click

import stretch4_mujoco.utils as utils
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.safe_motions.safe_motion import SafeMotion

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer
    from stretch4_mujoco.safe_motions.motion_overrides import MotionOverrides

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

    def __init__(self, mujoco_server: "MujocoServer", overrides: "MotionOverrides"):
        super().__init__(
            name="safe_motion_overtilt_avoid",
            mujoco_server=mujoco_server,
            overrides=overrides,
        )
        self.status = {"in_overtilt": False, "gravity_tilt": 0.0}
        self.ts_last_alert = 0.0

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
            self.overrides.freewheel_base(self.name)
            for actuator in SAFETY_ACTUATORS:
                self.overrides.hold_joint(actuator.name, self.name)
        elif was_in_overtilt:
            self.overrides.release_all(self.name)
            click.secho(
                f"SafeMotionOvertiltAvoid released: "
                f"gravity tilt {math.degrees(tilt_rad):.1f} deg",
                fg="green",
            )

        return in_overtilt
