from typing import TYPE_CHECKING

import mujoco
import numpy as np

from stretch4_mujoco.enums.actuators import Actuators

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer

WHEEL_ACTUATORS = (
    Actuators.left_wheel_vel,
    Actuators.right_wheel_vel,
    Actuators.back_wheel_vel,
)


class MotionOverrides:
    """The actions a safe motion can take on the actuators, shared between them.

    One instance per server, handed to every safe motion, because two of them can
    want the same joint stopped at the same time -- an overtilted robot whose
    gripper is also jammed against a table. Each request carries an `owner`, and
    an override lifts only once every owner has let go. Without that, whichever
    safe motion recovered first would hand the joint back while the other still
    needed it held.
    """

    def __init__(self, mujoco_server: "MujocoServer"):
        self.mujoco_server = mujoco_server

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
        # The wheels' own force limits, put back when freewheel lifts.
        self._wheel_forcerange: np.ndarray | None = None
        self._wheel_forcelimited: np.ndarray | None = None

        self._freewheel_owners: set[str] = set()
        self._hold_owners: dict[str, set[str]] = {}
        # Where each held joint came to rest, see `hold_joint`.
        self._held_positions: dict[str, float] = {}

    # -- base freewheel ----------------------------------------------------

    @property
    def is_freewheeling(self) -> bool:
        return bool(self._freewheel_owners)

    def freewheel_base(self, owner: str, clear_commands: bool = True) -> None:
        """Cut torque to the wheels, the sim's `omnibase.enable_freewheel_mode()`.

        Clamping the actuators' force to zero is the closest thing to a stepper
        in `MODE_FREEWHEEL`: the motors stop pushing, while joint damping and
        friction still act, so the base coasts rather than snapping to a halt.

        Args:
            owner: who is asking; freewheel lifts when every owner releases.
            clear_commands: also drop whatever the base was driving towards, so
                nothing resumes the moment torque comes back. An overtilting
                robot wants that. A guarded contact does not -- the caller
                leaning on the stick is how it knows contact is still being
                pushed into, and the release check reads that command.
        """
        if not self._wheel_actuator_ids:
            return
        self._freewheel_owners.add(owner)

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

        base_controller = self.mujoco_server.base_controller
        if clear_commands:
            base_controller._clear_command(is_stop_motion=False)
            for profile in self._wheel_profiles():
                profile.set_target_velocity(0.0)
        for actuator_id in self._wheel_actuator_ids:
            self.mujoco_server.mjdata.ctrl[actuator_id] = 0.0

    def release_base(self, owner: str) -> None:
        """Give the wheels their torque back, once nobody still wants them cut."""
        self._freewheel_owners.discard(owner)
        if self._freewheel_owners or self._wheel_forcerange is None:
            return

        mjmodel = self.mujoco_server.mjmodel
        mjmodel.actuator_forcerange[self._wheel_actuator_ids] = self._wheel_forcerange
        mjmodel.actuator_forcelimited[self._wheel_actuator_ids] = self._wheel_forcelimited
        self._wheel_forcerange = None
        self._wheel_forcelimited = None

        # Re-seed the wheel profiles off the wheels as they actually sit now;
        # they coasted while the base was freewheeling.
        self.mujoco_server.base_controller.profiles_initialized = False

    def _wheel_profiles(self):
        base_controller = self.mujoco_server.base_controller
        return (
            base_controller.left_wheel_profile,
            base_controller.right_wheel_profile,
            base_controller.back_wheel_profile,
        )

    # -- joint hold --------------------------------------------------------

    def is_holding(self, actuator_name: str) -> bool:
        return bool(self._hold_owners.get(actuator_name))

    def hold_joint(self, actuator_name: str, owner: str) -> None:
        """Bring one joint to a controlled stop and hold it there.

        The sim's `enable_safety()`: the stepper's `MODE_SAFETY` decelerates the
        joint and holds position. That is a zero-velocity goal on the joint's
        motion profile, re-run here so it overrides the setpoint
        `push_command()` wrote earlier this cycle. Call it every control cycle
        for as long as the hold should last.

        The hold parks the joint at the position it has actually reached, not at
        the setpoint it was chasing. That distinction is the whole point when a
        joint is stopped because it ran into something: a blocked joint's
        setpoint keeps advancing past where the joint physically is, and with
        the lift's kp of 8000 even 24 mm of that lead is 192 N. Holding the
        setpoint would therefore keep leaning on the obstacle with nearly full
        force while reporting the joint as stopped. Holding the measured
        position instead takes the position error -- and so the force -- to
        zero, which is what the stepper's `MODE_SAFETY` does on the robot.

        Once latched the joint stays pinned rather than being left on a
        zero-velocity goal. A caller that keeps re-commanding its original goal
        re-plans the profile every cycle, and each re-plan gets one tick of
        acceleration in before this override takes it back -- enough to walk
        the joint millimetres across the seconds a hold lasts. On the robot
        there is nothing to walk: the stepper is latched in safety mode until
        something takes it out.
        """
        profile = self.mujoco_server.joint_profiles.get(actuator_name)
        if profile is None:
            return
        self._hold_owners.setdefault(actuator_name, set()).add(owner)

        held = self._held_positions.get(actuator_name)
        if held is None:
            held = self.mujoco_server._measured_position(actuator_name)
            self._held_positions[actuator_name] = held
        profile.set_position(held)

        self.mujoco_server.mjdata.actuator(actuator_name).ctrl = held

    def release_joint(self, actuator_name: str, owner: str) -> None:
        """Let one joint move again, once nobody still wants it held."""
        owners = self._hold_owners.get(actuator_name)
        if not owners:
            return
        owners.discard(owner)
        if owners:
            return
        del self._hold_owners[actuator_name]
        self._held_positions.pop(actuator_name, None)

    def release_all(self, owner: str) -> None:
        """Drop every hold `owner` has, and its claim on freewheel."""
        for actuator_name in list(self._hold_owners):
            self.release_joint(actuator_name, owner)
        self.release_base(owner)
