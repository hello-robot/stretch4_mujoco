import math
import time
from typing import TYPE_CHECKING

import click

from stretch4_mujoco.safe_motions.motion_overrides import WHEEL_ACTUATORS
from stretch4_mujoco.safe_motions.safe_motion import SafeMotion

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer
    from stretch4_mujoco.safe_motions.motion_overrides import MotionOverrides
    from stretch4_mujoco.trapezoidal_profile import TrapezoidalProfile

WHEEL_NAMES = tuple(wheel.name for wheel in WHEEL_ACTUATORS)


class SafeMotionGuardedContact(SafeMotion):
    """Stop a joint that is driving into something.

    The sim counterpart of the steppers' guarded mode. On the robot the firmware
    watches motor current: once it passes `i_contact_pos`/`i_contact_neg` it
    raises `in_guarded_event` and drops the joint into safety mode. The base
    has a second layer on top, `SentryOmniBaseGuardedContact`, which freewheels
    the whole base as soon as any one wheel reports an event.

    The sim has no motor current, so effort stands in for it: an actuator's
    force as a fraction of its force range, which is what
    `Stepper.current_to_effort_pct` computes from current over `iMax`. The trip
    point is therefore `i_contact_a / i_max_a` of the force range, and the
    config carries the robot's own amp figures so the two stay in step.

    Without this the sim's position actuators drive at full stall force into
    whatever they hit -- enough to lever the robot over, since the arm needs
    only about 59 N at gripper height to beat the 50 N.m the robot's weight
    holds it down with.
    """

    def __init__(self, mujoco_server: "MujocoServer", overrides: "MotionOverrides"):
        super().__init__(
            name="safe_motion_guarded_contact",
            mujoco_server=mujoco_server,
            overrides=overrides,
        )
        self.status = {
            "guarded_events": 0,
            "in_guarded_event": {},
            "effort_pct": {},
            # Which way each tripped joint was being driven, so a caller can
            # tell which way out of the obstacle is.
            "trip_direction": {},
        }
        self.ts_last_alert = 0.0

        self._joint_params: dict[str, dict] = {}
        self._effort_pct: dict[str, float] = {}
        # Which way the joint was being driven when it tripped, so backing off
        # releases it but leaning harder does not.
        self._trip_direction: dict[str, int] = {}
        # The command counter as it stood at the trip, so a later command is
        # recognisable as new. See `MujocoServer.actuator_command_seq`.
        self._trip_command_seq: dict[str, int] = {}

        self._servo_params: dict[str, dict] = {}
        # When each servo first went over effort while stopped, in sim seconds.
        self._stall_since: dict[str, float | None] = {}

        for actuator_name, gains in self.params.get("joints", {}).items():
            if mujoco_server.joint_profiles.get(actuator_name) is None and (
                actuator_name not in WHEEL_NAMES
            ):
                continue
            self._joint_params[actuator_name] = gains
            self._register(actuator_name)

        for actuator_name, gains in self.params.get("servo_joints", {}).items():
            if mujoco_server.joint_profiles.get(actuator_name) is None:
                continue
            self._servo_params[actuator_name] = gains
            self._stall_since[actuator_name] = None
            self._register(actuator_name)

        self._base_in_guarded_event = False

    def _register(self, actuator_name: str) -> None:
        self._effort_pct[actuator_name] = 0.0
        self.status["in_guarded_event"][actuator_name] = False
        self.status["effort_pct"][actuator_name] = 0.0

    # -- thresholds --------------------------------------------------------

    @property
    def sensitivity(self) -> float:
        """Multiplier on the contact threshold, from the named profile.

        Mirrors `coeff_sensitivity_pos`/`_neg` in spirit: lower is more
        sensitive, i.e. the joint gives up sooner. The firmware's exact curve
        from coefficient to current is not in stretch4_body -- it lives on the
        stepper -- so this is a plain multiplier on `i_contact_a`, with
        `default` being the robot's documented contact current itself.
        """
        profile = self.params.get("sensitivity_profile", "default")
        return self.params.get("sensitivity", {}).get(profile, 1.0)

    def threshold_pct(self, actuator_name: str) -> float:
        """Effort, in percent of force range, at which contact is declared."""
        if actuator_name in self._servo_params:
            effort = self._servo_params[actuator_name]["stall_max_effort_pct"]
            return effort * self.sensitivity
        gains = self._joint_params[actuator_name]
        return 100.0 * (gains["i_contact_a"] / gains["i_max_a"]) * self.sensitivity

    def _is_driving(self, actuator_name: str, profile: "TrapezoidalProfile") -> int:
        """Which way the actuator is pushing: -1, 0 or +1.

        The lead of the setpoint over where the joint actually is, which is
        what the position controller turns into force. Taking the direction
        from here rather than from the caller's goal matters once a profile has
        run to completion against an obstacle: the setpoint has arrived, so the
        goal says "nothing left to do", while the joint is still metres of
        error behind it and leaning with everything it has.
        """
        error = profile.current_pos - self.mujoco_server._measured_position(actuator_name)
        if abs(error) <= self.params.get("drive_deadband", 0.001):
            return 0
        return 1 if error > 0 else -1

    def _is_stalled(self, actuator_name: str, effort_pct: float) -> bool:
        """The servos' guard: stopped, over effort, and stayed that way.

        Mirrors `feetech_SM_hello._unpack_status`. Timed on sim seconds rather
        than wall clock, so it behaves the same whether the sim runs faster or
        slower than real time.
        """
        gains = self._servo_params[actuator_name]
        velocity = abs(float(self.mujoco_server.mjdata.actuator(actuator_name).velocity[0]))
        stalled = velocity < gains["stall_min_vel"]
        over_effort = abs(effort_pct) > self.threshold_pct(actuator_name)

        if not (stalled and over_effort):
            self._stall_since[actuator_name] = None
            return False

        now = float(self.mujoco_server.mjdata.time)
        if self._stall_since[actuator_name] is None:
            self._stall_since[actuator_name] = now
        return now - self._stall_since[actuator_name] > gains["stall_max_time_s"]

    def _detect(self, actuator_name: str, direction: int, effort_pct: float) -> bool:
        """Whether this joint is in contact, by whichever guard it uses."""
        if direction == 0:
            return False
        if actuator_name in self._servo_params:
            return self._is_stalled(actuator_name, effort_pct)
        return direction * effort_pct > self.threshold_pct(actuator_name)

    # -- effort measurement ------------------------------------------------

    def _update_effort(self, actuator_name: str) -> float:
        """Low-pass the actuator's effort, as the firmware's `effort_LPF` does.

        A contact is a sustained push, not the single-cycle force spike the
        contact solver throws when two meshes first touch. Filtering is what
        keeps the solver's transients from reading as collisions.
        """
        mjdata = self.mujoco_server.mjdata
        mjmodel = self.mujoco_server.mjmodel
        actuator_id = mjdata.actuator(actuator_name).id
        force_limit = float(mjmodel.actuator_forcerange[actuator_id][1])
        if force_limit <= 0:
            return 0.0

        raw_pct = 100.0 * float(mjdata.actuator(actuator_name).force[0]) / force_limit

        cutoff_hz = self.params.get("effort_lpf_hz", 2.0)
        tau = 1.0 / (2.0 * math.pi * cutoff_hz) if cutoff_hz > 0 else 0.0
        alpha = self.dt / (tau + self.dt) if tau > 0 else 1.0
        filtered = self._effort_pct[actuator_name]
        filtered += alpha * (raw_pct - filtered)
        self._effort_pct[actuator_name] = filtered
        self.status["effort_pct"][actuator_name] = filtered
        return filtered

    # -- commanded direction -----------------------------------------------

    @staticmethod
    def _commanded_direction(profile: "TrapezoidalProfile") -> int:
        """Which way the caller is currently driving this joint: -1, 0 or +1."""
        if profile.mode == profile.VELOCITY:
            target = profile.target_vel
        else:
            target = profile.target_pos - profile.current_pos
        if abs(target) < 1e-9:
            return 0
        return 1 if target > 0 else -1

    # -- the cycle ---------------------------------------------------------

    def step(self) -> bool:
        triggered = False
        watched = list(self._joint_params) + list(self._servo_params)
        for actuator_name in watched:
            if actuator_name in WHEEL_NAMES:
                continue
            triggered |= self._step_joint(actuator_name)
        triggered |= self._step_base()
        return triggered

    def _step_joint(self, actuator_name: str) -> bool:
        profile = self.mujoco_server.joint_profiles.get(actuator_name)
        if profile is None:
            return False

        effort_pct = self._update_effort(actuator_name)
        # Read before the hold pins the profile, so both reflect the caller's
        # intent for this cycle rather than the override's.
        commanded = self._commanded_direction(profile)
        command_seq = self.mujoco_server.actuator_command_seq.get(actuator_name, 0)
        was_tripped = self.status["in_guarded_event"][actuator_name]

        if was_tripped:
            if command_seq == self._trip_command_seq[actuator_name]:
                # Nothing new commanded since the trip, so stay guarded.
                self.overrides.hold_joint(actuator_name, self.name)
                return True
            # A new command. Letting go, or driving back the way we came,
            # clears the guard.
            self._trip_command_seq[actuator_name] = command_seq
            if commanded == 0 or commanded == -self._trip_direction[actuator_name]:
                self._release_joint(actuator_name)
                return False
            # Still being driven into the obstacle. The robot would re-arm here
            # and let the joint have another go, but re-arming in sim means
            # handing the actuator back its full stall force for a cycle --
            # the very spike this exists to stop -- so the hold stands until
            # the caller stops or backs off.
            self.overrides.hold_joint(actuator_name, self.name)
            return True

        # A joint merely holding a load is not in contact; both of the robot's
        # guards are about driving into something.
        direction = self._is_driving(actuator_name, profile)
        if not self._detect(actuator_name, direction, effort_pct):
            return False

        self._trip_direction[actuator_name] = direction
        self._trip_command_seq[actuator_name] = command_seq
        self.status["trip_direction"][actuator_name] = direction
        self.status["in_guarded_event"][actuator_name] = True
        self.status["guarded_events"] += 1
        self.overrides.hold_joint(actuator_name, self.name)
        self._alert(f"{actuator_name} at {effort_pct:.0f}% effort "
                    f"(limit {self.threshold_pct(actuator_name):.0f}%)")
        return True

    def _release_joint(self, actuator_name: str) -> None:
        self.status["in_guarded_event"][actuator_name] = False
        self.status["trip_direction"].pop(actuator_name, None)
        self._trip_direction.pop(actuator_name, None)
        self._trip_command_seq.pop(actuator_name, None)
        if actuator_name in self._servo_params:
            self._stall_since[actuator_name] = None
        self.overrides.release_joint(actuator_name, self.name)

    def _step_base(self) -> bool:
        """Any wheel in contact freewheels the base, as the robot's sentry does."""
        wheels = [name for name in self._joint_params if name in WHEEL_NAMES]
        if not wheels:
            return False

        base_controller = self.mujoco_server.base_controller
        is_commanded = (
            base_controller.active_velocity is not None
            or base_controller.active_translate_x is not None
            or base_controller.active_translate_y is not None
            or base_controller.active_rotate is not None
        )

        over_threshold = False
        for name in wheels:
            effort_pct = self._update_effort(name)
            if abs(effort_pct) > self.threshold_pct(name):
                over_threshold = True

        if self._base_in_guarded_event:
            if not is_commanded:
                self._set_wheel_status(wheels, False)
                self._base_in_guarded_event = False
                self.overrides.release_base(self.name)
                return False
            # Keep the torque cut, but leave the caller's command in place --
            # it is what the release above reads.
            self.overrides.freewheel_base(self.name, clear_commands=False)
            return True

        if not (over_threshold and is_commanded):
            return False

        self._base_in_guarded_event = True
        self._set_wheel_status(wheels, True)
        self.status["guarded_events"] += 1
        self.overrides.freewheel_base(self.name, clear_commands=False)
        self._alert("base wheels in contact")
        return True

    def _set_wheel_status(self, wheels: list[str], value: bool) -> None:
        """Report every wheel as guarded, since the base freewheels as a unit.

        Once torque is cut the wheels read near-zero effort, so their own
        measurements stop being the thing that says contact is still live.
        """
        for name in wheels:
            self.status["in_guarded_event"][name] = value

    def _alert(self, detail: str) -> None:
        if time.time() - self.ts_last_alert <= self.params.get("alert_period", 2.0):
            return
        click.secho(f"SafeMotionGuardedContact triggered: {detail}", fg="yellow")
        self.ts_last_alert = time.time()
