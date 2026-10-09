"""Trapezoidal motion profiles, ported from the Stretch stepper firmware.

A sim move should take as long as the same move on hardware and follow the same
velocity curve while it does, so rather than approximate the shape,
`TrapezoidalProfile` is a transcription of the two generators that actually run
on the robot's stepper boards.
"""

import math
import time

import numpy as np


class TrapezoidalProfile:
    """Generates a trapezoidal velocity profile for a single joint.

    Two modes, both bounded by the same `max_vel`/`max_accel`:

    * velocity: `set_target_velocity()`, the position is whatever integrating
      the ramped velocity produces. Used for the wheels and for jogging.
    * position: `set_target_position()`, the profile plans a full
      brake/accelerate/cruise/decelerate move and evaluates it in closed form.
      Used to shape the position actuators' setpoints, which MuJoCo would
      otherwise step to instantly.

    The plan is recomputed whenever the goal moves, from wherever the profile
    has got to, exactly as the firmware does on a new `x_des` -- so a stream of
    setpoints is followed as well as a single discrete goal.
    """

    VELOCITY = "velocity"
    POSITION = "position"

    def __init__(self, max_vel: float = 10.0, max_accel: float = 10.0, dt: float = 0.002):
        self._max_vel = abs(max_vel)
        self._max_accel = abs(max_accel)
        self.dt = dt

        # MotionGenerator state. `current_pos`/`current_vel`/`current_accel` are
        # the firmware's pos/vel/acc, kept under the names the sim already uses.
        self.current_pos = 0.0
        self.current_vel = 0.0
        self.current_accel = 0.0
        self.old_pos = 0.0
        self.old_pos_ref = 0.0
        self.old_vel = 0.0

        self.d_brk = 0.0
        self.d_acc = 0.0
        self.d_vel = 0.0
        self.d_dec = 0.0
        self.d_tot = 0.0

        self.t_brk = 0.0
        self.t_acc = 0.0
        self.t_vel = 0.0
        self.t_dec = 0.0

        self.t = 0.0
        self.vel_st = 0.0
        self.sign_m = 1  # 1 = positive change, -1 = negative change
        self.sign_m_acc = 1
        self.shape = True  # True = trapezoidal, False = triangular
        self.is_finished = False
        self.force_recalc = False

        self.target_pos = 0.0
        self.target_vel = 0.0
        self.mode = self.VELOCITY
        self.last_update_time = time.perf_counter()

    # -- MotionGenerator::setMaxVelocity / setMaxAcceleration ----------------

    @property
    def max_vel(self) -> float:
        return self._max_vel

    @max_vel.setter
    def max_vel(self, value: float) -> None:
        value = abs(value)
        if value != self._max_vel:
            self._max_vel = value
            self.force_recalc = True

    @property
    def max_accel(self) -> float:
        return self._max_accel

    @max_accel.setter
    def max_accel(self, value: float) -> None:
        value = abs(value)
        if value != self._max_accel:
            self._max_accel = value
            self.force_recalc = True

    @staticmethod
    def _sign(value: float) -> int:
        """`MotionGenerator::sign` -- note it returns 0 for 0, not 1."""
        if value < 0:
            return -1
        if value > 0:
            return 1
        return 0

    def _safe_switch_on(self, pos: float, vel: float) -> None:
        """`MotionGenerator::safe_switch_on` -- adopt a state and replan onto it.

        The firmware calls this on every mode change so the new generator picks up
        the joint's measured position and velocity instead of resuming a plan made
        before the switch.
        """
        self.current_pos = pos
        self.current_vel = vel
        self.current_accel = 0.0
        self.force_recalc = True

    def set_target_velocity(self, target_vel: float):
        """
        Sets the target velocity for the profile.
        """
        if self.mode != self.VELOCITY:
            self.mode = self.VELOCITY
            # `vg.safe_switch_on(ywd, v_pll)` on the mode change: carry the
            # position and velocity across so a jog out of a move is continuous.
            self._safe_switch_on(self.current_pos, self.current_vel)
        self.target_vel = max(-self.max_vel, min(self.max_vel, target_vel))

    def set_target_position(self, target_pos: float):
        """
        Sets the position the profile should drive to, and switches to position mode.
        """
        if self.mode != self.POSITION:
            self.mode = self.POSITION
            # `mg.safe_switch_on(yw, v_pll)` on the mode change.
            self._safe_switch_on(self.current_pos, self.current_vel)
            self.is_finished = False
        self.target_pos = target_pos

    def update(self, dt: float | None = None) -> float:
        """
        Updates the profile state and returns the new position.
        Call this every simulation step.
        """
        if dt is None:
            now = time.perf_counter()
            dt = now - self.last_update_time
            self.last_update_time = now

        if self.mode == self.POSITION:
            return self._update_position(dt)
        return self._update_velocity(dt)

    def _update_velocity(self, dt: float) -> float:
        """`VelocityGenerator::update_relative` -- ramp to `target_vel`, integrate.

        The min/max land exactly on the target rather than dithering around it,
        so no epsilon snapping is needed to bring the joint to a true stop.
        """
        if self.current_vel < self.target_vel:  # accel up to desired vel
            self.current_vel = min(self.target_vel, self.current_vel + self.max_accel * dt)
        elif self.current_vel > self.target_vel:  # decel down to desired vel
            self.current_vel = max(self.target_vel, self.current_vel - self.max_accel * dt)

        self.current_accel = 0.0 if self.current_vel == self.target_vel else self.max_accel
        self.current_pos += self.current_vel * dt
        return self.current_pos

    def _update_position(self, dt: float) -> float:
        """`MotionGenerator::update` -- replan on a new goal, then evaluate at `t`.

        The whole move is laid out up front (how long to brake, accelerate,
        cruise and decelerate, and how far each phase covers) and then sampled in
        closed form. That is what makes the deceleration land exactly on the goal
        at exactly the planned time; a feedback shaper that picks
        `sqrt(2 * a * distance)` every step only approaches it asymptotically and
        drifts from hardware's timing.
        """
        pos_ref = self.target_pos

        if self.old_pos_ref != pos_ref or self.force_recalc:  # reference changed
            self.is_finished = False
            self.force_recalc = False

            # Shift state variables
            self.old_pos_ref = pos_ref
            self.old_pos = self.current_pos
            self.old_vel = self.current_vel
            self.t = 0.0

            # Calculate braking time and distance (in case is needed)
            self.t_brk = abs(self.old_vel) / self.max_accel
            self.d_brk = self.t_brk * abs(self.old_vel) / 2

            # Calculate sign of motion
            self.sign_m = self._sign(
                pos_ref - (self.old_pos + self._sign(self.old_vel) * self.d_brk)
            )
            self.sign_m_acc = self.sign_m

            if self.sign_m != self._sign(self.old_vel):  # means brake is needed
                self.t_acc = self.max_vel / self.max_accel
                self.d_acc = self.t_acc * (self.max_vel / 2)
            else:
                self.t_brk = 0.0
                self.d_brk = 0.0
                self.t_acc = abs(self.max_vel - abs(self.old_vel)) / self.max_accel
                self.d_acc = self.t_acc * (self.max_vel + abs(self.old_vel)) / 2
                if self.max_vel < abs(self.old_vel):  # need to decel in accel phase
                    self.sign_m_acc = self.sign_m_acc * -1

            # Calculate total distance to go after braking
            self.d_tot = abs(pos_ref - self.old_pos + self.sign_m * self.d_brk)

            self.t_dec = self.max_vel / self.max_accel
            self.d_dec = self.t_dec * self.max_vel / 2
            self.d_vel = self.d_tot - (self.d_acc + self.d_dec)
            self.t_vel = self.d_vel / self.max_vel

            if self.t_vel > 0:  # trapezoidal shape
                self.shape = True
            else:  # triangular shape
                self.shape = False
                # Recalculate distances and periods
                if self.sign_m != self._sign(self.old_vel):  # means brake is needed
                    self.vel_st = math.sqrt(self.max_accel * self.d_tot)
                    self.t_acc = self.vel_st / self.max_accel
                    self.d_acc = self.t_acc * (self.vel_st / 2)
                else:
                    self.t_brk = 0.0
                    self.d_brk = 0.0
                    self.d_tot = abs(pos_ref - self.old_pos)  # recalculate total distance
                    self.vel_st = math.sqrt(
                        (self.old_vel * self.old_vel) / 2 + self.max_accel * self.d_tot
                    )
                    self.t_acc = (self.vel_st - abs(self.old_vel)) / self.max_accel
                    self.d_acc = self.t_acc * (self.vel_st + abs(self.old_vel)) / 2
                self.t_dec = self.vel_st / self.max_accel
                self.d_dec = self.t_dec * self.vel_st / 2

        self.t = self.t + dt
        self._calculate_trapezoidal_profile(pos_ref)

        return self.current_pos

    def _calculate_trapezoidal_profile(self, pos_ref: float) -> None:
        """`MotionGenerator::calculateTrapezoidalProfile` -- sample the plan at `t`."""
        t = self.t
        max_vel = self.max_vel
        max_accel = self.max_accel
        t_brk, t_acc, t_vel, t_dec = self.t_brk, self.t_acc, self.t_vel, self.t_dec

        if self.shape:  # trapezoidal shape
            if t <= (t_brk + t_acc):
                self.current_pos = (
                    self.old_pos + self.old_vel * t + self.sign_m_acc * (max_accel / 2) * t * t
                )
                self.current_vel = self.old_vel + self.sign_m_acc * max_accel * t
                self.current_accel = self.sign_m_acc * max_accel
            elif t > (t_brk + t_acc) and t < (t_brk + t_acc + t_vel):
                self.current_pos = self.old_pos + self.sign_m * (
                    -self.d_brk + self.d_acc + max_vel * (t - t_brk - t_acc)
                )
                self.current_vel = self.sign_m * max_vel
                self.current_accel = 0.0
            elif t >= (t_brk + t_acc + t_vel) and t < (t_brk + t_acc + t_vel + t_dec):
                dt_dec = t - t_brk - t_acc - t_vel
                self.current_pos = self.old_pos + self.sign_m * (
                    -self.d_brk
                    + self.d_acc
                    + self.d_vel
                    + max_vel * dt_dec
                    - (max_accel / 2) * dt_dec * dt_dec
                )
                self.current_vel = self.sign_m * (max_vel - max_accel * dt_dec)
                self.current_accel = -self.sign_m * max_accel
            else:
                self.current_pos = pos_ref
                self.current_vel = 0.0
                self.current_accel = 0.0
                self.is_finished = True
        else:  # triangular shape
            if t <= (t_brk + t_acc):
                # NB: the firmware uses sign_m here where the trapezoidal branch
                # above uses sign_m_acc. Kept as-is so the two match.
                self.current_pos = (
                    self.old_pos + self.old_vel * t + self.sign_m * (max_accel / 2) * t * t
                )
                self.current_vel = self.old_vel + self.sign_m * max_accel * t
                self.current_accel = self.sign_m * max_accel
            elif t > (t_brk + t_acc) and t < (t_brk + t_acc + t_dec):
                dt_dec = t - t_brk - t_acc
                self.current_pos = self.old_pos + self.sign_m * (
                    -self.d_brk
                    + self.d_acc
                    + self.vel_st * dt_dec
                    - (max_accel / 2) * dt_dec * dt_dec
                )
                self.current_vel = self.sign_m * (self.vel_st - max_accel * dt_dec)
                self.current_accel = -self.sign_m * max_accel
            else:
                self.current_pos = pos_ref
                self.current_vel = 0.0
                self.current_accel = 0.0
                self.is_finished = True

    def is_accelerating(self) -> bool:
        """`MotionGenerator::isAccelerating` / `VelocityGenerator::isAccelerating`."""
        if self.mode == self.POSITION:
            return self.current_accel != 0
        return self.current_vel != self.target_vel

    def is_moving(self) -> bool:
        """`isMoving` in both generators."""
        return self.current_vel != 0

    @property
    def is_settled(self) -> bool:
        """Whether the profile has nothing left to do.

        A joint whose profile is still ramping has not finished moving even when
        it has not visibly moved yet -- a trapezoid starts from rest, so for the
        first tick of a move the position barely changes. Callers that decide
        "motion is over" from position stability need this, or they conclude a
        move is done before it has begun.

        The goal-reached test is exact because the firmware's final branch
        assigns `pos = posRef` outright rather than integrating into it.
        """
        if self.mode == self.POSITION:
            return (
                self.is_finished
                and self.current_vel == 0.0
                and self.current_pos == self.target_pos
            )
        return self.current_vel == 0.0 and self.target_vel == 0.0

    def set_position(self, pos: float):
        """Hard reset of the position (e.g. for initialization).

        `MotionGenerator::safe_switch_on(pos, 0)` plus a goal of `pos`, so the
        profile lands stopped and finished exactly here with nothing pending.
        """
        self._safe_switch_on(pos, 0.0)
        self.old_pos = pos
        self.old_pos_ref = pos
        self.old_vel = 0.0
        self.t = 0.0
        self.target_pos = pos
        self.is_finished = True


class TrapezoidalSetpointLimiter:
    """Shapes a *stream* of position setpoints so they respect vel/accel limits.
    Args:
        max_vel: per-joint velocity limit.
        max_accel: per-joint acceleration limit.
        angular: per-joint flag marking a continuous revolute DOF. Targets for
            those are unwrapped onto the revolution the setpoint is currently on
            before the error is taken, so a target the far side of +-pi is chased
            the short way round. Only ever the base yaw here -- the wrist joints
            are range-limited hinges whose travel exceeds pi, so wrapping them
            would fold a legitimate target back on itself.
    """

    def __init__(self, max_vel, max_accel, angular=None) -> None:
        max_vel = np.atleast_1d(np.asarray(max_vel, dtype=float))
        max_accel = np.atleast_1d(np.asarray(max_accel, dtype=float))
        if max_vel.shape != max_accel.shape:
            raise ValueError(
                f"max_vel {max_vel.shape} and max_accel {max_accel.shape} must match."
            )

        self._profiles = [
            TrapezoidalProfile(max_vel=v, max_accel=a)
            for v, a in zip(max_vel, max_accel, strict=True)
        ]
        if angular is None:
            self._angular = np.zeros(max_vel.shape, dtype=bool)
        else:
            self._angular = np.atleast_1d(np.asarray(angular, dtype=bool))
            if self._angular.shape != max_vel.shape:
                raise ValueError(
                    f"angular {self._angular.shape} must match max_vel {max_vel.shape}."
                )

    def __len__(self) -> int:
        return len(self._profiles)

    @property
    def position(self) -> np.ndarray:
        """The setpoint as it currently stands."""
        return np.array([p.current_pos for p in self._profiles])

    def reset(self, position) -> None:
        """Jump the setpoint to `position` and stop, with no ramp."""
        for profile, value in zip(
            self._profiles, np.atleast_1d(np.asarray(position, dtype=float)), strict=True
        ):
            profile.set_position(float(value))
            profile.current_vel = 0.0

    def step(self, target, dt: float) -> np.ndarray:
        """Advance the setpoint one control interval towards `target`."""
        target = np.atleast_1d(np.asarray(target, dtype=float))
        shaped = np.empty(len(self._profiles))
        for i, profile in enumerate(self._profiles):
            goal = float(target[i])
            if self._angular[i]:
                goal = profile.current_pos + _wrap_to_pi(goal - profile.current_pos)
            profile.set_target_position(goal)
            shaped[i] = profile.update(dt)
        return shaped


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
