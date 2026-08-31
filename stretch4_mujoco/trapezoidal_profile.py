import math
import time

import numpy as np


class TrapezoidalProfile:
    """
    Generates a trapezoidal velocity profile for a single joint.
    Mimics the behavior of a stepper motor driver.

    Two modes, both bounded by the same `max_vel`/`max_accel`:

    * velocity: `set_target_velocity()`, the position is whatever integrating
      the ramped velocity produces. Used for the wheels and for jogging.
    * position: `set_target_position()`, the profile accelerates towards the
      goal and decelerates into it. Used to shape the position actuators'
      setpoints, which MuJoCo would otherwise step to instantly.
    """

    VELOCITY = "velocity"
    POSITION = "position"

    def __init__(self, max_vel: float = 10.0, max_accel: float = 10.0, dt: float = 0.002):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.dt = dt

        self.current_pos = 0.0
        self.current_vel = 0.0
        self.target_vel = 0.0
        self.target_pos = 0.0
        self.mode = self.VELOCITY
        self.last_update_time = time.perf_counter()

    def set_target_velocity(self, target_vel: float):
        """
        Sets the target velocity for the profile.
        """
        self.mode = self.VELOCITY
        self.target_vel = max(-self.max_vel, min(self.max_vel, target_vel))

    def set_target_position(self, target_pos: float):
        """
        Sets the position the profile should drive to, and switches to position mode.
        """
        self.mode = self.POSITION
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
            self._update_position_target()

        # Calculate velocity error
        vel_error = self.target_vel - self.current_vel

        # Determine acceleration to apply
        if abs(vel_error) < 1e-6:
            accel = 0.0
        else:
            # Ramp velocity towards target
            accel_direction = 1.0 if vel_error > 0 else -1.0
            accel = accel_direction * self.max_accel

            # Don't overshoot target velocity in this step
            if abs(accel * dt) > abs(vel_error):
                accel = vel_error / dt

        # Update velocity
        self.current_vel += accel * dt

        # Clamp velocity (safety)
        self.current_vel = max(-self.max_vel, min(self.max_vel, self.current_vel))

        # Snap to zero if target is zero and we are close (prevents drift)
        if self.target_vel == 0.0 and abs(self.current_vel) < 1e-4:
            self.current_vel = 0.0

        # Update position
        previous_pos = self.current_pos
        self.current_pos += self.current_vel * dt

        if self.mode == self.POSITION:
            # Land exactly on the goal rather than dithering around it: with a
            # discrete step the deceleration ramp below can only get within
            # O(max_accel * dt^2) of it on its own.
            crossed = (self.target_pos - previous_pos) * (self.target_pos - self.current_pos) < 0
            if crossed or abs(self.target_pos - self.current_pos) < 1e-9:
                self.current_pos = self.target_pos
                self.current_vel = 0.0

        return self.current_pos

    def _update_position_target(self) -> None:
        """Pick the velocity that heads for `target_pos` and still stops on it.

        `sqrt(2 * a * distance)` is the fastest we can be going and still bleed
        off all of it within the distance left, so taking the smaller of that and
        `max_vel` gives the cruise-then-brake shape of a trapezoid without having
        to plan the whole move up front -- which matters because the goal can be
        moved on any step.
        """
        error = self.target_pos - self.current_pos
        if self.max_accel <= 0.0:
            braking_vel = self.max_vel
        else:
            braking_vel = math.sqrt(2.0 * self.max_accel * abs(error))
        self.target_vel = math.copysign(min(self.max_vel, braking_vel), error)

    @property
    def is_settled(self) -> bool:
        """Whether the profile has nothing left to do.

        A joint whose profile is still ramping has not finished moving even when
        it has not visibly moved yet -- a trapezoid starts from rest, so for the
        first tick of a move the position barely changes. Callers that decide
        "motion is over" from position stability need this, or they conclude a
        move is done before it has begun.
        """
        if self.mode == self.POSITION:
            return self.current_vel == 0.0 and self.current_pos == self.target_pos
        return self.current_vel == 0.0 and self.target_vel == 0.0

    def set_position(self, pos: float):
        """
        Hard reset of the position (e.g. for initialization).
        """
        self.current_pos = pos
        self.target_pos = pos


class SetpointRateLimiter:
    """Caps how fast, and how hard, a *stream* of position setpoints may move.

    For callers that re-issue an absolute target every step -- MolmoSpaces'
    controllers, which recompute one per control interval -- rather than the
    discrete goals `TrapezoidalProfile` is built for. The two cannot be swapped:
    a trapezoid plans to a full stop at whatever goal it currently has, so when
    the goal is a sample a few millimetres ahead it brakes into that sample, and
    a policy that derives its next target from the *measured* joint position (as
    the simple_ik expert's grasp controller does, at `measured + 0.03 rad` a
    step) never gets moving at all -- measured position and setpoint chase each
    other and the fingers crawl shut over hundreds of steps instead of the
    hardware's 1.2s.

    So this bounds the two things that were actually wrong -- top speed and how
    hard the setpoint may be made to speed up -- and leaves slowing down free:

        |v| <= max_vel,  d|v|/dt <= max_accel while speeding up

    Deceleration is unbounded, which shows up only at the end of a long move,
    where the setpoint lands on its goal rather than easing into it. The joint
    itself still decelerates under its own gains and `forcerange`, and the
    distance involved is one control interval's travel -- against the full-range
    step input this replaces, which is what made the joints too fast to begin
    with.

    Args:
        max_vel: per-joint velocity limit.
        max_accel: per-joint limit on how fast the setpoint may speed up.
        angular: per-joint flag marking a continuous revolute DOF. Targets for
            those are unwrapped onto the revolution the setpoint is currently on
            before the error is taken, so a target the far side of +-pi is chased
            the short way round.
    """

    def __init__(self, max_vel, max_accel, angular=None) -> None:
        self._max_vel = np.atleast_1d(np.asarray(max_vel, dtype=float))
        self._max_accel = np.atleast_1d(np.asarray(max_accel, dtype=float))
        if self._max_vel.shape != self._max_accel.shape:
            raise ValueError(
                f"max_vel {self._max_vel.shape} and max_accel "
                f"{self._max_accel.shape} must match."
            )

        if angular is None:
            self._angular = np.zeros(self._max_vel.shape, dtype=bool)
        else:
            self._angular = np.atleast_1d(np.asarray(angular, dtype=bool))
            if self._angular.shape != self._max_vel.shape:
                raise ValueError(
                    f"angular {self._angular.shape} must match "
                    f"max_vel {self._max_vel.shape}."
                )

        self._position = np.zeros(self._max_vel.shape)
        self._velocity = np.zeros(self._max_vel.shape)

    def __len__(self) -> int:
        return len(self._position)

    @property
    def position(self) -> np.ndarray:
        """The setpoint as it currently stands."""
        return self._position.copy()

    def reset(self, position) -> None:
        """Jump the setpoint to `position` and stop, with no ramp."""
        self._position = np.atleast_1d(np.asarray(position, dtype=float)).copy()
        self._velocity = np.zeros_like(self._position)

    def step(self, target, dt: float) -> np.ndarray:
        """Advance the setpoint one control interval towards `target`."""
        target = np.atleast_1d(np.asarray(target, dtype=float))

        error = target - self._position
        error = np.where(self._angular, _wrap_to_pi_array(error), error)

        # The velocity that would land on the target this step, capped.
        desired = np.clip(error / dt, -self._max_vel, self._max_vel)

        # Slowing down is free; speeding up, and reversing through zero, are not.
        slowing = (desired * self._velocity > 0) & (np.abs(desired) <= np.abs(self._velocity))
        ramped = self._velocity + np.clip(
            desired - self._velocity, -self._max_accel * dt, self._max_accel * dt
        )
        self._velocity = np.where(slowing, desired, ramped)

        # `desired` already cannot carry us past the target, and the ramp only
        # ever gives a velocity between the old one and `desired`, so this
        # advances at most to the target.
        self._position = self._position + self._velocity * dt
        return self._position.copy()


def _wrap_to_pi_array(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2 * math.pi) - math.pi
